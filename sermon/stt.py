"""Local speech-to-text.

Only one implementation in Phase 1 (faster-whisper on CPU), but it sits behind
a tiny interface so a different engine can be dropped in later without touching
the pipeline.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import STTConfig

# Whisper is known to emit these when fed near-silence or noise. Real sermon
# content that happens to match is a smaller cost than the false positives.
# Match is against the WHOLE output (case-insensitive, punctuation stripped);
# any single word that also matches a stripped form is filtered.
_HALLUCINATIONS_RAW = [
    "", ".", "you", "so", "okay", "yeah", "uh", "um", "mm",
    "thank you", "thanks for watching", "please subscribe", "[blank_audio]",
    "bye", "goodbye", "see you soon", "see you next time", "see you",
    "subtitles by the amara.org community",
    "amen", "hallelujah", "praise the lord", "praise god",
    "god bless you", "god bless", "jesus christ", "jesus", "lord",
    "yes sir", "yes ma'am", "yes",
    # counting sequences and single letters Whisper emits on noise
    "one two", "one two three", "one two one two", "one two one",
    "one", "two", "three", "a", "the",
    "tell me", "tell me that", "let's go", "let's go now",
    "see the wind", "somebody stop",
]


def _normalize(s: str) -> str:
    """Lowercase, strip punctuation/whitespace/exclamations for match."""
    import re as _re
    return _re.sub(r"[\s\.\,\!\?\;\:\-]+", " ", s.lower()).strip()


_HALLUCINATIONS = {_normalize(h) for h in _HALLUCINATIONS_RAW}


def _is_hallucination(text: str) -> bool:
    """True if the utterance is a known Whisper silence-hallucination.

    Also catches repetitive echoes like 'god bless you god bless you god bless
    you' - Whisper's rerun loop on noise - by collapsing repeated phrases.
    """
    n = _normalize(text)
    if not n or n in _HALLUCINATIONS:
        return True
    words = n.split()
    if len(words) < 3:
        return False
    # High word-repetition ratio => almost certainly a Whisper loop
    # (e.g. "one two one two one two one two").
    if len(set(words)) <= max(2, len(words) // 3):
        return True
    # A short phrase (2-3 words) that occupies most of the text via repetition.
    # Catches "see the wind see the wind see" and "god bless you god bless you
    # god bless you" - the actual signature of a stuck decoder, whether or not
    # the specific words are in the hallucination list.
    for chunk in (2, 3, 4):
        if len(words) < chunk * 2:
            continue
        first = tuple(words[:chunk])
        occurrences = sum(1 for i in range(0, len(words) - chunk + 1, chunk)
                          if tuple(words[i:i + chunk]) == first)
        if occurrences >= 2 and occurrences * chunk >= 0.6 * len(words):
            return True
    return False


@dataclass
class TranscriptionResult:
    text: str
    audio_seconds: float
    compute_seconds: float

    @property
    def rtf(self) -> float:
        """Real-time factor. < 1.0 means faster than real time."""
        return self.compute_seconds / max(self.audio_seconds, 1e-6)


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        ...


class ParakeetTranscriber(Transcriber):
    """NVIDIA Parakeet (FastConformer transducer) via sherpa-onnx.

    Runs on the same onnxruntime we already ship. Roughly 4x the throughput
    of Whisper base.en on this class of CPU because it's non-autoregressive:
    tokens are emitted by the joiner in parallel with the encoder rather
    than one at a time.

    Two variants:
      - 110M ("parakeet-110m") : ~180 MB disk, ~500 MB RSS, fastest
      - 0.6B ("parakeet-0.6b") : ~630 MB disk, ~1.5 GB RSS, best accuracy

    Trade-offs vs faster-whisper:
      * NO initial_prompt / vocabulary.txt biasing. Whatever the model was
        trained on is what it recognises; unknown words go through as
        acoustically-nearest guess.
      * NO no_speech_threshold / avg_logprob per-segment gates. Silero VAD
        gate is applied here as a pre-filter instead - same idea, different
        implementation.
      * Slightly less English coverage on rare vocabulary. Trained on
        LibriSpeech-scale English so it's excellent on standard prose;
        struggles with named entities as much as Whisper does.
    """

    PROMPT_TOKEN_LIMIT = 0     # no prompt mechanism

    def __init__(self, model_dir: str | Path, cfg: STTConfig):
        import sherpa_onnx

        self.cfg = cfg
        self.model_dir = Path(model_dir)
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(self.model_dir / "encoder.int8.onnx"),
            decoder=str(self.model_dir / "decoder.int8.onnx"),
            joiner=str(self.model_dir / "joiner.int8.onnx"),
            tokens=str(self.model_dir / "tokens.txt"),
            model_type="nemo_transducer",
            num_threads=cfg.cpu_threads,
            sample_rate=16000,
            feature_dim=80,
        )
        # Load Silero VAD as a pre-filter. Same job as the vad_filter=True in
        # faster-whisper - reject non-speech buffers so the recogniser never
        # even runs on room noise.
        vad_model = _download_silero()
        vad_conf = sherpa_onnx.VadModelConfig()
        vad_conf.silero_vad.model = str(vad_model)
        vad_conf.silero_vad.threshold = 0.6
        vad_conf.silero_vad.min_silence_duration = 0.5
        vad_conf.silero_vad.min_speech_duration = 0.5
        vad_conf.sample_rate = 16000
        vad_conf.provider = "cpu"
        vad_conf.num_threads = 1
        self.vad = sherpa_onnx.VoiceActivityDetector(vad_conf, buffer_size_in_seconds=30)

    def prompt_tokens(self) -> int:
        return 0

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        started = time.perf_counter()
        if sample_rate != 16000:
            # match Whisper interface but the mic source is 16 k already
            raise ValueError("Parakeet expects 16 kHz input")

        # Silero VAD: keep only the segments it flags as speech. Rejects a
        # buffer of noise before recognition is even attempted.
        self.vad.reset()
        # feed audio in ~30 ms chunks (480 samples) - sherpa-onnx's expected shape
        block = 512
        speech_chunks: list[np.ndarray] = []
        for i in range(0, len(audio), block):
            self.vad.accept_waveform(audio[i:i + block].astype(np.float32))
        # flush trailing state
        self.vad.flush()
        while not self.vad.empty():
            seg = self.vad.front
            speech_chunks.append(np.asarray(seg.samples, dtype=np.float32))
            self.vad.pop()

        if not speech_chunks:
            return TranscriptionResult(
                text="",
                audio_seconds=len(audio) / sample_rate,
                compute_seconds=time.perf_counter() - started,
            )

        speech = np.concatenate(speech_chunks)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=speech.tolist())
        self.recognizer.decode_stream(stream)
        text = stream.result.text.strip()

        if _is_hallucination(text):
            text = ""
        return TranscriptionResult(
            text=text,
            audio_seconds=len(audio) / sample_rate,
            compute_seconds=time.perf_counter() - started,
        )


def _download_silero() -> Path:
    """Silero VAD is a tiny (~2 MB) ONNX model. Should be fetched via
    fetch_models.py to avoid Python's cert-store issues on some Windows setups."""
    dst = Path(__file__).resolve().parent.parent / "models" / "silero_vad.onnx"
    if dst.exists() and dst.stat().st_size > 100_000:
        return dst
    raise FileNotFoundError(
        f"Silero VAD missing at {dst}. Fetch it once with:\n"
        "  curl -L https://github.com/snakers4/silero-vad/raw/v5.1/src/"
        "silero_vad/data/silero_vad.onnx -o models/silero_vad.onnx"
    )


class FasterWhisperTranscriber(Transcriber):
    def __init__(self, model_path: str | Path, cfg: STTConfig):
        from faster_whisper import WhisperModel

        self.cfg = cfg
        self.model_path = str(model_path)
        self.model = WhisperModel(
            self.model_path,
            device="cpu",
            compute_type=cfg.compute_type,
            cpu_threads=cfg.cpu_threads,
            num_workers=1,
        )

    PROMPT_TOKEN_LIMIT = 224   # Whisper uses half its 448-token context for the prompt

    def prompt_tokens(self) -> int:
        """Token count of the initial prompt; anything past the limit is dropped."""
        if not self.cfg.initial_prompt:
            return 0
        return len(self.model.hf_tokenizer.encode(" " + self.cfg.initial_prompt.strip()).ids)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        started = time.perf_counter()
        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            initial_prompt=self.cfg.initial_prompt,
            # Silero VAD as a SECOND gate: our energy VAD segments audio for
            # latency; Silero decides whether that segment is actually speech.
            # threshold=0.6 (up from Silero's 0.5) and min_speech_duration_ms
            # =500 (up from 250) reject brief noise transients that would
            # otherwise trigger Whisper into "One two, one two" hallucinations
            # when a phone mic streams gain-boosted silence.
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.6,
                min_speech_duration_ms=500,
                min_silence_duration_ms=600,
                speech_pad_ms=80,
            ),
            # Belt and braces confidence gates. If Whisper's own probability
            # of "no speech here" is above no_speech_threshold, or the mean
            # log-probability of what it emits is below log_prob_threshold,
            # the whole segment is dropped as noise.
            # Slightly stricter than defaults but not aggressive enough to
            # eat real speech. Prior tightening (0.85 / -0.5) blocked the
            # preacher's actual voice.
            no_speech_threshold=0.75,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,   # Whisper's default
            without_timestamps=False,          # timestamps let us inspect confidence
            condition_on_previous_text=False,  # stops runaway repetition loops
        )
        # Whisper reports per-segment confidence. We drop a segment only when
        # it's likely silence (no_speech_prob high) OR the transcription is
        # very low confidence. The bar has to be forgiving enough for real
        # accented speech to pass.
        keep = []
        for s in segments:
            if getattr(s, "no_speech_prob", 0.0) >= 0.65:
                continue
            if getattr(s, "avg_logprob", 0.0) < -1.0:
                continue
            keep.append(s.text.strip())
        text = " ".join(keep).strip()
        if _is_hallucination(text):
            text = ""
        return TranscriptionResult(
            text=text,
            audio_seconds=len(audio) / sample_rate,
            compute_seconds=time.perf_counter() - started,
        )
