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
