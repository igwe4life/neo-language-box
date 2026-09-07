"""Configuration objects. Everything tunable lives here, nothing is hardcoded
in the pipeline itself."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = Path(os.environ.get("SERMON_MODELS_DIR", str(ROOT / "models")))


def auto_threads() -> int:
    """Leave one logical core free so the audio callback never starves.

    On the 2-core / 4-thread i7-6650U this returns 3.
    """
    logical = os.cpu_count() or 2
    return max(1, min(logical - 1, 8))


def mt_threads() -> int:
    """Fewer threads for translation - it runs alongside STT and Piper for
    each utterance, and on a 4-thread CPU asking for 3 threads per model
    causes catastrophic contention. 1 thread per MT model + 3 for STT +
    1 per Piper fits comfortably."""
    logical = os.cpu_count() or 2
    return 1 if logical <= 4 else 2


@dataclass
class AudioConfig:
    sample_rate: int = 16000          # what Whisper wants; no resampling needed
    block_ms: int = 30                # VAD frame size
    device: int | None = None         # None = system default input
    channels: int = 1

    # --- utterance segmentation (energy VAD) ---
    preroll_ms: int = 320             # audio kept before speech onset, avoids clipped words
    min_speech_ms: int = 350          # shorter than this is treated as a noise blip
    smart_cut_ms: int = 1500          # window searched for the quietest split point

    # --- utterance length ---
    # Whisper's accuracy depends heavily on having context, and its encoder costs
    # the same whether you feed it 2 s or 12 s. Both argue for FEWER, LONGER
    # utterances. So a mid-sentence breath does not end an utterance: we keep
    # buffering through short pauses until there is enough material to recognise.
    silence_ms: int = 600             # pause that may end an utterance...
    min_utterance_s: float = 5.0      # ...but only once we have this much audio
    long_silence_ms: int = 1300       # a real stop: end it regardless of length
    max_utterance_s: float = 14.0     # hard flush; also bounds worst-case latency
    threshold_db: float = 7.0         # dB above the measured noise floor = speech
    absolute_floor_db: float = -52.0  # never trigger on anything quieter than this
    calibrate_s: float = 1.2          # ambient noise measurement at startup

    queue_size: int = 3               # pending utterances; oldest dropped when full

    def tune_for_low_latency(self) -> None:
        """Shorten utterances. Only safe on an engine with no fixed per-call cost.

        The defaults above deliberately buffer through pauses because
        faster-whisper pays its encoder on a 30 s padded mel no matter how
        little audio you give it: measured on this laptop, base.en costs 2.74 s
        for a 4.7 s clip and only 3.26 s for an 11.1 s one. Short utterances
        would spend nearly all their time on that fixed cost.

        Parakeet has no such floor - it is linear and ~8x cheaper (0.35 s for
        the same 4.7 s clip), so the reason for the long buffering disappears
        and the segmentation delay becomes the thing worth cutting. Waiting
        1.3 s to decide a sentence ended is most of the perceived lag.
        """
        self.min_utterance_s = 2.0     # was 5.0 - end short sentences promptly
        self.silence_ms = 450          # was 600
        self.long_silence_ms = 800     # was 1300 - dominates felt latency
        self.max_utterance_s = 8.0     # was 14.0 - bounds the worst case


# Whisper accepts a short prompt that biases its vocabulary. Naming the domain
# terms it would otherwise guess at is the cheapest accuracy win available -
# it is what stops "brought your Bibles" becoming "brought you by a boss".
SERMON_PROMPT = (
    # NOTE: do NOT put "amen" or "hallelujah" here. They're Whisper's default
    # silence-hallucination targets for religious content; naming them in the
    # prompt caused them to fire on quiet moments. Real utterances of those
    # words still transcribe fine WITHOUT the prompt hint, and are handled
    # correctly by the phrase override table on the way out.
    "A church sermon in English. Expect these words: Bible, Bibles, Scripture, "
    "Scriptures, God's Word, verse, chapter, God, Jehovah, Lord, Jesus Christ, "
    "Holy Spirit, faith, grace, prayer, worship, congregation, sermon, "
    "brothers and sisters, I implore you, come with your Bibles, righteousness, "
    "salvation, resurrection hope, everlasting life, kingdom, "
    "Genesis, Psalms, Matthew, John, Romans, Revelation."
)

DEFAULT_VOCAB_FILE = ROOT / "vocabulary.txt"


def phrase_file_for(target: str) -> Path:
    """Path to the per-target phrase-override table (may not exist)."""
    return ROOT / f"phrases.{target}.txt"


def build_prompt(vocab_file: Path | None = DEFAULT_VOCAB_FILE) -> str | None:
    """Base sermon prompt plus any user vocabulary.

    The vocabulary file is where this-week's proper nouns go. Whisper caps the
    prompt at 224 tokens and silently drops the rest, so `doctor` reports the
    count rather than letting it fail quietly.
    """
    prompt = SERMON_PROMPT
    if vocab_file and vocab_file.exists():
        terms = [
            line.strip()
            for line in vocab_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if terms:
            prompt = f"{prompt} Also expect: {', '.join(terms)}."
    return prompt


@dataclass
class STTConfig:
    engine: str = "whisper"           # "whisper" or "parakeet"
    # Whisper model names: tiny.en | base.en | small.en
    # Parakeet model names: parakeet-110m | parakeet-0.6b
    model: str = "base.en"
    compute_type: str = "int8"        # int8 is the right choice for AVX2 CPUs
    # Greedy decoding. beam=5 was previously fine when the pipeline ran alone,
    # but with --serve broadcasting to phones + Piper synthesising audio for
    # every language on every utterance, beam search was spiking STT compute
    # to 5-14 s on a 4-thread CPU. Greedy costs ~1 BLEU on English speech
    # and saves ~30% of STT time. Bump back to 5 with --beam 5 if wanted.
    beam_size: int = 1
    language: str = "en"
    initial_prompt: str | None = field(default_factory=build_prompt)
    cpu_threads: int = field(default_factory=auto_threads)


@dataclass
class MTConfig:
    # Every target listed here is translated in parallel from the same English.
    # Order matters: the first target is the "primary" - the one TTS speaks
    # aloud and the one the console prints first.
    targets: list[str] = field(default_factory=lambda: ["fr", "es"])
    beam_size: int = 2
    cpu_threads: int = field(default_factory=mt_threads)
    max_decoding_length: int = 320

    @property
    def target(self) -> str:
        """Backwards compat: the primary target (drives fr TTS, single-target APIs)."""
        return self.targets[0] if self.targets else "fr"


@dataclass
class TTSConfig:
    enabled: bool = False             # opt-in: --speak (laptop speaker playback)
    # One Piper voice per translation target. Every listed language gets audio
    # generated for phone-streaming; --speak still speaks only `voice` primary.
    # NOTE: default is French-only. Piper occasionally deadlocks loading a
    # second ONNX session on some Windows configs; Spanish listeners still get
    # text translations, and their phone's own TTS voice (if any) can speak.
    # To enable Spanish server audio explicitly, extend voices at runtime.
    voices: dict[str, str] = field(default_factory=lambda: {
        "fr": "fr_FR-siwis-medium",
    })
    length_scale: float = 1.0         # >1 slower delivery
    volume: float = 1.0
    output_device: int | None = None
    queue_size: int = 2               # pending phrases; oldest dropped when full
    # Playing translation out of the laptop speakers puts it straight back into
    # the microphone. Muting capture during playback is the only fix without
    # separate audio hardware.
    mute_mic_while_speaking: bool = True
    mic_resume_delay_ms: int = 250    # let the room's echo decay before listening

    @property
    def voice(self) -> str:
        """Primary voice (drives --speak). Reads from the primary target."""
        return next(iter(self.voices.values()))


@dataclass
class ServerConfig:
    enabled: bool = False
    host: str = "0.0.0.0"      # bind all interfaces so any phone on the LAN reaches us
    port: int = 8000
    history_size: int = 20     # translations replayed to a phone that joins mid-sermon
    tls: bool = False          # self-signed cert; needed for browser mic on non-localhost
    loopback_port: int = 8000  # only used with --tauri: HTTP bind for the desktop-app webview


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    mt: MTConfig = field(default_factory=MTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    models_dir: Path = DEFAULT_MODELS_DIR

    @property
    def whisper_dir(self) -> Path:
        return self.models_dir / "whisper" / self.stt.model

    @property
    def parakeet_dir(self) -> Path:
        return self.models_dir / self.stt.model

    @property
    def stt_model_dir(self) -> Path:
        return self.parakeet_dir if self.stt.engine == "parakeet" else self.whisper_dir

    def mt_dir_for(self, target: str) -> Path:
        return self.models_dir / "mt" / f"opus-mt-en-{target}"

    @property
    def mt_dir(self) -> Path:
        """Directory of the primary translation model. Kept for callers that
        translate to one language (subcommands, single-target diagnostics)."""
        return self.mt_dir_for(self.mt.target)

    @property
    def tts_model(self) -> Path:
        """Path to the primary voice (drives --speak)."""
        return self.tts_model_for(self.tts.voice)

    def tts_model_for(self, voice: str) -> Path:
        return self.models_dir / "tts" / voice / f"{voice}.onnx"

    def tts_voices_available(self) -> dict[str, Path]:
        """Voice model paths that are actually on disk, keyed by target lang."""
        return {lang: self.tts_model_for(v)
                for lang, v in self.tts.voices.items()
                if self.tts_model_for(v).exists()}
