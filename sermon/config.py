"""Configuration objects. Everything tunable lives here, nothing is hardcoded
in the pipeline itself."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = Path(os.environ.get("SERMON_MODELS_DIR", str(ROOT / "models")))

# Single source of truth for the app version on the Python side. Inno Setup
# cannot read this file at build time, so installer.iss's #define AppVersion
# is kept in sync by hand - bump both together.
APP_VERSION = "0.8.0"


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


# English is the SOURCE language, so it is a listener option with no
# translator behind it - the pipeline already has the text. It is listed here
# only so the listener UI can offer it; `targets` never contains it.
SOURCE_LANG = "en"

# Everything the UI can offer. `native` is what a speaker of the language
# calls it - a listener scanning the list should find their own language
# without reading English. `rtl` drives text direction on the listener page.
# `speech` is the BCP-47 tag handed to the phone's own voice when we have no
# server-synth clip for that language.
LANGUAGES = {
    "en": {"native": "English",  "flag": "\U0001F1EC\U0001F1E7", "speech": "en-GB"},
    "fr": {"native": "Français", "flag": "\U0001F1EB\U0001F1F7", "speech": "fr-FR"},
    "es": {"native": "Español",  "flag": "\U0001F1EA\U0001F1F8", "speech": "es-ES"},
    "pt": {"native": "Português", "flag": "\U0001F1E7\U0001F1F7", "speech": "pt-BR"},
    "de": {"native": "Deutsch",  "flag": "\U0001F1E9\U0001F1EA", "speech": "de-DE"},
    "it": {"native": "Italiano", "flag": "\U0001F1EE\U0001F1F9", "speech": "it-IT"},
    "ru": {"native": "Русский",  "flag": "\U0001F1F7\U0001F1FA", "speech": "ru-RU"},
    "ar": {"native": "العربية",   "flag": "\U0001F1F8\U0001F1E6", "speech": "ar-SA", "rtl": True},
    "hi": {"native": "हिन्दी",      "flag": "\U0001F1EE\U0001F1F3", "speech": "hi-IN"},
    "zh": {"native": "中文",      "flag": "\U0001F1E8\U0001F1F3", "speech": "zh-CN"},
    "sw": {"native": "Kiswahili", "flag": "\U0001F1F0\U0001F1EA", "speech": "sw-KE"},
    "ig": {"native": "Igbo",     "flag": "\U0001F1F3\U0001F1EC", "speech": "ig-NG"},
    "yo": {"native": "Yorùbá",   "flag": "\U0001F1F3\U0001F1EC", "speech": "yo-NG"},
    "ha": {"native": "Hausa",    "flag": "\U0001F1F3\U0001F1EC", "speech": "ha-NG"},

    # --- fourth language wave: same proven path as fr/es/.../sw above -
    # a Piper voice plus either OPUS-MT or NLLB-200 for translation, no
    # one-off model engineering. Re-pulled from the reverted 25-language
    # expansion (2026-10-01), where each of these five was individually
    # verified against the live Piper/OPUS-MT listings before being added.
    "ko": {"native": "한국어",      "flag": "\U0001F1F0\U0001F1F7", "speech": "ko-KR"},
    "vi": {"native": "Tiếng Việt", "flag": "\U0001F1FB\U0001F1F3", "speech": "vi-VN"},
    "id": {"native": "Bahasa Indonesia", "flag": "\U0001F1EE\U0001F1E9", "speech": "id-ID"},
    "tr": {"native": "Türkçe",    "flag": "\U0001F1F9\U0001F1F7", "speech": "tr-TR"},
    "pl": {"native": "Polski",    "flag": "\U0001F1F5\U0001F1F1", "speech": "pl-PL"},
    "bn": {"native": "বাংলা",      "flag": "\U0001F1E7\U0001F1E9", "speech": "bn-BD"},
    "ur": {"native": "اردو",       "flag": "\U0001F1F5\U0001F1F0", "speech": "ur-PK", "rtl": True},
}

DEFAULT_TARGETS = ["fr", "es", "pt", "de", "it", "ru", "ar", "hi", "zh",
                   "sw", "ig", "yo", "ha", "ko", "vi", "id", "tr", "pl", "bn", "ur"]
# Igbo was pulled from the default set 2026-10-04 while investigating reports
# of it sounding wrong in the field, and restored 2026-11-06 after the actual
# cause (BroadcastSink.emit() blocking the pipeline on synthesis - see the
# 2026-10-04 commit fixing that) was found and fixed. Re-tested via
# `python run.py doctor` on restoring it; watch for a recurrence of the
# original symptoms (a wrong voice cutting in, hangs) under real load.

# Targets served by NLLB-200 rather than a bilingual OPUS-MT model, mapped to
# NLLB's own language tags. OPUS-MT does publish models for all four, but they
# are low-resource pairs and a native speaker judged them clearly worse than
# NLLB in a blind comparison. Everything NOT listed here stays on OPUS-MT,
# which is better for those languages and about 13x faster.
NLLB_TARGETS = {
    "ig": "ibo_Latn",
    "yo": "yor_Latn",
    "ha": "hau_Latn",
    "sw": "swh_Latn",
    "pl": "pol_Latn",   # no official direct or tc-big OPUS-MT en->pl release
    "ko": "kor_Hang",   # only OPUS-MT release is tc-big, BLEU 13.7 - modest
    "bn": "ben_Beng",   # no official direct/tc-big OPUS-MT release
    "ur": "urd_Arab",   # direct OPUS-MT release exists but BLEU 12.1/chrF .39
}

# Voices that are plain VITS ONNX rather than Piper bundles, keyed by the
# directory under models/tts/. Piper has no voice for these three languages.
VITS_VOICES = {"ig": "ig", "yo": "yo", "ha": "ha"}


@dataclass
class MTConfig:
    # Every target listed here is translated from the same English.
    # Order matters: the first target is the "primary" - the one TTS speaks
    # aloud and the one the console prints first.
    targets: list[str] = field(default_factory=lambda: _targets_from_env())
    beam_size: int = 2
    cpu_threads: int = field(default_factory=mt_threads)
    max_decoding_length: int = 320

    @property
    def target(self) -> str:
        """Backwards compat: the primary target (drives fr TTS, single-target APIs)."""
        return self.targets[0] if self.targets else "fr"


LANGUAGE_PACKS_FILE = ROOT / "language-packs.json"


def installed_language_packs() -> list[str]:
    """Languages added after install via sermon/langpacks.py, not baked into
    this build. A code only lands here once its pack has been downloaded,
    verified and unzipped - this file is the record of that, written by
    langpacks.install(), read here so a restart picks it up with no code
    change. A code must already exist in LANGUAGES (fully configured, just
    not shipped by default) - a pack can activate a known-but-dormant
    language, not introduce one this build has never heard of."""
    try:
        data = json.loads(LANGUAGE_PACKS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return [c for c in data if c in LANGUAGES]


def _targets_from_env() -> list[str]:
    """Translation targets, optionally narrowed by SERMON_TARGETS.

    SERMON_TARGETS=fr,es  -> just those two, whatever else is installed.
    Useful for cutting per-utterance work on a weak machine, and for testing.
    """
    raw = (os.environ.get("SERMON_TARGETS") or "").strip()
    if not raw:
        base = list(DEFAULT_TARGETS)
        for code in installed_language_packs():
            if code not in base:
                base.append(code)
        return base
    picked = [c.strip().lower() for c in raw.split(",") if c.strip()]
    picked = [c for c in picked if c in LANGUAGES and c != SOURCE_LANG]
    return picked or list(DEFAULT_TARGETS)


ALL_VOICES = {
    "fr": "fr_FR-siwis-medium",
    "es": "es_ES-davefx-medium",
    "pt": "pt_BR-faber-medium",
    "de": "de_DE-thorsten-medium",
    "it": "it_IT-paola-medium",
    "ru": "ru_RU-dmitri-medium",
    "ar": "ar_JO-kareem-medium",
    "hi": "hi_IN-pratham-medium",
    "zh": "zh_CN-huayan-medium",
    "sw": "sw_CD-lanfrica-medium",
    "ko": "ko_KR-kss-medium",
    "vi": "vi_VN-vais1000-medium",
    "id": "id_ID-news_tts-medium",
    "tr": "tr_TR-dfki-medium",
    "pl": "pl_PL-darkman-medium",
    "bn": "bn_BD-google-medium",
    "ur": "ur_PK-fasih-medium",
    # No English voice on purpose: the room can already hear the speaker.
    # English listeners get the live transcript as text.
    #
    # Igbo, Yoruba and Hausa are absent here deliberately: Piper publishes no
    # voice for any of them, so they are served by VITS_VOICES instead.
}


def _voices_from_env() -> dict[str, str]:
    """Server-synth voices, optionally narrowed by SERMON_VOICES.

    SERMON_VOICES=fr      -> French audio only (Spanish falls back to the
                             listener's own phone voice)
    SERMON_VOICES=fr,es   -> both (the default)
    """
    raw = (os.environ.get("SERMON_VOICES") or "").strip()
    if not raw:
        return dict(ALL_VOICES)
    picked = {}
    for code in (c.strip().lower() for c in raw.split(",")):
        if code in ALL_VOICES:
            picked[code] = ALL_VOICES[code]
    return picked or dict(ALL_VOICES)


@dataclass
class TTSConfig:
    enabled: bool = False             # opt-in: --speak (laptop speaker playback)
    # One Piper voice per translation target. Every listed language gets audio
    # generated for phone-streaming; --speak still speaks only `voice` primary.
    #
    # French AND Spanish by default. This was French-only for a while because
    # Piper can deadlock loading a second ONNX session on some Windows configs,
    # which left Spanish listeners on their phone's own robotic voice. Loading
    # both is now guarded by a 30s-per-voice timeout in run.py, so the bad case
    # costs a slow startup and falls back to whatever loaded - it cannot hang
    # the app. Set SERMON_VOICES=fr to skip the attempt entirely on a machine
    # where it does misbehave.
    voices: dict[str, str] = field(default_factory=lambda: _voices_from_env())
    length_scale: float = 1.0         # >1 slower delivery
    # Per-language override of length_scale. Piper and the raw VITS voices
    # (Igbo, Yoruba, Hausa) are different model families with different
    # inherent pacing at the same length_scale - a listener flagged Yoruba
    # and Igbo specifically as rushed, not the Piper voices, so this slows
    # down only those two rather than every language.
    length_scale_overrides: dict[str, float] = field(default_factory=lambda: {
        "yo": 1.2,
        "ig": 1.2,
    })
    # noise_scale/noise_scale_w control a VITS voice's prosodic variation -
    # higher generally reads as more natural/expressive, lower as flatter or
    # more "robotic". Only the raw VITS voices (VitsSpeaker) read these;
    # Piper manages its own internally. 0.667/0.8 are the values this
    # project has always used (previously hardcoded in tts.py).
    noise_scale: float = 0.667
    noise_scale_w: float = 0.8
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

    def length_scale_for(self, code: str) -> float:
        return self.length_scale_overrides.get(code, self.length_scale)


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
