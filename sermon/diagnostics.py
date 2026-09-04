"""Diagnostic mode.

Answers, in order, the four things that actually go wrong on this kind of
machine: (1) is the environment right, (2) can we see the microphone and is it
picking up sound, (3) do the models load and how much RAM do they cost,
(4) is this CPU fast enough to keep up with live speech.

Run: python run.py doctor
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys
import time
from pathlib import Path

from .config import AppConfig

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"
INFO = "[    ]"

BENCH_SENTENCE = "Good morning everyone. Today we are going to talk about faith."


class Report:
    def __init__(self):
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []

    def line(self, status: str, label: str, detail: str = "") -> None:
        print(f"{status} {label:<34} {detail}")
        if status == FAIL:
            self.failures.append(f"{label}: {detail}")
        elif status == WARN:
            self.warnings.append(f"{label}: {detail}")

    def section(self, title: str) -> None:
        print()
        print(f"--- {title} " + "-" * max(0, 58 - len(title)))


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return float("nan")


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024


# --------------------------------------------------------------------- checks

def check_environment(rep: Report, cfg: AppConfig) -> None:
    rep.section("1. Environment")

    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro} ({platform.architecture()[0]})"
    if v < (3, 9):
        rep.line(FAIL, "Python version", detail + " - need 3.9 or newer")
    elif v >= (3, 15):
        rep.line(WARN, "Python version", detail + " - newer than tested; wheels may be missing")
    else:
        rep.line(OK, "Python version", detail)

    if platform.architecture()[0] != "64bit":
        rep.line(FAIL, "Python architecture", "32-bit Python cannot run these models")

    rep.line(INFO, "Operating system", f"{platform.system()} {platform.release()} ({platform.version()})")
    rep.line(INFO, "In virtualenv", "yes" if sys.prefix != sys.base_prefix else "NO - you are using system Python")

    logical = os.cpu_count() or 0
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or 0
        mem = psutil.virtual_memory()
        total_gb = mem.total / 1024 ** 3
        avail_gb = mem.available / 1024 ** 3

        rep.line(OK if logical >= 4 else WARN, "CPU threads",
                 f"{physical} cores / {logical} threads")
        if total_gb < 6.5:
            rep.line(FAIL, "Total RAM", f"{total_gb:.1f} GB - too little")
        else:
            rep.line(OK, "Total RAM", f"{total_gb:.1f} GB")

        if avail_gb < 1.2:
            rep.line(FAIL, "Available RAM", f"{avail_gb:.1f} GB free - close Chrome and retry")
        elif avail_gb < 2.0:
            rep.line(WARN, "Available RAM", f"{avail_gb:.1f} GB free - tight; close other apps")
        else:
            rep.line(OK, "Available RAM", f"{avail_gb:.1f} GB free")
    except Exception as exc:
        rep.line(WARN, "psutil", f"unavailable ({exc})")

    free_gb = shutil.disk_usage(str(cfg.models_dir.parent)).free / 1024 ** 3
    rep.line(OK if free_gb > 3 else FAIL, "Free disk space", f"{free_gb:.1f} GB")


VCREDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


def check_native_runtime(rep: Report) -> None:
    """The compiled wheels (ctranslate2, onnxruntime, sentencepiece) all link
    against the MSVC runtime. A clean Windows install does not have it, and the
    resulting error - 'Could not find module ctranslate2.dll (or one of its
    dependencies)' - names the wrong file, so check for it explicitly."""
    if sys.platform != "win32":
        return
    import ctypes.util

    missing = [d for d in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")
               if ctypes.util.find_library(d) is None]
    if missing:
        rep.line(FAIL, "MSVC runtime", f"missing {', '.join(missing)}")
        rep.notes.append(f"Install the C++ runtime, then re-run: {VCREDIST_URL}")
    else:
        rep.line(OK, "MSVC runtime", "vcruntime140 / msvcp140 present")


def check_packages(rep: Report) -> None:
    rep.section("2. Packages")
    check_native_runtime(rep)
    for mod, pretty in [
        ("numpy", "numpy"),
        ("sounddevice", "sounddevice"),
        ("ctranslate2", "ctranslate2"),
        ("faster_whisper", "faster-whisper"),
        ("sentencepiece", "sentencepiece"),
        ("psutil", "psutil"),
    ]:
        try:
            m = importlib.import_module(mod)
            rep.line(OK, pretty, getattr(m, "__version__", "installed"))
        except Exception as exc:
            # Collapse the very long "Could not find module <abs path>" text.
            detail = str(exc).splitlines()[0]
            if "dll" in detail.lower():
                detail = "native library failed to load -> install the MSVC runtime"
            detail = detail[:88]
            rep.line(FAIL, pretty, f"import failed: {detail}")

    try:
        import ctranslate2
        types = ctranslate2.get_supported_compute_types("cpu")
        if "int8" in types:
            rep.line(OK, "int8 CPU support", ", ".join(sorted(types)))
        else:
            rep.line(WARN, "int8 CPU support", f"not advertised: {sorted(types)}")
    except Exception as exc:
        rep.line(WARN, "CTranslate2 compute types", str(exc))


def check_models(rep: Report, cfg: AppConfig) -> None:
    rep.section("3. Models on disk")
    w = cfg.whisper_dir

    if (w / "model.bin").exists():
        rep.line(OK, f"Whisper ({cfg.stt.model})", f"{_dir_size_mb(w):.0f} MB at {w}")
    else:
        rep.line(FAIL, f"Whisper ({cfg.stt.model})",
                 f"missing at {w} - run: python scripts/fetch_models.py")

    for i, target in enumerate(cfg.mt.targets):
        m = cfg.mt_dir_for(target)
        # Primary target is required; secondaries just warn.
        status_missing = FAIL if i == 0 else WARN
        if (m / "model.bin").exists() and (m / "source.spm").exists():
            rep.line(OK, f"OPUS-MT en-{target}", f"{_dir_size_mb(m):.0f} MB at {m}")
        else:
            rep.line(status_missing, f"OPUS-MT en-{target}",
                     f"missing - run: python scripts/fetch_models.py --lang {target}")

    from .server import LISTENER_HTML, SPEAKER_HTML
    for label, p in [("Listener page", LISTENER_HTML),
                     ("Speaker page", SPEAKER_HTML)]:
        if p.exists():
            rep.line(OK, label, f"{p.stat().st_size // 1024} KB")
        else:
            rep.line(WARN, label, "missing - --serve will not serve it")

    for lang, voice in cfg.tts.voices.items():
        v = cfg.tts_model_for(voice)
        if v.exists() and v.with_suffix(".onnx.json").exists():
            rep.line(OK, f"Piper voice {voice}", f"{_dir_size_mb(v.parent):.0f} MB "
                     f"(target: {lang.upper()})")
        else:
            rep.line(WARN, f"Piper voice {voice}",
                     f"not downloaded - {lang.upper()} audio for phones disabled")


def _venv_firewall_ok() -> tuple[bool, str]:
    """On Windows, `--serve` needs a firewall rule for THIS python.exe.

    A rule for a different interpreter path (e.g. the system Python) does not
    apply - Windows keys rules by exact executable path. Loopback bypasses the
    firewall so a laptop-local curl will succeed even when phones are blocked;
    we have to check the rule itself, not connectivity.
    """
    if sys.platform != "win32":
        return True, "not applicable"
    exe = str(Path(sys.executable).resolve())
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-NetFirewallApplicationFilter | Where-Object { $_.Program -eq "
             f"'{exe}' " + "} | ForEach-Object { "
             "$r = Get-NetFirewallRule -AssociatedNetFirewallApplicationFilter $_ "
             "-ErrorAction SilentlyContinue; "
             "if ($r -and $r.Enabled -eq 'True' -and $r.Direction -eq 'Inbound' "
             "-and $r.Action -eq 'Allow') { 'yes' } }"],
            capture_output=True, text=True, timeout=10)
        if "yes" in out.stdout.lower():
            return True, "allowed for this interpreter"
        return False, "no inbound Allow rule for this exact python.exe"
    except Exception as exc:
        return True, f"could not check ({exc})"


VENV_ALLOW_CMD = (
    'New-NetFirewallRule -DisplayName "Sermon (venv Python)" -Direction Inbound '
    '-Program "{exe}" -Action Allow -Profile Any'
)


def check_network(rep: Report, cfg: AppConfig) -> None:
    """Report the URLs a phone on the same Wi-Fi should point its browser at,
    and warn if Windows Firewall will silently block them."""
    rep.section("4. Network (for --serve)")
    from .server import list_local_ips
    ips = list_local_ips()
    if not ips:
        rep.line(WARN, "LAN interfaces", "none detected - a phone can't reach this host")
    else:
        rep.line(OK, "LAN interfaces", f"{len(ips)} address(es)")
        for ip in ips:
            print(f"       listeners:  http://{ip}:{cfg.server.port}/")
            print(f"       speaker:    http://{ip}:{cfg.server.port}/speaker")

    ok, detail = _venv_firewall_ok()
    if ok:
        rep.line(OK, "Windows Firewall", detail)
    else:
        exe = str(Path(sys.executable).resolve())
        rep.line(WARN, "Windows Firewall", detail)
        rep.notes.append(
            "Phones will get 'site can't be reached' until you add the rule.\n"
            "       Open PowerShell as ADMIN and run:\n"
            "         " + VENV_ALLOW_CMD.format(exe=exe))


def check_audio(rep: Report, cfg: AppConfig, seconds: float):
    rep.section("5. Microphone")
    try:
        import sounddevice as sd
    except Exception as exc:
        rep.line(FAIL, "sounddevice import", str(exc))
        return None

    try:
        devices = sd.query_devices()
    except Exception as exc:
        rep.line(FAIL, "Enumerate audio devices", str(exc))
        return None

    inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    if not inputs:
        rep.line(FAIL, "Input devices", "none found - check Windows sound settings")
        return None

    rep.line(OK, "Input devices", f"{len(inputs)} found")
    for i, d in inputs:
        print(f"       [{i}] {d['name']}  ({d['max_input_channels']} ch @ {int(d['default_samplerate'])} Hz)")

    try:
        default_in = sd.default.device[0]
        rep.line(INFO, "Default input", f"[{default_in}] {devices[default_in]['name']}"
                 if default_in is not None and default_in >= 0 else "not set")
    except Exception:
        pass

    if seconds <= 0:
        return None

    import numpy as np

    from .audio import MicrophoneSource, rms_dbfs
    print()
    print(f"       Recording {seconds:.0f} s. SPEAK NOW, normal preaching volume:")
    print(f'       "{BENCH_SENTENCE}"')
    try:
        audio = MicrophoneSource(cfg.audio).record(seconds)
    except Exception as exc:
        rep.line(FAIL, "Microphone capture", f"{exc}  (check Windows mic privacy settings)")
        return None

    peak = 20 * np.log10(float(np.max(np.abs(audio))) + 1e-10)
    level = rms_dbfs(audio)
    loudest = max(
        rms_dbfs(audio[i:i + 4800]) for i in range(0, max(1, len(audio) - 4800), 2400)
    )

    rep.line(INFO, "Average level", f"{level:.1f} dBFS")
    rep.line(INFO, "Loudest 0.3 s window", f"{loudest:.1f} dBFS")

    from .audio import level_verdict
    _, advice, acceptable = level_verdict(peak)
    rep.line(OK if acceptable else WARN, "Signal level", f"peak {peak:.1f} dBFS - {advice}")

    return audio


def check_models_runtime(rep: Report, cfg: AppConfig, audio) -> None:
    rep.section("6. Model loading and speed")
    base_rss = _rss_mb()
    rep.line(INFO, "RSS before loading", f"{base_rss:.0f} MB")

    # --- STT ---
    transcriber = None
    try:
        from .stt import FasterWhisperTranscriber
        t0 = time.perf_counter()
        transcriber = FasterWhisperTranscriber(cfg.whisper_dir, cfg.stt)
        load_s = time.perf_counter() - t0
        after = _rss_mb()
        rep.line(OK, f"Whisper {cfg.stt.model} loaded",
                 f"{load_s:.1f}s, +{after - base_rss:.0f} MB "
                 f"({cfg.stt.compute_type}, {cfg.stt.cpu_threads} threads)")
        base_rss = after

        try:
            n = transcriber.prompt_tokens()
            limit = transcriber.PROMPT_TOKEN_LIMIT
            if n == 0:
                rep.line(WARN, "Vocabulary prompt", "disabled - accuracy will be lower")
            elif n > limit:
                rep.line(WARN, "Vocabulary prompt",
                         f"{n} tokens exceeds {limit} - trim vocabulary.txt, "
                         f"the excess is silently ignored")
            else:
                rep.line(OK, "Vocabulary prompt", f"{n}/{limit} tokens")
        except Exception as exc:
            rep.line(INFO, "Vocabulary prompt", f"could not measure ({exc})")
    except Exception as exc:
        rep.line(FAIL, "Whisper load", str(exc))

    # --- MT (one per configured target) ---
    from dataclasses import replace as _replace
    from .config import phrase_file_for
    from .mt import MarianCT2Translator, OverrideTranslator, load_phrase_overrides

    translators: dict[str, object] = {}
    for target in cfg.mt.targets:
        model_dir = cfg.mt_dir_for(target)
        if not (model_dir / "model.bin").exists():
            continue          # already flagged in check_models
        try:
            t0 = time.perf_counter()
            tr = MarianCT2Translator(model_dir, _replace(cfg.mt, targets=[target]))
            load_s = time.perf_counter() - t0
            after = _rss_mb()
            rules = load_phrase_overrides(phrase_file_for(target))
            extra = f" [+ {len(rules)} phrase overrides]" if rules else ""
            rep.line(OK, f"OPUS-MT en-{target} loaded",
                     f"{load_s:.1f}s, +{after - base_rss:.0f} MB{extra}")
            base_rss = after
            translators[target] = OverrideTranslator(tr, rules) if rules else tr
        except Exception as exc:
            rep.line(FAIL, f"OPUS-MT en-{target} load", str(exc))

    rep.line(INFO, "Total RSS with models", f"{_rss_mb():.0f} MB")

    # --- translation correctness + speed (no mic needed) ---
    if translators:
        print(f"       EN  {BENCH_SENTENCE}")
        for target, tr in translators.items():
            try:
                res = tr.translate(BENCH_SENTENCE)
                rep.line(OK, f"Translation en-{target}",
                         f"{res.compute_seconds * 1000:.0f} ms")
                print(f"       {target.upper():<3} {res.text}")
            except Exception as exc:
                rep.line(FAIL, f"Translation en-{target}", str(exc))

    # --- speech synthesis (one per voice that's on disk) ---
    from .tts import PiperSpeaker
    for lang_code, voice in cfg.tts.voices.items():
        model_path = cfg.tts_model_for(voice)
        if not model_path.exists():
            continue
        try:
            before = _rss_mb()
            t0 = time.perf_counter()
            speaker = PiperSpeaker(model_path, cfg.tts)
            load_s = time.perf_counter() - t0
            rep.line(OK, f"Piper {voice} loaded",
                     f"{load_s:.1f}s, +{_rss_mb() - before:.0f} MB, "
                     f"{speaker.sample_rate} Hz")
            phrase = "Bonjour tout le monde." if lang_code == "fr" \
                else "Buenos días a todos." if lang_code == "es" \
                else "Hello everyone."
            speaker.synthesize(phrase)                            # warm up
            sp = speaker.synthesize(phrase)
            status = OK if sp.rtf < 0.5 else (WARN if sp.rtf < 1.0 else FAIL)
            rep.line(status, f"Synthesis {lang_code}",
                     f"{sp.duration:.1f}s of {lang_code.upper()} in "
                     f"{sp.compute_seconds:.2f}s (rtf {sp.rtf:.2f})")
        except Exception as exc:
            rep.line(FAIL, f"Synthesis {lang_code}", str(exc)[:88])

    rep.line(INFO, "RSS with all models", f"{_rss_mb():.0f} MB")

    # --- transcription speed on the audio we just recorded ---
    if transcriber is not None and audio is not None:
        try:
            res = transcriber.transcribe(audio, cfg.audio.sample_rate)
            rtf = res.rtf
            status = OK if rtf < 0.7 else (WARN if rtf < 1.0 else FAIL)
            rep.line(status, "Transcription speed",
                     f"{res.compute_seconds:.1f}s for {res.audio_seconds:.1f}s audio "
                     f"(RTF {rtf:.2f})")
            print(f"       HEARD: {res.text or '(nothing recognised)'}")
            if not res.text:
                rep.warnings.append("Whisper recognised nothing in the test recording.")
            if rtf >= 1.0:
                rep.notes.append(
                    f"RTF {rtf:.2f} means transcription is SLOWER than real time. "
                    f"Run with --model tiny.en."
                )
            elif rtf >= 0.7:
                rep.notes.append(
                    f"RTF {rtf:.2f} leaves little headroom. --model tiny.en will feel snappier."
                )
            if translators and res.text:
                for target, tr in translators.items():
                    out = tr.translate(res.text)
                    print(f"       {target.upper()}: {out.text}")
        except Exception as exc:
            rep.line(FAIL, "Transcription test", str(exc))
    elif transcriber is not None:
        rep.line(INFO, "Transcription speed", "skipped (no microphone audio)")


def verdict(rep: Report) -> int:
    rep.section("Verdict")
    if rep.failures:
        print(f"{FAIL} {len(rep.failures)} blocking problem(s):")
        for f in rep.failures:
            print(f"       - {f}")
    if rep.warnings:
        print(f"{WARN} {len(rep.warnings)} warning(s):")
        for w in rep.warnings:
            print(f"       - {w}")
    for n in rep.notes:
        print(f"{INFO} {n}")
    if not rep.failures:
        print(f"{OK} Pipeline ready.  python run.py   (add --speak for French audio)")
    print()
    return 1 if rep.failures else 0


def run_doctor(cfg: AppConfig, mic_seconds: float = 6.0) -> int:
    print()
    print("=" * 70)
    print("  LOCAL SERMON TRANSLATION - DIAGNOSTICS")
    print("=" * 70)
    rep = Report()
    check_environment(rep, cfg)
    check_packages(rep)
    check_models(rep, cfg)
    check_network(rep, cfg)
    audio = check_audio(rep, cfg, mic_seconds)
    if not any("missing" in f for f in rep.failures):
        check_models_runtime(rep, cfg, audio)
    else:
        rep.section("6. Model loading and speed")
        print(f"{INFO} Skipped - download the models first.")
    return verdict(rep)
