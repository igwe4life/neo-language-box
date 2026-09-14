#!/usr/bin/env python
"""Local sermon translation - Phase 1 CLI.

  python run.py                    live: mic -> English -> French
  python run.py doctor             full diagnostic (env, mic, models, speed)
  python run.py devices            list audio input devices
  python run.py mic                microphone level meter only
  python run.py translate "text"   translation only, no microphone

Everything runs locally. No API keys, no network access after setup.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sermon.config import AppConfig  # noqa: E402
from sermon.console import enable_utf8_console  # noqa: E402


def build_config(args) -> AppConfig:
    cfg = AppConfig()
    if getattr(args, "stt", None):
        # e.g. --stt parakeet-110m, --stt whisper-base.en
        eng, _, m = args.stt.partition("-")
        if eng == "parakeet":
            cfg.stt.engine = "parakeet"
            cfg.stt.model = "parakeet-" + (m or "110m")
        elif eng == "whisper":
            cfg.stt.engine = "whisper"
            cfg.stt.model = m or "base.en"
        else:
            print(f"  !! unknown --stt {args.stt!r}. Use whisper-base.en, "
                  "parakeet-110m, or parakeet-0.6b.")
            return 2
    if getattr(args, "model", None):
        cfg.stt.model = args.model
    # Engine chosen; now pick a segmentation strategy that suits its cost
    # curve. Explicit --silence/--min-utterance flags below still win.
    if cfg.stt.engine == "parakeet" and not getattr(args, "long_utterances", False):
        cfg.audio.tune_for_low_latency()
    if getattr(args, "device", None) is not None:
        cfg.audio.device = args.device
    if getattr(args, "threads", None):
        cfg.stt.cpu_threads = args.threads
        cfg.mt.cpu_threads = args.threads
    if getattr(args, "silence", None):
        cfg.audio.silence_ms = args.silence
    if getattr(args, "threshold", None):
        cfg.audio.threshold_db = args.threshold
    if getattr(args, "min_utterance", None):
        cfg.audio.min_utterance_s = args.min_utterance
    if getattr(args, "max_utterance", None):
        cfg.audio.max_utterance_s = args.max_utterance
    if getattr(args, "beam", None):
        cfg.stt.beam_size = args.beam
    if getattr(args, "no_prompt", False):
        cfg.stt.initial_prompt = None
    if getattr(args, "targets", None):
        cfg.mt.targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    elif getattr(args, "no_yoruba", False):
        cfg.mt.targets = [t for t in cfg.mt.targets if t != "yo"]
    if getattr(args, "speak", False):
        cfg.tts.enabled = True
    if getattr(args, "voice", None):
        cfg.tts.voice = args.voice
    if getattr(args, "speed", None):
        cfg.tts.length_scale = args.speed
    if getattr(args, "output_device", None) is not None:
        cfg.tts.output_device = args.output_device
    if getattr(args, "no_mute", False):
        cfg.tts.mute_mic_while_speaking = False
    if getattr(args, "serve", False):
        cfg.server.enabled = True
    if getattr(args, "port", None) is not None:
        cfg.server.port = args.port
    if getattr(args, "bind", None):
        cfg.server.host = args.bind
    if getattr(args, "tls", False) or getattr(args, "no_tls", False):
        cfg.server.tls = bool(getattr(args, "tls", False))
    return cfg


def available_targets(cfg: AppConfig) -> list[str]:
    """Configured targets whose model is FULLY on disk (both weights AND the
    sentencepiece tokeniser). A partial directory - e.g. from an interrupted
    download - is treated as missing so we don't crash later in Translator init."""
    kept = []
    for t in cfg.mt.targets:
        d = cfg.mt_dir_for(t)
        if (d / "model.bin").exists() and (d / "source.spm").exists():
            kept.append(t)
        else:
            reason = "not downloaded" if not d.exists() else "partial download"
            print(f"  !! skipping '{t}': {reason} "
                  f"(fetch with:  python scripts/fetch_models.py --lang {t})")
    return kept


def require_models(cfg: AppConfig, need_stt: bool = True, need_tts: bool = False) -> None:
    missing = []
    if need_stt:
        d = cfg.stt_model_dir
        marker = "encoder.int8.onnx" if cfg.stt.engine == "parakeet" else "model.bin"
        if not (d / marker).exists():
            missing.append(f"{cfg.stt.engine} '{cfg.stt.model}' at {d}")
    # Only the PRIMARY target is required. Secondary targets fall through to
    # available_targets(), which just skips missing ones.
    if not (cfg.mt_dir / "model.bin").exists():
        missing.append(f"OPUS-MT en-{cfg.mt.target} at {cfg.mt_dir}")
    if need_tts and not cfg.tts_model.exists():
        missing.append(f"Piper voice '{cfg.tts.voice}' at {cfg.tts_model}")
    if missing:
        print("Models are not downloaded yet:")
        for m in missing:
            print(f"  - {m}")
        print("\nRun this once (needs internet):")
        print("  python scripts/fetch_models.py")
        sys.exit(2)


# --------------------------------------------------------------- subcommands

def cmd_devices(args) -> int:
    from sermon.audio import MicrophoneSource
    print(MicrophoneSource.list_devices())
    return 0


def cmd_mic(args) -> int:
    """Live level meter. Confirms the mic works without loading any model."""
    from sermon.audio import MicrophoneSource, level_verdict

    cfg = build_config(args)
    src = MicrophoneSource(cfg.audio)

    print()
    print("  MICROPHONE LEVEL METER - no models loaded.")
    print()
    print("  KEEP TALKING while this runs, at normal preaching volume.")
    print("  Adjust the Windows mic slider until PEAK sits in the target zone.")
    print()
    print("      too quiet        TARGET        clipping")
    print("     <---------->  <------------>  <-------->")
    print("    -60         -20  -18      -6   -3       0  dBFS")
    print()

    try:
        with src:
            while True:
                time.sleep(0.1)
                db, peak = src.last_db, src.peak_hold_db
                bars = int(max(0, min(44, (db + 60) * 44 / 60)))
                marker = int(max(0, min(43, (peak + 60) * 44 / 60)))
                meter = list("-" * 44)
                for i in range(bars):
                    meter[i] = "#"
                meter[marker] = "|"
                verdict, _, _ = level_verdict(peak, src.clip_blocks)
                print(f"\r  {''.join(meter)}  now {db:6.1f}  peak {peak:6.1f}  "
                      f"{verdict}  thr {src.threshold_db:6.1f}  "
                      f"{'SPEECH' if src.in_speech else '      '}",
                      end="", flush=True)
    except KeyboardInterrupt:
        pass

    peak = src.session_peak_db
    print("\n")
    print(f"  Loudest sample this session : {peak:.1f} dBFS")
    print(f"  Clipped audio blocks        : {src.clip_blocks}")
    print(f"  Measured noise floor        : {src.noise_floor_db:.1f} dBFS")
    print(f"  Dropped blocks / warnings   : {src.overflows} / {src.stream_warnings}")
    print()
    _, advice, acceptable = level_verdict(peak, src.clip_blocks)
    print(f"  VERDICT: {advice}")
    if acceptable:
        print("           Run it live:  python run.py")
    else:
        print("           Adjust in mmsys.cpl > Recording > Microphone >")
        print("           Properties > Levels, then run this meter again.")
    print()
    return 0


def cmd_doctor(args) -> int:
    from sermon.diagnostics import run_doctor
    cfg = build_config(args)
    return run_doctor(cfg, mic_seconds=0.0 if args.no_mic else args.seconds)


def cmd_translate(args) -> int:
    from sermon.mt import MarianCT2Translator
    cfg = build_config(args)
    # Show all configured targets, or just the one from --to.
    targets = [args.to] if args.to else available_targets(cfg)
    if not targets:
        return 2
    text = " ".join(args.text)
    print(f"EN  {text}")
    for t in targets:
        model_dir = cfg.mt_dir_for(t)
        if not (model_dir / "model.bin").exists():
            print(f"{t.upper():<3} (model not downloaded)")
            continue
        from sermon.config import phrase_file_for
        from sermon.mt import OverrideTranslator, load_phrase_overrides
        tr = MarianCT2Translator(model_dir, replace(cfg.mt, targets=[t]))
        rules = load_phrase_overrides(phrase_file_for(t))
        if rules:
            tr = OverrideTranslator(tr, rules)
        res = tr.translate(text)
        print(f"{t.upper():<3} {res.text}")
        print(f"    ({res.compute_seconds * 1000:.0f} ms, fully local)")
    return 0


def cmd_speak(args) -> int:
    """Translate text and speak it. No microphone, no Whisper."""
    import sounddevice as sd

    from sermon.mt import MarianCT2Translator
    from sermon.tts import PiperSpeaker

    cfg = build_config(args)
    require_models(cfg, need_tts=True)

    text = " ".join(args.text)
    tr = MarianCT2Translator(cfg.mt_dir, cfg.mt)
    french = tr.translate(text).text
    print(f"EN  {text}")
    print(f"FR  {french}")

    speaker = PiperSpeaker(cfg.tts_model, cfg.tts)
    speech = speaker.synthesize(french)
    print(f"    voice {cfg.tts.voice} | {speech.duration:.1f}s audio in "
          f"{speech.compute_seconds:.2f}s (rtf {speech.rtf:.2f}) | fully local")

    if args.save:
        import wave
        pcm = (speech.audio * 32767).clip(-32768, 32767).astype("<i2")
        with wave.open(args.save, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(speech.sample_rate)
            w.writeframes(pcm.tobytes())
        print(f"    saved {args.save} - open it in any player to check the audio")
        return 0

    out = cfg.tts.output_device
    name = sd.query_devices(out if out is not None else sd.default.device[1])["name"]
    print(f"    playing on [{out if out is not None else 'default'}] {name}")
    sd.play(speech.audio, speech.sample_rate, device=out)
    sd.wait()
    print("    done. If you heard nothing, check the Windows volume mixer, or try")
    print("    'run.py devices' and pass --output-device N, or --save out.wav")
    return 0


def cmd_live(args) -> int:
    from sermon.audio import (MicrophoneSource, NetworkMicrophoneSource,
                              SelectableSource, SystemAudioSource)
    from sermon.mt import MarianCT2Translator
    from sermon.pipeline import Pipeline
    from sermon.sinks import ConsoleSink
    from sermon.stt import FasterWhisperTranscriber, ParakeetTranscriber

    cfg = build_config(args)
    require_models(cfg, need_tts=cfg.tts.enabled)

    targets = available_targets(cfg)
    if not targets:
        print("No translation models available. Run: python scripts/fetch_models.py")
        return 2

    print("=" * 70)
    print("  LOCAL SERMON TRANSLATION (English -> "
          + " + ".join(t.upper() for t in targets) + ")")
    print("=" * 70)
    if cfg.stt.engine == "parakeet":
        print(f"  STT   Parakeet {cfg.stt.model} (int8, {cfg.stt.cpu_threads} threads)")
    else:
        print(f"  STT   faster-whisper {cfg.stt.model} ({cfg.stt.compute_type}, "
              f"{cfg.stt.cpu_threads} threads)")
    for t in targets:
        print(f"  MT    OPUS-MT en-{t} (CTranslate2 int8)")
    if cfg.tts.enabled:
        print(f"  TTS   Piper {cfg.tts.voice}  (speaks {cfg.mt.target})")
    print("  Net   offline - no API keys, nothing leaves this machine")
    print()

    t0 = time.perf_counter()
    print("  Loading speech model...", end="", flush=True)
    if cfg.stt.engine == "parakeet":
        transcriber = ParakeetTranscriber(cfg.parakeet_dir, cfg.stt)
    else:
        transcriber = FasterWhisperTranscriber(cfg.whisper_dir, cfg.stt)
    print(f" {time.perf_counter() - t0:.1f}s")

    from sermon.config import phrase_file_for
    from sermon.mt import OverrideTranslator, load_phrase_overrides

    translators: dict[str, "MarianCT2Translator"] = {}
    for t in targets:
        t0 = time.perf_counter()
        print(f"  Loading translation model en-{t}...", end="", flush=True)
        tr_cfg = replace(cfg.mt, targets=[t])
        try:
            tr = MarianCT2Translator(cfg.mt_dir_for(t), tr_cfg)
        except Exception as exc:
            # One bad target must not take the service down. With nine models
            # loaded at once a small-RAM machine hits `mkl_malloc: failed to
            # allocate memory` partway through, and a corrupt download raises
            # here too. Drop that language and carry on with the rest.
            print(f" FAILED ({type(exc).__name__}: {str(exc)[:60]})")
            print(f"     en-{t} unavailable; continuing without it.")
            continue
        rules = load_phrase_overrides(phrase_file_for(t))
        translators[t] = OverrideTranslator(tr, rules) if rules else tr
        extra = f" [+ {len(rules)} phrase overrides]" if rules else ""
        print(f" {time.perf_counter() - t0:.1f}s{extra}")

    if not translators:
        print("  !! no translation models could be loaded.")
        return 2
    # Targets narrows to what actually loaded, so the listener UI only ever
    # offers languages the server can really produce.
    targets = [t for t in targets if t in translators]

    source_kind = getattr(args, "source", "mic")
    use_network_mic = source_kind == "browser"
    use_system_audio = source_kind == "system"
    if use_network_mic and not cfg.server.enabled:
        print("  --source browser needs --serve. Turning --serve on.")
        cfg.server.enabled = True
    if use_network_mic and not cfg.server.tls and not getattr(args, "no_tls", False):
        print("  --source browser needs HTTPS for phone mic access. Turning --tls on.")
        cfg.server.tls = True
    if use_system_audio:
        # SelectableSource is a SystemAudioSource that can also drive a normal
        # input device, switched from the operator UI at runtime. Starting in
        # "system" mode keeps `--source system` behaving exactly as before,
        # but the app is no longer deaf to the microphone.
        source = SelectableSource(cfg.audio, kind="system")
        # System audio needs the explicit Start button; auto-serve so the
        # operator UI can talk to it.
        cfg.server.enabled = True
    elif use_network_mic:
        source = NetworkMicrophoneSource(cfg.audio)
    else:
        source = MicrophoneSource(cfg.audio)
    sinks = [ConsoleSink(show_timing=not args.quiet)]
    speech_sink = None
    server_handle = None
    broadcast_sink = None

    if cfg.server.enabled:
        from sermon.server import Bind, Broadcaster, ServerHandle, list_local_ips
        from sermon.sinks import BroadcastSink

        broadcaster = Broadcaster(history_size=cfg.server.history_size)

        ssl_ctx = None
        if cfg.server.tls:
            from sermon.tls import ssl_context
            from sermon.server import list_local_ips as _ips_for_cert
            ssl_ctx = ssl_context(_ips_for_cert())

        # Build the bind list. Normal case: one bind (the port, TLS or not).
        # Tauri desktop-app case: add a second bind on loopback HTTP so the
        # embedded webview can hit http://127.0.0.1:8000/speaker without a
        # self-signed cert warning (loopback = secure context regardless).
        binds = [Bind(cfg.server.host, cfg.server.port, ssl_ctx)]
        if getattr(args, "tauri", False):
            # Follow --port. The loopback bind used to be pinned to 8000 no
            # matter what, so moving the server off a busy port still left the
            # desktop window pointing at whatever else owned 8000.
            loopback = (cfg.server.port if getattr(args, "port", None)
                        else cfg.server.loopback_port)
            binds.append(Bind("127.0.0.1", loopback, None))

        # Every source the server may need to talk to (pause/resume, receive
        # network mic frames, read live level) gets handed in. Previously we
        # only did this for the network mic - which meant the operator UI's
        # Start/Stop couldn't reach the SystemAudioSource.
        server_handle = ServerHandle(
            broadcaster, binds=binds,
            mic_source=source)
        try:
            server_handle.start()
        except OSError as exc:
            print(f"  !! could not start server on port {cfg.server.port}: {exc}")
            print( "     Something else is probably already using that port.")
            print(f"     See what:  netstat -ano | findstr :{cfg.server.port}")
            print( "     Then either stop it, or run on another port:")
            print( "       python run.py --serve --port 8080")
            return 2
        # Server-side audio: one Piper voice per translation target, so any
        # listener language ends up as an audio clip they can fetch by HTTP.
        # A muted phone or a phone on a language we can't voice pays zero.
        audio_speakers: dict[str, "PiperSpeaker"] = {}
        if not args.no_audio:
            from sermon.tts import PiperSpeaker

            # Voices load CONCURRENTLY. Sequentially this is ~6s each, which
            # was tolerable for two languages and is a minute of staring at a
            # loading screen for nine. Each still gets its own daemon thread
            # and a hard timeout, because Piper can deadlock in espeak-ng init
            # on some Windows configurations; a stuck voice is skipped (text
            # still translates) rather than hanging startup.
            import threading
            wanted = [(t, cfg.tts_model_for(cfg.tts.voices[t]))
                      for t in targets if t in cfg.tts.voices]
            results: dict[str, dict] = {}
            threads = []
            for t, model_path in wanted:
                if not model_path.exists():
                    print(f"  !! Piper voice for {t} missing ({model_path.name}) - "
                          f"server audio for {t} disabled.")
                    continue
                slot: dict = {}
                results[t] = slot

                def _worker(mp=model_path, slot=slot):
                    try:
                        slot["speaker"] = PiperSpeaker(mp, cfg.tts)
                    except Exception as exc:
                        slot["error"] = exc

                th = threading.Thread(target=_worker, daemon=True,
                                      name=f"piper-load-{t}")
                th.start()
                threads.append((t, th))

            if threads:
                t0 = time.perf_counter()
                print(f"  Loading {len(threads)} server voices in parallel...",
                      end="", flush=True)
                deadline = time.monotonic() + 60
                for _, th in threads:
                    th.join(timeout=max(0.1, deadline - time.monotonic()))
                print(f" {time.perf_counter() - t0:.1f}s")
                for t, th in threads:
                    slot = results[t]
                    if "speaker" in slot:
                        audio_speakers[t] = slot["speaker"]
                    elif "error" in slot:
                        print(f"     {t}: failed ({slot['error']})")
                    else:
                        # Orphaned daemon thread stuck in C-level ONNX init.
                        # It dies with the process; the pipeline runs on.
                        print(f"     {t}: timed out; text still works.")

        # Tell listeners what they can actually pick. English first: it is the
        # source language, needs no model, and is what the room is hearing.
        # Only targets whose MT model really loaded are offered, so a missing
        # model shows up as an absent chip rather than a blank screen.
        from sermon.config import LANGUAGES, SOURCE_LANG
        offered = [SOURCE_LANG] + [t for t in targets if t in LANGUAGES]
        broadcaster.languages = offered
        broadcaster.language_meta = {
            code: {**LANGUAGES[code],
                   "voice": code in audio_speakers,
                   "source": code == SOURCE_LANG}
            for code in offered if code in LANGUAGES
        }

        broadcast_sink = BroadcastSink(broadcaster, speakers=audio_speakers)
        sinks.append(broadcast_sink)

        ips = list_local_ips()
        scheme = "https" if cfg.server.tls else "http"
        print()
        print("  SERVER   listener page (audience opens this on their phones):")
        for ip in ips or ["<no LAN interface found>"]:
            print(f"           {scheme}://{ip}:{cfg.server.port}/")
        if use_network_mic:
            print()
            print("  SPEAKER  page for the preacher (opens on their own phone):")
            for ip in ips or ["<no LAN interface found>"]:
                print(f"           {scheme}://{ip}:{cfg.server.port}/speaker")
            print("           Tap 'Start speaking' on that page to open the mic.")
        if cfg.server.tls:
            print()
            print("  TLS      Cert is self-signed. First time each phone visits,")
            print("           the browser will warn 'not secure' - tap Advanced,")
            print("           then 'Proceed anyway' / 'Visit this website'.")
        if audio_speakers:
            print(f"  AUDIO    server-synth audio available for: "
                  f"{', '.join(audio_speakers)}")
        print("  FIREWALL Windows may prompt on first run - click Allow.")
        print("           If phones cannot reach the URL, run `python run.py doctor`.")
        print()

    if cfg.tts.enabled:
        from sermon.sinks import SpeechSink
        from sermon.tts import PiperSpeaker

        t0 = time.perf_counter()
        print("  Loading French voice...", end="", flush=True)
        speaker = PiperSpeaker(cfg.tts_model, cfg.tts)
        print(f" {time.perf_counter() - t0:.1f}s")
        speech_sink = SpeechSink(
            speaker, cfg.tts,
            gate=source if cfg.tts.mute_mic_while_speaking else None,
            target=cfg.mt.target,   # primary target only; no Yoruba TTS
        )
        sinks.append(speech_sink)

    pipeline = Pipeline(transcriber, translators, sinks,
                        sample_rate=cfg.audio.sample_rate)

    try:
        with source:
            print(f"\n  Calibrating room noise ({cfg.audio.calibrate_s:.1f}s) - stay quiet...")
            deadline = time.monotonic() + cfg.audio.calibrate_s + 1.0
            while not source.calibrated and time.monotonic() < deadline:
                time.sleep(0.1)
            print(f"  Noise floor {source.noise_floor_db:.1f} dBFS, "
                  f"speech threshold {source.threshold_db:.1f} dBFS")
            if source.noise_floor_db > -50.0:
                print("  !! That noise floor is high - the room was not quiet during")
                print("     calibration. Quiet syllables may be cut. It will settle")
                print("     on its own after a real pause; restart for a clean start.")
            if cfg.tts.enabled and cfg.tts.mute_mic_while_speaking:
                print("  Microphone mutes while French plays (use headphones or a")
                print("  separate speaker to keep listening during playback).")
            print("\n  LISTENING. Speak in English. Press Ctrl+C to stop.")
            print("-" * 70)
            pipeline.run(source)
    except KeyboardInterrupt:
        pass
    finally:
        if speech_sink is not None:
            print("\n  Finishing speech playback...")
            speech_sink.close()
        if server_handle is not None:
            print("  Stopping server...")
            server_handle.stop()
        print("\n" + "-" * 70)
        print(f"  Utterances translated : {pipeline.processed}")
        print(f"  Discarded as non-speech: {pipeline.skipped_empty}")
        print(f"  Clipped input          : {pipeline.clipped}")
        print(f"  Dropped (CPU backlog)  : {source.dropped}")
        print(f"  Audio block overflows  : {source.overflows}")
        if pipeline.clipped:
            print("  -> Mic gain is too high. Lower it in mmsys.cpl > Recording >")
            print("     Microphone > Properties > Levels (and set Boost to 0 dB).")
        if source.dropped:
            print("  -> CPU could not keep up. Try: python run.py --silence 1200")
        if broadcast_sink is not None:
            b = broadcast_sink.broadcaster
            print(f"  Peak Wi-Fi listeners   : {b.peak_clients}")
            print(f"  Currently connected    : {b.client_count}")
            print(f"  Server audio generated : {broadcast_sink.audio_generated} clips "
                  f"({broadcast_sink.audio_seconds_generated:.1f}s of speech)")
            if broadcast_sink.audio_generated:
                print(f"  Server synth cost      : "
                      f"{broadcast_sink.synth_seconds:.1f}s "
                      f"(rtf {broadcast_sink.synth_seconds / max(broadcast_sink.audio_seconds_generated, 1e-6):.2f})")
                mb = b.audio_bytes_served / 1024 / 1024
                print(f"  Audio bytes served     : {b.audio_requests} requests, {mb:.1f} MB")
            if broadcast_sink.synth_failures:
                print(f"  Server synth failures  : {broadcast_sink.synth_failures}")
        if speech_sink is not None:
            print(f"  Phrases spoken        : {speech_sink.spoken}")
            print(f"  Phrases dropped (TTS)  : {speech_sink.dropped}")
            if speech_sink.failures:
                print(f"  Speech failures        : {speech_sink.failures}")
            if speech_sink.speech_seconds:
                print(f"  Synthesis speed        : "
                      f"{speech_sink.synth_seconds / speech_sink.speech_seconds:.2f} rtf")
            print(f"  Mic muted for          : {source.muted_seconds:.1f}s")
            if speech_sink.dropped:
                print("  -> Speech could not keep up with speech input. This is")
                print("     expected if you speak continuously; French playback")
                print("     takes as long as the original.")
        print("  Stopped.")
    return 0


# --------------------------------------------------------------------- parser

def main(argv=None) -> int:
    enable_utf8_console()

    p = argparse.ArgumentParser(
        prog="run.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["tiny.en", "base.en", "small.en"],
                   help="Whisper model size (default: base.en)")
    p.add_argument("--stt",
                   choices=["whisper-tiny.en", "whisper-base.en", "whisper-small.en",
                            "parakeet-110m", "parakeet-0.6b"],
                   help="STT engine + model. Overrides --model. "
                        "Whisper: supports vocabulary hints. "
                        "Parakeet: ~4x faster on CPU, no vocab support.")
    p.add_argument("--device", type=int, help="audio input device index (see: run.py devices)")
    p.add_argument("--threads", type=int, help="CPU threads for inference")
    p.add_argument("--silence", type=int, metavar="MS",
                   help="pause that can end an utterance (default 600)")
    p.add_argument("--min-utterance", type=float, metavar="S",
                   help="do not cut on a pause below this length (default 5); "
                        "raise it for better accuracy, lower it for less lag")
    p.add_argument("--max-utterance", type=float, metavar="S",
                   help="hard flush for unbroken speech (default 14)")
    p.add_argument("--long-utterances", action="store_true",
                   help="keep Whisper's longer-utterance segmentation even on "
                        "Parakeet: more context per utterance, more lag")
    p.add_argument("--threshold", type=float, metavar="DB",
                   help="dB above noise floor counted as speech (default 7)")
    p.add_argument("--beam", type=int, help="Whisper beam size (default 5; 1 = greedy)")
    p.add_argument("--no-prompt", action="store_true",
                   help="disable the sermon vocabulary prompt")
    p.add_argument("--quiet", action="store_true", help="hide per-utterance timings")

    mt = p.add_argument_group("translation targets")
    mt.add_argument("--no-yoruba", action="store_true",
                    help="skip Yoruba (French only)")
    mt.add_argument("--targets", metavar="fr,yo",
                    help="comma-separated list, overrides defaults")

    tts = p.add_argument_group("French speech output")
    tts.add_argument("--speak", action="store_true", help="speak the French translation")
    tts.add_argument("--voice", help="Piper voice (default fr_FR-siwis-medium)")
    tts.add_argument("--speed", type=float, metavar="X",
                     help="delivery speed: >1 slower, <1 faster (default 1.0)")
    tts.add_argument("--output-device", type=int, metavar="N",
                     help="audio output device index (see: run.py devices)")
    tts.add_argument("--no-mute", action="store_true",
                     help="do NOT mute the mic during playback (headphones only)")

    src = p.add_argument_group("audio source")
    src.add_argument("--source", choices=["mic", "browser", "system"], default="mic",
                     help="mic = laptop microphone (default). "
                          "browser = wait for a speaker page to open a phone mic. "
                          "system = capture what the laptop is currently playing "
                          "(WASAPI loopback; Windows only).")

    net = p.add_argument_group("Wi-Fi streaming to phones")
    net.add_argument("--serve", action="store_true",
                     help="serve translations to phones on the same Wi-Fi (default port 8000)")
    net.add_argument("--port", type=int, metavar="N", help="server port (default 8000)")
    net.add_argument("--bind", metavar="ADDR",
                     help="bind address (default 0.0.0.0 = every interface; "
                          "use 127.0.0.1 to keep it to this laptop)")
    net.add_argument("--no-audio", action="store_true",
                     help="do NOT synthesise server-side audio for phones "
                          "(text only; saves ~40 KB/s per active listener)")
    net.add_argument("--tls", action="store_true",
                     help="serve over HTTPS with a self-signed cert. Required "
                          "for phone mic access under --source browser. Auto-on "
                          "when --source browser.")
    net.add_argument("--no-tls", action="store_true",
                     help="force plain HTTP even under --source browser")
    net.add_argument("--tauri", action="store_true",
                     help="desktop-app mode: also bind HTTP on 127.0.0.1 "
                          "for the embedded webview (no cert warning); "
                          "LAN listeners still get HTTPS")

    p.set_defaults(func=cmd_live)

    sub = p.add_subparsers()

    d = sub.add_parser("doctor", help="run full diagnostics")
    d.add_argument("--seconds", type=float, default=6.0, help="mic test length")
    d.add_argument("--no-mic", action="store_true", help="skip the microphone test")
    d.set_defaults(func=cmd_doctor)

    sub.add_parser("devices", help="list audio devices").set_defaults(func=cmd_devices)
    sub.add_parser("mic", help="live microphone level meter").set_defaults(func=cmd_mic)

    t = sub.add_parser("translate", help="translate text without the microphone")
    t.add_argument("text", nargs="+")
    t.add_argument("--to", metavar="LANG",
                   help="single target (e.g. yo). Default: every configured target.")
    t.set_defaults(func=cmd_translate)

    sp = sub.add_parser("speak", help="translate text and say it out loud")
    sp.add_argument("text", nargs="+")
    sp.add_argument("--save", metavar="FILE.wav",
                    help="write the audio to a WAV instead of playing it")
    sp.set_defaults(func=cmd_speak)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
