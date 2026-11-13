"""Output destinations.

A sink is anything with `emit(result)`. Phase 1 ships one console sink; later
phases add a French TTS sink and a websocket broadcaster by appending to
`Pipeline.sinks` -- no change to the pipeline itself.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from .config import SOURCE_LANG, TTSConfig
from .console import enable_utf8_console  # re-exported for convenience
from .pipeline import PipelineResult
from .tts import Speaker

# Same reasoning as pipeline.py's fan-out pool: synthesis is CPU-bound C++
# (onnxruntime) work that releases the GIL, and VitsSpeaker now pins its own
# session to one thread specifically so several of these can run at once
# without each one reaching for every core.
_SYNTH_WORKERS = max(2, min(4, os.cpu_count() or 2))

__all__ = ["Sink", "ConsoleSink", "SpeechSink", "BroadcastSink",
           "enable_utf8_console"]


class Sink(ABC):
    @abstractmethod
    def emit(self, result: PipelineResult) -> None:
        ...


class ConsoleSink(Sink):
    """Prints English plus every translation, keyed by two-letter code."""

    def __init__(self, show_timing: bool = True):
        self.show_timing = show_timing
        self.counter = 0

    def emit(self, result: PipelineResult) -> None:
        # Printing must never be able to kill the pipeline. With Arabic, Hindi
        # and Chinese in the target set this stopped being hypothetical: a
        # console stuck on a legacy code page raises UnicodeEncodeError, and in
        # the windowed build stdout may not be a stream at all.
        try:
            self._emit(result)
        except Exception:
            self.counter += 0
            try:
                sys.stderr.write("  !! console output failed for this line\n")
            except Exception:
                pass

    def _emit(self, result: PipelineResult) -> None:
        self.counter += 1
        stamp = datetime.now().strftime("%H:%M:%S")
        print()
        print(f"[{stamp}] #{self.counter}")
        print(f"  EN  {result.english}")
        for code, text in result.translations.items():
            print(f"  {code.upper():<3} {text}")
        if result.clipped:
            print(f"  !!  input clipped ({result.peak_db:.1f} dBFS) - "
                  f"lower the mic gain, accuracy is degraded")
        if self.show_timing:
            mt_breakdown = " ".join(f"{c} {s:.2f}s"
                                    for c, s in result.mt_seconds.items())
            print(
                f"      ({result.audio_seconds:.1f}s audio | "
                f"stt {result.stt_seconds:.2f}s | mt {mt_breakdown} | "
                f"lag {result.total_latency:.2f}s | rtf {result.rtf:.2f})"
            )
        sys.stdout.flush()


class SpeechSink(Sink):
    """Speaks one target language out of the speakers.

    Synthesis and playback run on their own thread: a 6 s phrase takes 6 s to
    play, and the recognition pipeline must not wait for it. If speech backs up,
    the OLDEST pending phrase is dropped - during a live sermon, current text
    matters more than complete text.

    `target` picks which language of `PipelineResult.translations` to speak
    (default 'fr'). `gate` is anything with mute()/unmute() (i.e. the
    MicrophoneSource); without it, speakers feed straight back into the mic.
    """

    def __init__(self, speaker: Speaker, cfg: TTSConfig, gate=None, target: str = "fr"):
        self.speaker = speaker
        self.cfg = cfg
        self.gate = gate
        self.target = target
        self.spoken = 0
        self.dropped = 0
        self.failures = 0
        self.synth_seconds = 0.0
        self.speech_seconds = 0.0

        self._q: queue.Queue = queue.Queue(maxsize=cfg.queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="tts")
        self._thread.start()

    def emit(self, result: PipelineResult) -> None:
        text = result.translations.get(self.target, "")
        if not text:
            return
        while True:
            try:
                self._q.put_nowait(text)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def close(self, timeout: float = 5.0) -> None:
        """Let anything already queued finish, then stop the thread."""
        deadline = time.monotonic() + timeout
        while not self._q.empty() and time.monotonic() < deadline:
            time.sleep(0.1)
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self.gate is not None:
            self.gate.unmute()

    # ------------------------------------------------------------------ worker

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                speech = self.speaker.synthesize(text)
                if len(speech.audio) == 0:
                    continue
                self.synth_seconds += speech.compute_seconds
                self.speech_seconds += speech.duration
                self._play(speech)
                self.spoken += 1
            except Exception as exc:                      # never kill the thread
                self.failures += 1
                print(f"\n  !! speech failed: {exc}", file=sys.stderr)

    def _play(self, speech) -> None:
        import sounddevice as sd

        if self.gate is not None and self.cfg.mute_mic_while_speaking:
            self.gate.mute()
        try:
            sd.play(speech.audio, speech.sample_rate, device=self.cfg.output_device)
            sd.wait()
        finally:
            if self.gate is not None and self.cfg.mute_mic_while_speaking:
                time.sleep(self.cfg.mic_resume_delay_ms / 1000.0)
                self.gate.unmute()


class BroadcastSink(Sink):
    """Pushes each translation to phones connected over Wi-Fi.

    If a `speaker` is provided (typically the same PiperSpeaker used by
    SpeechSink), each French translation is also synthesised to a WAV and
    cached on the broadcaster. The WebSocket message then carries a URL like
    "/audio/42.fr.wav" which the phone fetches via HTTP.

    Why URL rather than base64-in-JSON: only phones with voice ON and set to
    French actually download the clip, so muted or Yoruba-only listeners cost
    zero audio bandwidth.

    Synthesis is done inline (Piper on this hardware is rtf ~0.09; a 6 s
    sentence takes ~0.4 s). The pipeline's console sink runs before us so text
    output is instant either way.
    """

    def __init__(self, broadcaster, speakers: dict[str, Speaker] | None = None,
                 recorder=None):
        """`speakers` maps target language ("fr", "es", ...) to a loaded
        PiperSpeaker. Every listed language gets audio synthesised and
        cached; a listener on that language fetches it by HTTP.

        `recorder`, if given, can ask for a language's audio to be
        synthesised for RECORDING even when no listener is on it - the
        Recorder is the one place that needs that, and it gets the clip this
        emits instead of synthesising its own copy.
        """
        self.broadcaster = broadcaster
        self.speakers = speakers or {}
        self.recorder = recorder
        self.counter = 0
        self.audio_generated = 0
        self.audio_seconds_generated = 0.0
        self.synth_seconds = 0.0
        self.synth_failures = 0
        self._pool = ThreadPoolExecutor(
            max_workers=_SYNTH_WORKERS, thread_name_prefix="tts-fanout")
        # One worker draining a FIFO queue, not one thread per utterance: it
        # keeps the per-language synthesis fan-out above concurrent (still
        # via _pool) while guaranteeing utterances reach broadcast_audio()
        # and the recorder's per-language files in the order they were
        # spoken. A thread per utterance could let a fast, late utterance's
        # audio land before a slow, earlier one's - fine for live playback
        # (each message carries its own id) but it would scramble the
        # recorded WAV files, which are appended to in whatever order calls
        # arrive.
        self._work_q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._broadcast_thread = threading.Thread(
            target=self._broadcast_worker, daemon=True, name="tts-broadcast")
        self._broadcast_thread.start()

    def close(self) -> None:
        self._stop.set()
        self._broadcast_thread.join(timeout=2.0)
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _broadcast_worker(self) -> None:
        while not self._stop.is_set():
            try:
                result, counter, pending = self._work_q.get(timeout=0.2)
            except queue.Empty:
                continue
            self._synthesize_and_broadcast(result, counter, pending)

    def emit(self, result: PipelineResult) -> None:
        self.counter += 1
        counter = self.counter
        # Synthesis is by far the most expensive stage - roughly 1s per
        # language against 0.5s for translation - and it is the only one that
        # scales badly with the target count. Text goes to everyone because it
        # is cheap and a listener switching language should see it instantly;
        # audio is produced ONLY for languages someone is actually listening
        # to, OR a language the recorder has been told to capture. Ten
        # languages offered costs the same as three in use (plus whatever is
        # being recorded).
        wanted = self.broadcaster.active_languages()
        if self.recorder is not None:
            wanted = wanted | self.recorder.wanted_languages()
        # list(): under on-demand loading another thread adds/removes voices.
        pending = [lang for lang in list(self.speakers)
                   if result.translations.get(lang) and lang in wanted]
        message = {
            "type": "translation",
            "id": counter,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "english": result.english,
            "audio_seconds": round(result.audio_seconds, 1),
            "latency": round(result.total_latency, 2),
        }
        message.update(result.translations)
        # English is the source language: there is no translator behind it,
        # but listeners can still select it, so expose it under its own code
        # the same way every target is exposed.
        message[SOURCE_LANG] = result.english
        # Tell the phone which languages have audio coming, so it shows the
        # text now and waits for the clip instead of racing ahead with the
        # browser's own robotic voice.
        if pending:
            message["audio_pending"] = pending

        # Send the TEXT first. Synthesis costs ~2s for two languages on this
        # laptop, and every phone used to sit on a blank screen for all of it.
        self.broadcaster.broadcast(message)

        if not pending:
            return

        # Queued, not awaited: this used to call fut.result() on this thread,
        # which is the SAME thread Pipeline.process() calls emit() on. That
        # made synthesis time for THIS utterance a hard floor on when the
        # NEXT utterance's transcription could even start - on a slower
        # machine, or under the fan-out pool's own CPU contention among
        # several heavy VITS voices (Igbo/Yoruba/Hausa), a few seconds of
        # synthesis lag compounded utterance after utterance into a real
        # backlog: captions freezing and the room's phones falling back to
        # their own (wrong-sounding) voice for a stretch before catching up.
        # The background worker drains this queue independently of STT/MT.
        self._work_q.put((result, counter, pending))

    def _synthesize_and_broadcast(self, result: PipelineResult, counter: int,
                                   pending: list[str]) -> None:
        # Captured now: a voice may be unloaded between here and its turn in
        # the pool, and holding the reference keeps this clip synthesizable.
        voices = {lang: self.speakers.get(lang) for lang in pending}
        pending = [lang for lang in pending if voices[lang] is not None]

        def _synthesize(lang):
            speech = voices[lang].synthesize(result.translations[lang])
            return lang, speech

        # Synthesising every pending language concurrently, not one after
        # another, is what keeps a language late in the list from missing the
        # listener page's 6s wait-for-server-audio window (see pipeline.py's
        # docstring - this was the other half of that same latency problem).
        futures = {self._pool.submit(_synthesize, lang): lang for lang in pending}
        urls: dict[str, str] = {}
        for fut in futures:
            lang = futures[fut]
            try:
                lang, speech = fut.result()
                if speech.audio.size:
                    wav = _encode_wav_int16(speech.audio, speech.sample_rate)
                    urls[f"audio_{lang}"] = self.broadcaster.store_audio(
                        counter, lang, "audio/wav", wav)
                    self.audio_generated += 1
                    self.audio_seconds_generated += speech.duration
                    self.synth_seconds += speech.compute_seconds
                    if self.recorder is not None:
                        label = (self.broadcaster.language_meta.get(lang) or {}).get("native", lang.upper())
                        self.recorder.write_lang(lang, speech.audio, speech.sample_rate, label)
            except Exception as exc:
                self.synth_failures += 1
                print(f"\n  !! server audio failed for {lang}: {exc}",
                      file=sys.stderr)
        # Sent even when every synthesis failed: the message is what releases
        # the phone from waiting, and an empty one tells it to fall back to
        # its own voice rather than stay silent.
        self.broadcaster.broadcast_audio(counter, urls)


def _encode_wav_int16(audio, sample_rate: int) -> bytes:
    """Serialise a float32 [-1, 1] mono array as a 16-bit PCM WAV."""
    import io
    import wave

    import numpy as np

    pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
