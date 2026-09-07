"""Output destinations.

A sink is anything with `emit(result)`. Phase 1 ships one console sink; later
phases add a French TTS sink and a websocket broadcaster by appending to
`Pipeline.sinks` -- no change to the pipeline itself.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime

from .config import TTSConfig
from .console import enable_utf8_console  # re-exported for convenience
from .pipeline import PipelineResult
from .tts import Speaker

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

    def __init__(self, broadcaster, speakers: dict[str, Speaker] | None = None):
        """`speakers` maps target language ("fr", "es", ...) to a loaded
        PiperSpeaker. Every listed language gets audio synthesised and
        cached; a listener on that language fetches it by HTTP."""
        self.broadcaster = broadcaster
        self.speakers = speakers or {}
        self.counter = 0
        self.audio_generated = 0
        self.audio_seconds_generated = 0.0
        self.synth_seconds = 0.0
        self.synth_failures = 0

    def emit(self, result: PipelineResult) -> None:
        self.counter += 1
        counter = self.counter
        pending = [lang for lang in self.speakers
                   if result.translations.get(lang)]
        message = {
            "type": "translation",
            "id": counter,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "english": result.english,
            "audio_seconds": round(result.audio_seconds, 1),
            "latency": round(result.total_latency, 2),
        }
        message.update(result.translations)
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
        urls: dict[str, str] = {}
        for lang in pending:
            try:
                speech = self.speakers[lang].synthesize(result.translations[lang])
                if speech.audio.size:
                    wav = _encode_wav_int16(speech.audio, speech.sample_rate)
                    urls[f"audio_{lang}"] = self.broadcaster.store_audio(
                        counter, lang, "audio/wav", wav)
                    self.audio_generated += 1
                    self.audio_seconds_generated += speech.duration
                    self.synth_seconds += speech.compute_seconds
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
