"""Microphone capture + energy-based utterance segmentation.

Design notes
------------
* The PortAudio callback does nothing but copy bytes into a queue. Any real work
  there would cause dropouts on a 2-core laptop.
* A segmenter thread turns the raw 30 ms block stream into complete utterances.
* Utterances go into a small bounded queue. If recognition falls behind, the
  OLDEST pending utterance is dropped and counted, rather than letting latency
  grow without bound. For a live sermon, fresh audio beats complete audio.
"""

from __future__ import annotations

import math
import queue
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import AudioConfig


# Recording-level thresholds, in dBFS of SAMPLE PEAK (not RMS). Defined once so
# the live meter, its exit summary and the doctor can never disagree.
CLIP_DB = -1.0       # at or above this the waveform is being flattened
HOT_DB = -3.0        # usable but one loud "Hallelujah" from clipping
GOOD_DB = -20.0      # comfortable working range for speech recognition
QUIET_DB = -35.0     # works, but raise the gain if you can
SILENT_DB = -45.0    # nothing usable arrived


def level_verdict(peak_db: float, clip_blocks: int = 0) -> tuple[str, str, bool]:
    """Map a session peak level to (short label, advice, acceptable)."""
    if peak_db < SILENT_DB:
        return ("no sig", "no signal. The mic is muted, the wrong device is selected, "
                          "or you did not speak.", False)
    if peak_db < QUIET_DB:
        return ("quiet", "too quiet. Raise the mic level until peaks reach about "
                         "-12 dBFS.", False)
    if clip_blocks or peak_db >= CLIP_DB:
        return ("CLIP!", "clipping. Lower the mic level and set Boost to 0.0 dB.", False)
    if peak_db >= HOT_DB:
        return (" hot ", "hot but usable. A little lower would be safer.", True)
    if peak_db < GOOD_DB:
        return (" ok  ", "usable, a bit low. Fine as is; raising the gain slightly "
                         "would give more margin.", True)
    return ("GOOD ", "good level.", True)


def _percentile(values: list[float], q: float) -> float:
    """Simple nearest-rank percentile; avoids a scipy/statistics dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _split_point(buf_db: list[float], lookback: int, threshold: float) -> int:
    """Where to cut an utterance that has run past the flush limit.

    Picks the MIDPOINT of the longest silent run in the recent window, so the
    emitted clip ends in silence and the carried remainder starts in silence -
    no word is split across the boundary. Taking the single quietest frame is
    not enough: it can be the last frame of a gap, leaving the cut hard against
    the next word. Falls back to the quietest frame when nothing dips below
    threshold (unbroken speech), and never returns 0.
    """
    start = max(1, len(buf_db) - lookback)
    window = buf_db[start:]
    if not window:
        return len(buf_db)

    best_len, best_mid, run_start = 0, None, None
    for i, db in enumerate(window + [float("inf")]):
        if db <= threshold:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            length = i - run_start
            if length >= best_len:          # >= prefers the latest such run
                best_len, best_mid = length, run_start + length // 2
            run_start = None

    if best_mid is not None:
        return max(1, start + best_mid)
    return start + min(range(len(window)), key=window.__getitem__)


def rms_dbfs(block: np.ndarray) -> float:
    """Level of one audio block in dBFS (0 = full scale, -60 = very quiet)."""
    x = np.asarray(block, dtype=np.float64)
    return 20.0 * math.log10(math.sqrt(float(np.mean(x * x)) + 1e-20) + 1e-10)


@dataclass
class Utterance:
    audio: np.ndarray          # float32 mono @ sample_rate
    duration: float            # seconds
    peak_db: float
    captured_at: float         # time.monotonic() when the utterance closed
    forced: bool = False       # True if cut by max_utterance_s rather than silence


class MicrophoneSource:
    """Yields Utterance objects from the default (or chosen) input device."""

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self._raw_q: queue.Queue = queue.Queue(maxsize=400)
        self._out_q: queue.Queue = queue.Queue(maxsize=cfg.queue_size)
        self._stop = threading.Event()
        self._stream = None
        self._thread: threading.Thread | None = None

        # observable state, used by the console UI and by `doctor`
        self.noise_floor_db: float = cfg.absolute_floor_db
        self.threshold_db: float = cfg.absolute_floor_db
        self.last_db: float = -120.0
        self.peak_hold_db: float = -120.0   # decaying peak, for level metering
        self.session_peak_db: float = -120.0
        self.clip_blocks: int = 0
        self.in_speech: bool = False
        self.calibrated: bool = False
        self.overflows: int = 0       # audio blocks lost before segmentation
        self.dropped: int = 0         # whole utterances dropped due to backlog
        self.stream_warnings: int = 0
        self.muted_seconds: float = 0.0
        self._muted = threading.Event()

    # ------------------------------------------------------------------ setup

    @staticmethod
    def list_devices():
        import sounddevice as sd
        return sd.query_devices()

    @property
    def blocksize(self) -> int:
        return int(self.cfg.sample_rate * self.cfg.block_ms / 1000)

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.cfg.device,
            callback=self._callback,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._segment_loop, daemon=True, name="segmenter")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    # ------------------------------------------------------------ mute gate

    def mute(self) -> None:
        """Stop consuming input. Used while French plays out of the speakers,
        so the translation is not fed back in and transcribed."""
        self._muted.set()

    def unmute(self) -> None:
        self._muted.clear()

    @property
    def muted(self) -> bool:
        return self._muted.is_set()

    # ------------------------------------------------------------- audio path

    def _callback(self, indata, frames, time_info, status):  # PortAudio thread
        if status:
            self.stream_warnings += 1
        try:
            self._raw_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self.overflows += 1

    def _segment_loop(self) -> None:
        cfg = self.cfg
        block_s = cfg.block_ms / 1000.0
        # NOTE: *_ms settings are milliseconds, block_s is seconds. Every
        # conversion below must divide by 1000 first.
        calib_frames = max(1, int(cfg.calibrate_s / block_s))
        silence_frames = max(1, int(cfg.silence_ms / 1000.0 / block_s))
        long_silence_frames = max(1, int(cfg.long_silence_ms / 1000.0 / block_s))
        preroll_frames = max(1, int(cfg.preroll_ms / 1000.0 / block_s))
        max_frames = max(1, int(cfg.max_utterance_s / block_s))
        min_frames = max(1, int(cfg.min_utterance_s / block_s))
        lookback_frames = max(1, int(cfg.smart_cut_ms / 1000.0 / block_s))

        preroll: deque = deque(maxlen=preroll_frames)   # holds (block, db) pairs
        calib: list[float] = []
        buf: list[np.ndarray] = []
        buf_db: list[float] = []
        speech_frames = 0
        silence_run = 0
        in_speech = False

        while not self._stop.is_set():
            try:
                block = self._raw_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if self._muted.is_set():
                # Discard input and abandon any part-built utterance: whatever
                # is on the microphone right now is our own output.
                self.muted_seconds += block_s
                buf, buf_db = [], []
                speech_frames = silence_run = 0
                in_speech = self.in_speech = False
                continue

            db = rms_dbfs(block)
            self.last_db = db

            # Sample peak, not RMS: clipping is about individual samples hitting
            # the ceiling, and RMS averaging hides it completely.
            sample_peak = float(np.max(np.abs(block)))
            peak_db = 20.0 * math.log10(sample_peak + 1e-10)
            if sample_peak >= 0.99:
                self.clip_blocks += 1
            self.session_peak_db = max(self.session_peak_db, peak_db)
            # ~2 s decay so a spoken peak stays readable on screen
            self.peak_hold_db = max(peak_db, self.peak_hold_db - block_s * 12.0)

            # --- startup: measure the room ---
            if len(calib) < calib_frames:
                calib.append(db)
                # 20th percentile, not median: if the room is not actually quiet
                # during calibration (a cough, a chair, someone talking) the
                # median tracks the noise. The low percentile tracks the floor.
                self.noise_floor_db = _percentile(calib, 0.20)
                preroll.append((block, db))
                if len(calib) == calib_frames:
                    self.calibrated = True
                continue

            threshold = max(self.noise_floor_db + cfg.threshold_db, cfg.absolute_floor_db)
            self.threshold_db = threshold

            if not in_speech:
                if db > threshold:
                    in_speech = self.in_speech = True
                    buf = [b for b, _ in preroll] + [block]
                    buf_db = [d for _, d in preroll] + [db]
                    preroll.clear()
                    speech_frames = 1
                    silence_run = 0
                else:
                    preroll.append((block, db))
                    # Asymmetric adaptation: fall towards a newly observed quiet
                    # level quickly, rise towards a noisy one very slowly. A bad
                    # calibration then self-corrects within a second of real
                    # silence, instead of poisoning the whole session.
                    if db < self.noise_floor_db:
                        self.noise_floor_db = 0.90 * self.noise_floor_db + 0.10 * db
                    else:
                        self.noise_floor_db = 0.9995 * self.noise_floor_db + 0.0005 * db
            else:
                buf.append(block)
                buf_db.append(db)
                if db > threshold:
                    speech_frames += 1
                    silence_run = 0
                else:
                    silence_run += 1

                # A short pause only ends the utterance once there is enough
                # audio to recognise well; otherwise keep buffering through it.
                # A long pause always ends it, however little we have.
                ended = (silence_run >= long_silence_frames
                         or (silence_run >= silence_frames and len(buf) >= min_frames))
                if ended or len(buf) >= max_frames:
                    if ended:
                        self._emit(buf, speech_frames * block_s, forced=False)
                        buf, buf_db = [], []
                        speech_frames = 0
                        in_speech = self.in_speech = False
                    else:
                        # Continuous speech hit the flush limit. Cutting at exactly
                        # max_frames lands mid-word and loses it; split at the
                        # quietest frame in the recent window instead, and carry
                        # the remainder into the next utterance.
                        split = _split_point(buf_db, lookback_frames, threshold)
                        self._emit(buf[:split], speech_frames * block_s, forced=True)
                        buf, buf_db = buf[split:], buf_db[split:]
                        speech_frames = sum(1 for d in buf_db if d > threshold)
                    silence_run = 0

    def _emit(self, blocks: list[np.ndarray], speech_s: float, forced: bool) -> None:
        if speech_s * 1000 < self.cfg.min_speech_ms:
            return  # a cough, a door, a chair scrape
        audio = np.concatenate(blocks).astype(np.float32, copy=False)
        utt = Utterance(
            audio=audio,
            duration=len(audio) / self.cfg.sample_rate,
            peak_db=20.0 * math.log10(float(np.max(np.abs(audio))) + 1e-10),
            captured_at=time.monotonic(),
            forced=forced,
        )
        while True:
            try:
                self._out_q.put_nowait(utt)
                return
            except queue.Full:
                try:
                    self._out_q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    # --------------------------------------------------------------- consumer

    def utterances(self):
        """Blocking generator over completed utterances."""
        while not self._stop.is_set():
            try:
                yield self._out_q.get(timeout=0.25)
            except queue.Empty:
                continue

    def record(self, seconds: float) -> np.ndarray:
        """Raw fixed-length capture. Used by the diagnostics, not the pipeline."""
        import sounddevice as sd
        frames = int(seconds * self.cfg.sample_rate)
        data = sd.rec(frames, samplerate=self.cfg.sample_rate,
                      channels=1, dtype="float32", device=self.cfg.device)
        sd.wait()
        return data[:, 0]


class NetworkMicrophoneSource(MicrophoneSource):
    """Audio arriving over a WebSocket instead of PortAudio.

    Structurally identical to MicrophoneSource - same segmenter, same
    Utterance output - so it plugs straight into Pipeline. The only
    difference is that audio blocks are fed by `submit_audio()` from the
    server's WebSocket handler instead of being pulled from PortAudio.

    A speaker's browser sends 16-bit PCM at 16 kHz mono; if it arrives at a
    different sample rate, we tell it so at handshake and it downsamples
    client-side (avoids importing scipy just for one linear resample).
    """

    def __init__(self, cfg: AudioConfig):
        super().__init__(cfg)
        self.speaker_connected = False

    def start(self) -> None:
        # No PortAudio stream to open; just spin up the segmenter thread.
        import threading
        self._thread = threading.Thread(target=self._segment_loop, daemon=True,
                                        name="segmenter-net")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    def submit_audio(self, block: np.ndarray) -> None:
        """Called from the server's asyncio loop with each decoded PCM frame."""
        if self._stop.is_set():
            return
        try:
            self._raw_q.put_nowait(block)
        except queue.Full:
            self.overflows += 1

    def notify_speaker(self, connected: bool) -> None:
        """Toggle the connected/disconnected state so UI can reflect it."""
        self.speaker_connected = connected


class SystemAudioSource(MicrophoneSource):
    """Captures whatever the laptop is currently playing (WASAPI loopback).

    On Windows any audio played by an application goes through the WASAPI
    audio engine. This source opens the DEFAULT output device in loopback
    mode - the same trick OBS Studio uses - so anything playing (a sermon
    video, a livestream, a Zoom call) becomes the pipeline's input.

    Uses PyAudioWPatch: a maintained PyAudio fork that exposes the
    Windows-only `paWinWasapiLoopback` flag which the stock `sounddevice`
    build doesn't ship. No new native install for the user - the wheel
    bundles its own PortAudio.

    On non-Windows platforms this class won't work (loopback is Windows-
    specific to WASAPI). The doctor and CLI should gate on OS.

    Loopback devices always report the OUTPUT sample rate + channel count.
    On this hardware that's 48 kHz stereo; we downmix to mono and
    resample to 16 kHz to keep the rest of the pipeline unchanged.
    """

    def __init__(self, cfg: AudioConfig, device_name: str | None = None):
        super().__init__(cfg)
        self.device_name = device_name        # None = follow default output
        self.paused = True                    # explicit Start button, per user spec
        self._pa = None
        self._pa_stream = None
        self._reader_thread = None
        self._pa_sample_rate = 48000
        self._pa_channels = 2
        # Separate from `self._stop` (which ends the segmenter) so the capture
        # backend can be torn down and rebuilt while the segmenter keeps
        # running - that is what lets SelectableSource swap mic <-> loopback
        # without dropping the pipeline.
        self._capture_stop = threading.Event()

    # ---- pause / resume for the explicit Start button ---------------------

    def resume(self) -> None:
        """Begin (or resume) forwarding captured audio into the segmenter."""
        self.paused = False

    def pause(self) -> None:
        """Stop forwarding audio. The stream stays open; toggling back is
        instant. Any buffered audio is discarded."""
        self.paused = True
        # Drop the partial utterance so resuming doesn't stitch to old audio
        try:
            while True: self._raw_q.get_nowait()
        except Exception:
            pass

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self._start_system_capture()

        # Segmenter thread (inherited from MicrophoneSource) consumes _raw_q
        self._thread = threading.Thread(target=self._segment_loop, daemon=True,
                                        name="segmenter-sysaudio")
        self._thread.start()

    def _start_system_capture(self) -> None:
        """Open the WASAPI loopback stream and spin up its reader thread.

        Split out from `start()` so a switchable source can rebuild just the
        capture backend without restarting the segmenter.
        """
        import pyaudiowpatch as pyaudio
        self._capture_stop = threading.Event()
        self._pa = pyaudio.PyAudio()

        # Find the loopback endpoint for the requested (or default) output.
        target = self._pick_loopback_device()
        self._pa_sample_rate = int(target["defaultSampleRate"])
        self._pa_channels = int(target["maxInputChannels"])
        # Read in ~30 ms chunks at whatever the device rate is
        frames_per_read = max(1, int(self._pa_sample_rate * 0.03))

        self._pa_stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._pa_channels,
            rate=self._pa_sample_rate,
            input=True,
            input_device_index=target["index"],
            frames_per_buffer=frames_per_read,
        )

        # Reader thread pulls from PortAudio, feeds the segmenter
        self._reader_thread = threading.Thread(
            target=self._reader_loop, args=(frames_per_read,),
            daemon=True, name="audio-loopback")
        self._reader_thread.start()

    def _stop_system_capture(self) -> None:
        # Signal reader thread FIRST so it exits its read loop, then join it
        # BEFORE closing the PortAudio stream. Closing while a read is in
        # flight is the source of PyAudio's Windows-shutdown segfaults.
        self._capture_stop.set()
        reader = self._reader_thread
        self._reader_thread = None
        if reader is not None:
            reader.join(timeout=2.0)
            if reader.is_alive():
                # The reader is wedged inside PortAudio. Closing the stream or
                # terminating PortAudio now would free memory it is actively
                # reading - an access violation that takes the whole process
                # down. Leak the handles instead: a stalled stream costs a few
                # MB, a segfault costs the service.
                self.stream_warnings += 1
                self._pa_stream = None
                self._pa = None
                return
        stream = self._pa_stream
        self._pa_stream = None
        if stream is not None:
            try: stream.stop_stream()
            except Exception: pass
            try: stream.close()
            except Exception: pass
        if self._pa is not None:
            try: self._pa.terminate()
            except Exception: pass
            self._pa = None

    def stop(self) -> None:
        self._stop.set()
        self._stop_system_capture()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    # ---- device discovery -------------------------------------------------

    def _pick_loopback_device(self):
        """Return the pyaudiowpatch device info for the loopback we should
        capture. If `self.device_name` is set, look for a loopback matching
        that name; otherwise the default output."""
        if self.device_name:
            for i in range(self._pa.get_device_count()):
                d = self._pa.get_device_info_by_index(i)
                if d.get("isLoopbackDevice") and self.device_name.lower() in d["name"].lower():
                    return d
            raise RuntimeError(f"No WASAPI loopback device matches {self.device_name!r}")
        return self._pa.get_default_wasapi_loopback()

    @staticmethod
    def list_loopback_devices() -> list[dict]:
        """Enumerate loopback-capable devices for the operator's UI."""
        import pyaudiowpatch as pyaudio
        pa = pyaudio.PyAudio()
        out = []
        try:
            for i in range(pa.get_device_count()):
                d = pa.get_device_info_by_index(i)
                if d.get("isLoopbackDevice"):
                    out.append({
                        "index": d["index"],
                        "name": d["name"],
                        "sample_rate": int(d["defaultSampleRate"]),
                        "channels": int(d["maxInputChannels"]),
                    })
        finally:
            pa.terminate()
        return out

    # ---- capture loop -----------------------------------------------------

    def _reader_loop(self, frames_per_read: int) -> None:
        """Reads PortAudio frames, downmixes stereo->mono, resamples to
        16 kHz mono float32, hands off to the segmenter (`_raw_q`)."""
        block16k = int(16000 * self.cfg.block_ms / 1000)
        in_sr = self._pa_sample_rate
        ch = self._pa_channels
        ratio = 16000 / in_sr

        while not self._capture_stop.is_set() and not self._stop.is_set():
            stream = self._pa_stream    # atomic snapshot; stop() may null it
            if stream is None:
                return
            try:
                # NEVER call read() without checking availability first. A
                # WASAPI loopback endpoint delivers no packets at all while
                # the machine is silent, so a blocking read() parks here
                # indefinitely - the stop flag goes unnoticed, the join times
                # out, and closing the stream under the in-flight read is an
                # access violation. Polling keeps us responsive within ~5 ms.
                if stream.get_read_available() < frames_per_read:
                    time.sleep(0.005)
                    continue
                data = stream.read(frames_per_read, exception_on_overflow=False)
            except Exception:
                # Stream was closed under us during shutdown - not a warning
                if self._capture_stop.is_set() or self._stop.is_set() \
                        or self._pa_stream is None:
                    return
                self.stream_warnings += 1
                continue
            if self.paused:
                continue        # explicit Start button not pressed yet

            arr = np.frombuffer(data, dtype=np.float32)
            if ch > 1:
                arr = arr.reshape(-1, ch).mean(axis=1)
            # Linear resample to 16 kHz mono
            if in_sr != 16000:
                n_out = int(arr.size * ratio)
                if n_out > 0:
                    src_x = np.linspace(0, 1, arr.size, endpoint=False)
                    dst_x = np.linspace(0, 1, n_out, endpoint=False)
                    arr = np.interp(dst_x, src_x, arr).astype(np.float32)

            # Feed the segmenter in `block16k`-sized chunks so its frame
            # math matches the local-mic case.
            for i in range(0, arr.size, block16k):
                chunk = arr[i:i + block16k]
                if chunk.size == 0:
                    continue
                try:
                    self._raw_q.put_nowait(chunk)
                except queue.Full:
                    self.overflows += 1


class SelectableSource(SystemAudioSource):
    """One source whose CAPTURE BACKEND can be swapped while it runs.

    The desktop app used to hard-wire WASAPI loopback, so it only heard audio
    the laptop was *playing*. A preacher talking into the laptop's microphone
    produced silence. This source keeps the segmenter, VAD and everything
    downstream fixed and swaps only the producer feeding `_raw_q`:

        "system" - WASAPI loopback: whatever the laptop is playing
        "mic"    - a normal input device (built-in mic, USB mic, mixer feed)

    Both backends deliver 16 kHz mono float32 blocks, so a switch is invisible
    to the rest of the pipeline. `paused` still gates everything, which keeps
    the operator UI's Start/Stop button working across both.
    """

    def __init__(self, cfg: AudioConfig, kind: str = "system",
                 device_name: str | None = None):
        super().__init__(cfg, device_name=device_name)
        self.kind = kind
        self.last_error: str | None = None
        self._switch_lock = threading.Lock()
        # Mic backend format. Set at open time: a device that refuses 16 kHz
        # mono is driven at its native format and converted in the callback.
        self._mic_rate = cfg.sample_rate
        self._mic_channels = 1
        self._mic_native = False
        # The HTTP API is serving ~30s before run.py reaches `with source:`
        # (model and voice loading sit in between), so the operator can pick a
        # source before any stream exists. Until then a switch only records the
        # choice; opening a device here would collide with the one start() is
        # about to open.
        self._started = False

    # ---- capture backend selection ---------------------------------------

    def _start_capture(self) -> None:
        if self.kind == "system":
            self._start_system_capture()
        else:
            self._start_mic_capture()

    def _stop_capture(self) -> None:
        if self.kind == "system":
            self._stop_system_capture()
        else:
            self._stop_mic_capture()

    @staticmethod
    def _resolve_input_device(name: str | None, fallback):
        """Device selector -> stable index.

        Windows lists the same hardware once per host API, so names are NOT
        unique - this laptop shows two inputs called "Microphone (4- High
        Definition Audio Device)" with different capabilities. The UI therefore
        sends the index, and a name is only accepted as a fallback for CLI use.
        """
        if name is None or name == "":
            return fallback
        import sounddevice as sd
        if isinstance(name, int) or str(name).isdigit():
            return int(name)
        devices = sd.query_devices()
        for i, d in enumerate(devices):                     # exact first
            if d["max_input_channels"] > 0 and d["name"] == name:
                return i
        low = str(name).lower()
        for i, d in enumerate(devices):                     # then substring
            if d["max_input_channels"] > 0 and low in d["name"].lower():
                return i
        raise RuntimeError(f"No input device matching {name!r}")

    def _start_mic_capture(self) -> None:
        import sounddevice as sd
        self._capture_stop = threading.Event()
        device = self._resolve_input_device(self.device_name, self.cfg.device)

        # Prefer 16 kHz mono: it is what the pipeline wants and costs no
        # conversion. But line-in on a mixer, and most USB capture cards, are
        # 48 kHz stereo devices that flatly refuse 16 kHz mono - on this laptop
        # alone two of six inputs reject it ("Invalid sample rate", "Sample
        # format not supported"). Every device accepts its NATIVE format, so
        # fall back to that and downmix/resample in the callback, exactly as
        # the WASAPI loopback path already does.
        rate, channels = self.cfg.sample_rate, self.cfg.channels
        try:
            sd.check_input_settings(device=device, channels=channels,
                                    samplerate=rate, dtype="float32")
        except Exception:
            info = sd.query_devices(device)
            rate = int(info["default_samplerate"])
            channels = min(2, int(info["max_input_channels"])) or 1
            sd.check_input_settings(device=device, channels=channels,
                                    samplerate=rate, dtype="float32")

        self._mic_rate = rate
        self._mic_channels = channels
        self._mic_native = (rate != self.cfg.sample_rate or channels != 1)
        self._stream = sd.InputStream(
            samplerate=rate,
            channels=channels,
            dtype="float32",
            blocksize=int(rate * self.cfg.block_ms / 1000),
            device=device,
            callback=self._callback,
        )
        self._stream.start()

    def _stop_mic_capture(self) -> None:
        self._capture_stop.set()
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    # ---- audio path -------------------------------------------------------

    def _callback(self, indata, frames, time_info, status):  # PortAudio thread
        # MicrophoneSource's callback has no pause gate; this source needs one
        # so the operator's Stop button silences the mic backend too.
        if status:
            self.stream_warnings += 1
        if self.paused:
            return
        if self._mic_native:
            block = indata.mean(axis=1) if self._mic_channels > 1 else indata[:, 0]
            block = self._to_16k(block)
        else:
            block = indata[:, 0].copy()
        try:
            self._raw_q.put_nowait(block)
        except queue.Full:
            self.overflows += 1

    def _to_16k(self, block: np.ndarray) -> np.ndarray:
        """Linear resample to the pipeline's 16 kHz. Cheap enough for the
        PortAudio thread: a 30 ms block is ~1400 samples."""
        target = self.cfg.sample_rate
        if self._mic_rate == target:
            return block.astype(np.float32, copy=True)
        n_out = int(block.size * target / self._mic_rate)
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        return np.interp(
            np.linspace(0, 1, n_out, endpoint=False),
            np.linspace(0, 1, block.size, endpoint=False),
            block,
        ).astype(np.float32)

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> None:
        with self._switch_lock:
            self._start_capture()
            self._started = True
        self._thread = threading.Thread(target=self._segment_loop, daemon=True,
                                        name="segmenter")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._switch_lock:
            self._started = False
            try:
                self._stop_capture()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    # ---- runtime switching ------------------------------------------------

    def set_input(self, kind: str, device_name: str | None = None) -> None:
        """Swap the capture backend. Raises if the new backend won't open,
        after restoring the previous one so the app is never left deaf."""
        kind = (kind or "").strip().lower()
        if kind not in ("system", "mic"):
            raise ValueError(f"unknown source kind: {kind!r}")

        with self._switch_lock:
            if kind == self.kind and device_name == self.device_name:
                return
            if not self._started:
                # Nothing is open yet; just record what start() should open.
                self.kind, self.device_name = kind, device_name
                self.last_error = None
                return
            prev_kind, prev_device = self.kind, self.device_name
            was_running = not self.paused

            # Gate the queue before tearing anything down so no block from the
            # old backend lands in the middle of the new one's audio.
            self.paused = True
            try:
                self._stop_capture()
            except Exception:
                pass
            self._drain_raw()

            self.kind, self.device_name = kind, device_name
            try:
                self._start_capture()
            except Exception as exc:
                self.last_error = f"{kind}: {exc}"
                # Roll back so the operator keeps whatever was working.
                self.kind, self.device_name = prev_kind, prev_device
                try:
                    self._start_capture()
                except Exception:
                    pass
                self.paused = not was_running
                raise
            self.last_error = None
            self.paused = not was_running

    def _drain_raw(self) -> None:
        try:
            while True:
                self._raw_q.get_nowait()
        except Exception:
            pass

    # ---- device discovery for the operator UI ----------------------------

    @staticmethod
    def list_input_devices() -> list[dict]:
        """Input devices suitable for the "mic" backend."""
        import sounddevice as sd
        out = []
        try:
            for i, d in enumerate(sd.query_devices()):
                if int(d.get("max_input_channels", 0)) > 0:
                    out.append({
                        "index": i,
                        "name": d["name"],
                        "sample_rate": int(d.get("default_samplerate", 0)),
                        "channels": int(d["max_input_channels"]),
                    })
        except Exception:
            pass
        return out
