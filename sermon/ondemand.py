"""Load a language's translation model and voice only when someone needs it.

Loading every configured language at startup does not fit a small machine:
measured on an 8 GB laptop, 15 languages left six OPUS-MT models failing with
`mkl_malloc: failed to allocate memory`, five voices failing with ONNX
`bad allocation`, and startup running past the desktop shell's 3-minute
limit. Five languages loaded cleanly on the same machine.

So under --serve nothing but speech recognition loads up front. One background
thread watches which languages are wanted (a listener has selected it, or the
recorder is capturing it), loads those one at a time - translator first, so
text reaches the phone while the voice is still loading - and drops a
language again once nobody has wanted it for a while.

`translators` and `speakers` are plain dicts that Pipeline and BroadcastSink
read live; this thread is their only writer.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Callable, Iterable, Optional

_POLL_SECONDS = 1.0
_VOICE_LOAD_TIMEOUT = 60.0


class OnDemandLanguages:
    def __init__(self, targets: Iterable[str],
                 make_translator: Callable[[str], object],
                 make_speaker: Optional[Callable[[str], object]],
                 wanted: Callable[[], set],
                 on_unload: Optional[Callable[[str], None]] = None,
                 idle_unload_seconds: float = 600.0,
                 retry_seconds: float = 60.0):
        self.targets = list(targets)
        self.translators: dict = {}
        self.speakers: dict = {}
        self._make_translator = make_translator
        self._make_speaker = make_speaker
        self._wanted = wanted
        self._on_unload = on_unload
        self._idle = idle_unload_seconds
        self._retry = retry_seconds
        self._last_wanted: dict[str, float] = {}
        self._mt_failed_at: dict[str, float] = {}
        self._voice_failed_at: dict[str, float] = {}
        self._loading: Optional[str] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="language-loader")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        """{code: "ready" | "loading" | "failed" | "idle"} for the dashboard."""
        out = {}
        for t in self.targets:
            if t in self.translators:
                out[t] = "ready"
            elif t == self._loading:
                out[t] = "loading"
            elif t in self._mt_failed_at:
                out[t] = "failed"
            else:
                out[t] = "idle"
        return out

    # ------------------------------------------------------------------ worker

    def _run(self) -> None:
        while not self._stop.wait(_POLL_SECONDS):
            try:
                self._tick()
            except Exception as exc:              # never let the loader die
                print(f"\n  !! language loader error: {exc}", file=sys.stderr)

    def _tick(self) -> None:
        now = time.monotonic()
        wanted = [t for t in self.targets if t in self._wanted()]
        for t in wanted:
            self._last_wanted[t] = now

        # Every wanted translator before any voice: text is what a listener
        # sees first, and a voice is useless without its translation anyway.
        for t in wanted:
            if t not in self.translators and self._may_retry(self._mt_failed_at, t):
                self._load_translator(t)
        if self._make_speaker is not None:
            for t in wanted:
                if (t in self.translators and t not in self.speakers
                        and self._may_retry(self._voice_failed_at, t)):
                    self._load_speaker(t)

        for t in list(self.translators):
            if t not in wanted and now - self._last_wanted.get(t, now) > self._idle:
                self._unload(t)

    def _may_retry(self, failed_at: dict, t: str) -> bool:
        at = failed_at.get(t)
        return at is None or time.monotonic() - at > self._retry

    def _load_translator(self, t: str) -> None:
        self._loading = t
        started = time.perf_counter()
        try:
            self.translators[t] = self._make_translator(t)
            self._mt_failed_at.pop(t, None)
            print(f"  [languages] {t}: translation loaded "
                  f"({time.perf_counter() - started:.1f}s)")
        except Exception as exc:
            self._mt_failed_at[t] = time.monotonic()
            print(f"  [languages] {t}: translation FAILED to load "
                  f"({type(exc).__name__}: {str(exc)[:80]}) - "
                  f"retrying in {self._retry:.0f}s", file=sys.stderr)
        finally:
            self._loading = None

    def _load_speaker(self, t: str) -> None:
        # Own thread with a hard timeout: Piper can deadlock in espeak-ng init
        # on some Windows configurations, and that must cost one voice, not
        # every language queued behind it.
        slot: dict = {}

        def _worker():
            try:
                slot["speaker"] = self._make_speaker(t)
            except Exception as exc:
                slot["error"] = exc

        th = threading.Thread(target=_worker, daemon=True, name=f"voice-load-{t}")
        started = time.perf_counter()
        th.start()
        th.join(_VOICE_LOAD_TIMEOUT)
        speaker = slot.get("speaker")
        if speaker is not None:
            self.speakers[t] = speaker
            self._voice_failed_at.pop(t, None)
            print(f"  [languages] {t}: voice loaded "
                  f"({time.perf_counter() - started:.1f}s)")
            return
        self._voice_failed_at[t] = time.monotonic()
        if "speaker" in slot:
            # The factory returned None: this language has no voice at all.
            self._voice_failed_at[t] = float("inf")
            return
        reason = slot.get("error", "timed out")
        print(f"  [languages] {t}: voice unavailable ({str(reason)[:80]}); "
              f"text still works", file=sys.stderr)

    def _unload(self, t: str) -> None:
        self.translators.pop(t, None)
        self.speakers.pop(t, None)
        self._last_wanted.pop(t, None)
        if self._on_unload is not None:
            self._on_unload(t)
        print(f"  [languages] {t}: unloaded (no listener for "
              f"{self._idle / 60:.0f} min)")
