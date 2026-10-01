"""Recording: save the source audio and each spoken translation to disk.

Taps data that ALREADY flows through the live pipeline - the raw utterance
audio in Pipeline.process(), and the Speech objects BroadcastSink already
synthesises for listeners - so recording costs one extra file write per
utterance per language, not a second recognition or synthesis pass. That is
what keeps it from interfering with the live stream: nothing is computed
twice, only written twice.

Threading: the writes themselves happen on the single pipeline thread (both
hook points run there already). start/pause/resume/stop arrive from the
asyncio server thread via the HTTP API, so every public method goes through
one lock - the file handles must never be touched from two threads at once.
"""

from __future__ import annotations

import json
import re
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .config import SOURCE_LANG


def slugify(text: str, fallback: str = "session") -> str:
    """URL- and filename-safe version of an operator-entered name.

    Used for both the recording folder name and the customisable listener URL
    - one helper, so "HSLHS 2026" becomes "hslhs-2026" consistently everywhere.
    """
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:60] or fallback


@dataclass
class _RecordedFile:
    code: str
    label: str
    path: Path
    sample_rate: int
    frames: int = 0

    @property
    def duration_seconds(self) -> float:
        return self.frames / max(self.sample_rate, 1)


def _to_pcm16(audio: np.ndarray) -> bytes:
    return np.clip(audio * 32767.0, -32768, 32767).astype("<i2").tobytes()


class Recorder:
    """Not a `Sink`: it needs the raw pre-synthesis source audio too, which
    `PipelineResult` does not carry. Wired directly into Pipeline.process()
    (for the source) and BroadcastSink.emit() (for each target language),
    both of which already have the audio in hand.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self._lock = threading.Lock()
        self.active = False
        self.paused = False
        self.languages: set = set()         # resolved concrete codes, SOURCE_LANG included or not
        self.session_dir: Optional[Path] = None
        self.started_at: Optional[float] = None
        self._writers: dict = {}
        self._files: dict = {}
        self._meta: dict = {}

    # ------------------------------------------------------------ queried by BroadcastSink
    def wanted_languages(self) -> set:
        """Non-source languages that should be synthesised for recording even
        if no listener is currently on them. Thread-safe snapshot."""
        with self._lock:
            if not self.active:
                return set()
            return {c for c in self.languages if c != SOURCE_LANG}

    # ------------------------------------------------------------ control (HTTP thread)
    def start(self, languages: set, language_meta: dict, event_label: str = "") -> dict:
        with self._lock:
            if self.active:
                raise RuntimeError("already recording")
            if not languages:
                raise ValueError("no languages selected")
            slug = slugify(event_label)
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            session_dir = self.base_dir / f"{stamp}_{slug}"
            session_dir.mkdir(parents=True, exist_ok=True)

            self.session_dir = session_dir
            self.languages = set(languages)
            self.active = True
            self.paused = False
            self.started_at = time.monotonic()
            self._writers = {}
            self._files = {}
            self._meta = {
                "event": event_label,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "languages": sorted(self.languages),
                "language_labels": {
                    c: (language_meta.get(c) or {}).get("native", c.upper())
                    for c in self.languages
                },
            }
            return {"session_dir": str(session_dir), "languages": sorted(self.languages)}

    def pause(self) -> None:
        with self._lock:
            if self.active:
                self.paused = True

    def resume(self) -> None:
        with self._lock:
            if self.active:
                self.paused = False

    def stop(self) -> dict:
        with self._lock:
            if not self.active:
                return {"ok": False, "error": "not recording"}
            for w in self._writers.values():
                try:
                    w.close()
                except Exception:
                    pass
            summary = {
                **self._meta,
                "stopped_at": datetime.now().isoformat(timespec="seconds"),
                "duration_seconds": round(time.monotonic() - self.started_at, 1),
                "files": [
                    {
                        "code": f.code, "label": f.label, "file": f.path.name,
                        "duration_seconds": round(f.duration_seconds, 1),
                        "size_bytes": f.path.stat().st_size if f.path.exists() else 0,
                    }
                    for f in self._files.values()
                ],
            }
            session_dir = self.session_dir
            try:
                (session_dir / "manifest.json").write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

            self.active = False
            self.paused = False
            self._writers = {}
            self.session_dir = None
            return {"ok": True, **summary, "session_dir": str(session_dir)}

    def status(self) -> dict:
        with self._lock:
            if not self.active:
                return {"active": False}
            return {
                "active": True,
                "paused": self.paused,
                "elapsed_seconds": round(time.monotonic() - self.started_at, 1),
                "languages": sorted(self.languages),
                "session_dir": str(self.session_dir) if self.session_dir else None,
            }

    def list_sessions(self, limit: int = 30) -> list:
        """Past recordings, newest first - timestamp-prefixed folder names sort
        correctly as strings, so no need to parse them."""
        if not self.base_dir.exists():
            return []
        dirs = sorted((d for d in self.base_dir.iterdir() if d.is_dir()),
                     key=lambda d: d.name, reverse=True)[:limit]
        out = []
        for d in dirs:
            manifest = d / "manifest.json"
            if manifest.exists():
                try:
                    data = json.loads(manifest.read_text(encoding="utf-8"))
                    data["folder"] = d.name
                    out.append(data)
                    continue
                except Exception:
                    pass
            # No manifest: likely interrupted (app closed mid-recording). Still
            # list it - the audio files are intact even though we never
            # finalised their durations - rather than hiding it from the
            # operator.
            out.append({"folder": d.name, "incomplete": True})
        return out

    # ------------------------------------------------------------ writes (pipeline thread)
    def _ensure_writer(self, code: str, sample_rate: int, label: str):
        w = self._writers.get(code)
        if w is not None:
            return w
        filename = f"source_{SOURCE_LANG}.wav" if code == SOURCE_LANG else f"{code}.wav"
        path = self.session_dir / filename
        w = wave.open(str(path), "wb")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        self._writers[code] = w
        self._files[code] = _RecordedFile(code=code, label=label, path=path,
                                          sample_rate=sample_rate)
        return w

    def write_source(self, audio: np.ndarray, sample_rate: int) -> None:
        if audio is None or not len(audio):
            return
        with self._lock:
            if not self.active or self.paused or SOURCE_LANG not in self.languages:
                return
            w = self._ensure_writer(SOURCE_LANG, sample_rate, "English (source)")
            pcm = _to_pcm16(audio)
            w.writeframes(pcm)
            self._files[SOURCE_LANG].frames += len(pcm) // 2

    def write_lang(self, code: str, audio: np.ndarray, sample_rate: int,
                   label: str = "") -> None:
        if audio is None or not len(audio):
            return
        with self._lock:
            if not self.active or self.paused or code not in self.languages:
                return
            w = self._ensure_writer(code, sample_rate, label or code.upper())
            pcm = _to_pcm16(audio)
            w.writeframes(pcm)
            self._files[code].frames += len(pcm) // 2
