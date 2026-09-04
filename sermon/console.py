"""Console helpers. Deliberately stdlib-only so `run.py` can start up and give
a useful error message even when the heavy dependencies are missing."""

from __future__ import annotations

import sys


def enable_utf8_console() -> None:
    """Make sure accented French renders correctly in cmd.exe / PowerShell."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
