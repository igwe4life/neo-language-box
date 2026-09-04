"""Entry point for the packaged desktop app.

Started by the Tauri shell (via sidecar) or double-clicked by a user for
local testing. Runs the pipeline with the desktop-specific defaults:

  * Both bindings: HTTP on 127.0.0.1:8000 (for the Tauri webview) AND
    HTTPS on 0.0.0.0:8443 (for phone listeners on the LAN)
  * Network mic source, so the webview's speaker page provides audio
  * Parakeet 0.6B STT (fastest quality option we've benched)
  * Models discovered relative to this executable (see _resolve_models_dir)

Command-line flags stay accepted so a technical user can still override:
    sermon.exe --stt whisper-base.en --port 8080 ...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bundled_root() -> Path:
    """Directory the .exe or launcher is running from.

    PyInstaller's one-dir bundle places the .exe next to the internal
    `_internal/` folder. Sibling `models/`, `certs/` and other resources
    are shipped next to the .exe by the installer.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _resolve_models_dir() -> Path:
    """Where to look for models when running as a packaged .exe.

    Search order:
      1. SERMON_MODELS_DIR env var (developer override)
      2. `models/` next to the .exe (production install layout)
      3. `models/` in the current working directory (development fallback)
    """
    env = os.environ.get("SERMON_MODELS_DIR")
    if env:
        return Path(env)
    beside_exe = _bundled_root() / "models"
    if beside_exe.exists():
        return beside_exe
    return Path.cwd() / "models"


def main() -> int:
    # Point config at the correct models dir BEFORE importing sermon.config
    os.environ.setdefault("SERMON_MODELS_DIR", str(_resolve_models_dir()))

    # Import here so the env var above is already set
    from run import main as run_main

    # If the user gave their own CLI args, respect them. Otherwise inject
    # the desktop-app defaults.
    if len(sys.argv) <= 1:
        sys.argv.extend([
            "--serve",
            "--source", "browser",
            "--tauri",
            "--stt", "parakeet-0.6b",
        ])
    return run_main() or 0


if __name__ == "__main__":
    sys.exit(main())
