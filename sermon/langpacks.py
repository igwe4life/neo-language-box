"""Download, verify and install a language pack after the app is already
installed - the client half of telemetry-server's /v1/manifest and
/v1/packs/<file>. See scripts/build_language_pack.py for how a pack is
built, and sermon/config.py's installed_language_packs() for how an
installed one becomes an active language on the next restart.

A pack is a zip of exactly the files one language needs beyond the base
install (its voice, and its own MT model if it doesn't use the shared
NLLB-200), unzipped directly into models/ with the same relative paths it
was built with. Nothing here touches DEFAULT_TARGETS or any other code -
the app just needs restarting to pick up a newly-installed language.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

from .config import LANGUAGE_PACKS_FILE, LANGUAGES, ROOT
from .telemetry import SERVER_URL, check_manifest

Progress = Optional[Callable[[int, int], None]]      # (bytes_done, bytes_total)


class LangPackError(Exception):
    pass


def _load_installed() -> dict:
    try:
        return json.loads(LANGUAGE_PACKS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_installed(data: dict) -> None:
    LANGUAGE_PACKS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def available_packs() -> list[dict]:
    """Every pack the manifest lists, each tagged with whether it's
    already installed and at what version. Empty if telemetry is
    disabled/unconfigured or the server can't be reached - this is the
    same "no connectivity assumed" rule telemetry.py already follows."""
    manifest = check_manifest()
    if not manifest:
        return []
    installed = _load_installed()
    packs = []
    for pack in manifest.get("language_packs", []):
        code = pack.get("code")
        packs.append({
            **pack,
            "installed": code in installed,
            "installed_version": installed.get(code),
            "update_available": code in installed and installed[code] != pack.get("version"),
        })
    return packs


def install(code: str, models_dir: Path = None, progress: Progress = None) -> None:
    """Download, verify and unzip one language pack by code. Raises
    LangPackError with a message fit to show an operator on anything that
    goes wrong - a bad download never leaves a half-installed pack lying
    around to confuse the next restart."""
    if not SERVER_URL:
        raise LangPackError("No update server is configured.")
    if models_dir is None:
        models_dir = ROOT / "models"

    manifest = check_manifest()
    if not manifest:
        raise LangPackError("Could not reach the update server. Check your internet connection.")

    pack = next((p for p in manifest.get("language_packs", []) if p.get("code") == code), None)
    if pack is None:
        raise LangPackError(f"No pack is published for {LANGUAGES.get(code, {}).get('native', code)!r}.")

    url = SERVER_URL.rstrip("/") + pack["url"]
    expected_sha256 = pack["sha256"]
    expected_size = pack["size_bytes"]

    with tempfile.TemporaryDirectory(prefix="langpack-") as tmp:
        zip_path = Path(tmp) / f"{code}.zip"
        _download(url, zip_path, expected_size, progress)
        _verify(zip_path, expected_sha256)
        _extract(zip_path, models_dir)

    installed = _load_installed()
    installed[code] = pack["version"]
    _save_installed(installed)


def _download(url: str, dest: Path, expected_size: int, progress: Progress) -> None:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp, dest.open("wb") as out:
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, expected_size)
    except OSError as exc:
        raise LangPackError(f"Download failed: {exc}") from exc

    actual_size = dest.stat().st_size
    if actual_size != expected_size:
        raise LangPackError(
            f"Download incomplete: got {actual_size} bytes, expected {expected_size}. "
            f"Check your connection and try again.")


def _verify(zip_path: Path, expected_sha256: str) -> None:
    h = hashlib.sha256()
    with zip_path.open("rb") as f:
        while True:
            block = f.read(8 << 20)
            if not block:
                break
            h.update(block)
    if h.hexdigest() != expected_sha256:
        raise LangPackError(
            "The downloaded file doesn't match what the server expects - "
            "it may have been corrupted in transit. Try again.")


def _extract(zip_path: Path, models_dir: Path) -> None:
    # Every member's path is checked before anything is written: a zip
    # entry that climbs out of models/ (e.g. "../../something") must never
    # be trusted just because it came from our own update server over
    # HTTPS with a verified checksum - defence in depth costs one check.
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            dest = (models_dir / member).resolve()
            if models_dir.resolve() not in dest.parents:
                raise LangPackError(f"Refusing to extract unsafe path: {member}")
        zf.extractall(models_dir)
