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
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

from .config import LANGUAGE_PACKS_FILE, LANGUAGES, ROOT
from .config import models_dir as _models_dir
from .telemetry import SERVER_URL, check_manifest, get_install_id

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
        models_dir = _models_dir()

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


def _fetch_pending(install_id: str) -> list[str]:
    url = SERVER_URL.rstrip("/") + f"/v1/installs/{install_id}/pending"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8")).get("pending", [])
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return []


def _ack_pending(install_id: str, code: str) -> None:
    url = SERVER_URL.rstrip("/") + f"/v1/installs/{install_id}/pending/{code}/ack"
    req = urllib.request.Request(url, data=b"", method="POST")
    try:
        urllib.request.urlopen(req, timeout=5).close()
    except (urllib.error.URLError, TimeoutError, OSError):
        pass          # best-effort - a missed ack just means a harmless re-delivery later


def apply_remote_triggers(progress: Progress = None) -> list[str]:
    """Install whatever this install's id has pending on the server, e.g.
    from `POST /v1/admin/trigger` fired at the PC builder's install_id once
    the lite test build is back. Best-effort and silent on any network
    failure, same as everything else in this module - there is no
    connectivity guarantee and none should be assumed. Returns the codes
    actually installed this call."""
    if not SERVER_URL:
        return []
    install_id = get_install_id()
    if not install_id:
        return []       # telemetry (and so this install's id) is opted out

    pending = _fetch_pending(install_id)
    if not pending:
        return []

    installed = _load_installed()
    done = []
    for code in pending:
        if code in installed:
            _ack_pending(install_id, code)     # already have it; just clear the backlog
            continue
        try:
            install(code, progress=progress)
        except LangPackError:
            continue        # leave it pending - a later restart/poll will retry
        _ack_pending(install_id, code)
        done.append(code)
    return done


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
