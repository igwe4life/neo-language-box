"""Three check-ins to the telemetry server, and nothing else.

No audio, no text, no names, no continuous "phone home" - just:
  - install:      the first time this copy of the app ever runs
  - update:       the first run after the app's version changed
  - session_end:  once, when a run ends, carrying only a peak listener
                   count and which language codes were used

All of it is opt-out (see `is_enabled`/`set_enabled`), all of it is
best-effort - a ping that fails to send is simply lost, never retried,
never allowed to block or crash anything it's called from. The server
side of this lives in telemetry-server/.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from .config import APP_VERSION, ROOT

STATE_PATH = ROOT / "telemetry.json"

# telemetry-server/, deployed on Fly.io (telemetry-server/fly.toml). Blank
# would make telemetry a silent no-op instead - see check_install_or_update.
SERVER_URL = "https://neo-language-box-telemetry.fly.dev"

_TIMEOUT = 3.0


def _load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"  !! could not save telemetry state: {exc}", file=sys.stderr)


def is_enabled() -> bool:
    """Defaults to on - matches the operator dashboard's toggle default.
    Flip it from there, or by setting "enabled": false in telemetry.json."""
    return _load_state().get("enabled", True)


def set_enabled(value: bool) -> None:
    state = _load_state()
    state["enabled"] = bool(value)
    _save_state(state)


def _install_id(state: dict) -> str:
    iid = state.get("install_id")
    if not iid:
        iid = uuid.uuid4().hex
        state["install_id"] = iid
    return iid


def _post(path: str, payload: dict) -> bool:
    if not SERVER_URL:
        return False
    url = SERVER_URL.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _run_in_background(fn, *args) -> None:
    threading.Thread(target=fn, args=args, daemon=True, name="telemetry").start()


# ------------------------------------------------------------- public calls

def check_install_or_update(app_version: str = APP_VERSION) -> None:
    """Call once near startup. Sends at most one ping: "install" the very
    first time this copy of the app runs, or "update" the first run after
    its version changed. Silent no-op on every run in between."""
    if not is_enabled():
        return

    def _go():
        state = _load_state()
        iid = _install_id(state)
        previous = state.get("last_known_version")

        if previous is None:
            _post("/v1/ping", {
                "install_id": iid, "type": "install", "app_version": app_version,
            })
        elif previous != app_version:
            _post("/v1/ping", {
                "install_id": iid, "type": "update", "app_version": app_version,
                "previous_version": previous,
            })
        state["last_known_version"] = app_version
        _save_state(state)

    _run_in_background(_go)


def send_session_end(peak_listeners: int, languages: list[str],
                      session_seconds: float, app_version: str = APP_VERSION,
                      wait: bool = False) -> None:
    """Call once when a run ends. `wait=True` blocks the caller (briefly,
    bounded by _TIMEOUT) - use that at shutdown, where there is no "later"
    for a background thread to finish in before the process exits."""
    if not is_enabled():
        return
    state = _load_state()
    iid = state.get("install_id")
    if not iid:
        return     # no install ping ever recorded; nothing to attribute this to

    def _go():
        _post("/v1/ping", {
            "install_id": iid, "type": "session_end", "app_version": app_version,
            "session": {
                "peak_listeners": peak_listeners,
                "languages": languages,
                "session_seconds": session_seconds,
            },
        })

    if wait:
        _go()
    else:
        _run_in_background(_go)


def check_manifest() -> Optional[dict]:
    """Fetch the current {app_version, app_download_url, language_packs}.
    Returns None if telemetry is off, unconfigured, or unreachable - callers
    treat "nothing to report" and "couldn't check" identically, since there
    is no connectivity guarantee here and none should be assumed."""
    if not is_enabled() or not SERVER_URL:
        return None
    url = SERVER_URL.rstrip("/") + "/v1/manifest"
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None
