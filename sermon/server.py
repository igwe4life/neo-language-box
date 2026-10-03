"""Local Wi-Fi broadcaster: phones on the same network open one HTML page in
their browser and get live French / Yoruba translations streamed to them over a
WebSocket.

Everything runs on the same machine that already does STT/MT/TTS. No cloud,
no accounts, no external network. Peak per-listener bandwidth is a few kB per
translation, so the practical ceiling on this laptop is dozens of listeners,
not the WebSocket count.

Architecture
------------
* One asyncio server (`websockets.serve`) on one port serves BOTH the HTML page
  (via process_request) and the WebSocket at /ws. Avoids a second port, avoids
  Windows Firewall asking twice.
* The server runs in its own thread so the recognition pipeline stays
  synchronous and simple. Broadcasts hop threads via
  `asyncio.run_coroutine_threadsafe`.
* `Broadcaster` holds an in-memory ring of the last N translations. Newly-
  connected phones receive it on join so a listener who arrives mid-sermon
  isn't staring at a blank screen.
* If a client's socket is dead, it is quietly dropped on the next broadcast;
  there is no per-client retry queue - a live sermon can't wait for a phone
  that has gone to sleep in someone's pocket.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from .config import LANGUAGES, ROOT, SOURCE_LANG

_LOGGER = logging.getLogger("sermon.server")
LISTENER_HTML = Path(__file__).parent / "templates" / "listen.html"
SPEAKER_HTML = Path(__file__).parent / "templates" / "speaker.html"
QR_HTML = Path(__file__).parent / "templates" / "qr.html"
HTML_PATH = LISTENER_HTML   # backwards compat for diagnostics


def _qr_svg(url: str) -> str:
    """QR code as inline SVG. Rendered once per HTTP request; tiny."""
    import io, segno
    q = segno.make(url, error="M")
    buf = io.BytesIO()
    q.save(buf, kind="svg", scale=10, dark="#000", light="#fff",
           xmldecl=False, svgns=False, omitsize=False)
    return buf.getvalue().decode()


# --------------------------------------------------------------------- broadcaster

class Broadcaster:
    """Thread-safe fan-out of translation messages to WebSocket clients, plus
    an in-memory ring of recent audio clips that phones fetch by HTTP.

    `broadcast()` is called from the pipeline thread; `_deliver()` runs on the
    server's asyncio loop. History is kept as a small ring so newcomers see
    context, not a blank page. Audio is kept in a separate ring keyed by
    (translation_id, lang) so a viewer's HTTP fetch never crosses the WebSocket
    fan-out - a muted or Yoruba-only listener costs zero audio bandwidth.
    """

    def __init__(self, history_size: int = 20, audio_size: int = 30):
        self._clients: set[ServerConnection] = set()
        # The operator's own dashboard also opens a /ws connection (to receive
        # live translations), but it is not a listener - counting it would
        # inflate "listeners connected" by one every time the dashboard is
        # open. Kept in a separate set so it still receives broadcasts.
        self._operator_clients: set[ServerConnection] = set()
        # Which language each connected phone is listening to, so the pipeline
        # can skip synthesising audio nobody will play. Keyed by connection so
        # a disconnect releases its claim automatically.
        self._client_lang: "dict[ServerConnection, str]" = {}
        self._history: deque = deque(maxlen=history_size)
        self._audio: "OrderedDict[tuple[int, str], tuple[str, bytes]]" = OrderedDict()
        self._audio_size = audio_size
        # What the listener UI should offer. Set by run.py once it knows which
        # models actually loaded, so a language whose model is missing is never
        # shown as a choice.
        self.languages: list[str] = []
        self.language_meta: dict = {}
        # Operator-editable session labels, shown on the dashboard. Purely
        # descriptive - never sent to a client, never affects the pipeline.
        self.session_meta: dict = {"event": "", "session": ""}
        # Vanity slug for /listeners/<slug> - e.g. "hslhs-2026". Empty means no
        # custom slug has been set; /listeners/<anything> still works (see the
        # route), this only controls what the dashboard suggests for sharing.
        self.listener_slug: str = ""
        # The language actually being spoken/transcribed. Only "en" has a
        # bundled speech-recognition model today (see /api/source-language),
        # so this exists mainly as the real, honest plumbing for when a
        # second model is added - not a feature that silently does nothing.
        self.source_lang: str = SOURCE_LANG
        self.started_at = time.monotonic()
        # Set by run.py when recording support is wired in. Optional: a
        # Broadcaster built without one (e.g. in a future test) just has no
        # recording routes respond usefully.
        self.recorder = None
        # In-memory listener-reported issues (translation wrong, no audio,
        # etc.) - also append-only logged to disk, see /api/report, so a
        # restart does not lose them.
        self.reports: list = []
        self._report_counter = 0
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.total_delivered = 0
        self.total_dropped = 0
        self.peak_clients = 0
        self.audio_bytes_served = 0
        self.audio_requests = 0

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self.started_at

    # -- server side (asyncio) --------------------------------------

    async def register(self, ws: ServerConnection, is_operator: bool = False) -> None:
        (self._operator_clients if is_operator else self._clients).add(ws)
        self.peak_clients = max(self.peak_clients, len(self._clients))
        with self._lock:
            history = list(self._history)
        await ws.send(json.dumps({"type": "welcome", "history": history,
                                  "targets": list(self.languages),
                                  "languages": self.language_meta,
                                  "event": self.session_meta.get("event", "")},
                                 ensure_ascii=False))

    def unregister(self, ws: ServerConnection) -> None:
        self._clients.discard(ws)
        self._operator_clients.discard(ws)
        self._client_lang.pop(ws, None)

    def set_client_language(self, ws: ServerConnection, lang: str) -> None:
        """Record what a listener selected. Drives which languages get audio."""
        self._client_lang[ws] = lang

    def active_languages(self) -> set:
        """Languages at least one connected listener has selected.

        Called from the pipeline thread while the asyncio loop may be mutating
        the map, so take a snapshot rather than iterating it live.
        """
        return set(list(self._client_lang.values()))

    def language_counts(self) -> dict:
        """How many connected listeners are on each language.

        A phone that has connected but not yet sent its `{type:"lang"}` message
        (still on the picker screen) counts under "" - shown to the operator as
        "choosing", not silently dropped from the total.
        """
        langs = list(self._client_lang.values())
        counts: dict = {}
        for lang in langs:
            counts[lang] = counts.get(lang, 0) + 1
        unset = len(self._clients) - len(langs)
        if unset > 0:
            counts[""] = unset
        return counts

    async def _deliver(self, payload: str) -> None:
        dead: list[ServerConnection] = []
        # The operator dashboard needs the same live-translation stream as a
        # listener, just not counted as one.
        for ws in list(self._clients) + list(self._operator_clients):
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)
            self._operator_clients.discard(ws)
        self.total_delivered += (len(self._clients))

    # -- pipeline side (any thread) ---------------------------------

    def broadcast(self, message: dict) -> None:
        with self._lock:
            self._history.append(message)
        loop = self._loop
        if loop is None:
            self.total_dropped += 1
            return
        payload = json.dumps(message, ensure_ascii=False)
        try:
            asyncio.run_coroutine_threadsafe(self._deliver(payload), loop)
        except RuntimeError:
            self.total_dropped += 1

    def broadcast_audio(self, translation_id: int, urls: dict) -> None:
        """Second-stage message: the audio clips for an already-sent sentence.

        Text goes out the moment translation finishes; synthesis follows and
        lands here. The matching history entry is patched in place so a phone
        that joins later still receives one complete message.
        """
        with self._lock:
            for entry in self._history:
                if entry.get("id") == translation_id:
                    entry.update(urls)
                    entry.pop("audio_pending", None)
                    break
        loop = self._loop
        if loop is None:
            self.total_dropped += 1
            return
        payload = json.dumps({"type": "audio", "id": translation_id, **urls},
                             ensure_ascii=False)
        try:
            asyncio.run_coroutine_threadsafe(self._deliver(payload), loop)
        except RuntimeError:
            self.total_dropped += 1

    def store_audio(self, translation_id: int, lang: str, mime: str, data: bytes) -> str:
        """Cache an audio clip and return the URL a phone should fetch."""
        key = (translation_id, lang)
        with self._lock:
            self._audio[key] = (mime, data)
            self._audio.move_to_end(key)
            while len(self._audio) > self._audio_size:
                self._audio.popitem(last=False)
        ext = {"audio/wav": "wav", "audio/mpeg": "mp3", "audio/ogg": "ogg"}.get(mime, "bin")
        return f"/audio/{translation_id}.{lang}.{ext}"

    def get_audio(self, translation_id: int, lang: str) -> tuple[str, bytes] | None:
        with self._lock:
            return self._audio.get((translation_id, lang))


# ------------------------------------------------------------------- HTTP + WS

def _read(path: Path) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"template missing: {path}")
    return path.read_bytes()


import re as _re

_AUDIO_PATH = _re.compile(r"^/audio/(\d+)\.([a-z]{2})\.([a-z0-9]+)$")


def _make_process_request(broadcaster: Broadcaster, mic_source=None, binds=None):
    async def _process_request(connection: ServerConnection, request):
        """HTTP handler. /ws and /mic upgrade to WebSocket (return None);
        other paths return a plain HTTP response."""
        path = request.path.split("?", 1)[0]
        if path in ("/ws", "/mic"):
            return None
        # /neo/<anything> is the shareable vanity URL (item 8): it always
        # serves the same listener page as "/", regardless of what the slug
        # is or whether one has even been configured. The slug is cosmetic -
        # there is no auth anywhere in this app - so a stale or mistyped slug
        # must never turn into a broken link for an arriving listener.
        # "/listeners/" is kept working too so an already-shared/bookmarked
        # link from before this rename doesn't 404.
        if (path in ("/", "/index.html", "/listen")
                or path.startswith("/neo/") or path.startswith("/listeners/")):
            body = _read(LISTENER_HTML)
            return Response(200, "OK", Headers([
                ("Content-Type", "text/html; charset=utf-8"),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
            ]), body)
        if path in ("/speaker", "/speak"):
            body = _read(SPEAKER_HTML)
            return Response(200, "OK", Headers([
                ("Content-Type", "text/html; charset=utf-8"),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
            ]), body)
        if path in ("/operator", "/op"):
            body = _read(Path(__file__).parent / "templates" / "operator.html")
            return Response(200, "OK", Headers([
                ("Content-Type", "text/html; charset=utf-8"),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
            ]), body)
        # Only these three legacy actions go through the generic handler below.
        # It used to be "any /api/* POST", which silently swallowed /api/session
        # with a {"ok": false, "error": "unknown action"} response - every new
        # route added after it needed its own exclusion, a bug waiting to
        # repeat. An allowlist means a new route below just works.
        if (path.startswith("/api/") and request.method == "POST"
                and path[len("/api/"):] in ("start", "stop", "source")):
            action = path[len("/api/"):]
            src = mic_source
            resp: dict = {"ok": False, "error": "no audio source is running"}
            if src is not None:
                # Prefer explicit pause/resume (SystemAudioSource); fall back
                # to mute/unmute on MicrophoneSource - functionally equivalent
                # for the operator (audio in is gated either way).
                if action == "start":
                    if hasattr(src, "resume"): src.resume()
                    elif hasattr(src, "unmute"): src.unmute()
                    resp = {"ok": True, "state": "running"}
                elif action == "stop":
                    if hasattr(src, "pause"): src.pause()
                    elif hasattr(src, "mute"): src.mute()
                    resp = {"ok": True, "state": "paused"}
                elif action == "source":
                    # Params ride in the query string: the websockets
                    # process_request hook does not expose the request body.
                    from urllib.parse import parse_qs, urlparse
                    q = parse_qs(urlparse(request.path).query)
                    kind = (q.get("kind") or [""])[0]
                    device = (q.get("device") or [""])[0] or None
                    if not hasattr(src, "set_input"):
                        resp = {"ok": False,
                                "error": "this audio source cannot be switched"}
                    else:
                        try:
                            src.set_input(kind, device)
                            resp = {"ok": True, "kind": src.kind,
                                    "device": src.device_name}
                        except Exception as exc:
                            resp = {"ok": False, "error": str(exc),
                                    "kind": getattr(src, "kind", None)}
                else:
                    resp = {"ok": False, "error": f"unknown action {action!r}"}
            body = json.dumps(resp).encode("utf-8")
            return Response(200, "OK", Headers([
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ]), body)
        if path == "/api/status":
            src = mic_source
            # SystemAudioSource uses `paused`; MicrophoneSource uses `muted`.
            # Either indicates "not currently forwarding audio downstream."
            if src is None:
                state = "n/a"
            elif getattr(src, "paused", False) or getattr(src, "muted", False):
                state = "paused"
            else:
                state = "running"
            body = json.dumps({
                "state": state,
                "listeners": broadcaster.client_count,
                "listener_langs": broadcaster.language_counts(),
                "languages_active": len(broadcaster.languages),
                "uptime_seconds": round(broadcaster.uptime_seconds, 1),
                "session": broadcaster.session_meta,
                "listener_slug": broadcaster.listener_slug,
                "recording": (broadcaster.recorder.status()
                             if broadcaster.recorder else {"active": False}),
                "reports_count": len(broadcaster.reports),
                "level_db": getattr(src, "last_db", -120.0) if src else -120.0,
                "peak_db": getattr(src, "peak_hold_db", -120.0) if src else -120.0,
                "in_speech": getattr(src, "in_speech", False) if src else False,
                "kind": getattr(src, "kind", None) if src else None,
                "device": getattr(src, "device_name", None) if src else None,
                "switchable": hasattr(src, "set_input") if src else False,
                "source_error": getattr(src, "last_error", None) if src else None,
            }).encode("utf-8")
            return Response(200, "OK", Headers([
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
            ]), body)
        if path == "/api/session" and request.method == "POST":
            # Purely descriptive labels for the dashboard's Session card - not
            # used anywhere in the pipeline, so no validation beyond a length
            # cap to keep a stray paste from blowing up the display.
            from urllib.parse import parse_qs, urlparse
            # keep_blank_values: a cleared field submits "event=" etc., and
            # without this flag parse_qs drops blank params entirely - the
            # "if ... in q" checks below would then never see them, so
            # clearing a field in Settings would silently do nothing.
            q = parse_qs(urlparse(request.path).query, keep_blank_values=True)
            if "event" in q:
                broadcaster.session_meta["event"] = q["event"][0][:80]
            if "session" in q:
                broadcaster.session_meta["session"] = q["session"][0][:80]
            if "slug" in q:
                from sermon.recorder import slugify
                raw = q["slug"][0].strip()
                # Blank clears the custom slug (falls back to "/" in the QR
                # page); anything else is slugified so a pasted "HSLHS 2026!"
                # cannot produce a URL with spaces or punctuation in it.
                broadcaster.listener_slug = slugify(raw, fallback="") if raw else ""
            body = json.dumps({"ok": True, "session": broadcaster.session_meta,
                               "listener_slug": broadcaster.listener_slug}).encode("utf-8")
            return Response(200, "OK", Headers([
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ]), body)
        if path == "/api/net":
            # The URLs a PHONE should use. The operator window itself is on
            # http://127.0.0.1:8000, so `location.origin` in the page is
            # useless for handing out - this is the authoritative answer.
            ips = list_local_ips()
            lan = []
            for b in (binds or []):
                if b.host in ("127.0.0.1", "localhost", "::1"):
                    continue
                for ip in ips:
                    lan.append({"url": f"{b.scheme}://{ip}:{b.port}/",
                                "scheme": b.scheme, "ip": ip, "port": b.port})
            body = json.dumps({
                "lan": lan,
                "ips": ips,
                "listening": [f"{b.scheme}://{b.host}:{b.port}"
                              for b in (binds or [])],
            }).encode("utf-8")
            return Response(200, "OK", Headers([
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
            ]), body)
        if path == "/api/devices":
            # Both backends the operator can pick between. Enumeration can
            # fail on odd driver states, so each side is guarded separately -
            # a broken loopback list should not hide the microphones.
            mics: list = []
            loopbacks: list = []
            try:
                from sermon.audio import SelectableSource
                mics = SelectableSource.list_input_devices()
            except Exception as exc:
                _LOGGER.warning("input device enumeration failed: %s", exc)
            try:
                from sermon.audio import SystemAudioSource
                loopbacks = SystemAudioSource.list_loopback_devices()
            except Exception as exc:
                _LOGGER.warning("loopback device enumeration failed: %s", exc)
            body = json.dumps({"mics": mics, "loopbacks": loopbacks}).encode("utf-8")
            return Response(200, "OK", Headers([
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
            ]), body)
        if path == "/api/source-language" and request.method != "POST":
            # Real endpoint, honestly scoped: ONLY English has a bundled
            # speech-recognition model today, so it is the only entry in
            # "available". The full registry is still returned so the
            # dashboard can show every language with a lock icon on the rest,
            # rather than pretending a switch exists only to do nothing.
            available = [SOURCE_LANG]
            body = json.dumps({
                "current": broadcaster.source_lang,
                "available": available,
                "languages": {c: m for c, m in LANGUAGES.items()},
            }, ensure_ascii=False).encode("utf-8")
            return Response(200, "OK", Headers([
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
            ]), body)
        if path == "/api/source-language" and request.method == "POST":
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(request.path).query)
            lang = (q.get("lang") or [""])[0]
            if lang == broadcaster.source_lang:
                resp = {"ok": True, "current": lang}
            elif lang == SOURCE_LANG:
                broadcaster.source_lang = lang
                resp = {"ok": True, "current": lang}
            else:
                name = (LANGUAGES.get(lang) or {}).get("native", lang or "that language")
                resp = {"ok": False,
                       "error": f"{name} needs its own speech-recognition model, "
                                 "which is not installed yet. Only English is "
                                 "supported as the spoken/source language right now."}
            body = json.dumps(resp, ensure_ascii=False).encode("utf-8")
            return Response(200, "OK", Headers([
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ]), body)
        if path.startswith("/api/recording/") and request.method == "POST":
            action = path[len("/api/recording/"):]
            recorder = broadcaster.recorder
            if recorder is None:
                resp = {"ok": False, "error": "recording is not available"}
            elif action == "start":
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(request.path).query)
                raw = (q.get("languages") or [""])[0].strip()
                if not raw or raw == "all":
                    wanted = set(broadcaster.languages)
                else:
                    wanted = {c.strip() for c in raw.split(",") if c.strip()}
                    wanted &= set(broadcaster.languages)   # ignore unknown codes
                try:
                    info = recorder.start(wanted, broadcaster.language_meta,
                                          event_label=broadcaster.session_meta.get("event", ""))
                    resp = {"ok": True, **info}
                except Exception as exc:
                    resp = {"ok": False, "error": str(exc)}
            elif action == "pause":
                recorder.pause(); resp = {"ok": True}
            elif action == "resume":
                recorder.resume(); resp = {"ok": True}
            elif action == "stop":
                resp = recorder.stop()
            elif action == "open-folder":
                # Loopback-only: this opens a window on the machine the server
                # runs on, so it must never be reachable from a phone on the LAN.
                try:
                    host_hdr = request.headers.get("Host", "")
                except Exception:
                    host_hdr = ""
                if not (host_hdr.startswith("127.") or host_hdr.startswith("localhost")):
                    resp = {"ok": False, "error": "only available from the operator window"}
                else:
                    try:
                        import os
                        recorder.base_dir.mkdir(parents=True, exist_ok=True)
                        os.startfile(recorder.base_dir)  # noqa: S606 - fixed, server-owned path
                        resp = {"ok": True}
                    except Exception as exc:
                        resp = {"ok": False, "error": str(exc)}
            else:
                resp = {"ok": False, "error": f"unknown action {action!r}"}
            body = json.dumps(resp, ensure_ascii=False).encode("utf-8")
            return Response(200, "OK", Headers([
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ]), body)
        if path == "/api/recording/sessions":
            recorder = broadcaster.recorder
            sessions = recorder.list_sessions() if recorder else []
            body = json.dumps({"sessions": sessions}, ensure_ascii=False).encode("utf-8")
            return Response(200, "OK", Headers([
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
            ]), body)
        if path == "/api/report" and request.method == "POST":
            # From the LISTENER page - untrusted input, so every field is
            # capped and only known-shape data is kept. Appended both to the
            # in-memory list the dashboard polls and to a log file, so a
            # report survives an app restart even if nobody saw it live.
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(request.path).query)
            kind = (q.get("type") or ["other"])[0][:32]
            note = (q.get("note") or [""])[0][:500]
            lang = (q.get("lang") or [""])[0][:8]
            broadcaster._report_counter += 1
            report = {
                "id": broadcaster._report_counter,
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": kind,
                "note": note,
                "lang": lang,
            }
            broadcaster.reports.append(report)
            # Unbounded growth over a multi-hour service is unlikely (this is
            # per-listener-initiated, not automatic), but cap it anyway so a
            # misbehaving client spamming the endpoint cannot grow this forever.
            del broadcaster.reports[:-200]
            try:
                with open(ROOT / "reports.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({**report, "date": datetime.now().isoformat(timespec="seconds")},
                                       ensure_ascii=False) + "\n")
            except Exception:
                pass
            body = json.dumps({"ok": True}).encode("utf-8")
            return Response(200, "OK", Headers([
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ]), body)
        if path == "/api/reports":
            body = json.dumps({"reports": broadcaster.reports}, ensure_ascii=False).encode("utf-8")
            return Response(200, "OK", Headers([
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
            ]), body)
        if path == "/api/reports/clear" and request.method == "POST":
            broadcaster.reports.clear()
            body = json.dumps({"ok": True}).encode("utf-8")
            return Response(200, "OK", Headers([
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ]), body)
        if path == "/qr":
            # Rendered on-the-fly with the ORIGIN the client used to reach us,
            # so if the operator opens it from LAN their phones scan the LAN
            # URL, not localhost. Falls back to first LAN IP if the request
            # arrived over loopback (e.g. the desktop app previewing it).
            try:
                host_hdr = request.headers.get("Host", "")
            except Exception:
                host_hdr = ""
            listener_url = ""
            if host_hdr and not host_hdr.startswith("127.") \
                    and not host_hdr.startswith("localhost"):
                # Use whatever origin the browser hit us on
                scheme = "https" if getattr(request, "extensions", None) is None else "https"
                listener_url = f"https://{host_hdr}/"
            else:
                # Reached over loopback (the desktop app's own webview), so the
                # Host header names an address no phone can use. Build the URL
                # from the real LAN bind instead of guessing a port.
                ips = list_local_ips()
                lan_bind = next((b for b in (binds or [])
                                 if b.host not in ("127.0.0.1", "localhost", "::1")),
                                None)
                if ips and lan_bind is not None:
                    listener_url = f"{lan_bind.scheme}://{ips[0]}:{lan_bind.port}/"
                elif ips:
                    _, _, port_str = host_hdr.partition(":")
                    listener_url = f"https://{ips[0]}:{port_str or '8443'}/"
            if not listener_url:
                listener_url = "https://<no-lan-ip>/"
            # The vanity path, when the operator has set one - purely cosmetic,
            # "/neo/<anything>" already works regardless (see above).
            if broadcaster.listener_slug:
                listener_url = f"{listener_url.rstrip('/')}/neo/{broadcaster.listener_slug}"
            page = _read(QR_HTML).decode("utf-8")
            page = page.replace("{{URL}}", listener_url)
            page = page.replace("{{QR_SVG}}", _qr_svg(listener_url))
            body = page.encode("utf-8")
            return Response(200, "OK", Headers([
                ("Content-Type", "text/html; charset=utf-8"),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
            ]), body)
        if path == "/health":
            body = b'{"status":"ok"}'
            return Response(200, "OK", Headers([
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ]), body)

        m = _AUDIO_PATH.match(path)
        if m:
            tid, lang, _ext = int(m.group(1)), m.group(2), m.group(3)
            hit = broadcaster.get_audio(tid, lang)
            if hit is None:
                body = b"gone"
                return Response(404, "Not Found", Headers([
                    ("Content-Type", "text/plain"),
                    ("Content-Length", str(len(body))),
                ]), body)
            mime, data = hit
            broadcaster.audio_requests += 1
            broadcaster.audio_bytes_served += len(data)
            return Response(200, "OK", Headers([
                ("Content-Type", mime),
                ("Content-Length", str(len(data))),
                ("Cache-Control", "public, max-age=300"),
                # Some phones only play range-served audio; advertise support.
                ("Accept-Ranges", "bytes"),
            ]), data)

        body = b"Not Found"
        return Response(404, "Not Found", Headers([
            ("Content-Type", "text/plain"),
            ("Content-Length", str(len(body))),
        ]), body)
    return _process_request


def _make_handler(broadcaster: Broadcaster, mic_source=None):
    """Routes both listener connections (/ws) and speaker mic streams (/mic).

    Speaker frames are int16 PCM at 16 kHz mono, arriving as binary
    WebSocket messages. The handler feeds them into `mic_source` (a
    NetworkMicrophoneSource); segmentation and downstream pipeline are
    unchanged from the local-mic case.
    """
    async def _handler(ws: ServerConnection) -> None:
        # websockets 15+ exposes the requested path via ws.request.path
        path = getattr(getattr(ws, "request", None), "path", "/ws")
        if path.startswith("/mic"):
            await _mic_handler(ws, mic_source)
        else:
            # The operator dashboard marks itself with ?role=operator so it is
            # not counted as a listener (see Broadcaster._operator_clients).
            _, _, query = path.partition("?")
            is_operator = "role=operator" in query
            await _listener_handler(ws, broadcaster, is_operator)
    return _handler


async def _listener_handler(ws: ServerConnection, broadcaster: Broadcaster,
                            is_operator: bool = False) -> None:
    await broadcaster.register(ws, is_operator=is_operator)
    try:
        async for raw in ws:
            # The only thing a listener tells us is which language it is on,
            # so the pipeline can skip synthesising audio nobody will play.
            # Anything else is ignored - this is an untrusted input.
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if isinstance(msg, dict) and msg.get("type") == "lang":
                lang = msg.get("lang")
                if isinstance(lang, str) and 0 < len(lang) <= 8:
                    broadcaster.set_client_language(ws, lang)
    except Exception:
        pass
    finally:
        broadcaster.unregister(ws)


async def _mic_handler(ws: ServerConnection, mic_source) -> None:
    """Receive PCM frames from the speaker's browser and hand them to the
    network mic source. First frame is JSON: {"sampleRate": N}. Following
    frames are little-endian int16 PCM at that rate. Server treats the
    stream as authoritative - if a second speaker connects, the first is
    silently displaced (last-writer-wins, no coordination protocol needed).
    """
    if mic_source is None:
        try:
            await ws.send(json.dumps({"error": "no network mic configured "
                                      "(server started without --source browser)"}))
        finally:
            await ws.close(code=1011, reason="no network mic")
        return

    import numpy as np
    sample_rate = 16000
    try:
        # First message: handshake
        first = await asyncio.wait_for(ws.recv(), timeout=5.0)
        if isinstance(first, str):
            try:
                info = json.loads(first)
                sample_rate = int(info.get("sampleRate", 16000))
            except Exception:
                pass
        else:
            # No handshake, treat as PCM at default rate
            _submit_pcm(mic_source, first, 16000)

        mic_source.notify_speaker(True)
        _LOGGER.info("speaker connected, sample rate %d Hz", sample_rate)

        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                _submit_pcm(mic_source, msg, sample_rate)
            # ignore stray text frames after handshake
    except Exception:
        pass
    finally:
        mic_source.notify_speaker(False)
        _LOGGER.info("speaker disconnected")


def _submit_pcm(mic_source, raw: bytes, sample_rate: int) -> None:
    """Convert int16 PCM to float32 [-1,1] and resample to 16 kHz if needed."""
    import numpy as np
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_rate != 16000 and pcm.size > 0:
        # Simple linear resample. Speech through Whisper is tolerant enough
        # that this is fine; avoids pulling in scipy just for this hop.
        n_out = int(pcm.size * 16000 / sample_rate)
        if n_out > 0:
            src_x = np.linspace(0, 1, pcm.size, endpoint=False)
            dst_x = np.linspace(0, 1, n_out, endpoint=False)
            pcm = np.interp(dst_x, src_x, pcm).astype(np.float32)
    # Feed the segmenter in blocks matching the local-mic block size, so its
    # frame math stays consistent.
    block = mic_source.blocksize
    for start in range(0, pcm.size, block):
        mic_source.submit_audio(pcm[start:start + block])


# ------------------------------------------------------------------ runner

def _quiet_exception_handler(loop, context):
    """Suppress the noisy 'ConnectionResetError [WinError 10054]' spam that
    Windows' proactor loop emits when a WebSocket client vanishes abruptly.
    Everything else falls through to the default handler."""
    exc = context.get("exception")
    if isinstance(exc, ConnectionResetError):
        return
    msg = context.get("message", "")
    if "shutdown" in msg.lower() and "10054" in str(context):
        return
    loop.default_exception_handler(context)


@dataclass
class Bind:
    """One (host, port, ssl) endpoint the server should listen on.

    The packaged desktop app runs TWO binds: HTTPS on 0.0.0.0:8443 for phones
    on the LAN, and HTTP on 127.0.0.1:8000 for the Tauri window's own webview
    (loopback = secure context, so no self-signed cert warning inside the app).
    """
    host: str
    port: int
    ssl_ctx: Optional[object] = None       # ssl.SSLContext, kept as object to avoid the import here

    @property
    def scheme(self) -> str:
        return "https" if self.ssl_ctx is not None else "http"


async def _run_async(broadcaster: Broadcaster, binds: list[Bind],
                     stop: asyncio.Event, mic_source=None,
                     ready: "threading.Event | None" = None) -> None:
    asyncio.get_running_loop().set_exception_handler(_quiet_exception_handler)
    handler = _make_handler(broadcaster, mic_source)
    process_request = _make_process_request(broadcaster, mic_source, binds)

    from contextlib import AsyncExitStack
    async with AsyncExitStack() as stack:
        for b in binds:
            server = await stack.enter_async_context(serve(
                handler, b.host, b.port,
                process_request=process_request,
                ssl=b.ssl_ctx,
                ping_interval=20, ping_timeout=20,
                max_size=None,
                server_header="neo-sermon",
            ))
            _LOGGER.info("listening on %s://%s:%d", b.scheme, b.host, b.port)
        broadcaster.bind_loop(asyncio.get_running_loop())
        # Only NOW is the server actually serving. Signalling readiness any
        # earlier makes a failed bind invisible: start() returns happily, the
        # operator is told the listener URL, and nothing is listening on it.
        if ready is not None:
            ready.set()
        await stop.wait()


class ServerHandle:
    """Runs the server in a background thread. Call `.stop()` to shut down.

    Accepts either the single-bind shape (host, port, ssl_ctx) for backwards
    compatibility, or an explicit `binds=[Bind(...), Bind(...)]` list for the
    dual-bind desktop-app case.
    """

    def __init__(self, broadcaster: Broadcaster,
                 host: str | None = None, port: int | None = None,
                 mic_source=None, ssl_ctx=None,
                 binds: Optional[list[Bind]] = None):
        self.broadcaster = broadcaster
        self.mic_source = mic_source
        if binds is not None:
            self.binds = list(binds)
        else:
            assert host is not None and port is not None, \
                "ServerHandle needs either binds=[...] or host+port"
            self.binds = [Bind(host, port, ssl_ctx)]
        # Kept as attributes for callers that read .host/.port after construction
        self.host = self.binds[0].host
        self.port = self.binds[0].port
        self.ssl_ctx = self.binds[0].ssl_ctx
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._error: Optional[BaseException] = None
        self._ready = threading.Event()

    def start(self, timeout: float = 15.0) -> None:
        def _target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._stop_event = asyncio.Event()
            try:
                loop.run_until_complete(
                    _run_async(self.broadcaster, self.binds,
                               self._stop_event, self.mic_source,
                               self._ready))
            except BaseException as exc:
                self._error = exc
            finally:
                self._ready.set()      # unblock start() on success OR failure
                try:
                    loop.close()
                except Exception:
                    pass

        self._thread = threading.Thread(target=_target, daemon=True,
                                        name="sermon-server")
        self._thread.start()
        self._ready.wait(timeout=timeout)
        if self._error is not None:
            raise self._error
        if not self.listening:
            raise OSError(
                f"server did not start within {timeout:.0f}s on "
                + ", ".join(f"{b.scheme}://{b.host}:{b.port}" for b in self.binds))

    @property
    def listening(self) -> bool:
        """True once every bind is accepting connections."""
        return self._ready.is_set() and self._error is None and (
            self._thread is not None and self._thread.is_alive())

    def stop(self, timeout: float = 3.0) -> None:
        if self._loop is None or self._stop_event is None:
            return
        self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout=timeout)


# ------------------------------------------------------------------ discovery

def list_local_ips() -> list[str]:
    """Best-effort list of IPv4 addresses on which phones could reach us.
    Excludes loopback."""
    ips: set[str] = set()

    # Trick: connecting a UDP socket doesn't send anything, but Windows picks
    # the outbound interface, whose IP is the one on the phone's network.
    for target in ("8.8.8.8", "1.1.1.1"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((target, 80))
            ips.add(s.getsockname()[0])
            s.close()
        except Exception:
            pass

    # Everything else the OS knows about, in case we're on multiple networks
    # (Wi-Fi + Ethernet + phone hotspot).
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass

    return sorted(ips)
