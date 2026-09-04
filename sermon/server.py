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
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Response

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
        self._history: deque = deque(maxlen=history_size)
        self._audio: "OrderedDict[tuple[int, str], tuple[str, bytes]]" = OrderedDict()
        self._audio_size = audio_size
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

    # -- server side (asyncio) --------------------------------------

    async def register(self, ws: ServerConnection) -> None:
        self._clients.add(ws)
        self.peak_clients = max(self.peak_clients, len(self._clients))
        with self._lock:
            history = list(self._history)
        await ws.send(json.dumps({"type": "welcome", "history": history,
                                  "targets": ["fr", "es"]}))

    def unregister(self, ws: ServerConnection) -> None:
        self._clients.discard(ws)

    async def _deliver(self, payload: str) -> None:
        dead: list[ServerConnection] = []
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)
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


def _make_process_request(broadcaster: Broadcaster, mic_source=None):
    async def _process_request(connection: ServerConnection, request):
        """HTTP handler. /ws and /mic upgrade to WebSocket (return None);
        other paths return a plain HTTP response."""
        path = request.path.split("?", 1)[0]
        if path in ("/ws", "/mic"):
            return None
        if path in ("/", "/index.html", "/listen"):
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
        if path.startswith("/api/") and request.method == "POST":
            action = path[len("/api/"):]
            src = mic_source
            resp: dict = {"ok": False}
            if src is not None and hasattr(src, "resume") and hasattr(src, "pause"):
                if action == "start":
                    src.resume(); resp = {"ok": True, "state": "running"}
                elif action == "stop":
                    src.pause(); resp = {"ok": True, "state": "paused"}
                else:
                    resp = {"ok": False, "error": f"unknown action {action!r}"}
            else:
                resp = {"ok": False, "error": "current source doesn't support start/stop"}
            body = json.dumps(resp).encode("utf-8")
            return Response(200, "OK", Headers([
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ]), body)
        if path == "/api/status":
            src = mic_source
            paused = getattr(src, "paused", None) if src else None
            body = json.dumps({
                "state": ("paused" if paused else "running") if paused is not None else "n/a",
                "listeners": broadcaster.client_count,
                "level_db": getattr(src, "last_db", -120.0) if src else -120.0,
                "peak_db": getattr(src, "peak_hold_db", -120.0) if src else -120.0,
                "in_speech": getattr(src, "in_speech", False) if src else False,
            }).encode("utf-8")
            return Response(200, "OK", Headers([
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
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
                ips = list_local_ips()
                if ips:
                    # First non-loopback address; assume the LAN HTTPS port
                    from urllib.parse import urlparse
                    host_only, _, port_str = host_hdr.partition(":")
                    port = port_str or "8443"
                    listener_url = f"https://{ips[0]}:{port}/"
            if not listener_url:
                listener_url = "https://<no-lan-ip>/"
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
            await _listener_handler(ws, broadcaster)
    return _handler


async def _listener_handler(ws: ServerConnection, broadcaster: Broadcaster) -> None:
    await broadcaster.register(ws)
    try:
        async for _ in ws:
            pass   # ignore anything the client sends
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
                     stop: asyncio.Event, mic_source=None) -> None:
    asyncio.get_running_loop().set_exception_handler(_quiet_exception_handler)
    handler = _make_handler(broadcaster, mic_source)
    process_request = _make_process_request(broadcaster, mic_source)

    from contextlib import AsyncExitStack
    async with AsyncExitStack() as stack:
        for b in binds:
            server = await stack.enter_async_context(serve(
                handler, b.host, b.port,
                process_request=process_request,
                ssl=b.ssl_ctx,
                ping_interval=20, ping_timeout=20,
                max_size=None,
                server_header="sermon-translation",
            ))
            _LOGGER.info("listening on %s://%s:%d", b.scheme, b.host, b.port)
        broadcaster.bind_loop(asyncio.get_running_loop())
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

    def start(self, timeout: float = 5.0) -> None:
        def _target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._stop_event = asyncio.Event()
            try:
                loop.call_soon(self._ready.set)
                loop.run_until_complete(
                    _run_async(self.broadcaster, self.binds,
                               self._stop_event, self.mic_source))
            except BaseException as exc:
                self._error = exc
                self._ready.set()
            finally:
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
