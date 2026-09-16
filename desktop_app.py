"""Native desktop app for the Speaker.

Launches the Python server in a background thread (system-audio source,
TLS on for LAN phones, HTTP loopback on 127.0.0.1:8000 for the local
webview), waits for it to accept requests, then opens a WebView2 window
pointed at http://127.0.0.1:8000/operator - the purpose-built operator UI.

Nothing else in the pipeline changes; this file is a thin shell.

Run as source: `python desktop_app.py`
Package via PyInstaller: single .exe + Inno Setup wraps it as an installer.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path


APP_TITLE = "Neo Sermon"
LOOPBACK_HOST = "127.0.0.1"
PREFERRED_PORT = 8000
WINDOW_WIDTH = 980
WINDOW_HEIGHT = 720

# Resolved at startup by _pick_port(); the window and the status poll both
# need to agree with whatever the server actually bound.
LOOPBACK_PORT = PREFERRED_PORT


def _pick_port(preferred: int = PREFERRED_PORT, tries: int = 12) -> int:
    """First port free on BOTH the LAN and loopback binds.

    8000 is a popular port - a local PHP or Node project sitting on it used to
    make this app start, announce a listener URL, and serve nothing, because
    the bind failure happened on a background thread. Rather than fail, take
    the next free port; phones are handed the real address by /api/net and the
    QR page, so nothing downstream cares which one it is.
    """
    import socket
    for port in range(preferred, preferred + tries):
        try:
            for host in ("0.0.0.0", LOOPBACK_HOST):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    s.bind((host, port))
                finally:
                    s.close()
            return port
        except OSError:
            continue
    return preferred          # nothing free; let the server report the error


def _bundled_root() -> Path:
    """Where the .exe (or this script) lives."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _resolve_models_dir() -> Path:
    """Same discovery rules as desktop_main.py."""
    env = os.environ.get("SERMON_MODELS_DIR")
    if env:
        return Path(env)
    beside_exe = _bundled_root() / "models"
    if beside_exe.exists():
        return beside_exe
    return Path.cwd() / "models"


def _missing_models(models: Path) -> list:
    """What the app needs but cannot find.

    Models ship separately from the installer, so "not copied yet" is a normal
    first-run state rather than a corrupt install. Saying exactly which pieces
    are absent beats a stack trace ninety seconds into a loading screen.
    """
    needed = [
        ("speech recognition", models / "parakeet-0.6b" / "encoder.int8.onnx"),
        ("translation (French)", models / "mt" / "opus-mt-en-fr" / "model.bin"),
        ("voices", models / "tts"),
    ]
    return [label for label, path in needed if not path.exists()]


DEFAULT_STT = "parakeet-0.6b"
# Only engines whose weights are actually in the installer. whisper-small.en
# is deliberately absent: it needed 9.3s to transcribe a 4.7s clip on this
# hardware, so shipping 464MB for it bought nothing. whisper-tiny.en was
# never bundled at all.
VALID_STT = {"parakeet-0.6b", "parakeet-110m", "whisper-base.en"}


def _stt_choice() -> str:
    """Which recognition engine to run.

    Default is Parakeet 0.6b. Measured on this laptop, end to end from the
    moment a sentence finishes to its translation being ready:

        whisper-base.en   2.6 - 2.7 s      (previous default)
        parakeet-0.6b     1.2 - 1.7 s      accuracy held up
        parakeet-110m     0.5 - 0.9 s      fastest, but dropped proper nouns

    Whisper's encoder always runs on a 30 s padded mel, so a short sentence
    pays the same as a long one - which is why it lagged. The trade-off going
    to Parakeet is that it has no vocabulary-prompt support, so terms in
    vocabulary.txt no longer bias recognition; the phrase-override table still
    applies on the way out.

    Overridable without a rebuild:  set SERMON_STT before launching.
    """
    choice = (os.environ.get("SERMON_STT") or "").strip()
    if choice in VALID_STT:
        return choice
    if choice:
        print(f"[desktop] ignoring SERMON_STT={choice!r}; "
              f"expected one of {sorted(VALID_STT)}")
    return DEFAULT_STT


def _start_server_thread() -> threading.Thread:
    """Kick off the pipeline in a daemon thread.

    We import inside the thread so any heavy startup (Whisper load,
    OPUS-MT load) happens off the main thread, letting the webview open
    quickly and show "starting..." while models come up.
    """
    def _run():
        # Set up args as if the user typed them on the command line
        sys.argv = [
            "run.py",
            "--serve",
            "--port", str(LOOPBACK_PORT),
            "--source", "system",
            "--tls",
            "--tauri",              # add loopback HTTP bind for the webview
            "--stt", _stt_choice(),
        ]
        from run import main as run_main
        try:
            run_main()
        except SystemExit:
            pass
        except Exception as exc:
            print(f"server thread crashed: {exc}", file=sys.stderr)

    t = threading.Thread(target=_run, daemon=True, name="sermon-server")
    t.start()
    return t


def _wait_for_server(timeout: float = 60.0) -> bool:
    """Poll the loopback HTTP endpoint until it answers, or give up."""
    deadline = time.monotonic() + timeout
    url = f"http://{LOOPBACK_HOST}:{LOOPBACK_PORT}/api/status"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> int:
    global LOOPBACK_PORT
    # Flush diagnostics line by line. Redirected output is block-buffered, and
    # the frozen app ignores PYTHONUNBUFFERED, so anything printed just before a
    # hang or a forced close was silently lost - exactly when it matters.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass
    models = _resolve_models_dir()
    os.environ.setdefault("SERMON_MODELS_DIR", str(models))

    missing = _missing_models(models)
    if missing:
        import webview
        webview.create_window(
            APP_TITLE, html=_models_missing_page(models, missing),
            width=WINDOW_WIDTH, height=WINDOW_HEIGHT,
            resizable=True, confirm_close=False, text_select=True)
        webview.start()
        return 1

    LOOPBACK_PORT = _pick_port()
    if LOOPBACK_PORT != PREFERRED_PORT:
        print(f"[desktop] port {PREFERRED_PORT} is in use; serving on {LOOPBACK_PORT}")

    # Open the window IMMEDIATELY with a loading page. Pywebview owns the main
    # thread from webview.start() onward; `_startup` runs on its own thread
    # once the window exists.
    import webview

    window = webview.create_window(
        APP_TITLE,
        html=_loading_page(),
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        resizable=True,
        confirm_close=False,
        text_select=True,
    )

    def _status(text: str) -> None:
        try:
            window.evaluate_js(
                "document.getElementById('status').textContent = "
                + json.dumps(text))
        except Exception:
            pass

    def _startup():
        # Verify BEFORE starting the server: a damaged model does not fail
        # cleanly, so it is far better to stop here with a list of files than
        # to load it and serve garbage.
        if not _check_models(window, models, _status):
            return

        print(f"[desktop] launching server (models dir: {models})")
        _status("Loading speech and translation models. This can take a "
                "minute with every language enabled.")
        _start_server_thread()
        if _wait_for_server(timeout=180):
            print("[desktop] server ready, swapping to /operator")
            window.load_url(f"http://{LOOPBACK_HOST}:{LOOPBACK_PORT}/operator")
        else:
            print("[desktop] server did not become ready in 180s")
            window.load_html(_error_page("Server did not start within 3 minutes. "
                                         "Check the console for details."))

    webview.start(_startup)
    return 0


def _check_models(window, models: Path, status) -> bool:
    """Verify the models against their manifest; show a page and stop if not.

    The full hash runs once and is remembered in a stamp beside the app, so
    later launches only pay for it again if a model file actually changed.
    """
    from sermon.integrity import verify

    last = [-1]

    def progress(done, total):
        pct = int(done * 100 / max(total, 1))
        if pct != last[0]:
            last[0] = pct
            status(f"Checking the language models ({pct}%). This happens once, "
                   "after they are copied.")

    stamp = _bundled_root() / "models-verified.json"
    try:
        report = verify(models, stamp_path=stamp, progress=progress)
    except Exception as exc:
        # A broken manifest must not block a service that might otherwise run.
        print(f"[desktop] model check skipped: {exc}")
        return True

    if not report.has_manifest:
        print("[desktop] no models manifest; skipping integrity check")
        return True
    if report.ok:
        print("[desktop] models verified" + (" (cached)" if report.cached else ""))
        return True

    print(f"[desktop] models damaged: {len(report.missing)} missing, "
          f"{len(report.corrupt)} corrupt")
    window.load_html(_models_damaged_page(models, report))
    return False


def _loading_page() -> str:
    return """<!doctype html>
<meta charset="utf-8">
<title>Loading Neo Sermon</title>
<style>
  body { margin:0; background:#0b0b0b; color:#e0e0e0;
         font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
         display:flex; align-items:center; justify-content:center;
         height:100vh; text-align:center; }
  .center { max-width: 360px; padding: 0 20px; }
  h1 { font-size:22px; margin:0 0 12px; font-weight:600; }
  p { color:#8a8a8a; font-size:14px; line-height:1.55; margin:0; }
  .dot { display:inline-block; width:10px; height:10px; margin-right:8px;
         background:#2b8dff; border-radius:50%%; animation:pulse 1.2s infinite; }
  @keyframes pulse { 0%%,100%%{opacity:0.3;} 50%%{opacity:1;} }
</style>
<div class="center">
  <h1><span class="dot"></span>Starting Neo Sermon</h1>
  <p id="status">Loading speech and translation models. This can take a
     minute with every language enabled.</p>
</div>
""".replace("%%", "%")


def _models_missing_page(models: Path, missing: list) -> str:
    items = "".join(f"<li>{m}</li>" for m in missing)
    return f"""<!doctype html>
<meta charset="utf-8">
<style>
  body {{ margin:0; background:#0b0b0b; color:#e0e0e0;
          font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
          display:flex; align-items:center; justify-content:center;
          height:100vh; }}
  .box {{ max-width:620px; padding:34px; }}
  h1 {{ color:#e8b53a; font-size:21px; margin:0 0 14px; }}
  p  {{ color:#b0b0b0; line-height:1.6; font-size:14px; }}
  code {{ background:#171717; border:1px solid #262626; border-radius:6px;
          padding:3px 7px; font-size:13px; color:#f2f2f2;
          word-break:break-all; }}
  ul {{ color:#b0b0b0; font-size:14px; line-height:1.7; }}
  .path {{ margin:18px 0; }}
</style>
<div class="box">
  <h1>The language models have not been copied yet</h1>
  <p>The app installs without them - they are about 5.8&nbsp;GB and are copied
     separately, once per machine.</p>
  <p>Missing:</p>
  <ul>{items}</ul>
  <p class="path">Copy the <code>models</code> folder here:</p>
  <p><code>{models}</code></p>
  <p>Then start the app again. Nothing needs downloading and no internet is
     required once the folder is in place.</p>
</div>
"""


def _models_damaged_page(models: Path, report) -> str:
    import html
    rows = ([("Missing", r) for r in report.missing]
            + [("Damaged", r) for r in report.corrupt])
    shown = rows[:15]
    items = "".join(f"<li><b>{kind}</b> &nbsp;<code>{html.escape(rel)}</code></li>"
                    for kind, rel in shown)
    more = (f"<p>…and {len(rows) - len(shown)} more.</p>"
            if len(rows) > len(shown) else "")
    return f"""<!doctype html>
<meta charset="utf-8">
<style>
  body {{ margin:0; background:#0b0b0b; color:#e0e0e0;
          font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
          display:flex; align-items:center; justify-content:center;
          min-height:100vh; }}
  .box {{ max-width:680px; padding:34px; }}
  h1 {{ color:#e05656; font-size:21px; margin:0 0 14px; }}
  p  {{ color:#b0b0b0; line-height:1.6; font-size:14px; }}
  ul {{ color:#b0b0b0; font-size:13px; line-height:1.8; padding-left:18px; }}
  b  {{ color:#e8b53a; font-weight:600; }}
  code {{ background:#171717; border:1px solid #262626; border-radius:6px;
          padding:2px 6px; font-size:12px; color:#f2f2f2; word-break:break-all; }}
</style>
<div class="box">
  <h1>Some language model files did not copy correctly</h1>
  <p>Neo Sermon checked every model file against the list it was shipped with,
     and these do not match. This usually means a download or copy was cut
     short. Starting anyway could produce wrong translations, so it has
     stopped here.</p>
  <ul>{items}</ul>
  {more}
  <p>Copy those files again from the original into:</p>
  <p><code>{html.escape(str(models))}</code></p>
  <p>Then start Neo Sermon again. Only the files listed need replacing.</p>
</div>
"""


def _error_page(msg: str) -> str:
    return f"""<!doctype html>
<meta charset="utf-8">
<style>
  body {{ margin:0; background:#0b0b0b; color:#e0e0e0;
          font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
          display:flex; align-items:center; justify-content:center; height:100vh; }}
  .center {{ max-width:500px; padding:30px; text-align:center; }}
  h1 {{ color:#e05656; font-size:20px; }}
  p {{ color:#8a8a8a; line-height:1.55; }}
</style>
<div class="center">
  <h1>Could not start the translation server</h1>
  <p>{msg}</p>
</div>
"""


if __name__ == "__main__":
    sys.exit(main())
