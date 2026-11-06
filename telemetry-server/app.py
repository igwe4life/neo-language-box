"""Neo Language Box's companion server: three jobs, kept deliberately small.

1. POST /v1/ping   - accepts the app's three check-ins (install, update,
   session_end) and logs them. No audio, no text, no names - an anonymous
   install id, an app version, and for session_end a peak listener count
   and the list of language codes that were active. Country is never sent
   by the client; it is guessed server-side from the request's own IP.

2. GET /v1/manifest - what update.py and the language-pack downloader on
   the client poll: the current app version and the list of available
   language packs with their version/url/checksum. Edit manifest.json by
   hand to add a language - this file is the "simple server where you can
   add more language modules" from the original ask.

3. GET /v1/stats   - a plain HTML dashboard for the project owner: install
   count, countries, last-seen, and aggregate session numbers. No login,
   because nothing it shows is sensitive - but see the README before
   putting this on a host you don't want it guessed to by URL alone.

4. GET /v1/packs/<file> - serves the actual language-pack zip files (built
   by scripts/build_language_pack.py) from PACKS_DIR. This was meant to be
   a separate static host - see telemetry-server/README.md - but nothing
   else was set up yet, and these packs are modest (tens to a couple
   hundred MB), not the multi-GB the installer itself deals with, so they
   live on this server's own volume for now. Revisit if that stops being
   true.

Storage is one SQLite file plus, now, a directory of zip files. At this
data volume (a ping every so often from every install, not every request
on every page load) SQLite is not a simplification made to cut corners -
it is sized correctly for the job.
"""

from __future__ import annotations

import ipaddress
import json
import os
import sqlite3
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("TELEMETRY_DB", ROOT / "telemetry.db"))
MANIFEST_PATH = Path(os.environ.get("TELEMETRY_MANIFEST", ROOT / "manifest.json"))
PACKS_DIR = Path(os.environ.get("TELEMETRY_PACKS_DIR", ROOT / "packs"))

app = FastAPI(title="Neo Language Box telemetry", docs_url=None, redoc_url=None)


# --------------------------------------------------------------------- storage

def _init_db() -> None:
    with _db() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS pings (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                install_id      TEXT NOT NULL,
                type            TEXT NOT NULL,          -- install | update | session_end
                app_version     TEXT NOT NULL,
                previous_version TEXT,                  -- set for type=update
                country         TEXT,                   -- server-inferred, 2-letter or NULL
                peak_listeners  INTEGER,                 -- set for type=session_end
                languages       TEXT,                    -- JSON array, type=session_end
                session_seconds REAL,                    -- set for type=session_end
                received_at     TEXT NOT NULL
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_pings_install ON pings(install_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_pings_type ON pings(type)")


@contextmanager
def _db():
    con = sqlite3.connect(DB_PATH)
    try:
        yield con
        con.commit()
    finally:
        con.close()


_init_db()


# ------------------------------------------------------------- country lookup

# Best-effort, no API key, no bundled GeoIP database to keep the server a
# single file. ip-api.com's free tier is good enough for "roughly which
# countries are our installs in" - it is not held to account-level accuracy.
# Swap this one function out if that stops being good enough.
def _country_for_ip(ip: str) -> Optional[str]:
    try:
        addr = ipaddress.ip_address(ip)
        if addr.is_private or addr.is_loopback:
            return None
    except ValueError:
        return None
    try:
        req = urllib.request.Request(
            f"http://ip-api.com/json/{ip}?fields=countryCode",
            headers={"User-Agent": "neo-language-box-telemetry"},
        )
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("countryCode") or None
    except Exception:
        return None


def _client_ip(request: Request) -> str:
    # A forwarding proxy (most PaaS hosts put one in front) sets this; fall
    # back to the raw socket peer for a direct deployment.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


# ------------------------------------------------------------------- /v1/ping

class SessionStats(BaseModel):
    peak_listeners: int = Field(ge=0)
    languages: list[str] = Field(default_factory=list)
    session_seconds: float = Field(ge=0)


class Ping(BaseModel):
    install_id: str = Field(min_length=8, max_length=64)
    type: str                      # "install" | "update" | "session_end"
    app_version: str
    previous_version: Optional[str] = None
    session: Optional[SessionStats] = None


@app.post("/v1/ping")
async def ping(body: Ping, request: Request) -> dict:
    if body.type not in ("install", "update", "session_end"):
        return JSONResponse({"ok": False, "error": "unknown ping type"}, status_code=400)
    if body.type == "update" and not body.previous_version:
        return JSONResponse({"ok": False, "error": "update ping needs previous_version"}, status_code=400)
    if body.type == "session_end" and body.session is None:
        return JSONResponse({"ok": False, "error": "session_end ping needs session"}, status_code=400)

    country = _country_for_ip(_client_ip(request))
    session = body.session

    with _db() as con:
        con.execute(
            "INSERT INTO pings (install_id, type, app_version, previous_version, "
            "country, peak_listeners, languages, session_seconds, received_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                body.install_id, body.type, body.app_version, body.previous_version,
                country,
                session.peak_listeners if session else None,
                json.dumps(session.languages) if session else None,
                session.session_seconds if session else None,
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ),
        )
    # Always 200: a listener-less reply still releases the client. The one
    # thing worse than losing a ping is the app blocking a shutdown on it.
    return {"ok": True}


# --------------------------------------------------------------- /v1/manifest

@app.get("/v1/manifest")
async def manifest() -> JSONResponse:
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {"app_version": "0.0.0", "language_packs": []}
    return JSONResponse(data)


# --------------------------------------------------------------------- /v1/packs

@app.get("/v1/packs/{filename}")
async def pack(filename: str) -> FileResponse:
    # filename comes straight off the URL path - reject anything that
    # isn't a bare filename before it ever touches the filesystem, so a
    # path like "../../etc/passwd" can't escape PACKS_DIR.
    if "/" in filename or "\\" in filename or filename in (".", ".."):
        raise HTTPException(404)
    path = PACKS_DIR / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="application/zip", filename=filename)


# ------------------------------------------------------------------- /v1/stats

@app.get("/v1/stats", response_class=HTMLResponse)
async def stats() -> str:
    with _db() as con:
        total_installs = con.execute(
            "SELECT COUNT(DISTINCT install_id) FROM pings").fetchone()[0]

        by_country = con.execute(
            "SELECT COALESCE(country, 'Unknown'), COUNT(DISTINCT install_id) "
            "FROM pings GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()

        last_seen = con.execute(
            "SELECT install_id, MAX(received_at) FROM pings "
            "GROUP BY install_id ORDER BY 2 DESC LIMIT 25"
        ).fetchall()

        sessions = con.execute(
            "SELECT COUNT(*), AVG(peak_listeners), MAX(peak_listeners), "
            "SUM(session_seconds) FROM pings WHERE type = 'session_end'"
        ).fetchone()
        session_count, avg_listeners, max_listeners, total_seconds = sessions

        lang_rows = con.execute(
            "SELECT languages FROM pings WHERE type = 'session_end' AND languages IS NOT NULL"
        ).fetchall()
        lang_counts: dict[str, int] = {}
        for (raw,) in lang_rows:
            for code in json.loads(raw):
                lang_counts[code] = lang_counts.get(code, 0) + 1
        top_langs = sorted(lang_counts.items(), key=lambda kv: -kv[1])[:15]

        version_rows = con.execute(
            "SELECT app_version, COUNT(DISTINCT install_id) FROM pings "
            "GROUP BY app_version ORDER BY 2 DESC"
        ).fetchall()

    def esc(s) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    country_rows = "".join(
        f"<tr><td>{esc(c)}</td><td>{n}</td></tr>" for c, n in by_country
    ) or "<tr><td colspan=2>No data yet</td></tr>"

    last_seen_rows = "".join(
        f"<tr><td class='mono'>{esc(iid[:8])}…</td><td>{esc(ts)}</td></tr>"
        for iid, ts in last_seen
    ) or "<tr><td colspan=2>No data yet</td></tr>"

    lang_rows_html = "".join(
        f"<tr><td class='mono'>{esc(code)}</td><td>{n}</td></tr>" for code, n in top_langs
    ) or "<tr><td colspan=2>No data yet</td></tr>"

    version_rows_html = "".join(
        f"<tr><td class='mono'>{esc(v)}</td><td>{n}</td></tr>" for v, n in version_rows
    ) or "<tr><td colspan=2>No data yet</td></tr>"

    hours = round((total_seconds or 0) / 3600, 1)

    return f"""<!doctype html>
<meta charset="utf-8">
<title>Neo Language Box - usage stats</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", sans-serif; background:#0b1220; color:#eef2f8;
         max-width: 880px; margin: 40px auto; padding: 0 20px; }}
  h1 {{ font-size: 20px; }}
  h2 {{ font-size: 14px; text-transform: uppercase; letter-spacing: .06em; color: #4ea1ff;
        margin: 32px 0 10px; }}
  .tiles {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .tile {{ background: #121b2e; border: 1px solid #223049; border-radius: 10px;
           padding: 14px 18px; min-width: 140px; }}
  .tile .n {{ font-size: 24px; font-weight: 700; }}
  .tile .l {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #223049; }}
  .mono {{ font-family: ui-monospace, monospace; }}
  .cols {{ display: flex; gap: 30px; flex-wrap: wrap; }}
  .cols > div {{ flex: 1; min-width: 260px; }}
</style>
<h1>Neo Language Box - usage stats</h1>
<div class="tiles">
  <div class="tile"><div class="n">{total_installs}</div><div class="l">Installs</div></div>
  <div class="tile"><div class="n">{session_count or 0}</div><div class="l">Sessions reported</div></div>
  <div class="tile"><div class="n">{round(avg_listeners or 0, 1)}</div><div class="l">Avg peak listeners</div></div>
  <div class="tile"><div class="n">{max_listeners or 0}</div><div class="l">Largest session</div></div>
  <div class="tile"><div class="n">{hours}</div><div class="l">Total hours run</div></div>
</div>

<div class="cols">
  <div>
    <h2>Installs by country</h2>
    <table>{country_rows}</table>
  </div>
  <div>
    <h2>App version adoption</h2>
    <table>{version_rows_html}</table>
  </div>
</div>

<div class="cols">
  <div>
    <h2>Most-used languages</h2>
    <table>{lang_rows_html}</table>
  </div>
  <div>
    <h2>Most recently active installs</h2>
    <table>{last_seen_rows}</table>
  </div>
</div>
"""
