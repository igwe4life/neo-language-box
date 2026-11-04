# Neo Language Box - telemetry & manifest server

One small FastAPI app, one SQLite file. Three endpoints:

- `POST /v1/ping` - the desktop app's check-ins (install, update, end of a
  session). See `app.py`'s docstring for exactly what each one contains -
  short version: an anonymous install id, the app version, and for a
  session only a peak listener count and which language codes were active.
  No audio, no text, no names.
- `GET /v1/manifest` - what the app polls to learn the current version and
  which language packs exist. This is **how you add a language**: edit
  `manifest.json` and redeploy (or mount it from a volume you can edit
  in place - see below).
- `GET /v1/stats` - a plain HTML dashboard: install count, countries,
  version adoption, most-used languages, last-seen per install. No login.
  Put it behind your host's access control, or a reverse-proxy basic-auth
  rule, if you don't want the URL alone to be enough to view it.

## Running it

Locally, no Docker:

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

With Docker, with the database persisted outside the container:

```bash
docker build -t neo-language-box-telemetry .
docker run -d -p 8080:8080 -v neo-telemetry-data:/data neo-language-box-telemetry
```

## Editing the manifest without rebuilding the image

Two options:

1. **Rebuild and redeploy** whenever you add a language - simplest, fine at
   this scale. Edit `manifest.json`, push, redeploy.
2. **Mount it from outside the container** so you can edit it in place:
   add `-v $(pwd)/manifest.json:/app/manifest.json:ro` to the `docker run`
   command (or the equivalent volume mount on whatever platform you use),
   then edit the file on the host - no rebuild needed.

### `manifest.json` shape

```json
{
  "app_version": "0.9.0",
  "app_download_url": "https://example.com/NeoLanguageBoxSetup.exe",
  "language_packs": [
    {
      "code": "ko",
      "name": "Korean",
      "version": "1",
      "url": "https://example.com/packs/ko-v1.zip",
      "sha256": "...",
      "size_bytes": 184000000
    }
  ]
}
```

`app_download_url` is what the client's "a new version is available" notice
links to - point it at wherever `NeoLanguageBoxSetup.exe` is hosted (a GitHub
Release, an S3/R2 bucket, anywhere with a stable URL). This server does not
host the installer or the language-pack zips itself; it only hosts the
*manifest that says where they are*. Keeping big binary files off this
service is deliberate - it's meant to stay a single small container.

## Where to host this

Any of these work with zero code changes, since it's one process reading
one env var for the DB path:

- A small VPS (Hetzner/DigitalOcean/Lightsail, ~$5/mo) - `docker run` as
  above, point a domain at it.
- Render, Railway, or Fly.io - all support "here's a Dockerfile" deploys
  with a free or near-free tier; attach a persistent volume for the SQLite
  file or the DB resets on every redeploy.
- Anything that can run a container and give you one persistent volume.

Whatever you pick, note the public URL - the desktop app needs it to know
where to send pings and check the manifest (see `sermon/telemetry.py`).
