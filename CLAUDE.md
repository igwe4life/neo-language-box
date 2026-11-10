# Neo Language Box

Offline, on-device live-sermon translation. A speaker's English is
transcribed, translated into many languages at once, and streamed as text +
synthesized speech to listeners' phones over the venue's own Wi-Fi. No
cloud, no accounts, no internet dependency for the translation itself - that
offline-first positioning is the product's actual pitch, not an incidental
detail, and should weigh in architecture decisions (see telemetry below).

## Architecture

```
mic/system audio -> STT (Whisper or Parakeet) -> MT (OPUS-MT / NLLB-200, fanned
out per language) -> TTS (Piper or VITS, per language) -> WebSocket -> phones
```

- `sermon/pipeline.py` - STT -> per-language MT fan-out. Translators sharing
  NLLB-200 batch into one call (`translate_many`); everything else runs
  concurrently via a thread pool (CTranslate2/onnxruntime release the GIL,
  so this is real parallelism, not just interleaving).
- `sermon/sinks.py` - `BroadcastSink` fans synthesis out the same way, but
  on a **separate background worker thread**, not the pipeline thread. This
  was a real, fixed bug: synthesis used to block `emit()`, which blocked the
  *next* utterance's transcription - a slow or contended voice stalled the
  whole room, not just its own audio. If synthesis ever looks like it's
  blocking something again, check here first.
- `sermon/server.py` - one asyncio `websockets` server serves both the
  listener HTML (via `process_request`) and the `/ws` socket, so there's
  only one port and one firewall prompt. `Broadcaster` holds the in-memory
  listener state (who's connected, what language, recent history for
  late joiners).
- `sermon/templates/operator.html` / `listen.html` - the two real UIs. No
  frontend framework; vanilla JS reading the WebSocket + a few `/api/*`
  endpoints documented inline in `server.py`.
- `desktop_app.py` - the packaged desktop shell: opens a WebView2 window,
  runs `run.py --serve` in a background thread, waits for it to answer
  `/api/status`, then swaps the window to `/operator`.

## Languages

`sermon/config.py` is the single source of truth: `LANGUAGES` (metadata),
`DEFAULT_TARGETS` (what a fresh install actually bundles),
`ALL_VOICES`/`VITS_VOICES` (which TTS family), `NLLB_TARGETS` (which
languages route through the shared NLLB-200 model instead of their own
bilingual OPUS-MT model).

Two totally different risk profiles when adding a language:
- **Piper + OPUS-MT (or NLLB)** - the safe, proven path. fr/es/pt/de/it/ru/
  ar/hi/zh/sw/id/tr/pl/bn/ur all work this way. Verify the voice and MT
  model actually exist (check the live `rhasspy/piper-voices` tree and
  OPUS-MT/Tatoeba-MT listings - `scripts/fetch_models.py`'s comments show
  how) before adding config entries.
- **VITS/MMS one-off models** - Igbo, Yoruba, Hausa. Every real voice-quality
  complaint this project has had came from this bucket. Don't add another
  language here without good reason; prefer the safe path.

A language can ship two ways:
- **Baked into the installer** - listed in `DEFAULT_TARGETS`.
- **Pack-only** - fully configured in `LANGUAGES`/`ALL_VOICES`/etc, but *not*
  in `DEFAULT_TARGETS`. Downloaded after install via `sermon/langpacks.py`,
  which activates it on next restart with zero code changes
  (`installed_language_packs()` in config.py folds it into the target list).
  Korean and Vietnamese are the first real example of this - see
  `scripts/build_language_pack.py` to build a new one.

`python run.py doctor`'s synthesis test only covers Piper voices - it does
NOT exercise VITS/MMS synthesis (ig/yo/ha). Don't trust a clean `doctor` run
alone as proof a VITS-family voice works; test it directly (see git history
around the Igbo restoration for the pattern: load `VitsSpeaker` and
`BroadcastSink` directly, recreate concurrent synthesis).

## Release process

```
python -m PyInstaller --clean --noconfirm desktop.spec   # app -> dist/NeoLanguageBox/
python scripts/build_models_ship.py --clean              # models -> models-ship/ (hardlinked, instant)
tools\innosetup\ISCC.exe installer.iss                    # -> dist/NeoLanguageBoxSetup.exe (+ .bin slices)
```

The installer is **disk-spanned** (`DiskSpanning=yes` in `installer.iss`) -
20 languages already pushed the compressed payload past Inno Setup's 4.2 GB
single-file limit (measured: these already-quantised model files barely
compress, ~0.91 ratio). Expect more `.bin` slices as more languages get
baked in; this is why pack-only releases (above) matter - every new
language doesn't have to grow the base installer forever.

`AppId` in `installer.iss` is intentionally never changed, across two
renames so far (Sermon Translation Speaker -> Neo Sermon -> Neo Language
Box) - it's what makes a new build an *upgrade* of an existing install
rather than a second copy. `sermon/config.py`'s `APP_VERSION` and
`installer.iss`'s `AppVersion` are kept in sync by hand (Inno can't read
the Python file at build time).

## Telemetry & language-pack server

`telemetry-server/` - a small FastAPI + SQLite service, deployed on Fly.io
(`neo-language-box-telemetry`, account: frontlineconsults@gmail.com).
Three jobs: `POST /v1/ping` (install/update/session_end check-ins from the
app, all opt-in via `sermon/telemetry.py`, no audio/text/names), `GET
/v1/manifest` (app version + language-pack list - edit `manifest.json` to
add a pack), `GET /v1/stats` (plain HTML dashboard, no login).

Pack zip files live on the same Fly volume for now (`/data/packs/`) -
originally meant to be a separate static host (see the server's own
README), but nothing else was set up and these packs are modest (tens to
~150 MB). `flyctl ssh sftp put` is unreliable for anything over ~60 MB on
this connection - it drops mid-transfer. Workaround that's worked twice:
`split -b 20m` locally, upload each chunk, `cat` them back together on the
server, verify the sha256 matches before cleaning up.

Fly machine auto-suspends when idle (`min_machines_running = 0`, cost
tradeoff) - a request after a quiet period needs a wake-up call first
(any `curl` to it) before `flyctl ssh console`/`sftp` will connect.

## Branching

`master` - stable/release. `develop` - active work, pushed as it goes; PR
into `master` opened when there's something real to merge, not
automatically after every commit (explicit instruction - don't open one
preemptively).

## Known landmines

- This dev machine runs low on RAM under load (Chrome + Claude Code +
  models can leave <2 GB free) - `doctor`'s RAM warnings and occasional
  `mkl_malloc` crashes here are this machine's limitation, not necessarily
  a real bug. Don't over-index on them without checking available memory
  first.
- A background build/compile task can get silently killed by a session
  boundary (not a real failure) - if a long `ISCC.exe`/`PyInstaller` run
  shows no error and no completion line, it was probably interrupted, not
  broken. Just rerun it.
- `gh repo create` (and other repo-creation-shaped commands) is blocked by
  the auto-mode classifier by default - see `.claude/settings.local.json`
  (gitignored, per-machine) for the narrow allow rule already added here.
  It will need re-adding on a fresh machine/session if blocked again.

## Reference docs

- Install/setup guide for operators: `docs/setup-guide.html` /
  `docs/setup-guide.pdf`
- Platform brief (non-technical, for management): `docs/platform-one-pager.html`
  / `.pdf`
- Telemetry dashboard: https://neo-language-box-telemetry.fly.dev/v1/stats
