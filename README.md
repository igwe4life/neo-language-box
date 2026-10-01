# Neo Sermon

Live interpretation into 24 languages, running entirely on one laptop.

* The operator opens the desktop app, picks an audio source (the laptop's own
  audio, a microphone, or line-in from a mixer) and presses Start.
* Listeners scan the QR code, pick a language, and hear it live on their phones.

Everything runs locally: no cloud, no accounts, no API keys, and **no LLMs**.
Once installed the machine can stay offline.

## Stack

| Stage | Component | Why |
|---|---|---|
| Capture | PyAudioWPatch (WASAPI loopback) or `sounddevice`, resampled to 16 kHz mono | captures system audio, a mic, or line-in; switchable while running |
| Segmentation | energy VAD in numpy + Silero VAD | ~0 % CPU for the first pass; Silero stops silence becoming invented sentences |
| STT | NVIDIA Parakeet TDT 0.6B, int8 (sherpa-onnx) | ~8x faster than Whisper `base.en` on the same clip |
| MT | OPUS-MT (14 languages) + NLLB-200 1.3B (10 languages), CTranslate2 int8 | one engine each where it wins; NLLB decodes its ten in one batch |
| TTS | Piper (18 languages), Meta MMS (5), VITS-OpenBible-Igbo (1), all ONNX | no source covers all of them; one onnxruntime for all four |
| Streaming | `websockets` server + one HTML page | pure Python, no CDN |
| Output | console + speech + broadcast sinks | one per phase, `pipeline.py` unchanged |

Full inventory, fallbacks, licences and the reasoning behind each choice:
**[docs/MODELS.md](docs/MODELS.md)**.

## Install

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

## Use

```powershell
.\.venv\Scripts\Activate.ps1
python run.py doctor        # check everything
python run.py               # live translation, Ctrl+C to stop
```

## Commands

| Command | Purpose |
|---|---|
| `python run.py` | live microphone → EN + FR + YO in terminal |
| `python run.py --no-yoruba` | French only (~250 ms less latency per utterance) |
| `python run.py --speak` | as above, and say the French out loud |
| `python run.py --serve` | as above, and stream to phones on this Wi-Fi (port 8000) |
| `python run.py doctor` | environment, packages, models, mic, RAM, CPU speed |
| `python run.py devices` | list audio input devices |
| `python run.py mic` | live level meter, no models loaded |
| `python run.py translate "text"` | translate to every configured target |
| `python run.py translate --to yo "text"` | one target only |
| `python run.py speak "text"` | translate and say it, no microphone |

Useful flags: `--model tiny.en`, `--device 2`, `--min-utterance 5`,
`--max-utterance 14`, `--silence 900`, `--threshold 9`, `--beam 5`,
`--no-prompt`, `--targets fr,yo`, `--quiet`.

## Accuracy: edit `vocabulary.txt` before each sermon

The biggest accuracy lever on CPU-only hardware is telling the recogniser which
proper nouns to expect. Add this week's names — people, places, books — to
[vocabulary.txt](vocabulary.txt), one per line. Without it, Whisper substitutes
acoustically similar everyday words ("Samson" becomes "Psalms", "Philistines"
becomes "peasants").

The prompt is capped at 224 tokens and the excess is silently discarded;
`python run.py doctor` reports the current count.

Longer utterances are also markedly more accurate, because Whisper uses
surrounding context. `--min-utterance` trades latency for accuracy: raise it
while `Dropped (CPU backlog)` stays at 0.

## Layout

```
run.py                  CLI
vocabulary.txt          per-sermon proper nouns fed to the recogniser
sermon/config.py        all tunables
sermon/audio.py         mic capture + utterance segmentation
sermon/stt.py           Transcriber interface + faster-whisper
sermon/mt.py            Translator interface + OPUS-MT/CTranslate2
sermon/tts.py           Speaker interface + Piper French voice
sermon/server.py        Wi-Fi broadcaster (asyncio + websockets)
sermon/templates/listen.html   the page phones open
sermon/sinks.py         output destinations (console, speech, broadcast)
sermon/pipeline.py      source -> stt -> mt -> sinks
sermon/diagnostics.py   doctor
scripts/fetch_models.py one-time download + int8 conversion
models/                 downloaded models (offline after setup)
```

Extension points for later phases: add a `Sink` for the websocket broadcaster;
replace `MicrophoneSource` with a network source. `pipeline.py` doesn't need to
change to add a third language — it already runs one translator per configured
target in parallel.

## Streaming to phones

`python run.py --serve` starts a local web server on port 8000 alongside the
usual pipeline. Listeners on the same Wi-Fi open the URL printed at startup in
their phone browser — no app to install, no CDN, no account. Each phone picks
FR or YO from a toggle at the top; both are pushed as they arrive.

Nothing leaves the room. The server binds to the LAN, not the internet.

### Windows firewall

On the first `--serve` run Windows shows a firewall prompt for the port —
click **Allow**. If phones can't reach the URL:

* You clicked Block: `wf.msc` → Inbound Rules → find `python.exe`, Enable.
* Phones are on a different network. Point them at the same Wi-Fi as the
  laptop, or turn the laptop's mobile hotspot on and have phones join it.
* Corporate Wi-Fi with client isolation blocks it — use a personal hotspot.

### Capacity

Broadcast messages are a few hundred bytes each. The laptop can serve dozens
of listeners without touching the recognition pipeline's CPU budget. What
would eventually cap this is Windows' per-process socket limit, not model
inference.

### What the phone page can do

Half a screen of text, big-text-friendly. Language toggle. Optional "speak the
French with the phone's own TTS voice" (uses `SpeechSynthesis`; works on
Android and iOS if a French voice is installed on the device). Optional
"show English underneath" for bilingual listeners. Screen wake-lock so the
display doesn't sleep during a sermon. Auto-reconnect if the phone loses Wi-Fi.

## Speech output and the feedback problem

Playing French from the laptop speakers puts it straight back into the
microphone, where it would be transcribed as if the preacher had said it. The
microphone is therefore **muted during playback** by default. Two consequences:

* Speech you make while French is playing is not captured. Half-duplex is
  unavoidable with one speaker and one mic in the same room.
* French playback takes roughly as long as the original English, so a preacher
  who speaks continuously will outrun it. Excess phrases are dropped oldest-first
  and counted in the exit summary.

With headphones or a separate output device, pass `--no-mute` to keep listening
during playback. Streaming to phones (a later phase) removes the problem
entirely, since nothing is played into the room.
