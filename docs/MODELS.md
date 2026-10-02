# Models

Neo Sermon uses **no LLMs**. Nothing like GPT or Claude is involved, and nothing
leaves the machine. Every stage is a small single-purpose model running locally
on CPU.

27 models ship in the installer: 4 for recognition (3 engines plus the voice
activity detector), 10 for translation, and 13 voices. 14 languages are offered;
English, the source language, needs neither a translator nor a voice.

## 1. Speech recognition — English speech to English text

| Model | On disk | Role |
|---|---|---|
| **NVIDIA Parakeet TDT 0.6B** | `parakeet-0.6b/` | **Default.** ~1.3 s per sentence on the dev laptop. |
| NVIDIA Parakeet TDT 110M | `parakeet-110m/` | Fallback: ~4x faster, drops proper nouns more often. |
| OpenAI Whisper `base.en` | `whisper/base.en/` | Fallback: ~2 s slower per sentence, but the **only** engine that uses `vocabulary.txt` word hints. |
| **Silero VAD** | `silero_vad.onnx` | Decides when someone is actually speaking, so silence and room noise are not transcribed as invented sentences. |

Switch engine without rebuilding: `SERMON_STT=whisper-base.en` (also
`parakeet-110m`, `parakeet-0.6b`).

Parakeet cannot take a vocabulary prompt. That capability was traded for speed
when it replaced Whisper as the default; the phrase-override tables below still
apply on the way out.

## 2. Translation — English text to 13 languages

Two engines, because the one that is better for European and Asian languages is
not better for African ones.

| Engine | On disk | Languages |
|---|---|---|
| **OPUS-MT / Tatoeba-MT** (Helsinki-NLP, Marian), one model per pair | `mt/opus-mt-en-<xx>/` | French, Spanish, Portuguese, German, Italian, Russian, Arabic, Hindi, Chinese |
| **Meta NLLB-200 distilled 1.3B**, one model for all four | `nllb-1.3b/` | Igbo, Yoruba, Hausa, Swahili |

OPUS-MT publishes bilingual models for the African languages too, but a native
Igbo speaker judged them clearly worse than NLLB in a blind comparison
(`scripts/compare_mt.py` rebuilds it). NLLB is roughly 13x slower per sentence,
so it is used only where it won. All four of its targets are decoded in one
batched pass, which is what makes carrying a 1.3B model affordable.

English is the source language: there is no translator behind it, but listeners
can select it to read the live transcript.

Narrow the set with `SERMON_TARGETS=fr,es,ig`.

## 3. Voices — translated text to spoken audio

No single source covers these languages, so three are used.

| Engine | On disk | Languages |
|---|---|---|
| **Piper** (VITS, ONNX) | `tts/<xx_XX-voice-medium>/` | French, Spanish, Portuguese, German, Italian, Russian, Arabic, Hindi, Chinese, Swahili |
| **Meta MMS-TTS** (VITS) | `tts/yo/`, `tts/ha/` | Yoruba, Hausa |
| **VITS-OpenBible-Igbo** (Coqui VITS, converted to ONNX) | `tts/ig/` | Igbo |

Igbo needed its own source: Meta's MMS-TTS has no Igbo at all, and every
community MMS-style Igbo model shares a 43-character vocabulary with no
`ị` or `ụ` — two of the commonest vowels — so "ụmụnna" comes out as "mnna".
The OpenBible checkpoint has the full orthography. It ships only as a Coqui
`.pth`, so `scripts/export_vits_onnx.py` converts it once at build time in the
separate `.venv-export`; the shipped app carries onnxruntime and no torch.

There is deliberately **no English voice** — the room already hears the speaker,
so English listeners get the transcript as text. If a server voice fails, the
listener's phone falls back to its own built-in voice for that language.

Restrict with `SERMON_VOICES=fr,es`.

## Not models

* `sermon/enlex.py` — a common-English word list that rejects transcriptions
  which are not plausible English, e.g. the speakers' own output picked up again.
* `phrases.<xx>.txt` — fixed translations for set phrases, applied after
  translation. Yoruba ships 30.
* `vocabulary.txt` — proper-noun hints, used only by the Whisper engines.

## Engines that run them

CTranslate2 (Whisper, OPUS-MT, NLLB), onnxruntime (Piper, MMS, the Igbo voice),
sherpa-onnx (Parakeet, Silero VAD). All int8 on CPU.

## In the repo but NOT shipped

| Path | Why kept |
|---|---|
| `models/nllb-600m/` | Lost the blind comparison to 1.3B. Its tokenizer was copied into `nllb-1.3b/`, which ships without one. |
| `models/mt/opus-mt-en-{ig,yo,sw}/` | The comparison's losing side. |
| `models/mms/ibo-openbible/` | The Coqui `.pth` source for the Igbo voice — needed only to re-run the ONNX export. |
| `models/whisper/small.en/` (in `models_unused/`) | 9.3 s to transcribe a 4.7 s clip. Never the right choice. |

`scripts/build_models_ship.py` assembles the shipped subset from
`sermon/config.py`, so this list cannot drift from what the app loads.

## Licences

The installer redistributes third-party models, each under its own licence.
**NLLB-200 and the Meta MMS voices are CC-BY-NC — non-commercial use only.**
`models-ship/README.txt` ships alongside the app with the full list and
sources. Check the source pages before any commercial use; that file is a
pointer, not legal advice.
