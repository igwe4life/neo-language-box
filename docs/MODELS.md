# Models

Neo Sermon uses **no LLMs**. Nothing like GPT or Claude is involved, and nothing
leaves the machine. Every stage is a small single-purpose model running locally
on CPU.

42 models ship in the installer: 4 for recognition (3 engines plus the voice
activity detector), 15 for translation (NLLB-200 now covers ten of the
languages off one shared model), and 23 voices. 25 languages are offered;
English, the source language, needs neither a translator nor a voice, and
Japanese has no voice at all (see the TTS section below) - every other
language gets one.

**RAM note for the full 24-target configuration.** Every target the operator
enables loads its own translator and, if available, its own voice into RAM at
once - narrowing with `SERMON_TARGETS=fr,es,ig` (see MTConfig in
`sermon/config.py`) controls this. All 24 together need roughly 5 GB: the
shared NLLB-200 model is ~1.4 GB, the 14 OPUS-MT models are ~100-250 MB each,
Parakeet is a few hundred MB, and the 17 Piper + 6 MMS/Coqui voices are
~60-130 MB each. On an 8 GB machine that is close enough to the ceiling that
other running programs (a browser, an editor) can push a late model over it;
tested on an 8 GB/4-thread laptop already busy with other software, several
OPUS-MT and Piper loads failed with `mkl_malloc: failed to allocate memory` /
`bad allocation` under that combined load. Every load is wrapped in a
try/except (see `run.py`), so this degrades a language to text-only rather
than crashing the server - but it means the realistic recommendation for an
8 GB machine is the same one `SERMON_TARGETS` was already built for: pick the
handful of languages an actual service needs rather than enabling all 24, or
run on a machine with more headroom (16 GB is comfortable).

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

## 2. Translation — English text to 24 languages

Two engines, because the one that is better for most European and Asian
languages is not better for low-resource ones - or, in a few cases, there is
simply no usable direct OPUS-MT release to begin with.

| Engine | On disk | Languages |
|---|---|---|
| **OPUS-MT / Tatoeba-MT** (Helsinki-NLP, Marian), one model per pair | `mt/opus-mt-en-<xx>/` | French, Spanish, Portuguese, German, Italian, Russian, Arabic, Hindi, Chinese, Japanese, Vietnamese, Tagalog, Indonesian, Turkish |
| **Meta NLLB-200 distilled 1.3B**, one model for all ten | `nllb-1.3b/` | Igbo, Yoruba, Hausa, Swahili, Polish, Amharic, Somali, Bengali, Korean, Urdu |

OPUS-MT publishes bilingual models for Igbo, Yoruba, Hausa and Swahili too, but
a native Igbo speaker judged them clearly worse than NLLB in a blind comparison
(`scripts/compare_mt.py` rebuilds it). Polish, Amharic, Somali and Bengali have
no official direct (or Tatoeba "tc-big") OPUS-MT release at all - only
low-BLEU multi-target group models. Korean and Urdu do have a direct release,
but their own model cards report modest scores (BLEU 13.7 and 12.1
respectively), so they default to NLLB too pending a blind comparison the way
the African languages got one. NLLB is roughly 13x slower per sentence, so it
is used only where OPUS-MT is not clearly better. All ten of its targets are
decoded in one batched pass, which is what makes carrying a 1.3B model
affordable.

Turkish uses the larger "tc-big" transformer release (~830 MB rather than the
usual ~260 MB) deliberately, for the BLEU gain - translation accuracy was
prioritised over install size for this wave.

English is the source language: there is no translator behind it, but listeners
can select it to read the live transcript.

Narrow the set with `SERMON_TARGETS=fr,es,ig`.

## 3. Voices — translated text to spoken audio

No single source covers these languages, so four are used.

| Engine | On disk | Languages |
|---|---|---|
| **Piper** (VITS, ONNX) | `tts/<xx_XX-voice-medium>/` | French, Spanish, Portuguese, German, Italian, Russian, Arabic, Hindi, Chinese, Swahili, Korean, Vietnamese, Indonesian, Turkish, Polish, Bengali, Urdu |
| **Meta MMS-TTS**, pre-converted ONNX (VITS) | `tts/yo/`, `tts/ha/`, `tts/tl/`, `tts/so/` | Yoruba, Hausa, Tagalog, Somali |
| **Meta MMS-TTS**, converted here (VITS) | `tts/am/` | Amharic |
| **VITS-OpenBible-Igbo** (Coqui VITS, converted to ONNX) | `tts/ig/` | Igbo |

Igbo needed its own source: Meta's MMS-TTS has no Igbo at all, and every
community MMS-style Igbo model shares a 43-character vocabulary with no
`ị` or `ụ` — two of the commonest vowels — so "ụmụnna" comes out as "mnna".
The OpenBible checkpoint has the full orthography. It ships only as a Coqui
`.pth`, so `scripts/export_vits_onnx.py` converts it once at build time in the
separate `.venv-export`; the shipped app carries onnxruntime and no torch.

Amharic needed its own conversion too, for a different reason: the willwade
pre-converted MMS repository used for Yoruba/Hausa/Tagalog/Somali has no
Amharic folder, so `scripts/export_mms_amharic.py` converts
`facebook/mms-tts-amh` directly (same separate `.venv-export`). That
checkpoint's vocabulary is also not what you'd expect: 28 plain Latin letters,
no Ge'ez script at all, because Meta trained it on *romanised* Amharic. NLLB's
Amharic output is in native Ge'ez script, so `VitsSpeaker` romanises it with
the `uroman` package first - see the `romanize.flag` file the export script
writes, which is what tells it to. This voice has not been checked by a native
Amharic speaker the way Igbo's was; it is verified only to produce
non-silent, duration-proportional audio rather than silence or noise.

There is also no Japanese voice, but not by design: Piper's only Japanese
voice (`ja_JP-hi_fi_captain-medium`) needs the `pyopenjtalk` package for
phonemisation, and `pyopenjtalk` ships no prebuilt wheel for any platform -
installing it means compiling a C++ library with cmake on every machine that
runs this, which does not fit an unsigned, no-admin-elevation installer.
Japanese listeners get the transcript as text, like English, until that
changes.

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
That now covers translation for ten languages (Igbo, Yoruba, Hausa, Swahili,
Polish, Amharic, Somali, Bengali, Korean, Urdu) and voices for five (Yoruba,
Hausa, Tagalog, Somali, Amharic) - a larger share of the app than before this
wave, since several of the new languages had no non-NC OPUS-MT/Piper option at
all. `models-ship/README.txt` ships alongside the app with the full list and
sources. Check the source pages before any commercial use; that file is a
pointer, not legal advice.
