#!/usr/bin/env python
"""One-time model download + conversion. Needs internet ONCE; after this the
application runs fully offline.

  python scripts/fetch_models.py                 base.en + en-fr  (default)
  python scripts/fetch_models.py --model tiny.en
  python scripts/fetch_models.py --model tiny.en --model base.en
  python scripts/fetch_models.py --keep-temp     keep the raw OPUS-MT download

What it does:
  1. Downloads the CTranslate2 Whisper weights from Hugging Face into
     models/whisper/<size>/  (self-contained, not the shared HF cache, so the
     project folder can be copied to another offline machine).
  2. Downloads the official OPUS-MT en-fr Marian model from the University of
     Helsinki, converts it to CTranslate2 int8, and keeps only the ~78 MB result.

No PyTorch and no transformers are involved anywhere in this process.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

WHISPER_REPOS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base.en": "Systran/faster-whisper-base.en",
    "small.en": "Systran/faster-whisper-small.en",
}

OPUS_MT_URLS = {
    "fr": "https://object.pouta.csc.fi/OPUS-MT-models/en-fr/opus-2020-02-26.zip",
    # NOTE: the older `OPUS-MT-models/en-es/opus-2019-12-04.zip` uses BPE
    # tokenisation, not SentencePiece, so it won't run through our
    # sentencepiece-based tokeniser. The `Tatoeba-MT-models/eng-spa` release
    # uses spm32k. Same footprint (~290 MB), better BLEU (55.1 vs 50.1).
    "es": "https://object.pouta.csc.fi/Tatoeba-MT-models/eng-spa/opus+bt-2021-04-10.zip",
    # Low-resource: only ~4k genuine parallel sentences plus back-translation.
    # Validated on JW300 (religious text), which is at least the right domain.
    "yo": "https://object.pouta.csc.fi/OPUS-MT-models/en-yo/opus+bt-2020-05-23.zip",

    # --- second language wave -------------------------------------------
    # All Tatoeba-MT releases, chosen because every one was verified to ship
    # source.spm/target.spm (SentencePiece) rather than BPE - checked by
    # reading each zip's central directory over an HTTP Range request before
    # downloading, which is the trap the en-es note above records.
    #
    # Each pair also has a `opusTCv20210807+bt_transformer-big_*` release at
    # ~870 MB. They score better but convert to ~300 MB of int8 each, which
    # would put the installer far past its 2 GB budget for a quality gain that
    # does not survive being read off a phone in a noisy hall.
    "ar": "https://object.pouta.csc.fi/Tatoeba-MT-models/eng-ara/opus-2021-02-23.zip",
    "pt": "https://object.pouta.csc.fi/Tatoeba-MT-models/eng-por/opus+bt-2021-04-14.zip",
    "zh": "https://object.pouta.csc.fi/Tatoeba-MT-models/eng-zho/opus-2021-02-23.zip",
    "hi": "https://object.pouta.csc.fi/Tatoeba-MT-models/eng-hin/opus+bt-2021-04-10.zip",
    "de": "https://object.pouta.csc.fi/Tatoeba-MT-models/eng-deu/opus-2021-02-22.zip",
    "it": "https://object.pouta.csc.fi/Tatoeba-MT-models/eng-ita/opus+bt-2021-04-14.zip",
    "ru": "https://object.pouta.csc.fi/Tatoeba-MT-models/eng-rus/opus+bt-2021-04-14.zip",

    # --- African languages ----------------------------------------------
    # All SentencePiece (verified the same way). Treat the quality of these
    # with caution: they are low-resource pairs trained largely on JW300,
    # which is religious text - a good domain match for sermons, but built
    # from far less parallel data than the European pairs above. NLLB-200 is
    # the alternative worth comparing against before trusting them.
    "yo": "https://object.pouta.csc.fi/Tatoeba-MT-models/eng-yor/opus+bt-2021-04-10.zip",
    "ig": "https://object.pouta.csc.fi/OPUS-MT-models/en-ig/opus+bt-2020-05-23.zip",
    "ha": "https://object.pouta.csc.fi/OPUS-MT-models/en-ha/opus+bt-2020-05-23.zip",
    "zu": "https://object.pouta.csc.fi/Tatoeba-MT-models/eng-zul/opus+bt-2021-03-12.zip",
    "sw": "https://object.pouta.csc.fi/Tatoeba-MT-models/eng-swa/opus+bt-2021-04-10.zip",
}

# Piper TTS voices. ~60 MB each; runs on the onnxruntime we already have.
_PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
PIPER_VOICES = {
    "fr_FR-siwis-medium":  f"{_PIPER_BASE}/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx",
    "fr_FR-tom-medium":    f"{_PIPER_BASE}/fr/fr_FR/tom/medium/fr_FR-tom-medium.onnx",
    "fr_FR-upmc-medium":   f"{_PIPER_BASE}/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx",
    "fr_FR-gilles-low":    f"{_PIPER_BASE}/fr/fr_FR/gilles/low/fr_FR-gilles-low.onnx",
    "es_ES-davefx-medium": f"{_PIPER_BASE}/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx",
    "es_ES-sharvard-medium": f"{_PIPER_BASE}/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx",
    "es_MX-ald-medium":    f"{_PIPER_BASE}/es/es_MX/ald/medium/es_MX-ald-medium.onnx",

    # --- second language wave -------------------------------------------
    # One medium voice per new target. Names taken from the published
    # voices.json index, not guessed. Portuguese uses pt_BR rather than the
    # pt_PT alternative, which carries a non-ASCII character in its filename
    # and is the smaller audience.
    "ar_JO-kareem-medium": f"{_PIPER_BASE}/ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx",
    "pt_BR-faber-medium":  f"{_PIPER_BASE}/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx",
    "zh_CN-huayan-medium": f"{_PIPER_BASE}/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx",
    "hi_IN-pratham-medium": f"{_PIPER_BASE}/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx",
    "de_DE-thorsten-medium": f"{_PIPER_BASE}/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
    "it_IT-paola-medium":  f"{_PIPER_BASE}/it/it_IT/paola/medium/it_IT-paola-medium.onnx",
    "ru_RU-dmitri-medium": f"{_PIPER_BASE}/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx",
    # Swahili is the only one of the African targets Piper covers. Igbo,
    # Yoruba and Hausa use VITS voices instead - see MMS_VOICES below.
    "sw_CD-lanfrica-medium": f"{_PIPER_BASE}/sw/sw_CD/lanfrica/medium/sw_CD-lanfrica-medium.onnx",
}

# Non-Piper VITS voices, for languages Piper has none. Same onnxruntime, a
# different input signature - see sermon/tts.py VitsSpeaker.
#
#   Igbo    Meta's MMS-TTS has no Igbo at all, and every community MMS-style
#           Igbo model shares a 43-character vocabulary missing i-dot-below
#           and u-dot-below, so they cannot spell most Igbo words. This is a
#           Coqui checkpoint trained on Igbo Bible readings with the full
#           orthography; it needs the one-off conversion in
#           scripts/export_vits_onnx.py before it can be used.
#   Yoruba,
#   Hausa   Meta's own MMS models, already published as ONNX.
MMS_VOICES = {
    "yor": ("willwade/mms-tts-multilingual-models-onnx",
            ["yor/model.onnx", "yor/tokens.txt"]),
    "hau": ("willwade/mms-tts-multilingual-models-onnx",
            ["hau/model.onnx", "hau/tokens.txt"]),
}
COQUI_VOICES = {
    "ibo": ("multilingual-tts/VITS-OpenBible-Igbo",
            ["config.json", "model_last.pth"]),
}


def human(n: float) -> str:
    return f"{n / 1024 / 1024:.0f} MB"


def remote_size(url: str) -> int:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return int(resp.headers.get("content-length", 0))
    except Exception:
        return 0


def download(url: str, dest: Path) -> Path:
    """Download unless a complete copy is already cached at `dest`.

    The cache matters: the OPUS-MT archive is 265 MB and the conversion step
    after it can fail for reasons unrelated to the download (a missing C++
    runtime, for one). Re-downloading on every retry is needless.
    """
    expected = remote_size(url)
    if dest.exists() and expected and dest.stat().st_size == expected:
        print(f"  reusing cached download ({human(expected)}) at {dest}")
        return dest
    if dest.exists():
        dest.unlink()

    print(f"  downloading {url}")
    started = time.time()
    last = [-1.0]

    def hook(blocks, block_size, total):
        done = blocks * block_size
        if total <= 0:
            return
        pct = min(100.0, done * 100.0 / total)
        if pct - last[0] < 1.0 and pct < 100.0:
            return          # only redraw each 1 %, keeps logs readable
        last[0] = pct
        speed = done / max(time.time() - started, 0.01) / 1024 / 1024
        print(f"\r    {pct:5.1f}%  {human(done)} / {human(total)}  ({speed:.1f} MB/s)",
              end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print()
    return dest


def fetch_whisper(size: str) -> None:
    from huggingface_hub import snapshot_download

    target = MODELS / "whisper" / size
    if (target / "model.bin").exists():
        print(f"[skip] Whisper {size} already present at {target}")
        return

    print(f"[1/2] Whisper {size} -> {target}")
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=WHISPER_REPOS[size],
        local_dir=str(target),
        allow_patterns=["*.bin", "*.json", "*.txt"],
    )
    size_mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"      done, {size_mb:.0f} MB")


def fetch_translation(lang: str, keep_temp: bool) -> None:
    target = MODELS / "mt" / f"opus-mt-en-{lang}"
    if (target / "model.bin").exists() and (target / "source.spm").exists():
        print(f"[skip] OPUS-MT en-{lang} already present at {target}")
        return

    print(f"[2/2] OPUS-MT en-{lang} -> {target}")
    cache = MODELS / "_download"
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / f"opus-mt-en-{lang}.zip"
    work = Path(tempfile.mkdtemp(prefix=f"opus-en-{lang}-", dir=str(cache)))

    converted = False
    try:
        download(OPUS_MT_URLS[lang], zip_path)

        extract = work / "raw"
        extract.mkdir()
        print("  extracting...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract)

        # Some archives nest everything one level deep.
        if not (extract / "decoder.yml").exists():
            nested = [p for p in extract.iterdir() if p.is_dir()]
            if len(nested) == 1 and (nested[0] / "decoder.yml").exists():
                extract = nested[0]
        if not (extract / "decoder.yml").exists():
            raise SystemExit(f"decoder.yml not found in {extract}; archive layout changed")

        print("  converting to CTranslate2 int8 (this takes ~1 minute)...")
        try:
            from ctranslate2.converters.opus_mt import OpusMTConverter
        except (ImportError, FileNotFoundError, OSError) as exc:
            raise SystemExit(
                f"\nCould not load CTranslate2: {exc}\n\n"
                "The usual cause on a clean Windows install is a missing\n"
                "Microsoft Visual C++ Runtime. Install it, then re-run this\n"
                "script (the 265 MB download is cached, so the retry is fast):\n"
                "    https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
            ) from exc

        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        OpusMTConverter(str(extract)).convert(str(target), quantization="int8", force=True)

        # The tokenizer lives outside the CTranslate2 model, so copy it in.
        for name in ("source.spm", "target.spm"):
            src = extract / name
            if src.exists():
                shutil.copy2(src, target / name)
        if not (target / "source.spm").exists():
            raise SystemExit("source.spm missing after conversion")

        converted = True
        size_mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1024 / 1024
        print(f"      done, {size_mb:.0f} MB")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if converted and not keep_temp:
            # Only discard the big archive once we have the converted model.
            zip_path.unlink(missing_ok=True)
        elif not converted:
            print(f"  download kept at {zip_path} so the retry does not re-fetch it")
        try:
            cache.rmdir()
        except OSError:
            pass


def fetch_voice(name: str) -> None:
    target = MODELS / "tts" / name
    onnx = target / f"{name}.onnx"
    if onnx.exists() and (target / f"{name}.onnx.json").exists():
        print(f"[skip] Piper voice {name} already present at {target}")
        return

    print(f"[3/3] Piper voice {name} -> {target}")
    target.mkdir(parents=True, exist_ok=True)
    url = PIPER_VOICES[name]
    download(url, onnx)
    download(url + ".json", target / f"{name}.onnx.json")
    size_mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"      done, {size_mb:.0f} MB")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", choices=list(WHISPER_REPOS),
                    help="Whisper size to download (repeatable). Default: base.en")
    ap.add_argument("--lang", action="append", choices=list(OPUS_MT_URLS),
                    help="translation target (repeatable). Default: fr")
    ap.add_argument("--voice", action="append", choices=list(PIPER_VOICES),
                    help="Piper TTS voice (repeatable). Default: fr_FR-siwis-medium")
    ap.add_argument("--no-voice", action="store_true", help="skip the TTS voice download")
    ap.add_argument("--keep-temp", action="store_true",
                    help="keep the raw 278 MB OPUS-MT download")
    args = ap.parse_args(argv)

    sizes = args.model or ["base.en"]
    langs = args.lang or ["fr", "es"]
    voices = [] if args.no_voice else (args.voice or
                                       ["fr_FR-siwis-medium", "es_ES-davefx-medium"])

    print()
    print("Downloading models. This is the ONLY step that needs internet.")
    print(f"Destination: {MODELS}")
    print()

    for s in sizes:
        fetch_whisper(s)
    for l in langs:
        fetch_translation(l, args.keep_temp)
    for v in voices:
        fetch_voice(v)

    total = sum(f.stat().st_size for f in MODELS.rglob("*") if f.is_file()) / 1024 / 1024
    print()
    print(f"All models ready. models/ is {total:.0f} MB.")
    print("You can disconnect from the internet now.")
    print()
    print("Next:  python run.py doctor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
