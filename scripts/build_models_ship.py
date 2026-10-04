"""Build the models folder to hand to other operators: only what the app loads.

The development models/ folder carries extras the shipped app never touches -
the Igbo voice's Coqui source checkpoint, the NLLB-600M model that lost to
1.3B, bilingual OPUS-MT models kept from the blind comparison. That is ~2.3 GB
nobody receiving the app needs to download.

The file list is derived from sermon/config.py rather than written out by
hand, so adding a language to the app automatically adds it here.

Files are HARD-LINKED where the filesystem allows, so building this costs no
extra disk space and is near-instant; it falls back to copying otherwise. The
result is a normal folder to any uploader.

    python scripts/build_models_ship.py            # -> models-ship/
    python scripts/build_models_ship.py --clean    # rebuild from scratch
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sermon.config import (ALL_VOICES, DEFAULT_TARGETS, NLLB_TARGETS,  # noqa: E402
                           VITS_VOICES)
from sermon.integrity import (MANIFEST_NAME, verify,  # noqa: E402
                              write_manifest)

README = """\
NEO LANGUAGE BOX - LANGUAGE MODELS
===================================

WHERE THIS GOES
Copy the `models` folder (the folder itself, not its contents) into the
Neo Language Box install directory, normally:

    %LOCALAPPDATA%\\NeoLanguageBox\\models

Or keep it anywhere and set the environment variable SERMON_MODELS_DIR to its
full path. On first launch Neo Language Box checks every file against
models-manifest.json and names anything missing or damaged in transit.

WHAT IS IN IT AND WHERE IT CAME FROM
This folder redistributes third-party models. Each keeps its own licence;
check the source before using or passing them on - this list is a pointer,
not legal advice. Two of them are NON-COMMERCIAL:

  nllb-1.3b\\            Meta NLLB-200 (distilled 1.3B), CTranslate2 build
                        from OpenNMT/nllb-200-distilled-1.3B-ct2-int8.
                        Licence: CC-BY-NC-4.0 - non-commercial use only.
  tts\\yo, tts\\ha        Meta MMS-TTS voices, ONNX builds from
                        willwade/mms-tts-multilingual-models-onnx.
                        Licence: CC-BY-NC-4.0 - non-commercial use only.
  tts\\ig                multilingual-tts/VITS-OpenBible-Igbo, converted to
                        ONNX. See that page for its licence.
  tts\\<xx_XX-...>       Piper voices, rhasspy/piper-voices. Licences vary by
                        voice - see each voice's model card.
  mt\\opus-mt-en-*       OPUS-MT / Tatoeba-MT, Helsinki-NLP. See the Tatoeba
                        Challenge / OPUS-MT pages for terms.
  parakeet-*            NVIDIA Parakeet, via sherpa-onnx (k2-fsa). See the
                        model pages.
  whisper\\base.en       OpenAI Whisper, faster-whisper build (Systran).
  silero_vad.onnx       Silero VAD (snakers4/silero-vad).
"""


def required(models: Path) -> list:
    """Every path the shipped app can load, relative to models/."""
    rels = [
        Path("silero_vad.onnx"),
        Path("parakeet-0.6b"),          # default speech recognition
        Path("parakeet-110m"),          # SERMON_STT fallback
        Path("whisper") / "base.en",    # SERMON_STT fallback
        Path("nllb-1.3b"),              # the NLLB_TARGETS languages
    ]
    rels += [Path("mt") / f"opus-mt-en-{t}"
             for t in DEFAULT_TARGETS if t not in NLLB_TARGETS]
    rels += [Path("tts") / name for name in ALL_VOICES.values()]
    rels += [Path("tts") / name for name in VITS_VOICES.values()]
    return rels


def _place(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--out", default=str(ROOT / "models-ship"))
    ap.add_argument("--clean", action="store_true",
                    help="delete an existing output folder first")
    args = ap.parse_args()

    src = Path(args.models)
    out = Path(args.out)
    dest = out / "models"

    if dest.exists() and any(dest.iterdir()):
        if not args.clean:
            print(f"{dest} already exists. Pass --clean to rebuild it.")
            return 2
        # Removing a hard link leaves the original file untouched.
        shutil.rmtree(out)

    missing = [r for r in required(src) if not (src / r).exists()]
    if missing:
        print("Cannot build - these are missing from the source models folder:")
        for r in missing:
            print(f"  {r.as_posix()}")
        return 1

    started = time.monotonic()
    placed, size, how = [], 0, {"link": 0, "copy": 0}
    for rel in required(src):
        item = src / rel
        files = [item] if item.is_file() else sorted(
            p for p in item.rglob("*")
            if p.is_file() and ".cache" not in p.relative_to(src).parts)
        for f in files:
            how[_place(f, dest / f.relative_to(src))] += 1
            placed.append(f.relative_to(src).as_posix())
            size += f.stat().st_size

    (out / "README.txt").write_text(README, encoding="utf-8")
    print(f"placed {len(placed)} files, {size / 1e9:.2f} GB "
          f"({how['link']} linked, {how['copy']} copied) "
          f"in {time.monotonic() - started:.1f}s")

    source_manifest = src / MANIFEST_NAME
    if source_manifest.exists():
        # Take the hashes from the SOURCE's manifest, then check the new folder
        # against them. That verifies every link or copy really matches the
        # original - writing a fresh manifest from the copies would only prove
        # the copies agree with themselves.
        entries = json.loads(source_manifest.read_text(encoding="utf-8"))["files"]
        absent = [r for r in placed if r not in entries]
        if absent:
            print("!! the source manifest is out of date (no entry for "
                  f"{absent[0]}). Regenerate it with:")
            print("     python scripts/verify_models.py --write")
            return 1
        (dest / MANIFEST_NAME).write_text(
            json.dumps({"files": {r: entries[r] for r in placed}}, indent=1),
            encoding="utf-8")
        print("verifying against the source manifest...")
        report = verify(dest)
        if not report.ok:
            print(f"!! {len(report.missing)} missing, {len(report.corrupt)} "
                  "differ from the source - is the source manifest stale?")
            for r in (report.missing + report.corrupt)[:10]:
                print(f"     {r}")
            return 1
    else:
        print("no source manifest; hashing the new folder...")
        write_manifest(dest)

    print(f"done. Ship this folder: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
