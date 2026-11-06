"""Build one language's downloadable pack: a zip of exactly the files that
language needs beyond the base install, plus the manifest entry for it.

A language needs some subset of:
  - a Piper or VITS voice directory under models/tts/
  - its own OPUS-MT directory under models/mt/ (bilingual languages only -
    an NLLB_TARGETS language has nothing here, since NLLB-200 itself is a
    1.4 GB shared/core model, already in the base install, never
    duplicated per pack)

This deliberately does NOT touch the telemetry server or upload anything -
see telemetry-server/README.md for that step. This script only answers
"what does this language need, zipped up, with a checksum".

    python scripts/build_language_pack.py ko
    python scripts/build_language_pack.py vi --out telemetry-server/packs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sermon.config import ALL_VOICES, VITS_VOICES, LANGUAGES  # noqa: E402
import scripts.fetch_models as fm  # noqa: E402

PACK_VERSION = 1   # bump when a language's pack contents change


def pack_files(code: str, models_dir: Path) -> list[Path]:
    """Every file this language's pack needs to contain, or raises if the
    language isn't shippable as a pack (no voice at all - text-only
    languages like Japanese have nothing to download)."""
    files: list[Path] = []

    voice = ALL_VOICES.get(code)
    if voice:
        voice_dir = models_dir / "tts" / voice
    elif code in VITS_VOICES:
        voice_dir = models_dir / "tts" / VITS_VOICES[code]
    else:
        voice_dir = None

    if voice_dir is not None:
        if not voice_dir.is_dir():
            raise FileNotFoundError(f"{voice_dir} missing - run fetch_models.py first")
        files += sorted(p for p in voice_dir.rglob("*") if p.is_file())

    if code in fm.OPUS_MT_URLS:
        mt_dir = models_dir / "mt" / f"opus-mt-en-{code}"
        if not mt_dir.is_dir():
            raise FileNotFoundError(f"{mt_dir} missing - run fetch_models.py first")
        files += sorted(p for p in mt_dir.rglob("*") if p.is_file())

    if not files:
        raise ValueError(
            f"{code!r} has no voice and no own MT model - nothing to pack "
            f"(if it uses NLLB-200 for translation, that's core, not a pack)")
    return files


def build(code: str, models_dir: Path, out_dir: Path) -> dict:
    if code not in LANGUAGES:
        raise ValueError(f"{code!r} is not a known language in sermon/config.py")

    files = pack_files(code, models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_name = f"{code}-v{PACK_VERSION}.zip"
    zip_path = out_dir / zip_name

    print(f"[{code}] packing {len(files)} file(s):")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        # ZIP_STORED, not DEFLATE: these files are already quantised model
        # weights - int8 CTranslate2/onnxruntime binaries - which earlier
        # measurement (see installer.iss) showed barely compress (~0.91
        # ratio) at real cost in build time. Storing uncompressed is
        # honest about that rather than spending minutes for ~9%.
        for f in files:
            arcname = f.relative_to(models_dir).as_posix()
            zf.write(f, arcname)
            print(f"      {arcname}  ({f.stat().st_size / 1e6:.1f} MB)")

    size = zip_path.stat().st_size
    sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    print(f"[{code}] -> {zip_path}  ({size / 1e6:.1f} MB, sha256 {sha256[:12]}...)")

    return {
        "code": code,
        "name": LANGUAGES[code]["native"],
        "version": str(PACK_VERSION),
        "url": f"/v1/packs/{zip_name}",
        "sha256": sha256,
        "size_bytes": size,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("code", help="language code, e.g. ko")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--out", type=Path, default=ROOT / "telemetry-server" / "packs")
    args = ap.parse_args(argv)

    entry = build(args.code, args.models_dir, args.out)
    print()
    print("Manifest entry (append to telemetry-server/manifest.json's language_packs):")
    print(json.dumps(entry, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
