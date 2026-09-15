"""Checksum manifest for the models folder, so a transfer can be trusted.

Moving ~6 GB over Drive, a USB stick or a flaky link can truncate or corrupt a
file without any visible error. A half-written model.bin does not fail loudly:
onnxruntime or CTranslate2 will usually raise something obscure minutes into
startup, or - worse - load and produce nonsense.

    Before sending:   python scripts/verify_models.py --write
    After receiving:  python scripts/verify_models.py --check

Hashes are BLAKE2b over the file contents, read in chunks so a 1.4 GB file
does not need to fit in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_NAME = "models-manifest.json"


def digest(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def walk(models: Path):
    for p in sorted(models.rglob("*")):
        if p.is_file() and p.name != MANIFEST_NAME:
            yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--write", action="store_true", help="create the manifest")
    ap.add_argument("--check", action="store_true", help="verify against it")
    args = ap.parse_args()
    if args.write == args.check:
        ap.error("choose exactly one of --write or --check")

    models = Path(args.models)
    if not models.is_dir():
        print(f"No models folder at {models}")
        return 2
    manifest_path = models / MANIFEST_NAME

    files = list(walk(models))
    total = sum(f.stat().st_size for f in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB")

    if args.write:
        entries = {}
        done = 0
        started = time.monotonic()
        for f in files:
            entries[f.relative_to(models).as_posix()] = {
                "size": f.stat().st_size, "blake2b": digest(f)}
            done += f.stat().st_size
            print(f"\r  hashing {done / max(total, 1) * 100:5.1f}%", end="")
        print(f"\r  hashed in {time.monotonic() - started:.0f}s")
        manifest_path.write_text(
            json.dumps({"files": entries}, indent=1), encoding="utf-8")
        print(f"wrote {manifest_path}")
        return 0

    if not manifest_path.exists():
        print(f"No manifest at {manifest_path} - run --write on the source first.")
        return 2
    want = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    have = {f.relative_to(models).as_posix(): f for f in files}

    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    bad, done = [], 0
    for rel, meta in want.items():
        f = have.get(rel)
        if f is None:
            continue
        # Size first: it is free, and catches every truncated transfer.
        if f.stat().st_size != meta["size"] or digest(f) != meta["blake2b"]:
            bad.append(rel)
        done += meta["size"]
        print(f"\r  checking {done / max(total, 1) * 100:5.1f}%", end="")
    print("\r" + " " * 24)

    for rel in missing:
        print(f"  MISSING   {rel}")
    for rel in bad:
        print(f"  CORRUPT   {rel}")
    for rel in extra:
        print(f"  (extra)   {rel}")

    if missing or bad:
        print(f"\nFAILED: {len(missing)} missing, {len(bad)} corrupt. "
              "Re-copy those files.")
        return 1
    print("\nAll files present and intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
