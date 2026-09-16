"""Checksum manifest for a models folder, so a transfer can be trusted.

    Before sending:   python scripts/verify_models.py --write
    After receiving:  python scripts/verify_models.py --check

The installed app runs the same check on its first launch; this is the
command-line version for a developer's machine. Logic lives in
sermon/integrity.py so the two can never disagree.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sermon.integrity import iter_files, verify, write_manifest  # noqa: E402


def _bar(label):
    last = [-1]

    def show(done, total):
        pct = int(done * 100 / max(total, 1))
        if pct != last[0]:
            last[0] = pct
            print(f"\r  {label} {pct:3d}%", end="", flush=True)
    return show


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

    files = list(iter_files(models))
    print(f"{len(files)} files, {sum(f.stat().st_size for f in files) / 1e9:.2f} GB")

    if args.write:
        started = time.monotonic()
        path = write_manifest(models, _bar("hashing"))
        print(f"\r  hashed in {time.monotonic() - started:.0f}s        ")
        print(f"wrote {path}")
        return 0

    report = verify(models, progress=_bar("checking"))
    print("\r" + " " * 24)
    if not report.has_manifest:
        print("No manifest in that folder - run --write on the source first.")
        return 2
    for rel in report.missing:
        print(f"  MISSING   {rel}")
    for rel in report.corrupt:
        print(f"  CORRUPT   {rel}")
    if not report.ok:
        print(f"\nFAILED: {len(report.missing)} missing, "
              f"{len(report.corrupt)} corrupt. Re-copy those files.")
        return 1
    print("All files present and intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
