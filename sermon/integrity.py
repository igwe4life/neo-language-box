"""Model-folder integrity: a manifest of sizes and hashes, and a check against it.

The models travel separately from the installer - over Drive, SwissTransfer, a
USB stick - and a truncated or corrupted file does not fail loudly. CTranslate2
and onnxruntime tend to raise something obscure minutes into startup, or load
and quietly produce nonsense. Checking against a manifest turns that into a
clear "re-copy these files" before anything else runs.

Hashing ~4 GB takes around ten seconds, too slow for every launch. So a full
check runs once, and a small stamp records the size and modification time of
every file it passed; later launches only re-hash if something changed. Sizes
are always compared, which is free and catches every truncated transfer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

MANIFEST_NAME = "models-manifest.json"

Progress = Optional[Callable[[int, int], None]]      # (bytes_done, bytes_total)


def digest(path: Path, chunk: int = 8 << 20) -> str:
    """BLAKE2b of a file, read in chunks so a 1.4 GB model need not fit in RAM."""
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def iter_files(models: Path):
    for p in sorted(models.rglob("*")):
        if not p.is_file() or p.name == MANIFEST_NAME:
            continue
        # Hugging Face download metadata, not part of any model.
        if ".cache" in p.relative_to(models).parts:
            continue
        yield p


def write_manifest(models: Path, progress: Progress = None) -> Path:
    files = list(iter_files(models))
    total = sum(f.stat().st_size for f in files)
    entries, done = {}, 0
    for f in files:
        size = f.stat().st_size
        entries[f.relative_to(models).as_posix()] = {"size": size,
                                                     "blake2b": digest(f)}
        done += size
        if progress:
            progress(done, total)
    path = models / MANIFEST_NAME
    path.write_text(json.dumps({"files": entries}, indent=1), encoding="utf-8")
    return path


@dataclass
class Report:
    has_manifest: bool
    missing: list = field(default_factory=list)
    corrupt: list = field(default_factory=list)
    cached: bool = False                 # passed on a previous launch, unchanged

    @property
    def ok(self) -> bool:
        return not self.missing and not self.corrupt


def verify(models: Path, stamp_path: Optional[Path] = None,
           progress: Progress = None) -> Report:
    """Check `models` against its manifest.

    No manifest means nothing to check against - a developer's working folder,
    say - and is reported as such rather than as a failure.
    """
    manifest_path = models / MANIFEST_NAME
    if not manifest_path.exists():
        return Report(has_manifest=False)
    want = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]

    report = Report(has_manifest=True)
    observed = {}
    for rel, meta in want.items():
        try:
            st = (models / rel).stat()
        except OSError:
            report.missing.append(rel)
            continue
        if st.st_size != meta["size"]:
            report.corrupt.append(rel)
            continue
        observed[rel] = [st.st_size, st.st_mtime_ns]

    # Anything that could make a previous pass stale is part of the key: the
    # folder, the manifest itself, and every file's size and mtime.
    key = {"models": str(models.resolve()),
           "manifest": digest(manifest_path),
           "files": observed}
    if report.ok and stamp_path is not None:
        try:
            if json.loads(stamp_path.read_text(encoding="utf-8")) == key:
                report.cached = True
                return report
        except (OSError, ValueError):
            pass

    # Hash everything still in the running, EVEN IF something is already known
    # to be missing. Stopping at the first problem means someone re-downloads
    # one file, relaunches, and only then learns about the next - over a slow
    # link that is far worse than the extra seconds spent here.
    total = sum(want[rel]["size"] for rel in observed)
    done = 0
    for rel in observed:
        if digest(models / rel) != want[rel]["blake2b"]:
            report.corrupt.append(rel)
        done += want[rel]["size"]
        if progress:
            progress(done, total)

    if report.ok and stamp_path is not None:
        try:
            stamp_path.write_text(json.dumps(key), encoding="utf-8")
        except OSError:
            pass            # unwritable location: just check again next time
    return report
