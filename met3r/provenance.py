"""Runtime and local-source provenance for reproducible experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


RUNTIME_PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "Pillow",
    "opencv-python",
    "matplotlib",
    "pandas",
    "transformers",
    "huggingface-hub",
    "natten",
    "depth-anything-3",
)


def runtime_environment() -> dict[str, object]:
    packages: dict[str, str | None] = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        import torch

        cuda = torch.version.cuda
        cudnn = torch.backends.cudnn.version()
    except ImportError:
        cuda = None
        cudnn = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda": cuda,
        "cudnn": cudnn,
        "packages": packages,
    }


def provenance_fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def repository_state(path: str | Path) -> dict[str, object]:
    repository = Path(path).expanduser().resolve()
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dirty = bool(status.strip())
    dirty_fingerprint = None
    if dirty:
        digest = hashlib.sha256()
        digest.update(
            subprocess.run(
                ["git", "-C", str(repository), "diff", "--binary", "HEAD", "--", "."],
                check=True,
                capture_output=True,
            ).stdout
        )
        untracked = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        for encoded_path in sorted(path for path in untracked if path):
            digest.update(encoded_path)
            digest.update((repository / encoded_path.decode()).read_bytes())
        dirty_fingerprint = digest.hexdigest()
    return {
        "path": str(repository),
        "revision": revision,
        "dirty": dirty,
        "dirty_fingerprint": dirty_fingerprint,
    }
