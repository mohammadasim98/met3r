from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).parents[1]


@pytest.mark.skipif(
    not (REPOSITORY / "mast3r" / "mast3r" / "__init__.py").is_file(),
    reason="MASt3R/DUSt3R submodules are not initialized",
)
def test_wheel_keeps_legacy_nested_packages(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            str(REPOSITORY),
            "-w",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(tmp_path.glob("met3r-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

    assert "met3r/met3r.py" in names
    assert "mast3r/mast3r/__init__.py" in names
    assert "mast3r/dust3r/dust3r/__init__.py" in names
    assert "mast3r/dust3r/croco/models/curope/__init__.py" in names
