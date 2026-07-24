from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from met3r.job_registry import WorkerSummary
from met3r.upper_cli import main


def test_help_lists_exactly_six_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    for command in (
        "prepare",
        "geometry-worker",
        "feature-worker",
        "aggregate",
        "validate-pilot",
        "status",
    ):
        assert command in output


def test_geometry_cli_forwards_local_cuda_device_and_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, str, str, Path]] = []
    monkeypatch.setattr(
        "met3r.upper_cli.run_geometry_worker",
        lambda root, shoe, device, repository: calls.append(
            (Path(root), shoe, device, Path(repository))
        )
        or WorkerSummary(1, 0, 0),
    )
    main(
        [
            "geometry-worker",
            str(tmp_path),
            "--shoe-id",
            "shoe-a",
            "--device",
            "cuda:0",
            "--da3-repository",
            "/fake/da3",
        ]
    )
    assert calls == [(tmp_path.resolve(), "shoe-a", "cuda:0", Path("/fake/da3"))]


def test_status_require_complete_returns_nonzero_for_pending(
    tmp_path: Path,
) -> None:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    from met3r.job_registry import JobDefinition, JobRegistry

    registry = JobRegistry(registry_dir / "jobs.sqlite")
    registry.register(JobDefinition("job", "feature", {"x": 1}))
    with pytest.raises(SystemExit) as error:
        main(["status", str(tmp_path), "--require-complete"])
    assert error.value.code == 1


def test_status_require_complete_revalidates_completed_output_hashes(
    tmp_path: Path,
) -> None:
    from met3r.job_registry import JobDefinition, JobRegistry, sha256_file

    registry = JobRegistry(tmp_path / "registry/jobs.sqlite")
    registry.register(JobDefinition("job", "feature", {"x": 1}))
    claimed = registry.claim_next("test", "feature", {})
    assert claimed is not None
    output = tmp_path / "part.parquet"
    output.write_bytes(b"valid")
    registry.complete(
        claimed.job_id,
        {
            "files": [
                {"path": str(output.resolve()), "sha256": sha256_file(output)}
            ]
        },
    )
    output.write_bytes(b"corrupt")

    with pytest.raises(SystemExit) as error:
        main(["status", str(tmp_path), "--require-complete"])

    assert error.value.code == 1


@pytest.mark.parametrize("reason", ("", "   "))
def test_status_require_complete_rejects_unexplained_abstention(
    tmp_path: Path,
    reason: str,
) -> None:
    from met3r.job_registry import JobDefinition, JobRegistry

    registry = JobRegistry(tmp_path / "registry/jobs.sqlite")
    registry.register(JobDefinition("job", "feature", {"x": 1}))
    with sqlite3.connect(registry.path) as connection:
        connection.execute(
            "UPDATE jobs SET state = 'abstained', error = ? WHERE job_id = 'job'",
            (reason,),
        )
    with pytest.raises(SystemExit) as error:
        main(["status", str(tmp_path), "--require-complete"])
    assert error.value.code == 1


def test_pilot_launcher_is_upper_only_gpu4_and_exact_paths() -> None:
    path = Path("scripts/run_upper_pilot.sh")
    text = path.read_text(encoding="utf-8")
    assert "PYTHON=/opt/conda/envs/pixal3d/bin/python" in text
    assert (
        "CONFIG=/root/dev/met3r-worktrees/known-camera-met3r/"
        "configs/upper_dino_fidelity.json"
    ) in text
    # The geometry prefix appears once in a loop that executes for two shoes.
    assert text.count("CUDA_VISIBLE_DEVICES=4") == 3
    assert text.count("--device cuda:0") == 2
    assert "air_huarache_run_ultra_111449" in text
    assert "air_max_90_surplus_cq7743_300" in text
    assert "bottom" not in text.lower()
    assert "met3r.upper_cli validate-pilot" in text
    assert "\"$RUN_ROOT\" --rerun-sentinel" in text
