from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from met3r.job_registry import WorkerSummary
from met3r.upper_cli import main


def test_prepare_forwards_physical_gpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, str, tuple[int, ...] | None]] = []

    def fake_prepare(
        config: Path,
        scope: str,
        *,
        physical_gpus: tuple[int, ...] | None = None,
    ) -> SimpleNamespace:
        calls.append((config, scope, physical_gpus))
        return SimpleNamespace(
            root=tmp_path / scope,
            scope=scope,
            config_sha256="f" * 64,
        )

    monkeypatch.setattr("met3r.upper_cli.prepare_run", fake_prepare)
    config = tmp_path / "config.json"

    assert (
        main(
            [
                "prepare",
                str(config),
                "--scope",
                "full",
                "--physical-gpus",
                "0,1,2,3",
            ]
        )
        == 0
    )
    assert calls == [(config.resolve(), "full", (0, 1, 2, 3))]


def test_help_lists_exactly_seven_commands(capsys: pytest.CaptureFixture[str]) -> None:
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
        "analyze",
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


def test_pilot_launcher_defaults_to_gpu1_and_v2_root(
    tmp_path: Path,
) -> None:
    path = Path("scripts/run_upper_pilot.sh")
    text = path.read_text(encoding="utf-8")
    fake_python = tmp_path / "python"
    call_log = tmp_path / "calls.log"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s|%s\\n' \"${CUDA_VISIBLE_DEVICES-}\" \"$*\" >> \"$CALL_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    safe_copy = tmp_path / "run_upper_pilot.sh"
    safe_copy.write_text(
        text.replace(
            "PYTHON=/opt/conda/envs/pixal3d/bin/python",
            f"PYTHON={fake_python}",
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(PYTHON=str(fake_python), CALL_LOG=str(call_log))

    subprocess.run(
        ["bash", str(safe_copy)],
        cwd=Path.cwd(),
        env=environment,
        check=True,
    )

    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 6
    gpu_calls = [line for line in calls if line.split("|", 1)[0]]
    assert [line.split("|", 1)[0] for line in gpu_calls] == ["1"] * 4
    assert sum("geometry-worker" in line for line in calls) == 2
    assert sum("feature-worker" in line for line in calls) == 1
    assert sum("validate-pilot" in line and "--rerun-sentinel" in line for line in calls) == 1
    assert all(
        "20260724_upper_v2/pilot" in line
        for line in calls
        if " prepare " not in f" {line} "
    )
    assert any("--shoe-id air_huarache_run_ultra_111449" in line for line in calls)
    assert any("--shoe-id air_max_90_surplus_cq7743_300" in line for line in calls)
    assert not any("bottom" in line.lower() for line in calls)


def test_analyze_cli_forwards_run_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(
        "met3r.upper_cli.analyze_upper_run",
        lambda root: calls.append(Path(root))
        or type(
            "Outputs",
            (),
            {
                "tables": (tmp_path / "table.csv",),
                "figures": (tmp_path / "figure.png",),
                "report_path": tmp_path / "report.md",
            },
        )(),
    )
    assert main(["analyze", str(tmp_path)]) == 0
    assert calls == [tmp_path.resolve()]


def test_full_launcher_maps_exact_shoes_and_backbones_to_gpus(
    tmp_path: Path,
) -> None:
    text = Path("scripts/run_upper_full.sh").read_text(encoding="utf-8")
    fake_python = tmp_path / "python"
    call_log = tmp_path / "calls.log"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s|%s\\n' \"${CUDA_VISIBLE_DEVICES-}\" \"$*\" >> \"$CALL_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    run_root = tmp_path / "full"
    (run_root / "logs").mkdir(parents=True)
    config = tmp_path / "config.json"
    da3_repository = tmp_path / "da3"
    safe_copy = tmp_path / "run_upper_full.sh"
    for old, new in (
        (
            "PYTHON=/opt/conda/envs/pixal3d/bin/python",
            "PYTHON=${PYTHON:-/opt/conda/envs/pixal3d/bin/python}",
        ),
        (
            "CONFIG=/root/dev/met3r-worktrees/known-camera-met3r/"
            "configs/upper_dino_fidelity.json",
            "CONFIG=${CONFIG:-/root/dev/met3r-worktrees/known-camera-met3r/"
            "configs/upper_dino_fidelity.json}",
        ),
        (
            "RUN_ROOT=/root/dev/ByteSize/Reconstruction/"
            "met3r_upper_experiments/20260724_upper_v2/full",
            "RUN_ROOT=${RUN_ROOT:-/root/dev/ByteSize/Reconstruction/"
            "met3r_upper_experiments/20260724_upper_v2/full}",
        ),
        (
            "DA3_REPO=/root/dev/ByteSize/Reconstruction/third_party/"
            "Depth-Anything-3",
            "DA3_REPO=${DA3_REPO:-/root/dev/ByteSize/Reconstruction/"
            "third_party/Depth-Anything-3}",
        ),
    ):
        text = text.replace(old, new)
    safe_copy.write_text(text, encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        PYTHON=str(fake_python),
        CONFIG=str(config),
        RUN_ROOT=str(run_root),
        DA3_REPO=str(da3_repository),
        FULL_GPUS="0,1,2,3",
        CALL_LOG=str(call_log),
    )

    subprocess.run(
        ["bash", str(safe_copy)],
        cwd=Path.cwd(),
        env=environment,
        check=True,
    )

    calls = call_log.read_text(encoding="utf-8").splitlines()
    prepare_calls = [
        line for line in calls if " met3r.upper_cli prepare " in f" {line} "
    ]
    assert len(prepare_calls) == 1
    assert (
        f"prepare {config} --scope full --physical-gpus 0,1,2,3"
        in prepare_calls[0]
    )
    geometry = {}
    features = {}
    for line in calls:
        gpu, arguments = line.split("|", 1)
        tokens = arguments.split()
        if "geometry-worker" in tokens:
            geometry[tokens[tokens.index("--shoe-id") + 1]] = gpu
        if "feature-worker" in tokens:
            features[tokens[tokens.index("--backbone") + 1]] = gpu
    assert geometry == {
        "air_huarache_run_ultra_111449": "0",
        "air_max_90_nrg_dc6083_500": "1",
        "air_max_90_surplus_cq7743_300": "2",
        "air_tuned_max_cv6984_001": "3",
    }
    assert features == {
        "dino_vitb16": "0",
        "dinov2_vitb14": "1",
        "dinov3_vitb16": "2",
        "dinov3_vitl16_pixal3d": "3",
    }
    assert sum(" met3r.upper_cli aggregate " in f" {line} " for line in calls) == 1
    assert sum(" met3r.upper_cli analyze " in f" {line} " for line in calls) == 1
    assert not any("bottom" in line.lower() for line in calls)
