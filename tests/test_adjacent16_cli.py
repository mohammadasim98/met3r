from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from met3r.job_registry import WorkerSummary


EXPECTED_COMMANDS = {
    "validate-inputs",
    "prepare-stress",
    "prepare",
    "geometry-worker",
    "geometry-preflight",
    "sentinel-manifest",
    "feature-worker",
    "runtime-gate",
    "aggregate",
    "final-validate",
    "analyze",
    "visualize",
    "status",
}
SHOES = (
    "air_huarache_run_ultra_111449",
    "air_max_90_nrg_dc6083_500",
    "air_max_90_surplus_cq7743_300",
    "air_tuned_max_cv6984_001",
)
BACKBONES = (
    "dino_vitb16",
    "dinov2_vitb14",
    "dinov3_vitb16",
    "dinov3_vitl16_pixal3d",
)


def _fake_python(tmp_path: Path) -> tuple[Path, Path]:
    executable = tmp_path / "fake-python"
    calls = tmp_path / "calls.log"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "${1-}" == "-c" ]]; then\n'
        '  exec "$REAL_PYTHON" "$@"\n'
        "fi\n"
        'printf "%s|%s\\n" "${CUDA_VISIBLE_DEVICES-}" "$*" >> "$CALL_LOG"\n'
        'if [[ -n "${FAKE_DELAY_CONTAINS-}" && "$*" == *"$FAKE_DELAY_CONTAINS"* ]]; then\n'
        '  "$REAL_PYTHON" -c "import pathlib,sys,time; time.sleep(float(sys.argv[1])); pathlib.Path(sys.argv[2]).write_text(\\"done\\")" "$FAKE_DELAY_SECONDS" "$FAKE_DELAY_MARKER"\n'
        "fi\n"
        'if [[ -n "${FAKE_FAIL_CONTAINS-}" && "$*" == *"$FAKE_FAIL_CONTAINS"* ]]; then\n'
        "  exit 23\n"
        "fi\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, calls


def _launcher_environment(
    tmp_path: Path,
    *,
    fail_contains: str | None = None,
) -> tuple[dict[str, str], Path, Path]:
    fake_python, calls = _fake_python(tmp_path)
    run_root = tmp_path / "adjacent-run"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON": str(fake_python),
            "REAL_PYTHON": "/opt/conda/envs/pixal3d/bin/python",
            "CALL_LOG": str(calls),
            "CONFIG": str(tmp_path / "config.json"),
            "RUN_ROOT": str(run_root),
            "DA3_REPO": str(tmp_path / "da3"),
        }
    )
    if fail_contains is not None:
        environment["FAKE_FAIL_CONTAINS"] = fail_contains
    return environment, run_root, calls


def _run_launcher(
    phase: str,
    environment: dict[str, str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/run_adjacent16_core.sh", phase],
        cwd=Path.cwd(),
        env=environment,
        check=check,
        text=True,
        capture_output=True,
    )


def _write_gate_reports(run_root: Path, *, geometry: bool, runtime: bool) -> None:
    reports = run_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "geometry_preflight.json").write_text(
        json.dumps({"passed": geometry}),
        encoding="utf-8",
    )
    (reports / "runtime_projection.json").write_text(
        json.dumps({"passed": runtime}),
        encoding="utf-8",
    )


def _calls(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    return [
        tuple(line.split("|", 1))  # type: ignore[misc]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_help_exposes_exact_adjacent16_subcommands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from met3r.adjacent16_cli import main

    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    output = capsys.readouterr().out
    command_list = output.split("{", 1)[1].split("}", 1)[0]
    assert set(command_list.split(",")) == EXPECTED_COMMANDS


def test_feature_worker_uses_optional_job_id_file_and_defaults_to_all_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from met3r.adjacent16_cli import main

    calls: list[tuple[Path, str, str, tuple[str, ...] | None]] = []

    def worker(
        root: Path,
        backbone: str,
        device: str,
        *,
        job_ids: tuple[str, ...] | None = None,
    ) -> WorkerSummary:
        calls.append((Path(root), backbone, device, job_ids))
        return WorkerSummary(completed=2, failed=0, abstained=0)

    monkeypatch.setattr("met3r.adjacent16_cli.run_feature_worker", worker)
    manifest = tmp_path / "sentinel.json"
    manifest.write_text(
        json.dumps(
            {
                "backbone": "dinov3_vitb16",
                "config_sha256": "f" * 64,
                "job_ids": ["feature-a", "feature-b"],
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "feature-worker",
            str(tmp_path),
            "--backbone",
            "dinov3_vitb16",
            "--device",
            "cuda:0",
            "--job-id-file",
            str(manifest),
        ]
    ) == 0
    assert main(
        [
            "feature-worker",
            str(tmp_path),
            "--backbone",
            "dinov3_vitb16",
            "--device",
            "cuda:0",
        ]
    ) == 0
    assert calls == [
        (
            tmp_path.resolve(),
            "dinov3_vitb16",
            "cuda:0",
            ("feature-a", "feature-b"),
        ),
        (tmp_path.resolve(), "dinov3_vitb16", "cuda:0", None),
    ]


def test_final_validate_stops_before_final_gate_when_ownership_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from met3r.adjacent16_cli import main

    design = object()
    calls: list[str] = []
    monkeypatch.setattr(
        "met3r.adjacent16_cli._load_adjacent16_design",
        lambda root: calls.append("design") or design,
    )

    def fail_ownership(root: Path, actual_design: object) -> None:
        assert Path(root) == tmp_path.resolve()
        assert actual_design is design
        calls.append("ownership")
        raise ValueError("wrong owner")

    monkeypatch.setattr(
        "met3r.adjacent16_cli.validate_adjacent16_metric_ownership",
        fail_ownership,
    )
    monkeypatch.setattr(
        "met3r.adjacent16_cli.validate_adjacent16_final_gate",
        lambda root: calls.append("final"),
    )

    with pytest.raises(ValueError, match="wrong owner"):
        main(["final-validate", str(tmp_path)])

    assert calls == ["design", "ownership"]


def test_aggregate_validates_ownership_before_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from met3r.adjacent16_cli import main

    design = object()
    calls: list[str] = []
    monkeypatch.setattr(
        "met3r.adjacent16_cli._load_adjacent16_design",
        lambda root: calls.append("design") or design,
    )
    monkeypatch.setattr(
        "met3r.adjacent16_cli.validate_adjacent16_metric_ownership",
        lambda root, actual: calls.append("ownership"),
    )
    monkeypatch.setattr(
        "met3r.adjacent16_cli.aggregate_all_metric_parts",
        lambda root: calls.append("aggregate") or (tmp_path / "metrics.parquet",),
    )

    assert main(["aggregate", str(tmp_path)]) == 0
    assert calls == ["design", "ownership", "aggregate"]


def test_launcher_default_gpu_mapping_is_deterministic_and_preserves_run(
    tmp_path: Path,
) -> None:
    environment, run_root, calls_path = _launcher_environment(tmp_path)
    marker = run_root / "keep-existing-run"
    marker.parent.mkdir(parents=True)
    marker.write_text("completed registry state", encoding="utf-8")

    _run_launcher("geometry", environment)
    _write_gate_reports(run_root, geometry=True, runtime=True)
    _run_launcher("full", environment)

    calls = _calls(calls_path)
    geometry = [
        (gpu, command)
        for gpu, command in calls
        if " geometry-worker " in f" {command} "
    ]
    features = [
        (gpu, command)
        for gpu, command in calls
        if " feature-worker " in f" {command} "
    ]
    assert {
        next(shoe for shoe in SHOES if f"--shoe-id {shoe}" in command): gpu
        for gpu, command in geometry
    } == dict(zip(SHOES, ("0", "1", "2", "3"), strict=True))
    assert {
        next(backbone for backbone in BACKBONES if f"--backbone {backbone}" in command): gpu
        for gpu, command in features
    } == dict(zip(BACKBONES, ("0", "1", "2", "3"), strict=True))
    assert marker.read_text(encoding="utf-8") == "completed registry state"
    assert all("20260724_upper_v2" not in command for _, command in calls)


@pytest.mark.parametrize(
    ("geometry_passed", "runtime_passed"),
    ((False, True), (True, False)),
)
def test_full_refuses_to_launch_without_both_persisted_gates(
    tmp_path: Path,
    geometry_passed: bool,
    runtime_passed: bool,
) -> None:
    environment, run_root, calls_path = _launcher_environment(tmp_path)
    _write_gate_reports(
        run_root,
        geometry=geometry_passed,
        runtime=runtime_passed,
    )

    completed = _run_launcher("full", environment, check=False)

    assert completed.returncode != 0
    assert not any(
        " feature-worker " in f" {command} "
        for _, command in _calls(calls_path)
    )


def test_launcher_waits_for_every_child_and_propagates_failure(
    tmp_path: Path,
) -> None:
    environment, run_root, calls_path = _launcher_environment(
        tmp_path,
        fail_contains="dinov2_vitb14",
    )
    delayed_marker = tmp_path / "delayed-child-finished"
    environment.update(
        {
            "FAKE_DELAY_CONTAINS": "dinov3_vitl16_pixal3d",
            "FAKE_DELAY_SECONDS": "0.3",
            "FAKE_DELAY_MARKER": str(delayed_marker),
        }
    )
    _write_gate_reports(run_root, geometry=True, runtime=True)

    started = time.monotonic()
    completed = _run_launcher("full", environment, check=False)
    elapsed = time.monotonic() - started

    feature_calls = [
        command
        for _, command in _calls(calls_path)
        if " feature-worker " in f" {command} "
    ]
    assert completed.returncode != 0
    assert len(feature_calls) == 4
    assert all(backbone in " ".join(feature_calls) for backbone in BACKBONES)
    assert delayed_marker.read_text(encoding="utf-8") == "done"
    assert elapsed >= 0.25


@pytest.mark.parametrize("phase", ("stress", "geometry", "sentinel", "full", "finalize"))
def test_launcher_supports_each_frozen_phase_with_per_phase_logs(
    tmp_path: Path,
    phase: str,
) -> None:
    environment, run_root, _calls_path = _launcher_environment(tmp_path)
    _write_gate_reports(run_root, geometry=True, runtime=True)

    _run_launcher(phase, environment)

    phase_logs = list((run_root / "logs").glob(f"{phase}_*.log"))
    assert phase_logs
