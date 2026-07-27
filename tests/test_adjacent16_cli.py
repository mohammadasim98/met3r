from __future__ import annotations

import hashlib
import json
import os
import sqlite3
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


def test_temp_run_override_uses_one_persisted_stress_root_across_stress_and_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from met3r.adjacent16_cli import main

    config = tmp_path / "adjacent16.json"
    config.write_bytes(b'{"frozen":"config"}\n')
    run_root = tmp_path / "custom-run"
    expected_stress_root = run_root / "inputs" / "stress"
    seen_stress_roots: list[Path | None] = []
    contract = SimpleNamespace(
        output_root=tmp_path / "configured-run",
        stress_root=tmp_path / "configured-stress",
        physical_gpus=(0, 1, 2, 3),
    )

    def fake_replace(value: SimpleNamespace, **changes: object) -> SimpleNamespace:
        return SimpleNamespace(**{**vars(value), **changes})

    def fake_manifest(
        actual_contract: SimpleNamespace,
        *,
        require_stress: bool,
    ) -> object:
        assert not require_stress
        seen_stress_roots.append(Path(actual_contract.stress_root))
        return object()

    def fake_prepare(
        config_path: Path,
        *,
        output_root: Path,
        stress_root: Path,
    ) -> SimpleNamespace:
        assert config_path == config.resolve()
        assert output_root == run_root.resolve()
        seen_stress_roots.append(Path(stress_root))
        frozen = run_root / "config" / "frozen_config.json"
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(config.read_bytes())
        (run_root / "run_manifest.json").write_text(
            json.dumps(
                {
                    "experiment_id": "20260727_adjacent16_core_v1",
                    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        manifests = run_root / "manifests"
        manifests.mkdir(parents=True, exist_ok=True)
        (manifests / "adjacent16_manifest.json").write_text(
            json.dumps({"stress_root": str(stress_root)}),
            encoding="utf-8",
        )
        return SimpleNamespace(
            root=run_root.resolve(),
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        )

    monkeypatch.setattr("met3r.adjacent16_cli.load_adjacent16_contract", lambda path: contract)
    monkeypatch.setattr("met3r.adjacent16_cli.replace", fake_replace)
    monkeypatch.setattr("met3r.adjacent16_cli.build_adjacent16_manifest", fake_manifest)
    monkeypatch.setattr(
        "met3r.adjacent16_cli.render_stress_references",
        lambda actual_contract, manifest: (),
    )
    monkeypatch.setattr(
        "met3r.adjacent16_cli.build_stress_tasks",
        lambda actual_contract, manifest, references: (),
    )
    monkeypatch.setattr(
        "met3r.adjacent16_cli.run_stress_tasks",
        lambda tasks, **kwargs: {"accepted": 0, "rejected": 0, "tasks": {}},
    )
    monkeypatch.setattr("met3r.adjacent16_cli.prepare_adjacent16_run", fake_prepare)

    assert main(["prepare-stress", str(config), "--run-root", str(run_root)]) == 0
    assert main(["prepare", str(config), "--output-root", str(run_root)]) == 0

    assert seen_stress_roots == [expected_stress_root, expected_stress_root]
    assert json.loads((run_root / "config" / "run_paths.json").read_text()) == {
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "output_root": str(run_root.resolve()),
        "stress_root": str(expected_stress_root.resolve()),
    }


def test_invalid_initial_config_does_not_poison_corrected_stress_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from met3r.adjacent16_cli import main

    config = tmp_path / "config.json"
    config.write_text("invalid", encoding="utf-8")
    run_root = tmp_path / "run"
    contract = SimpleNamespace(
        output_root=run_root,
        stress_root=run_root / "inputs" / "stress",
        physical_gpus=(0, 1, 2, 3),
    )

    def load(path: Path) -> SimpleNamespace:
        if path.read_text(encoding="utf-8") == "invalid":
            raise ValueError("invalid frozen contract")
        return contract

    monkeypatch.setattr("met3r.adjacent16_cli.load_adjacent16_contract", load)
    monkeypatch.setattr(
        "met3r.adjacent16_cli.replace",
        lambda value, **changes: SimpleNamespace(**{**vars(value), **changes}),
    )
    monkeypatch.setattr(
        "met3r.adjacent16_cli.build_adjacent16_manifest",
        lambda actual, require_stress: object(),
    )
    monkeypatch.setattr(
        "met3r.adjacent16_cli.render_stress_references",
        lambda actual, manifest: (),
    )
    monkeypatch.setattr(
        "met3r.adjacent16_cli.build_stress_tasks",
        lambda actual, manifest, references: (),
    )
    monkeypatch.setattr(
        "met3r.adjacent16_cli.run_stress_tasks",
        lambda tasks, **kwargs: {"accepted": 0, "rejected": 0, "tasks": {}},
    )

    with pytest.raises(ValueError, match="invalid frozen contract"):
        main(["prepare-stress", str(config), "--run-root", str(run_root)])
    assert not (run_root / "config" / "run_paths.json").exists()

    config.write_text("corrected", encoding="utf-8")
    assert main(["prepare-stress", str(config), "--run-root", str(run_root)]) == 0
    assert (
        json.loads((run_root / "config" / "run_paths.json").read_text())[
            "config_sha256"
        ]
        == hashlib.sha256(config.read_bytes()).hexdigest()
    )


def _legacy_run(
    run_root: Path,
    *,
    consistent: bool,
    with_hash: bool = True,
) -> SimpleNamespace:
    config = run_root / "config" / "frozen_config.json"
    config.parent.mkdir(parents=True)
    config.write_bytes(b'{"legacy":"adjacent16"}\n')
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    if with_hash:
        (config.parent / "frozen_config.sha256").write_text(
            f"{digest}\n",
            encoding="utf-8",
        )
    stress_root = run_root / "inputs" / "stress"
    reconstruction_root = run_root / "reconstruction"
    (run_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "20260727_adjacent16_core_v1",
                "config_sha256": digest if consistent else "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    manifests = run_root / "manifests"
    manifests.mkdir()
    views = [
        {
            "shoe_id": "shoe-a",
            "camera_dataset_id": "Camera A",
            "quartet_id": "q0",
            "view_id": view_id,
        }
        for view_id in ("0", "1", "2", "3")
    ]
    (manifests / "adjacent16_manifest.json").write_text(
        json.dumps(
            {
                "reconstruction_root": str(reconstruction_root),
                "stress_root": str(
                    stress_root if consistent else run_root / "wrong-stress"
                ),
                "shoes": [
                    {
                        "shoe_id": "shoe-a",
                        "camera_dataset_id": "Camera A",
                        "views": views,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        experiment_id="20260727_adjacent16_core_v1",
        output_root=run_root.resolve(),
        stress_root=stress_root.resolve(),
        reconstruction_root=reconstruction_root.resolve(),
        physical_gpus=(0, 1, 2, 3),
        raw={"shoes": {"shoe-a": "Camera A"}},
        quartets=(SimpleNamespace(quartet_id="q0", view_ids=("0", "1", "2", "3")),),
    )


def _stub_stress_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    seen_contracts: list[SimpleNamespace],
) -> None:
    monkeypatch.setattr(
        "met3r.adjacent16_cli.replace",
        lambda value, **changes: SimpleNamespace(**{**vars(value), **changes}),
    )

    def manifest(contract: SimpleNamespace, *, require_stress: bool) -> object:
        assert not require_stress
        seen_contracts.append(contract)
        return object()

    monkeypatch.setattr(
        "met3r.adjacent16_cli.build_adjacent16_manifest",
        manifest,
    )
    monkeypatch.setattr(
        "met3r.adjacent16_cli.render_stress_references",
        lambda contract, manifest: (),
    )
    monkeypatch.setattr(
        "met3r.adjacent16_cli.build_stress_tasks",
        lambda contract, manifest, references: (),
    )
    monkeypatch.setattr(
        "met3r.adjacent16_cli.run_stress_tasks",
        lambda tasks, **kwargs: {"accepted": 0, "rejected": 0, "tasks": {}},
    )


def test_prepare_stress_refuses_true_prefix_run_missing_authoritative_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from met3r.adjacent16_cli import main

    run_root = tmp_path / "legacy-run"
    contract = _legacy_run(run_root, consistent=True, with_hash=False)
    requested = tmp_path / "requested.json"
    requested.write_text("valid requested config", encoding="utf-8")
    seen: list[SimpleNamespace] = []
    monkeypatch.setattr(
        "met3r.adjacent16_cli.load_adjacent16_contract",
        lambda path: contract,
    )
    _stub_stress_pipeline(monkeypatch, seen)

    with pytest.raises(ValueError, match="legacy run-path recovery failed"):
        main(["prepare-stress", str(requested), "--run-root", str(run_root)])

    assert not (run_root / "config" / "run_paths.json").exists()
    assert not list((run_root / "config").glob(".run_paths.*.tmp"))
    assert seen == []


def test_requested_config_bytes_cannot_poison_true_prefix_frozen_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from met3r.adjacent16_cli import main

    run_root = tmp_path / "legacy-run"
    contract = _legacy_run(run_root, consistent=True, with_hash=False)
    frozen = run_root / "config" / "frozen_config.json"
    requested = tmp_path / "requested.json"
    requested.write_bytes(frozen.read_bytes() + b" ")
    requested_digest = hashlib.sha256(requested.read_bytes()).hexdigest()
    seen: list[SimpleNamespace] = []
    monkeypatch.setattr(
        "met3r.adjacent16_cli.load_adjacent16_contract",
        lambda path: contract,
    )
    _stub_stress_pipeline(monkeypatch, seen)

    with pytest.raises(ValueError, match="legacy run-path recovery failed"):
        main(["prepare-stress", str(requested), "--run-root", str(run_root)])

    assert hashlib.sha256(frozen.read_bytes()).hexdigest() != requested_digest
    assert not (run_root / "config" / "run_paths.json").exists()
    assert seen == []


def test_prepare_stress_migrates_authoritative_run_without_requested_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from met3r.adjacent16_cli import main

    run_root = tmp_path / "legacy-run"
    frozen_contract = _legacy_run(run_root, consistent=True, with_hash=True)
    frozen = run_root / "config" / "frozen_config.json"
    requested = tmp_path / "requested.json"
    requested.write_bytes(frozen.read_bytes() + b" ")
    requested_contract = SimpleNamespace(**vars(frozen_contract))
    requested_contract.authority = "requested"
    frozen_contract.authority = "frozen"
    seen: list[SimpleNamespace] = []

    def load(path: Path) -> SimpleNamespace:
        return frozen_contract if path.resolve() == frozen.resolve() else requested_contract

    monkeypatch.setattr("met3r.adjacent16_cli.load_adjacent16_contract", load)
    _stub_stress_pipeline(monkeypatch, seen)

    assert main(
        ["prepare-stress", str(requested), "--run-root", str(run_root)]
    ) == 0

    sidecar = json.loads(
        (run_root / "config" / "run_paths.json").read_text(encoding="utf-8")
    )
    assert sidecar["config_sha256"] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert sidecar["config_sha256"] != hashlib.sha256(requested.read_bytes()).hexdigest()
    assert [contract.authority for contract in seen] == ["frozen"]


def test_production_shaped_legacy_run_atomically_migrates_before_later_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from met3r.adjacent16_cli import main

    run_root = tmp_path / "legacy-run"
    contract = _legacy_run(run_root, consistent=True)
    calls: list[Path] = []
    monkeypatch.setattr(
        "met3r.adjacent16_cli.load_adjacent16_contract",
        lambda path: contract,
    )
    monkeypatch.setattr(
        "met3r.adjacent16_cli.run_feature_worker",
        lambda root, backbone, device: calls.append(Path(root))
        or WorkerSummary(completed=0, abstained=0, failed=0),
    )

    assert main(
        [
            "feature-worker",
            str(run_root),
            "--backbone",
            "dino_vitb16",
            "--device",
            "cuda:0",
        ]
    ) == 0

    sidecar = run_root / "config" / "run_paths.json"
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {
        "config_sha256": hashlib.sha256(
            (run_root / "config" / "frozen_config.json").read_bytes()
        ).hexdigest(),
        "output_root": str(run_root.resolve()),
        "stress_root": str((run_root / "inputs" / "stress").resolve()),
    }
    assert not list(sidecar.parent.glob(".run_paths.*.tmp"))
    assert calls == [run_root.resolve()]


def test_inconsistent_legacy_metadata_fails_recovery_without_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from met3r.adjacent16_cli import main

    run_root = tmp_path / "legacy-run"
    contract = _legacy_run(run_root, consistent=False)
    calls: list[str] = []
    monkeypatch.setattr(
        "met3r.adjacent16_cli.load_adjacent16_contract",
        lambda path: contract,
    )
    monkeypatch.setattr(
        "met3r.adjacent16_cli.run_feature_worker",
        lambda *args: calls.append("worker"),
    )

    with pytest.raises(ValueError, match="legacy run-path recovery failed"):
        main(
            [
                "feature-worker",
                str(run_root),
                "--backbone",
                "dino_vitb16",
                "--device",
                "cuda:0",
            ]
        )

    assert not (run_root / "config" / "run_paths.json").exists()
    assert not list((run_root / "config").glob(".run_paths.*.tmp"))
    assert calls == []


def test_run_path_sidecar_rejects_rerun_drift_before_stress_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from met3r.adjacent16_cli import main

    config = tmp_path / "config.json"
    config.write_text("frozen", encoding="utf-8")
    run_root = tmp_path / "run"
    sidecar = run_root / "config" / "run_paths.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(
        json.dumps(
            {
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "output_root": str(tmp_path / "other-run"),
                "stress_root": str(run_root / "inputs" / "stress"),
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []
    monkeypatch.setattr(
        "met3r.adjacent16_cli.load_adjacent16_contract",
        lambda path: SimpleNamespace(
            output_root=run_root,
            stress_root=run_root / "inputs" / "stress",
            physical_gpus=(0, 1, 2, 3),
        ),
    )
    monkeypatch.setattr(
        "met3r.adjacent16_cli.build_adjacent16_manifest",
        lambda *args, **kwargs: calls.append("stress-work"),
    )

    with pytest.raises(ValueError, match="persisted run paths differ"):
        main(["prepare-stress", str(config), "--run-root", str(run_root)])

    assert calls == []


def test_nondefault_physical_gpus_fail_before_cli_or_launcher_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from met3r.adjacent16_cli import main

    config = tmp_path / "config.json"
    config.write_text("unused", encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        "met3r.adjacent16_cli.load_adjacent16_contract",
        lambda path: calls.append("stress-work"),
    )

    with pytest.raises(SystemExit) as error:
        main(
            [
                "prepare-stress",
                str(config),
                "--run-root",
                str(tmp_path / "run"),
                "--physical-gpus",
                "4,5,6,7",
            ]
        )
    assert error.value.code == 2
    assert "physical GPUs must be exactly 0,1,2,3" in capsys.readouterr().err
    assert calls == []

    environment, _run_root, calls_path = _launcher_environment(tmp_path)
    environment["PHYSICAL_GPUS"] = "4,5,6,7"
    completed = _run_launcher("stress", environment, check=False)
    assert completed.returncode == 2
    assert "PHYSICAL_GPUS must be exactly 0,1,2,3" in completed.stderr
    assert _calls(calls_path) == []


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


@pytest.mark.parametrize("state", ("pending", "abstained"))
def test_adjacent_status_requires_the_exact_completed_graph(
    tmp_path: Path,
    state: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from met3r.adjacent16_cli import main
    from met3r.job_registry import JobDefinition, JobRegistry

    registry = JobRegistry(tmp_path / "registry" / "jobs.sqlite")
    registry.register(JobDefinition("feature-one", "feature", {"backbone": "fake"}))
    if state == "abstained":
        with sqlite3.connect(registry.path) as connection:
            connection.execute(
                "UPDATE jobs SET state = 'abstained', error = 'explained' "
                "WHERE job_id = 'feature-one'"
            )

    assert main(["status", str(tmp_path), "--require-complete"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["complete"] is False
    assert payload["counts"] == {"feature": {state: 1}}


def test_adjacent_status_revalidates_completed_output_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from met3r.adjacent16_cli import main
    from met3r.job_registry import JobRegistry

    registry = JobRegistry(tmp_path / "registry" / "jobs.sqlite")
    payload_json = "{}"
    payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
    missing_manifest = json.dumps(
        {
            "files": [
                {
                    "path": str(tmp_path / "missing.parquet"),
                    "sha256": "0" * 64,
                }
            ]
        }
    )
    feature_ids = [f"feature-{index:04d}" for index in range(3472)]
    geometry_ids = [f"geometry-{index:02d}" for index in range(32)]
    with sqlite3.connect(registry.path) as connection:
        connection.executemany(
            """
            INSERT INTO jobs(
                job_id, phase, state, payload_json, payload_hash,
                output_manifest_json
            ) VALUES (?, ?, 'completed', ?, ?, ?)
            """,
            [
                (
                    job_id,
                    "feature" if job_id.startswith("feature") else "geometry",
                    payload_json,
                    payload_hash,
                    missing_manifest,
                )
                for job_id in (*feature_ids, *geometry_ids)
            ],
        )
    monkeypatch.setattr(
        "met3r.adjacent16_cli._load_contract",
        lambda root: (object(), "f" * 64),
    )
    monkeypatch.setattr(
        "met3r.adjacent16_cli._registry_rows",
        lambda root: (SimpleNamespace(job_id=feature_ids[0]),),
    )
    monkeypatch.setattr(
        "met3r.adjacent16_cli._validate_registry_graph",
        lambda *args: True,
    )

    assert main(["status", str(tmp_path), "--require-complete"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["complete"] is False
    assert payload["counts"]["feature"] == {"completed": 3471, "pending": 1}


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
