"""Command line orchestration for the frozen adjacent 16-view experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from . import adjacent16_gate as _gate_module
from .adjacent16_analysis import analyze_adjacent16_run
from .adjacent16_contract import Adjacent16Contract, load_adjacent16_contract
from .adjacent16_data import Adjacent16Manifest, build_adjacent16_manifest
from .adjacent16_design import Adjacent16Design
from .adjacent16_gate import (
    _load_contract,
    _registry_rows,
    _validate_registry_graph,
    _frozen_design,
    validate_adjacent16_final_gate,
    validate_adjacent16_geometry_preflight,
    validate_timed_runtime_projection,
    write_timed_sentinel_manifest,
)
from .adjacent16_prepare import prepare_adjacent16_run
from .adjacent16_stress import (
    build_stress_tasks,
    render_stress_references,
    run_stress_tasks,
)
from .adjacent16_visualization import render_adjacent16_audit
from .pilot_gate import (
    aggregate_all_metric_parts,
    validate_adjacent16_metric_ownership,
)
from .job_registry import JobRegistry
from .upper_geometry import run_geometry_worker
from .upper_metric_worker import run_feature_worker


def _resolved(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _gpu_indices(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "physical GPUs must be comma-separated integers"
        ) from error
    if result != (0, 1, 2, 3):
        raise argparse.ArgumentTypeError(
            "physical GPUs must be exactly 0,1,2,3"
        )
    return result


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _print_json(value: object) -> None:
    print(json.dumps(_jsonable(value), sort_keys=True))


def _manifest_summary(manifest: Adjacent16Manifest) -> dict[str, int]:
    selected_views = sum(len(shoe.views) for shoe in manifest.shoes)
    return {
        "all_source_views": len(manifest.shoes) * 48,
        "selected_views": selected_views,
        "selected_pair_units": selected_views * 3 // 2,
    }


def _run_paths_payload(config_path: Path, run_root: Path) -> dict[str, str]:
    root = run_root.expanduser().resolve()
    return {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "output_root": str(root),
        "stress_root": str((root / "inputs" / "stress").resolve()),
    }


def _publish_run_paths(
    run_root: Path,
    payload: Mapping[str, str],
) -> Mapping[str, str]:
    destination = run_root.expanduser().resolve() / "config" / "run_paths.json"
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("persisted run paths are unreadable") from error
        if existing != payload:
            raise ValueError("persisted run paths differ from requested run")
        return payload
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".run_paths.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing != payload:
                raise ValueError("persisted run paths differ from requested run")
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload


def _persist_run_paths(config_path: Path, run_root: Path) -> Mapping[str, str]:
    return _publish_run_paths(
        run_root,
        _run_paths_payload(config_path, run_root),
    )


def _persist_frozen_config_hash(run_root: Path, digest: str) -> None:
    destination = run_root / "config" / "frozen_config.sha256"
    content = f"{digest}\n".encode("ascii")
    if destination.exists():
        if destination.read_bytes() != content:
            raise ValueError("persisted frozen config hash differs")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".frozen_config.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != content:
                raise ValueError("persisted frozen config hash differs")
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _legacy_run_paths(run_root: Path) -> Mapping[str, str]:
    root = run_root.expanduser().resolve()
    frozen = root / "config" / "frozen_config.json"
    hash_path = root / "config" / "frozen_config.sha256"
    run_manifest_path = root / "run_manifest.json"
    source_manifest_path = root / "manifests" / "adjacent16_manifest.json"
    try:
        content = frozen.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if hash_path.read_text(encoding="ascii") != f"{digest}\n":
            raise ValueError("frozen config hash file differs")
        contract = load_adjacent16_contract(frozen)
        expected_stress = (root / "inputs" / "stress").resolve()
        if (
            Path(contract.output_root).expanduser().resolve() != root
            or Path(contract.stress_root).expanduser().resolve() != expected_stress
        ):
            raise ValueError("frozen config roots differ from legacy run root")
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(run_manifest, Mapping)
            or run_manifest.get("experiment_id") != contract.experiment_id
            or run_manifest.get("config_sha256") != digest
        ):
            raise ValueError("legacy run manifest differs from frozen config")
        source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        expected_shoes = contract.raw.get("shoes")
        if (
            not isinstance(source, Mapping)
            or frozenset(source) != frozenset(
                {"reconstruction_root", "stress_root", "shoes"}
            )
            or source.get("reconstruction_root")
            != str(Path(contract.reconstruction_root).expanduser().resolve())
            or source.get("stress_root") != str(expected_stress)
            or not isinstance(expected_shoes, Mapping)
            or not isinstance(source.get("shoes"), list)
            or len(source["shoes"]) != len(expected_shoes)
        ):
            raise ValueError("legacy source manifest roots or shoes differ")
        expected_views = tuple(
            (quartet.quartet_id, view_id)
            for quartet in contract.quartets
            for view_id in quartet.view_ids
        )
        for shoe, (shoe_id, camera_dataset_id) in zip(
            source["shoes"],
            expected_shoes.items(),
            strict=True,
        ):
            if (
                not isinstance(shoe, Mapping)
                or shoe.get("shoe_id") != shoe_id
                or shoe.get("camera_dataset_id") != camera_dataset_id
                or not isinstance(shoe.get("views"), list)
                or len(shoe["views"]) != len(expected_views)
            ):
                raise ValueError("legacy source shoe metadata differs")
            for view, (quartet_id, view_id) in zip(
                shoe["views"],
                expected_views,
                strict=True,
            ):
                if (
                    not isinstance(view, Mapping)
                    or view.get("shoe_id") != shoe_id
                    or view.get("camera_dataset_id") != camera_dataset_id
                    or view.get("quartet_id") != quartet_id
                    or view.get("view_id") != view_id
                ):
                    raise ValueError("legacy source view metadata differs")
        payload = {
            "config_sha256": digest,
            "output_root": str(root),
            "stress_root": str(expected_stress),
        }
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError(
            "legacy run-path recovery failed; restore authoritative frozen "
            "config/hash and prepared run/source metadata"
        ) from error
    return _publish_run_paths(root, payload)


def _validated_run_paths(run_root: Path) -> Mapping[str, str] | None:
    root = run_root.expanduser().resolve()
    destination = root / "config" / "run_paths.json"
    frozen = root / "config" / "frozen_config.json"
    if not destination.is_file():
        if frozen.is_file():
            return _legacy_run_paths(root)
        return None
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("persisted run paths are unreadable") from error
    expected_paths = {
        "output_root": str(root),
        "stress_root": str((root / "inputs" / "stress").resolve()),
    }
    if (
        not isinstance(payload, Mapping)
        or frozenset(payload)
        != frozenset({"config_sha256", "output_root", "stress_root"})
        or any(payload.get(key) != value for key, value in expected_paths.items())
        or not isinstance(payload.get("config_sha256"), str)
    ):
        raise ValueError("persisted run paths differ from requested run")
    if (
        frozen.is_file()
        and hashlib.sha256(frozen.read_bytes()).hexdigest()
        != payload["config_sha256"]
    ):
        raise ValueError("persisted run paths differ from frozen config")
    source_manifest = root / "manifests" / "adjacent16_manifest.json"
    if source_manifest.is_file():
        try:
            source = json.loads(source_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("persisted adjacent source manifest is unreadable") from error
        if (
            not isinstance(source, Mapping)
            or source.get("stress_root") != payload["stress_root"]
        ):
            raise ValueError("persisted stress root differs from run paths")
    return payload


@contextmanager
def _effective_gate_paths(run_root: Path) -> Iterator[None]:
    payload = _validated_run_paths(run_root)
    if payload is None:
        yield
        return
    original = _gate_module.load_adjacent16_contract

    def load_effective(path: str | Path) -> Adjacent16Contract:
        contract = original(path)
        return replace(
            contract,
            output_root=Path(payload["output_root"]),
            stress_root=Path(payload["stress_root"]),
        )

    _gate_module.load_adjacent16_contract = load_effective
    try:
        yield
    finally:
        _gate_module.load_adjacent16_contract = original


def _validate_prepared_paths(
    config_path: Path,
    run_root: Path,
    prepared_config_sha256: str,
) -> None:
    payload = _run_paths_payload(config_path, run_root)
    frozen = run_root / "config" / "frozen_config.json"
    if (
        not frozen.is_file()
        or frozen.read_bytes() != config_path.read_bytes()
        or prepared_config_sha256 != payload["config_sha256"]
    ):
        raise ValueError("persisted frozen config differs from requested config")
    try:
        run_manifest = json.loads(
            (run_root / "run_manifest.json").read_text(encoding="utf-8")
        )
        source_manifest = json.loads(
            (run_root / "manifests" / "adjacent16_manifest.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("persisted adjacent run metadata is unreadable") from error
    if (
        not isinstance(run_manifest, Mapping)
        or run_manifest.get("config_sha256") != payload["config_sha256"]
        or not isinstance(source_manifest, Mapping)
        or source_manifest.get("stress_root") != payload["stress_root"]
    ):
        raise ValueError("persisted adjacent run metadata differs from run paths")
    _persist_frozen_config_hash(run_root, payload["config_sha256"])
    _publish_run_paths(run_root, payload)


def _registry_counts(path: Path) -> dict[str, dict[str, int]]:
    with sqlite3.connect(path, timeout=60) as connection:
        rows = connection.execute(
            """
            SELECT phase, state, COUNT(*)
            FROM jobs
            GROUP BY phase, state
            ORDER BY phase, state
            """
        ).fetchall()
    counts: dict[str, dict[str, int]] = {}
    for phase, state, count in rows:
        counts.setdefault(str(phase), {})[str(state)] = int(count)
    return counts


def _adjacent_status(run_root: Path) -> tuple[dict[str, dict[str, int]], bool]:
    registry_path = run_root / "registry" / "jobs.sqlite"
    if not registry_path.is_file():
        raise ValueError("missing adjacent16 job registry")
    counts = _registry_counts(registry_path)
    expected = {
        "feature": {"completed": 3472},
        "geometry": {"completed": 32},
    }
    if counts != expected:
        return counts, False
    try:
        contract, config_sha256 = _load_contract(run_root)
        rows = _registry_rows(run_root)
        if not _validate_registry_graph(
            run_root,
            contract,
            config_sha256,
            rows,
        ):
            return counts, False
        registry = JobRegistry(registry_path)
        complete = all(
            registry.validated_state(row.job_id) == "completed"
            for row in rows
        )
    except (OSError, ValueError, KeyError, sqlite3.Error):
        complete = False
    refreshed = _registry_counts(registry_path)
    return refreshed, complete and refreshed == expected


def _load_adjacent16_design(run_root: str | Path) -> Adjacent16Design:
    root = Path(run_root).expanduser().resolve()
    contract = load_adjacent16_contract(root / "config" / "frozen_config.json")
    paths = _validated_run_paths(root)
    if paths is not None:
        contract = replace(
            contract,
            output_root=Path(paths["output_root"]),
            stress_root=Path(paths["stress_root"]),
        )
    return _frozen_design(root, contract)


def _job_ids(path: Path, backbone: str) -> tuple[str, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load feature job-ID file: {path}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("backbone") != backbone
        or not isinstance(payload.get("config_sha256"), str)
        or not isinstance(payload.get("job_ids"), list)
        or not payload["job_ids"]
        or any(not isinstance(item, str) or not item for item in payload["job_ids"])
        or len(set(payload["job_ids"])) != len(payload["job_ids"])
    ):
        raise ValueError("feature job-ID file is malformed or for another backbone")
    return tuple(payload["job_ids"])


def _prepare_stress(
    config_path: Path,
    *,
    run_root: Path | None,
    physical_gpus: tuple[int, ...] | None,
    max_parallel: int,
    max_outer_attempts: int,
) -> Mapping[str, object]:
    contract: Adjacent16Contract = load_adjacent16_contract(config_path)
    effective_root = Path(
        run_root if run_root is not None else contract.output_root
    ).expanduser().resolve()
    _persist_run_paths(config_path, effective_root)
    contract = replace(
        contract,
        output_root=effective_root,
        stress_root=effective_root / "inputs" / "stress",
    )
    if physical_gpus is not None:
        contract = replace(contract, physical_gpus=physical_gpus)
    manifest = build_adjacent16_manifest(contract, require_stress=False)
    references = render_stress_references(contract, manifest)
    tasks = build_stress_tasks(contract, manifest, references)
    return run_stress_tasks(
        tasks,
        max_parallel=max_parallel,
        max_outer_attempts=max_outer_attempts,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="met3r-adjacent16",
        description="Resumable frozen adjacent 16-view fidelity experiment",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate_inputs = commands.add_parser(
        "validate-inputs",
        help="validate the frozen source and optional stress inputs",
    )
    validate_inputs.add_argument("config", type=_resolved)
    validate_inputs.add_argument("--require-stress", action="store_true")

    prepare_stress = commands.add_parser(
        "prepare-stress",
        help="render references and generate audited stress inputs",
    )
    prepare_stress.add_argument("config", type=_resolved)
    prepare_stress.add_argument("--run-root", type=_resolved)
    prepare_stress.add_argument("--physical-gpus", type=_gpu_indices)
    prepare_stress.add_argument("--max-parallel", type=int, default=4)
    prepare_stress.add_argument("--max-outer-attempts", type=int, default=3)

    prepare = commands.add_parser("prepare", help="prepare the immutable run tree")
    prepare.add_argument("config", type=_resolved)
    prepare.add_argument("--output-root", type=_resolved)

    geometry = commands.add_parser(
        "geometry-worker",
        help="run one shoe's resumable DA3 geometry jobs",
    )
    geometry.add_argument("run_root", type=_resolved)
    geometry.add_argument("--shoe-id", required=True)
    geometry.add_argument("--device", required=True)
    geometry.add_argument("--da3-repository", type=_resolved, required=True)

    geometry_preflight = commands.add_parser(
        "geometry-preflight",
        help="persist the frozen geometry preflight decision",
    )
    geometry_preflight.add_argument("run_root", type=_resolved)

    sentinel = commands.add_parser(
        "sentinel-manifest",
        help="write the exact timed sentinel job manifests",
    )
    sentinel.add_argument("run_root", type=_resolved)

    feature = commands.add_parser(
        "feature-worker",
        help="run one backbone's pending or selected feature jobs",
    )
    feature.add_argument("run_root", type=_resolved)
    feature.add_argument("--backbone", required=True)
    feature.add_argument("--device", required=True)
    feature.add_argument("--job-id-file", type=_resolved)

    runtime = commands.add_parser(
        "runtime-gate",
        help="persist the frozen timed runtime projection decision",
    )
    runtime.add_argument("run_root", type=_resolved)

    aggregate = commands.add_parser(
        "aggregate",
        help="ownership-validate and atomically publish all metric tracks",
    )
    aggregate.add_argument("run_root", type=_resolved)

    final_validate = commands.add_parser(
        "final-validate",
        help="ownership-validate and evaluate the final frozen gate",
    )
    final_validate.add_argument("run_root", type=_resolved)

    analyze = commands.add_parser(
        "analyze",
        help="publish the frozen adjacent-view analysis",
    )
    analyze.add_argument("run_root", type=_resolved)
    analyze.add_argument("--bootstrap-repeats", type=int, default=10_000)
    analyze.add_argument("--seed", type=int, default=42)

    visualize = commands.add_parser(
        "visualize",
        help="render the posthoc PCA audit PNGs",
    )
    visualize.add_argument("run_root", type=_resolved)
    visualize.add_argument("--device", required=True)

    status = commands.add_parser("status", help="show resumable registry counts")
    status.add_argument("run_root", type=_resolved)
    status.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-inputs":
        contract = load_adjacent16_contract(args.config)
        manifest = build_adjacent16_manifest(
            contract,
            require_stress=args.require_stress,
        )
        _print_json(_manifest_summary(manifest))
        return 0
    if args.command == "prepare-stress":
        result = _prepare_stress(
            args.config,
            run_root=args.run_root,
            physical_gpus=args.physical_gpus,
            max_parallel=args.max_parallel,
            max_outer_attempts=args.max_outer_attempts,
        )
        _print_json(result)
        return 0 if result.get("rejected") == 0 else 1
    if args.command == "prepare":
        contract = load_adjacent16_contract(args.config)
        run_root = Path(
            args.output_root
            if args.output_root is not None
            else contract.output_root
        ).expanduser().resolve()
        paths = _run_paths_payload(args.config, run_root)
        if (run_root / "config" / "run_paths.json").exists():
            _publish_run_paths(run_root, paths)
        prepared = prepare_adjacent16_run(
            args.config,
            output_root=run_root,
            stress_root=Path(paths["stress_root"]),
        )
        if prepared.root != run_root:
            raise ValueError("prepared run root differs from requested run root")
        _validate_prepared_paths(
            args.config,
            run_root,
            prepared.config_sha256,
        )
        _print_json(
            {
                "run_root": prepared.root,
                "config_sha256": prepared.config_sha256,
            }
        )
        return 0
    if args.command == "geometry-worker":
        _validated_run_paths(args.run_root)
        summary = run_geometry_worker(
            args.run_root,
            args.shoe_id,
            args.device,
            args.da3_repository,
        )
        _print_json(summary)
        return 0 if summary.failed == 0 else 1
    if args.command == "geometry-preflight":
        with _effective_gate_paths(args.run_root):
            report = validate_adjacent16_geometry_preflight(args.run_root)
        _print_json(report)
        return 0 if report.passed else 1
    if args.command == "sentinel-manifest":
        with _effective_gate_paths(args.run_root):
            path = write_timed_sentinel_manifest(args.run_root)
        _print_json({"manifest_root": path})
        return 0
    if args.command == "feature-worker":
        _validated_run_paths(args.run_root)
        job_ids = (
            _job_ids(args.job_id_file, args.backbone)
            if args.job_id_file is not None
            else None
        )
        if job_ids is None:
            summary = run_feature_worker(
                args.run_root,
                args.backbone,
                args.device,
            )
        else:
            summary = run_feature_worker(
                args.run_root,
                args.backbone,
                args.device,
                job_ids=job_ids,
            )
        _print_json(summary)
        return 0 if summary.failed == 0 else 1
    if args.command == "runtime-gate":
        with _effective_gate_paths(args.run_root):
            report = validate_timed_runtime_projection(args.run_root)
        _print_json(report)
        return 0 if report.passed else 1
    if args.command == "aggregate":
        _validated_run_paths(args.run_root)
        design = _load_adjacent16_design(args.run_root)
        validate_adjacent16_metric_ownership(args.run_root, design)
        _print_json(aggregate_all_metric_parts(args.run_root))
        return 0
    if args.command == "final-validate":
        design = _load_adjacent16_design(args.run_root)
        validate_adjacent16_metric_ownership(args.run_root, design)
        with _effective_gate_paths(args.run_root):
            report = validate_adjacent16_final_gate(args.run_root)
        _print_json(report)
        return 0 if report.passed else 1
    if args.command == "analyze":
        _validated_run_paths(args.run_root)
        outputs = analyze_adjacent16_run(
            args.run_root,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed,
        )
        _print_json(outputs)
        return 0
    if args.command == "visualize":
        _validated_run_paths(args.run_root)
        _print_json(render_adjacent16_audit(args.run_root, device=args.device))
        return 0
    if args.command == "status":
        with _effective_gate_paths(args.run_root):
            counts, complete = _adjacent_status(args.run_root)
        _print_json({"complete": complete, "counts": counts})
        if args.require_complete and not complete:
            return 1
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
