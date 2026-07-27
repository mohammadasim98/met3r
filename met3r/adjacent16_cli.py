"""Command line orchestration for the frozen adjacent 16-view experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from .adjacent16_analysis import analyze_adjacent16_run
from .adjacent16_contract import Adjacent16Contract, load_adjacent16_contract
from .adjacent16_data import Adjacent16Manifest, build_adjacent16_manifest
from .adjacent16_design import Adjacent16Design
from .adjacent16_gate import (
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
from .upper_cli import _status
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
    if len(result) != 4 or len(set(result)) != 4 or min(result) < 0:
        raise argparse.ArgumentTypeError(
            "adjacent16 requires four distinct nonnegative physical GPUs"
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


def _load_adjacent16_design(run_root: str | Path) -> Adjacent16Design:
    root = Path(run_root).expanduser().resolve()
    contract = load_adjacent16_contract(root / "config" / "frozen_config.json")
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
    if run_root is not None:
        contract = replace(
            contract,
            output_root=run_root,
            stress_root=run_root / "inputs" / "stress",
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
        prepared = prepare_adjacent16_run(
            args.config,
            output_root=args.output_root,
        )
        _print_json(
            {
                "run_root": prepared.root,
                "config_sha256": prepared.config_sha256,
            }
        )
        return 0
    if args.command == "geometry-worker":
        summary = run_geometry_worker(
            args.run_root,
            args.shoe_id,
            args.device,
            args.da3_repository,
        )
        _print_json(summary)
        return 0 if summary.failed == 0 else 1
    if args.command == "geometry-preflight":
        report = validate_adjacent16_geometry_preflight(args.run_root)
        _print_json(report)
        return 0 if report.passed else 1
    if args.command == "sentinel-manifest":
        path = write_timed_sentinel_manifest(args.run_root)
        _print_json({"manifest_root": path})
        return 0
    if args.command == "feature-worker":
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
        report = validate_timed_runtime_projection(args.run_root)
        _print_json(report)
        return 0 if report.passed else 1
    if args.command == "aggregate":
        design = _load_adjacent16_design(args.run_root)
        validate_adjacent16_metric_ownership(args.run_root, design)
        _print_json(aggregate_all_metric_parts(args.run_root))
        return 0
    if args.command == "final-validate":
        design = _load_adjacent16_design(args.run_root)
        validate_adjacent16_metric_ownership(args.run_root, design)
        report = validate_adjacent16_final_gate(args.run_root)
        _print_json(report)
        return 0 if report.passed else 1
    if args.command == "analyze":
        outputs = analyze_adjacent16_run(
            args.run_root,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed,
        )
        _print_json(outputs)
        return 0
    if args.command == "visualize":
        _print_json(render_adjacent16_audit(args.run_root, device=args.device))
        return 0
    if args.command == "status":
        counts, complete = _status(args.run_root)
        _print_json({"complete": complete, "counts": counts})
        if args.require_complete and not complete:
            return 1
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
