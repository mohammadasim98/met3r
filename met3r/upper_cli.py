"""Dedicated command line surface for the upper-view experiment."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .pilot_gate import (
    aggregate_all_metric_parts,
    validate_pilot,
)
from .upper_analysis import analyze_upper_run
from .job_registry import JobRegistry
from .upper_geometry import run_geometry_worker
from .upper_metric_worker import run_feature_worker
from .upper_prepare import prepare_run


def _resolved(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="met3r-upper-experiment",
        description="Resumable upper-view DINO fidelity experiment",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="prepare an immutable run tree")
    prepare.add_argument("config", type=_resolved)
    prepare.add_argument("--scope", choices=("pilot", "full"), required=True)

    geometry = commands.add_parser(
        "geometry-worker", help="run one shoe's long-lived DA3 worker"
    )
    geometry.add_argument("run_root", type=_resolved)
    geometry.add_argument("--shoe-id", required=True)
    geometry.add_argument("--device", required=True)
    geometry.add_argument("--da3-repository", type=_resolved, required=True)

    feature = commands.add_parser(
        "feature-worker", help="run one backbone's streaming feature worker"
    )
    feature.add_argument("run_root", type=_resolved)
    feature.add_argument("--backbone", required=True)
    feature.add_argument("--device", required=True)

    aggregate = commands.add_parser(
        "aggregate", help="atomically aggregate all registered metric parts"
    )
    aggregate.add_argument("run_root", type=_resolved)

    validate = commands.add_parser(
        "validate-pilot", help="evaluate and persist all hard pilot gates"
    )
    validate.add_argument("run_root", type=_resolved)
    validate.add_argument("--rerun-sentinel", action="store_true")

    status = commands.add_parser("status", help="show registry state counts")
    status.add_argument("run_root", type=_resolved)
    status.add_argument("--require-complete", action="store_true")

    analyze = commands.add_parser(
        "analyze", help="generate scale-safe endpoint tables, figures, and report"
    )
    analyze.add_argument("run_root", type=_resolved)
    return parser


def _status(run_root: Path) -> tuple[dict[str, dict[str, int]], bool]:
    registry_path = run_root / "registry" / "jobs.sqlite"
    registry = JobRegistry(registry_path)
    with sqlite3.connect(registry_path, timeout=60) as connection:
        completed_ids = [
            str(row[0])
            for row in connection.execute(
                "SELECT job_id FROM jobs WHERE state = 'completed' ORDER BY job_id"
            ).fetchall()
        ]
    for job_id in completed_ids:
        registry.validated_state(job_id)
    with sqlite3.connect(registry_path, timeout=60) as connection:
        unexplained_abstentions = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM jobs
                WHERE state = 'abstained'
                  AND (error IS NULL OR TRIM(error) = '')
                """
            ).fetchone()[0]
        )
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
    complete = unexplained_abstentions == 0 and bool(rows) and all(
        state in {"completed", "abstained"} for _, state, _ in rows
    )
    return counts, complete


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepared = prepare_run(args.config, args.scope)
        print(
            json.dumps(
                {
                    "run_root": str(prepared.root),
                    "scope": prepared.scope,
                    "config_sha256": prepared.config_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "geometry-worker":
        summary = run_geometry_worker(
            args.run_root,
            args.shoe_id,
            args.device,
            args.da3_repository,
        )
        print(json.dumps(asdict(summary), sort_keys=True))
        return 0 if summary.failed == 0 else 1
    if args.command == "feature-worker":
        summary = run_feature_worker(args.run_root, args.backbone, args.device)
        print(json.dumps(asdict(summary), sort_keys=True))
        return 0 if summary.failed == 0 else 1
    if args.command == "aggregate":
        outputs = aggregate_all_metric_parts(args.run_root)
        print(json.dumps([str(path) for path in outputs]))
        return 0
    if args.command == "validate-pilot":
        report = validate_pilot(
            args.run_root,
            rerun_sentinel=args.rerun_sentinel,
        )
        print(json.dumps(asdict(report), sort_keys=True))
        return 0 if report.passed else 1
    if args.command == "status":
        counts, complete = _status(args.run_root)
        print(json.dumps({"complete": complete, "counts": counts}, sort_keys=True))
        if args.require_complete and not complete:
            raise SystemExit(1)
        return 0
    if args.command == "analyze":
        outputs = analyze_upper_run(args.run_root)
        print(
            json.dumps(
                {
                    "tables": [str(path) for path in outputs.tables],
                    "figures": [str(path) for path in outputs.figures],
                    "report": str(outputs.report_path),
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
