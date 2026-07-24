from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from met3r.job_registry import JobDefinition, JobRegistry, sha256_file


def test_claim_is_transactional_and_unique_across_connections(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite"
    registry = JobRegistry(path)
    for index in range(20):
        registry.register(
            JobDefinition(f"job-{index:02d}", "geometry", {"shoe_id": "shoe-a"})
        )

    def claim(index: int) -> str | None:
        claimed = JobRegistry(path).claim_next(
            f"worker-{index}", "geometry", {"shoe_id": "shoe-a"}
        )
        return claimed.job_id if claimed is not None else None

    with ThreadPoolExecutor(max_workers=8) as executor:
        claimed = list(executor.map(claim, range(32)))

    job_ids = [job_id for job_id in claimed if job_id is not None]
    assert len(job_ids) == 20
    assert len(set(job_ids)) == 20


def test_registration_is_idempotent_but_rejects_job_id_redefinition(
    tmp_path: Path,
) -> None:
    registry = JobRegistry(tmp_path / "jobs.sqlite")
    original = JobDefinition("a", "feature", {"shoe_id": "shoe-a", "n": 1})

    registry.register(original)
    registry.register(original)

    assert registry.counts_by_phase() == {"feature": 1}
    with pytest.raises(ValueError, match="different definition"):
        registry.register(
            JobDefinition("a", "feature", {"shoe_id": "shoe-a", "n": 2})
        )


def test_retryable_failure_stops_after_three_claim_attempts(tmp_path: Path) -> None:
    registry = JobRegistry(tmp_path / "jobs.sqlite")
    registry.register(JobDefinition("a", "geometry", {}))

    for attempt in range(1, 4):
        claimed = registry.claim_next("worker", "geometry", {})
        assert claimed is not None and claimed.attempts == attempt
        registry.fail("a", f"attempt {attempt}", retryable=True)

    assert registry.validated_state("a") == "failed"
    assert registry.claim_next("worker", "geometry", {}) is None


def test_completed_job_requires_valid_output_hash(tmp_path: Path) -> None:
    output = tmp_path / "part.parquet"
    output.write_bytes(b"metric")
    registry = JobRegistry(tmp_path / "jobs.sqlite")
    registry.register(JobDefinition("a", "feature", {}))
    registry.claim_next("worker", "feature", {})
    registry.complete(
        "a",
        {"files": [{"path": str(output), "sha256": sha256_file(output)}]},
    )
    assert registry.validated_state("a") == "completed"

    output.write_bytes(b"changed")

    assert registry.validated_state("a") == "pending"
    assert registry.claim_next("replacement", "feature", {}).attempts == 1


def test_invalid_third_attempt_completion_resets_attempt_budget(
    tmp_path: Path,
) -> None:
    output = tmp_path / "part.parquet"
    output.write_bytes(b"metric")
    registry = JobRegistry(tmp_path / "jobs.sqlite")
    registry.register(JobDefinition("a", "feature", {}))
    for attempt in (1, 2):
        claimed = registry.claim_next("worker", "feature", {})
        assert claimed is not None and claimed.attempts == attempt
        registry.fail("a", "transient", retryable=True)
    claimed = registry.claim_next("worker", "feature", {})
    assert claimed is not None and claimed.attempts == 3
    registry.complete(
        "a",
        {"files": [{"path": str(output), "sha256": sha256_file(output)}]},
    )
    output.write_bytes(b"corrupted")

    assert registry.validated_state("a") == "pending"
    replacement = registry.claim_next("replacement", "feature", {})
    assert replacement is not None
    assert replacement.attempts == 1


def test_complete_rejects_missing_or_mismatched_outputs(tmp_path: Path) -> None:
    output = tmp_path / "part.parquet"
    output.write_bytes(b"metric")
    registry = JobRegistry(tmp_path / "jobs.sqlite")
    registry.register(JobDefinition("a", "feature", {}))
    registry.claim_next("worker", "feature", {})

    with pytest.raises(ValueError, match="output"):
        registry.complete(
            "a",
            {"files": [{"path": str(output), "sha256": "0" * 64}]},
        )
    assert registry.validated_state("a") == "running"


def test_stale_requeue_is_phase_specific_and_respects_cutoff(tmp_path: Path) -> None:
    registry = JobRegistry(tmp_path / "jobs.sqlite")
    registry.register(JobDefinition("feature-old", "feature", {}))
    registry.register(JobDefinition("geometry-old", "geometry", {}))
    registry.claim_next("dead-feature", "feature", {})
    registry.claim_next("dead-geometry", "geometry", {})

    count = registry.requeue_stale(
        phase="feature",
        max_age_seconds=21_600,
        now=datetime.now(timezone.utc) + timedelta(hours=7),
    )

    assert count == 1
    replacement = registry.claim_next("replacement", "feature", {})
    assert replacement is not None and replacement.job_id == "feature-old"
    assert registry.claim_next("other", "geometry", {}) is None


def test_stale_requeue_requires_aware_utc_time_and_positive_age(tmp_path: Path) -> None:
    registry = JobRegistry(tmp_path / "jobs.sqlite")

    with pytest.raises(ValueError, match="UTC"):
        registry.requeue_stale(
            "feature", 21_600, datetime.now()
        )
    with pytest.raises(ValueError, match="positive"):
        registry.requeue_stale(
            "feature", 0, datetime.now(timezone.utc)
        )


def test_registry_enables_wal_and_foreign_keys(tmp_path: Path) -> None:
    registry = JobRegistry(tmp_path / "jobs.sqlite")

    pragmas = registry.sqlite_pragmas()

    assert pragmas["journal_mode"] == "wal"
    assert pragmas["foreign_keys"] == 1
