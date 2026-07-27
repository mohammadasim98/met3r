"""SQLite-backed transactional job ownership for resumable experiment workers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  phase TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('pending','running','completed','abstained','failed')),
  payload_json TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  worker_id TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  started_at TEXT,
  finished_at TEXT,
  output_manifest_json TEXT,
  error TEXT
);
CREATE INDEX IF NOT EXISTS jobs_phase_state ON jobs(phase, state);
"""


@dataclass(frozen=True)
class JobDefinition:
    job_id: str
    phase: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class ClaimedJob:
    job_id: str
    phase: str
    payload: Mapping[str, object]
    attempts: int


@dataclass(frozen=True)
class WorkerSummary:
    completed: int
    abstained: int
    failed: int


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: Mapping[str, object]) -> str:
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("job payload must be JSON serializable") from error


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _validated_output_manifest(output_manifest: Mapping[str, object]) -> str:
    files = output_manifest.get("files")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)) or not files:
        raise ValueError("output manifest requires a non-empty files list")
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("output manifest file entries must be objects")
        raw_path = item.get("path")
        expected = item.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("output file path must be a non-empty string")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError("output sha256 must be a lowercase SHA-256 digest")
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"output hash validation failed: {path}")
    return _canonical_json(output_manifest)


class JobRegistry:
    """Authoritative job registry with one SQLite connection per operation."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def register(self, job: JobDefinition) -> None:
        if not job.job_id or not job.phase:
            raise ValueError("job_id and phase must be non-empty")
        payload_json = _canonical_json(job.payload)
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT phase, payload_json, payload_hash FROM jobs WHERE job_id = ?",
                    (job.job_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO jobs(job_id, phase, state, payload_json, payload_hash)
                        VALUES (?, ?, 'pending', ?, ?)
                        """,
                        (job.job_id, job.phase, payload_json, payload_hash),
                    )
                elif (
                    row["phase"] != job.phase
                    or row["payload_json"] != payload_json
                    or row["payload_hash"] != payload_hash
                ):
                    raise ValueError(
                        f"job_id {job.job_id!r} already has a different definition"
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def claim_next(
        self,
        worker_id: str,
        phase: str,
        filters: Mapping[str, str],
    ) -> ClaimedJob | None:
        if not worker_id or not phase:
            raise ValueError("worker_id and phase must be non-empty")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    """
                    SELECT job_id, phase, payload_json, attempts
                    FROM jobs
                    WHERE phase = ? AND state = 'pending' AND attempts < 3
                    ORDER BY job_id
                    """,
                    (phase,),
                ).fetchall()
                selected: sqlite3.Row | None = None
                payload: Mapping[str, object] | None = None
                for row in rows:
                    candidate = json.loads(row["payload_json"])
                    if all(candidate.get(key) == value for key, value in filters.items()):
                        selected = row
                        payload = candidate
                        break
                if selected is None or payload is None:
                    connection.commit()
                    return None
                attempts = int(selected["attempts"]) + 1
                updated = connection.execute(
                    """
                    UPDATE jobs
                    SET state = 'running', worker_id = ?, attempts = ?,
                        started_at = ?, finished_at = NULL, error = NULL
                    WHERE job_id = ? AND state = 'pending'
                    """,
                    (
                        worker_id,
                        attempts,
                        _utc_text(datetime.now(timezone.utc)),
                        selected["job_id"],
                    ),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("transactional job claim lost ownership")
                connection.commit()
                return ClaimedJob(
                    job_id=str(selected["job_id"]),
                    phase=str(selected["phase"]),
                    payload=payload,
                    attempts=attempts,
                )
            except Exception:
                connection.rollback()
                raise

    def claim_exact(
        self,
        worker_id: str,
        phase: str,
        job_id: str,
    ) -> ClaimedJob | None:
        """Transactionally claim one named pending job, or return ``None``."""

        if not worker_id or not phase or not job_id:
            raise ValueError("worker_id, phase, and job_id must be non-empty")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT job_id, phase, payload_json, attempts
                    FROM jobs
                    WHERE job_id = ? AND phase = ? AND state = 'pending'
                      AND attempts < 3
                    """,
                    (job_id, phase),
                ).fetchone()
                if row is None:
                    connection.commit()
                    return None
                attempts = int(row["attempts"]) + 1
                updated = connection.execute(
                    """
                    UPDATE jobs
                    SET state = 'running', worker_id = ?, attempts = ?,
                        started_at = ?, finished_at = NULL, error = NULL
                    WHERE job_id = ? AND phase = ? AND state = 'pending'
                    """,
                    (
                        worker_id,
                        attempts,
                        _utc_text(datetime.now(timezone.utc)),
                        job_id,
                        phase,
                    ),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("transactional exact job claim lost ownership")
                connection.commit()
                payload = json.loads(str(row["payload_json"]))
                if not isinstance(payload, Mapping):
                    raise ValueError("stored job payload is malformed")
                return ClaimedJob(job_id, phase, payload, attempts)
            except Exception:
                connection.rollback()
                raise

    def complete(
        self, job_id: str, output_manifest: Mapping[str, object]
    ) -> None:
        output_json = _validated_output_manifest(output_manifest)
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE jobs
                SET state = 'completed', finished_at = ?,
                    output_manifest_json = ?, error = NULL
                WHERE job_id = ? AND state = 'running'
                """,
                (_utc_text(datetime.now(timezone.utc)), output_json, job_id),
            )
            if updated.rowcount != 1:
                raise ValueError(f"job {job_id!r} is not running")

    def abstain(self, job_id: str, error: str) -> None:
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE jobs
                SET state = 'abstained', finished_at = ?, error = ?
                WHERE job_id = ? AND state = 'running'
                """,
                (_utc_text(datetime.now(timezone.utc)), error, job_id),
            )
            if updated.rowcount != 1:
                raise ValueError(f"job {job_id!r} is not running")

    def fail(self, job_id: str, error: str, retryable: bool) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT attempts FROM jobs WHERE job_id = ? AND state = 'running'",
                    (job_id,),
                ).fetchone()
                if row is None:
                    raise ValueError(f"job {job_id!r} is not running")
                pending = retryable and int(row["attempts"]) < 3
                connection.execute(
                    """
                    UPDATE jobs
                    SET state = ?, worker_id = NULL, started_at = NULL,
                        finished_at = ?, output_manifest_json = NULL, error = ?
                    WHERE job_id = ? AND state = 'running'
                    """,
                    (
                        "pending" if pending else "failed",
                        None if pending else _utc_text(datetime.now(timezone.utc)),
                        error,
                        job_id,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def validated_state(self, job_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state, output_manifest_json FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        state = str(row["state"])
        if state != "completed":
            return state
        try:
            manifest = json.loads(row["output_manifest_json"])
            if not isinstance(manifest, Mapping):
                raise ValueError("stored output manifest is not an object")
            _validated_output_manifest(manifest)
            return "completed"
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE jobs
                    SET state = 'pending', worker_id = NULL, started_at = NULL,
                        finished_at = NULL, output_manifest_json = NULL,
                        attempts = 0,
                        error = 'completed output validation failed'
                    WHERE job_id = ? AND state = 'completed'
                    """,
                    (job_id,),
                )
            return "pending"

    def requeue_stale(
        self,
        phase: str,
        max_age_seconds: int,
        now: datetime,
    ) -> int:
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        if now.tzinfo is None or now.utcoffset() != timedelta(0):
            raise ValueError("now must be an aware UTC datetime")
        cutoff = _utc_text(now - timedelta(seconds=max_age_seconds))
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE jobs
                SET state = CASE WHEN attempts < 3 THEN 'pending' ELSE 'failed' END,
                    worker_id = NULL, started_at = NULL,
                    finished_at = CASE
                        WHEN attempts < 3 THEN NULL
                        ELSE ?
                    END,
                    output_manifest_json = NULL,
                    error = 'stale running job requeued'
                WHERE phase = ? AND state = 'running' AND started_at <= ?
                """,
                (_utc_text(now), phase, cutoff),
            )
            return updated.rowcount

    def counts_by_phase(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT phase, COUNT(*) AS count FROM jobs GROUP BY phase ORDER BY phase"
            ).fetchall()
        return {str(row["phase"]): int(row["count"]) for row in rows}

    def sqlite_pragmas(self) -> dict[str, object]:
        with self._connect() as connection:
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        return {"journal_mode": str(journal_mode), "foreign_keys": int(foreign_keys)}
