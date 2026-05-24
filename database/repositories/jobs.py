"""Job run logging helpers."""

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from database.models import JobRun


def start_job(db: Session, job_name: str, summary: dict[str, Any] | None = None) -> JobRun:
    job = JobRun(
        job_name=job_name,
        status="running",
        started_at=datetime.now(timezone.utc),
        summary_json=json.dumps(summary or {}, sort_keys=True),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def finish_job(
    db: Session,
    job: JobRun,
    *,
    status: str = "success",
    summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    job.status = status
    job.finished_at = datetime.now(timezone.utc)
    if summary is not None:
        job.summary_json = json.dumps(summary, sort_keys=True, default=str)
    job.error = error
    db.commit()
