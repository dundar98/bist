"""Admin operational monitoring endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.auth import require_roles
from app.schemas.operations import JobRunRead
from database.models import JobRun, User, UserRole
from database.session import get_db

router = APIRouter(prefix="/operations", tags=["operations"])


@router.get("/jobs", response_model=list[JobRunRead])
def list_job_runs(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ADMIN)),
) -> list[JobRun]:
    """Return latest production job runs for admin monitoring."""

    return list(db.scalars(select(JobRun).order_by(JobRun.started_at.desc()).limit(limit)).all())

