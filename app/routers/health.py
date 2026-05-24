"""Health and readiness endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import AppSettings, get_settings
from database.session import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health(settings: AppSettings = Depends(get_settings)) -> dict:
    """Basic liveness check."""

    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
    }


@router.get("/ready")
def ready(db: Session = Depends(get_db)) -> dict:
    """Readiness check that verifies database connectivity."""

    db.execute(text("SELECT 1"))
    return {"status": "ready"}
