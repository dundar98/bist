"""Tests for operations monitoring endpoints."""

from collections.abc import Iterator
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import hash_password
from app.main import app
from database.base import Base
from database.models import JobRun, User, UserRole
from database.session import get_db
from fastapi.testclient import TestClient


@pytest.fixture()
def client() -> Iterator[TestClient]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as db:
        db.add_all(
            [
                User(
                    username="admin",
                    email="admin@example.com",
                    password_hash=hash_password("admin123"),
                    role=UserRole.ADMIN,
                    is_active=True,
                ),
                User(
                    username="viewer",
                    email="viewer@example.com",
                    password_hash=hash_password("viewer123"),
                    role=UserRole.VIEWER,
                    is_active=True,
                ),
                JobRun(
                    job_name="daily_pipeline",
                    status="success",
                    started_at=datetime(2026, 5, 24, 18),
                    finished_at=datetime(2026, 5, 24, 18, 10),
                    summary_json='{"ok": true}',
                ),
            ]
        )
        db.commit()

    def override_get_db() -> Iterator[Session]:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _token(client: TestClient, username: str, password: str) -> str:
    response = client.post("/api/auth/login", json={"username_or_email": username, "password": password})
    return response.json()["access_token"]


def test_operations_jobs_requires_admin(client: TestClient):
    viewer_token = _token(client, "viewer", "viewer123")

    response = client.get("/api/operations/jobs", headers={"Authorization": f"Bearer {viewer_token}"})

    assert response.status_code == 403


def test_admin_can_list_job_runs(client: TestClient):
    admin_token = _token(client, "admin", "admin123")

    response = client.get("/api/operations/jobs", headers={"Authorization": f"Bearer {admin_token}"})

    assert response.status_code == 200
    assert response.json()[0]["job_name"] == "daily_pipeline"
