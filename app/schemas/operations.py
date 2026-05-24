"""Operational monitoring schemas."""

from datetime import datetime

from pydantic import BaseModel


class JobRunRead(BaseModel):
    id: int
    job_name: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    summary_json: str | None = None
    error: str | None = None

    model_config = {"from_attributes": True}

