"""Runtime configuration for the API platform."""

from functools import lru_cache
from typing import Annotated, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class AppSettings(BaseSettings):
    """Settings loaded from environment variables and optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="BIST_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "BIST Signal Platform"
    environment: str = "local"
    database_url: str = "sqlite:///./bist.db"
    secret_key: str = "change-me-before-production"
    access_token_expire_minutes: int = 60
    cors_origins: Annotated[List[str], NoDecode] = Field(default_factory=lambda: ["http://localhost:3000"])

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        """Accept comma-separated CORS origins in .env files."""

        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached application settings."""

    return AppSettings()
