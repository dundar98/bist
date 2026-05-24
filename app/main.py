"""FastAPI entrypoint for the BIST signal platform."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routers import auth, dashboard, health, market, operations, portfolios, signals, symbols, users, watchlist


def create_app() -> FastAPI:
    """Create and configure the API application."""

    settings = get_settings()
    app = FastAPI(title=settings.app_name)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(auth.router, prefix="/api")
    app.include_router(symbols.router, prefix="/api")
    app.include_router(signals.router, prefix="/api")
    app.include_router(portfolios.router, prefix="/api")
    app.include_router(users.router, prefix="/api")
    app.include_router(dashboard.router, prefix="/api")
    app.include_router(market.router, prefix="/api")
    app.include_router(watchlist.router, prefix="/api")
    app.include_router(operations.router, prefix="/api")
    return app


app = create_app()
