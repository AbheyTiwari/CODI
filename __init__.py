# core/__init__.py


"""
app/main.py
FastAPI application factory.
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import get_settings
from app.core.logging import setup_logging, logger
from app.core.embedder import get_embedder
from app.core.vectorstore import get_vectorstore
from app.api import chat, health, ingest


# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    os.makedirs("logs", exist_ok=True)
    os.makedirs(cfg.chroma_path, exist_ok=True)
    setup_logging()

    logger.info("Starting RAGChat API…")

    # Warm up singletons — fail fast if something is broken
    get_embedder()
    get_vectorstore()
    logger.info("All subsystems initialised")

    yield  # ← app is running

    logger.info("Shutting down RAGChat API")


# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title="RAGChat API",
        description="Retrieval-Augmented Generation over a glossary dataset.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS — allow the frontend origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.origins_list,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    # Global exception handler — never leak stack traces to client
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: {}", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred."},
        )

    # Routers
    app.include_router(chat.router)
    app.include_router(health.router)
    app.include_router(ingest.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "RAGChat API — see /docs"}

    return app


app = create_app()
