"""
FastAPI application. Minimal — everything interesting lives in agent.py.

The API is a thin adapter: validate request, call the agent, return the result.
One AsyncAnthropic client is created at startup and reused across requests
(lifespan context manager).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import pandas as pd
from anthropic import AsyncAnthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agent import identify_acquirers
from .config import settings
from .evidence import load_csv
from .observability import configure_logging, get_logger
from .schemas import RationaleSet, TargetProfile

log = get_logger("api")

# Path to the bundled frontend (frontend/index.html sits at project root)
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"


# ==============================================================================
# LIFESPAN — startup & shutdown
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging, preload the CSV, instantiate the LLM client once."""
    configure_logging()
    log.info("api_starting", model=settings.model)

    app.state.df = load_csv(settings.csv_path)
    app.state.client = AsyncAnthropic(api_key=settings.api_key)

    log.info("api_ready", rows=len(app.state.df))
    yield

    log.info("api_shutting_down")
    await app.state.client.close()


app = FastAPI(
    title="Acquirer Identification Engine",
    version="1.0.0",
    description="Agentic M&A acquirer screening for William Blair IB.",
    lifespan=lifespan,
)

# CORS — allow local frontends for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in prod
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# ROUTES
# ==============================================================================

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": settings.model}


@app.post("/api/v1/identify-acquirers", response_model=RationaleSet)
async def post_identify_acquirers(target: TargetProfile) -> RationaleSet:
    """Run the full agentic pipeline for the given target profile."""
    try:
        return await identify_acquirers(
            df=app.state.df,
            target=target,
            client=app.state.client,
        )
    except Exception as exc:
        log.exception("pipeline_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# Serve the UI at root. Mounted last so API routes take precedence.
if FRONTEND_DIR.exists():
    @app.get("/")
    async def serve_ui() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
