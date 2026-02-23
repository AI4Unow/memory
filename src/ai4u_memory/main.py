"""FastAPI application for memory.ai4u.now.

Thin wrapper around graphiti — initializes the client on startup,
registers routers, and provides health/info endpoints.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ai4u_memory.client import create_graphiti
from ai4u_memory.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API Key Middleware
# ---------------------------------------------------------------------------

# Paths that don't require authentication
PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require X-API-Key header for /v1/* endpoints."""

    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        # Skip auth for public paths
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Skip auth if no key configured (open mode)
        if not self.api_key:
            return await call_next(request)

        # Check X-API-Key header
        provided_key = request.headers.get("X-API-Key", "")
        if provided_key != self.api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)


# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize graphiti on startup, close on shutdown."""
    logger.info("Initializing memory service...")

    graphiti = await create_graphiti(settings)
    app.state.graphiti = graphiti

    if settings.memory_api_key:
        logger.info("API key authentication enabled")
    else:
        logger.info("API key authentication disabled (MEMORY_API_KEY not set)")

    logger.info("Memory service ready")
    yield

    logger.info("Shutting down...")
    await graphiti.close()


app = FastAPI(
    title="memory.ai4u.now",
    description="Agent Memory Service — graphiti temporal knowledge graph + FalkorDB",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key auth — must be added after CORS
app.add_middleware(APIKeyMiddleware, api_key=settings.memory_api_key)

# Routers
from ai4u_memory.routers.ingest import router as ingest_router  # noqa: E402
from ai4u_memory.routers.recall import router as recall_router  # noqa: E402

app.include_router(ingest_router)
app.include_router(recall_router)


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "service": "memory.ai4u.now", "version": "0.2.0"}


@app.get("/")
async def root():
    """Service info."""
    return {
        "service": "memory.ai4u.now",
        "version": "0.2.0",
        "stack": "graphiti + FalkorDB",
        "endpoints": {
            "health": "/health",
            "ingest": "/v1/ingest",
            "recall": "/v1/recall",
            "entities": "/v1/entities",
            "episodes": "/v1/episodes",
        },
    }
