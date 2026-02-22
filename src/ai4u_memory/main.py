"""FastAPI application for memory.ai4u.now.

Thin wrapper around graphiti — initializes the client on startup,
registers routers, and provides health/info endpoints.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai4u_memory.client import create_graphiti
from ai4u_memory.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize graphiti on startup, close on shutdown."""
    logger.info("Initializing memory service...")

    graphiti = await create_graphiti(settings)
    app.state.graphiti = graphiti

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
