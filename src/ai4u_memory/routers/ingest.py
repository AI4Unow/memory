"""Ingestion router — POST /v1/ingest.

Receives raw text from agents and feeds it through graphiti's
extraction pipeline with custom entity types.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from pydantic import BaseModel, Field

from ai4u_memory.entity_types import ENTITY_TYPES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["ingest"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Request body for ingesting a memory."""

    content: str = Field(..., description="Text content to extract knowledge from")
    user_id: str = Field(..., description="User ID for graph scoping (→ group_id)")
    agent_id: Optional[str] = Field(None, description="Agent ID")
    session_id: Optional[str] = Field(
        None, description="Session/run ID (→ graphiti saga)"
    )
    source: str = Field(
        "message", description="Source type: message | text | json"
    )
    reference_time: Optional[str] = Field(
        None, description="ISO timestamp for when this happened (default: now)"
    )


class BulkIngestRequest(BaseModel):
    """Request body for bulk ingestion."""

    episodes: list[IngestRequest] = Field(
        ..., description="List of episodes to ingest"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_graphiti(request: Request) -> Graphiti:
    """Extract graphiti client from app state."""
    graphiti = getattr(request.app.state, "graphiti", None)
    if graphiti is None:
        raise HTTPException(status_code=503, detail="Memory service not initialized")
    return graphiti


def _sanitize_id(value: str) -> str:
    """Sanitize an ID for graphiti — only alphanumeric, dashes, underscores."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


def _build_group_id(user_id: str, agent_id: Optional[str] = None) -> str:
    """Build graphiti group_id from user/agent scope."""
    uid = _sanitize_id(user_id)
    if agent_id:
        return f"{uid}_{_sanitize_id(agent_id)}"
    return uid


def _parse_source_type(source: str) -> EpisodeType:
    """Map source string to graphiti EpisodeType."""
    mapping = {
        "message": EpisodeType.message,
        "text": EpisodeType.text,
        "json": EpisodeType.json,
    }
    return mapping.get(source, EpisodeType.message)


def _parse_reference_time(ref_time: Optional[str]) -> datetime:
    """Parse ISO timestamp or return now."""
    if ref_time:
        try:
            return datetime.fromisoformat(ref_time)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/ingest")
async def ingest(request: Request, body: IngestRequest):
    """Ingest a single memory episode.

    Runs graphiti's full extraction pipeline:
    1. Extract entities (classified as Fact/Decision/Failure/Reflection)
    2. Extract relationships with temporal edges
    3. Resolve against existing knowledge graph
    4. Embed for vector search
    """
    graphiti = _get_graphiti(request)
    group_id = _build_group_id(body.user_id, body.agent_id)
    ref_time = _parse_reference_time(body.reference_time)
    source_type = _parse_source_type(body.source)

    episode_name = f"memory_{body.user_id}_{ref_time.strftime('%Y%m%d_%H%M%S')}"

    try:
        result = await graphiti.add_episode(
            name=episode_name,
            episode_body=body.content,
            source_description=f"Agent memory for {body.user_id}",
            reference_time=ref_time,
            source=source_type,
            group_id=group_id,
            saga=body.session_id,
            entity_types=ENTITY_TYPES,
        )

        # Format response
        entities = []
        for node in result.nodes:
            entities.append(
                {
                    "uuid": node.uuid,
                    "name": node.name,
                    "type": node.label if hasattr(node, "label") else "Entity",
                    "group_id": node.group_id,
                }
            )

        edges = []
        for edge in result.edges:
            edges.append(
                {
                    "uuid": edge.uuid,
                    "fact": edge.fact,
                    "source_node": edge.source_node_uuid,
                    "target_node": edge.target_node_uuid,
                    "valid_at": (
                        edge.valid_at.isoformat() if edge.valid_at else None
                    ),
                    "invalid_at": (
                        edge.invalid_at.isoformat() if edge.invalid_at else None
                    ),
                }
            )

        return {
            "status": "ok",
            "episode": episode_name,
            "entities_extracted": len(entities),
            "edges_extracted": len(edges),
            "entities": entities,
            "edges": edges,
        }

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/bulk")
async def ingest_bulk(request: Request, body: BulkIngestRequest):
    """Ingest multiple episodes in one request.

    Processes sequentially to maintain temporal ordering.
    """
    graphiti = _get_graphiti(request)
    results = []

    for episode in body.episodes:
        group_id = _build_group_id(episode.user_id, episode.agent_id)
        ref_time = _parse_reference_time(episode.reference_time)
        source_type = _parse_source_type(episode.source)
        episode_name = (
            f"memory_{episode.user_id}_{ref_time.strftime('%Y%m%d_%H%M%S')}"
        )

        try:
            result = await graphiti.add_episode(
                name=episode_name,
                episode_body=episode.content,
                source_description=f"Agent memory for {episode.user_id}",
                reference_time=ref_time,
                source=source_type,
                group_id=group_id,
                saga=episode.session_id,
                entity_types=ENTITY_TYPES,
            )
            results.append(
                {
                    "episode": episode_name,
                    "entities": len(result.nodes),
                    "edges": len(result.edges),
                    "status": "ok",
                }
            )
        except Exception as e:
            logger.error(f"Bulk ingestion failed for {episode_name}: {e}")
            results.append(
                {"episode": episode_name, "status": "error", "error": str(e)}
            )

    return {"status": "ok", "results": results}
