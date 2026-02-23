"""Recall router — retrieval endpoints.

POST /v1/recall     — Hybrid search (vector + BM25 + graph traversal + reranking)
GET  /v1/entities   — List entities for a scope
GET  /v1/entities/{uuid} — Get entity details
GET  /v1/episodes   — List recent episodes
DELETE /v1/entities — Clear graph data for a scope
"""

import re

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from graphiti_core import Graphiti
from graphiti_core.search.search_filters import SearchFilters
from pydantic import BaseModel, Field

from ai4u_memory.utils.salience import rank_by_salience

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["recall"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class RecallRequest(BaseModel):
    """Request body for recalling memories."""

    query: str = Field(..., description="What to search for")
    user_id: str = Field(..., description="User ID for scoping")
    agent_id: Optional[str] = Field(None, description="Agent ID filter")
    limit: int = Field(10, ge=1, le=100, description="Max results")
    min_salience: Optional[int] = Field(
        None, ge=1, le=10, description="Only return memories with salience >= this"
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


def _build_group_ids(
    user_id: str, agent_id: Optional[str] = None
) -> list[str]:
    """Build graphiti group_ids for search scoping."""
    uid = _sanitize_id(user_id)
    group_ids = [uid]
    if agent_id:
        group_ids.append(f"{uid}_{_sanitize_id(agent_id)}")
    return group_ids


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/recall")
async def recall(request: Request, body: RecallRequest):
    """Search memories using graphiti's hybrid search.

    In graphiti 0.28, search() returns list[EntityEdge] directly.
    Each edge represents a fact connecting two entities.
    """
    graphiti = _get_graphiti(request)
    group_ids = _build_group_ids(body.user_id, body.agent_id)

    try:
        # graphiti 0.28: search() returns list[EntityEdge]
        edges = await graphiti.search(
            query=body.query,
            group_ids=group_ids,
            num_results=body.limit,
            search_filter=SearchFilters(),
        )

        # Format edges (temporal facts)
        results = []
        for edge in edges:
            results.append(
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

        # Apply salience ranking
        ranked = rank_by_salience(results, body.min_salience)

        return {
            "status": "ok",
            "query": body.query,
            "edges": ranked[: body.limit],
        }

    except Exception as e:
        logger.error(f"Recall failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities")
async def list_entities(
    request: Request,
    user_id: str = Query(..., description="User ID"),
    agent_id: Optional[str] = Query(None, description="Agent ID filter"),
    limit: int = Query(50, ge=1, le=500),
):
    """List all entities in the knowledge graph for a given scope."""
    graphiti = _get_graphiti(request)
    group_id = _sanitize_id(user_id)
    if agent_id:
        group_id = f"{_sanitize_id(user_id)}_{_sanitize_id(agent_id)}"

    try:
        # Query graph directly for entity nodes
        records, _, _ = await graphiti.driver.execute_query(
            """
            MATCH (n:Entity {group_id: $group_id})
            RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary, n.group_id AS group_id
            LIMIT $limit
            """,
            group_id=group_id,
            limit=limit,
        )

        nodes = []
        for r in records:
            nodes.append(
                {
                    "uuid": r["uuid"],
                    "name": r["name"],
                    "summary": r.get("summary", ""),
                    "group_id": r.get("group_id", ""),
                }
            )

        return {"status": "ok", "entities": nodes}

    except Exception as e:
        logger.error(f"List entities failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/episodes")
async def list_episodes(
    request: Request,
    user_id: str = Query(..., description="User ID"),
    agent_id: Optional[str] = Query(None, description="Agent ID filter"),
    limit: int = Query(20, ge=1, le=200),
):
    """List recent episodes for a given scope."""
    graphiti = _get_graphiti(request)
    group_ids = _build_group_ids(user_id, agent_id)

    try:
        episodes = await graphiti.retrieve_episodes(
            reference_time=datetime.now(timezone.utc),
            last_n=limit,
            group_ids=group_ids,
        )

        result = []
        for ep in episodes:
            result.append(
                {
                    "uuid": ep.uuid,
                    "name": ep.name,
                    "content": ep.content if hasattr(ep, "content") else "",
                    "created_at": (
                        ep.created_at.isoformat() if ep.created_at else None
                    ),
                    "group_id": ep.group_id,
                }
            )

        return {"status": "ok", "episodes": result}

    except Exception as e:
        logger.error(f"List episodes failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/entities")
async def delete_entities(
    request: Request,
    user_id: str = Query(..., description="User ID"),
    agent_id: Optional[str] = Query(None, description="Agent ID filter"),
):
    """Delete all graph data for a given user/agent scope."""
    graphiti = _get_graphiti(request)

    group_id = _sanitize_id(user_id)
    if agent_id:
        group_id = f"{_sanitize_id(user_id)}_{_sanitize_id(agent_id)}"

    try:
        await graphiti.driver.execute_query(
            """
            MATCH (n {group_id: $group_id})
            DETACH DELETE n
            """,
            group_id=group_id,
        )
        return {"status": "ok", "deleted_scope": group_id}

    except Exception as e:
        logger.error(f"Delete failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
