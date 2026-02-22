"""Recall router — retrieval endpoints.

POST /v1/recall     — Hybrid search (vector + BM25 + graph traversal + reranking)
GET  /v1/entities   — List entities for a scope
GET  /v1/entities/{uuid} — Get entity details
GET  /v1/episodes   — List recent episodes
DELETE /v1/entities — Clear graph data for a scope
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from graphiti_core import Graphiti
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
)
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


def _build_group_ids(
    user_id: str, agent_id: Optional[str] = None
) -> list[str]:
    """Build graphiti group_ids for search scoping."""
    group_ids = [user_id]
    if agent_id:
        group_ids.append(f"{user_id}:{agent_id}")
    return group_ids


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/recall")
async def recall(request: Request, body: RecallRequest):
    """Search memories using graphiti's hybrid search.

    Combines cosine similarity, BM25 fulltext, and graph traversal
    with cross-encoder reranking for maximum accuracy.

    Returns edges (facts), nodes (entities), episodes (raw context),
    and communities (cluster summaries), ranked by salience.
    """
    graphiti = _get_graphiti(request)
    group_ids = _build_group_ids(body.user_id, body.agent_id)

    try:
        results = await graphiti.search(
            query=body.query,
            group_ids=group_ids,
            config=COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
            search_filter=SearchFilters(),
        )

        # Format edges (temporal facts)
        edges = []
        for i, edge in enumerate(results.edges):
            score = (
                results.edge_reranker_scores[i]
                if i < len(results.edge_reranker_scores)
                else 0.0
            )
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
                    "score": score,
                }
            )

        # Format nodes (entities)
        nodes = []
        for i, node in enumerate(results.nodes):
            score = (
                results.node_reranker_scores[i]
                if i < len(results.node_reranker_scores)
                else 0.0
            )
            nodes.append(
                {
                    "uuid": node.uuid,
                    "name": node.name,
                    "type": node.label if hasattr(node, "label") else "Entity",
                    "summary": node.summary if hasattr(node, "summary") else "",
                    "score": score,
                }
            )

        # Format episodes (raw context)
        episodes = []
        for ep in results.episodes:
            episodes.append(
                {
                    "uuid": ep.uuid,
                    "name": ep.name,
                    "content": ep.content if hasattr(ep, "content") else "",
                    "created_at": (
                        ep.created_at.isoformat() if ep.created_at else None
                    ),
                }
            )

        # Format communities (cluster summaries)
        communities = []
        for comm in results.communities:
            communities.append(
                {
                    "uuid": comm.uuid,
                    "name": comm.name,
                    "summary": comm.summary if hasattr(comm, "summary") else "",
                }
            )

        # Apply salience ranking
        ranked_edges = rank_by_salience(edges, body.min_salience)

        return {
            "status": "ok",
            "query": body.query,
            "edges": ranked_edges[: body.limit],
            "nodes": nodes[: body.limit],
            "episodes": episodes[: body.limit],
            "communities": communities,
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
    group_ids = _build_group_ids(user_id, agent_id)

    try:
        # Search with a broad query to get entities
        results = await graphiti.search(
            query="*",
            group_ids=group_ids,
            config=COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
            search_filter=SearchFilters(),
        )

        nodes = []
        for node in results.nodes[:limit]:
            nodes.append(
                {
                    "uuid": node.uuid,
                    "name": node.name,
                    "type": node.label if hasattr(node, "label") else "Entity",
                    "summary": node.summary if hasattr(node, "summary") else "",
                    "group_id": node.group_id,
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

    group_id = user_id
    if agent_id:
        group_id = f"{user_id}:{agent_id}"

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
