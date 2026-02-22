"""Tests for the FastAPI application endpoints.

Tests API round-trip with mocked graphiti backend.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_mock_graphiti():
    """Create a mock graphiti instance."""
    mock = MagicMock()

    # Mock add_episode result
    mock_add_result = SimpleNamespace(
        episode=SimpleNamespace(uuid="ep-1", name="test"),
        nodes=[
            SimpleNamespace(
                uuid="n-1", name="CTO", label="Entity", group_id="nad"
            ),
        ],
        edges=[
            SimpleNamespace(
                uuid="e-1",
                fact="CTO deployed CRM",
                source_node_uuid="n-1",
                target_node_uuid="n-2",
                valid_at=None,
                invalid_at=None,
            ),
        ],
        communities=[],
        community_edges=[],
        episodic_edges=[],
    )
    mock.add_episode = AsyncMock(return_value=mock_add_result)

    # Mock search result
    mock_search_result = SimpleNamespace(
        edges=[
            SimpleNamespace(
                uuid="e-1",
                fact="CTO deployed CRM",
                source_node_uuid="n-1",
                target_node_uuid="n-2",
                valid_at=None,
                invalid_at=None,
            ),
        ],
        edge_reranker_scores=[0.85],
        nodes=[
            SimpleNamespace(
                uuid="n-1", name="CTO", label="Entity", summary="", group_id="nad"
            ),
        ],
        node_reranker_scores=[0.8],
        episodes=[],
        episode_reranker_scores=[],
        communities=[],
        community_reranker_scores=[],
    )
    mock.search = AsyncMock(return_value=mock_search_result)

    # Mock retrieve_episodes
    mock.retrieve_episodes = AsyncMock(return_value=[])

    # Mock close
    mock.close = AsyncMock()

    # Mock driver for delete
    mock.driver = MagicMock()
    mock.driver.execute_query = AsyncMock(return_value=([], None, None))

    return mock


@pytest.fixture
def client():
    """Create a test client with mocked graphiti."""
    mock_graphiti = _make_mock_graphiti()

    with patch("ai4u_memory.main.create_graphiti", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_graphiti

        from ai4u_memory.main import app

        app.state.graphiti = mock_graphiti

        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Health & Root
# ---------------------------------------------------------------------------


class TestHealthEndpoints:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stack"] == "graphiti + FalkorDB"
        assert "ingest" in data["endpoints"]
        assert "recall" in data["endpoints"]


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class TestIngestEndpoints:
    def test_ingest_single(self, client):
        resp = client.post(
            "/v1/ingest",
            json={
                "content": "CTO deployed CRM to production",
                "user_id": "nad",
                "agent_id": "cto",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["entities_extracted"] >= 0
        assert data["edges_extracted"] >= 0

    def test_ingest_with_session(self, client):
        resp = client.post(
            "/v1/ingest",
            json={
                "content": "Fixed auth bug in api.ai4u.now",
                "user_id": "nad",
                "session_id": "deploy-session-1",
            },
        )
        assert resp.status_code == 200

    def test_ingest_bulk(self, client):
        resp = client.post(
            "/v1/ingest/bulk",
            json={
                "episodes": [
                    {"content": "First event", "user_id": "nad"},
                    {"content": "Second event", "user_id": "nad"},
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


class TestRecallEndpoints:
    def test_recall(self, client):
        resp = client.post(
            "/v1/recall",
            json={"query": "What happened with deployments?", "user_id": "nad"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "edges" in data
        assert "nodes" in data

    def test_recall_with_min_salience(self, client):
        resp = client.post(
            "/v1/recall",
            json={
                "query": "failures",
                "user_id": "nad",
                "min_salience": 8,
            },
        )
        assert resp.status_code == 200

    def test_list_episodes(self, client):
        resp = client.get("/v1/episodes?user_id=nad")
        assert resp.status_code == 200

    def test_delete_entities(self, client):
        resp = client.delete("/v1/entities?user_id=nad")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
