# memory.ai4u.now

**Agent Memory Service** — temporal knowledge graph powered by [graphiti](https://github.com/getzep/graphiti) + [FalkorDB](https://www.falkordb.com/).

Gives AI agents persistent memory with structured episodic recall, temporal awareness, and progressive disclosure.

## Quick Start

```bash
# Clone
git clone https://github.com/AI4Unow/memory.git
cd memory

# Configure
cp .env.example .env
# Edit .env — set LLM_API_KEY

# Deploy
docker compose up -d

# Verify
curl http://localhost:8000/health
# → {"status":"ok","service":"memory.ai4u.now","version":"0.2.0"}
```

## Architecture

```
Agent → FastAPI → graphiti → FalkorDB
                   ↕              ↕
              LLM extraction   graph + vector + fulltext
              (async, idle)    (single database)
```

- **Single database** — FalkorDB handles graph, vector, and fulltext search
- **No adapters** — thin FastAPI wrapper around graphiti
- **~600 lines** of custom code

## API Reference

### Ingest Memory

Store a memory. Runs graphiti's full extraction pipeline: entity/relationship extraction, temporal edges, deduplication, embedding.

```bash
curl -X POST https://memory.ai4u.now/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "CTO deployed CRM after fixing auth bug. Root cause was race condition in session locking.",
    "user_id": "nad",
    "agent_id": "cto",
    "session_id": "deploy-2026-02-22"
  }'
```

**Response:**
```json
{
  "status": "ok",
  "episode": "memory_nad_20260222_153000",
  "entities_extracted": 3,
  "edges_extracted": 2,
  "entities": [
    {"uuid": "...", "name": "CTO", "type": "Entity"},
    {"uuid": "...", "name": "CRM", "type": "Entity"}
  ],
  "edges": [
    {"uuid": "...", "fact": "CTO deployed CRM", "valid_at": "2026-02-22T15:30:00+00:00"}
  ]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `content` | string | ✅ | Text to extract knowledge from |
| `user_id` | string | ✅ | User scope (maps to graphiti `group_id`) |
| `agent_id` | string | | Agent scope (combined as `user_id:agent_id`) |
| `session_id` | string | | Session grouping (maps to graphiti `saga`) |
| `source` | string | | `message` \| `text` \| `json` (default: `message`) |
| `reference_time` | string | | ISO timestamp for when this happened (default: now) |

### Bulk Ingest

```bash
curl -X POST https://memory.ai4u.now/v1/ingest/bulk \
  -H "Content-Type: application/json" \
  -d '{
    "episodes": [
      {"content": "First event", "user_id": "nad"},
      {"content": "Second event", "user_id": "nad"}
    ]
  }'
```

### Recall Memory

Search memories using hybrid retrieval: cosine similarity + BM25 fulltext + graph traversal, reranked by cross-encoder.

```bash
curl -X POST https://memory.ai4u.now/v1/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What went wrong with deployments?",
    "user_id": "nad",
    "limit": 10
  }'
```

**Response:**
```json
{
  "status": "ok",
  "query": "What went wrong with deployments?",
  "edges": [
    {
      "uuid": "...",
      "fact": "CTO deployed CRM after fixing auth bug",
      "valid_at": "2026-02-22T15:30:00+00:00",
      "invalid_at": null,
      "score": 0.92
    }
  ],
  "nodes": [
    {"uuid": "...", "name": "CTO", "type": "Entity", "score": 0.88}
  ],
  "episodes": [
    {"uuid": "...", "name": "memory_nad_...", "content": "CTO deployed CRM..."}
  ],
  "communities": []
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | ✅ | What to search for |
| `user_id` | string | ✅ | User scope |
| `agent_id` | string | | Agent scope filter |
| `limit` | int | | Max results (default: 10, max: 100) |
| `min_salience` | int | | Only return memories with salience ≥ this (1-10) |

### List Episodes

```bash
curl "https://memory.ai4u.now/v1/episodes?user_id=nad&limit=20"
```

### Delete Data

```bash
curl -X DELETE "https://memory.ai4u.now/v1/entities?user_id=nad"
```

## Memory Types

Graphiti automatically classifies extracted entities into structured types:

| Type | Default Salience | What It Captures |
|---|---|---|
| **Fact** | 5 | Preferences, attributes, observations |
| **Decision** | 8 | Reasoning, alternatives considered, outcome |
| **Failure** | 9 | Root cause, prevention steps, severity |
| **Reflection** | 9 | Lessons learned, recognized patterns |

Higher salience → surfaces first in recall results (progressive disclosure).

## Retrieval Accuracy

Optimized for retrieval accuracy over extraction speed:

1. **Hybrid search** — cosine similarity + BM25 keyword + graph traversal (BFS)
2. **Cross-encoder reranking** — final pass for precision
3. **Temporal awareness** — bi-temporal edges (`valid_at` / `invalid_at`)
4. **Multi-hop** — graph traversal finds related entities not in the query text

## Scoping

| Scope | Maps To | Example |
|---|---|---|
| `user_id` | graphiti `group_id` | `"nad"` |
| `user_id` + `agent_id` | graphiti `group_id` | `"nad:cto"` |
| `session_id` | graphiti `saga` | `"deploy-2026-02-22"` |

Agents with `agent_id="cto"` see their own memories + the user's shared memories.

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `FALKORDB_HOST` | `localhost` | FalkorDB host |
| `FALKORDB_PORT` | `6379` | FalkorDB port |
| `LLM_API_BASE` | `https://api.ai4u.now/v1` | OpenAI-compatible API base |
| `LLM_API_KEY` | | API key for LLM/embedding calls |
| `LLM_MODEL` | `gemini-2.5-flash` | Model for extraction |
| `EMBEDDING_MODEL` | `text-embedding-004` | Model for embeddings |
| `RERANKER_MODEL` | `gpt-4o-mini` | Model for cross-encoder reranking |

## Stack

- **[graphiti-core](https://github.com/getzep/graphiti)** — temporal knowledge graph engine
- **[FalkorDB](https://www.falkordb.com/)** — graph + vector database (Redis-based)
- **[FastAPI](https://fastapi.tiangolo.com/)** — API framework
- **Docker Compose** — 2 containers (FalkorDB + API)

## License

MIT
