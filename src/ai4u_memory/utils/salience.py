"""Salience-based ranking utilities for post-retrieval ordering."""


def rank_by_salience(
    edges: list[dict],
    min_salience: int | None = None,
) -> list[dict]:
    """Rank search results by salience, filtering below threshold.

    Salience is stored as entity type metadata on nodes connected
    to edges. For now, we use the reranker score as a proxy —
    edges with higher cross-encoder scores are more relevant.

    When entity type metadata is available on edges (e.g., from
    Decision.salience=8 or Failure.salience=9), this function
    will incorporate that for progressive disclosure.

    Args:
        edges: List of edge result dicts from recall.
        min_salience: If set, filter out edges below this threshold.

    Returns:
        Sorted list of edges, highest relevance first.
    """
    if min_salience is not None:
        edges = [
            e for e in edges if e.get("salience", e.get("score", 5)) >= min_salience
        ]

    # Sort by score descending (cross-encoder reranker score)
    return sorted(edges, key=lambda e: e.get("score", 0), reverse=True)
