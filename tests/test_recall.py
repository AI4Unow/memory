"""Tests for the recall router's salience ranking."""

from ai4u_memory.utils.salience import rank_by_salience


class TestSalienceRanking:
    """Verify salience-based ranking and filtering."""

    def test_sorts_by_score_descending(self):
        edges = [
            {"uuid": "1", "fact": "low", "score": 0.3},
            {"uuid": "2", "fact": "high", "score": 0.9},
            {"uuid": "3", "fact": "mid", "score": 0.6},
        ]
        ranked = rank_by_salience(edges)
        assert ranked[0]["uuid"] == "2"
        assert ranked[1]["uuid"] == "3"
        assert ranked[2]["uuid"] == "1"

    def test_filters_by_min_salience(self):
        edges = [
            {"uuid": "1", "fact": "low", "score": 0.3, "salience": 3},
            {"uuid": "2", "fact": "high", "score": 0.9, "salience": 9},
            {"uuid": "3", "fact": "mid", "score": 0.6, "salience": 6},
        ]
        ranked = rank_by_salience(edges, min_salience=5)
        assert len(ranked) == 2
        assert ranked[0]["uuid"] == "2"
        assert ranked[1]["uuid"] == "3"

    def test_empty_list(self):
        assert rank_by_salience([]) == []

    def test_no_filter(self):
        edges = [{"uuid": "1", "fact": "x", "score": 0.5}]
        ranked = rank_by_salience(edges, min_salience=None)
        assert len(ranked) == 1

    def test_uses_score_as_salience_fallback(self):
        """When salience field is missing, use score as fallback."""
        edges = [
            {"uuid": "1", "fact": "no salience", "score": 7},
        ]
        ranked = rank_by_salience(edges, min_salience=5)
        assert len(ranked) == 1
