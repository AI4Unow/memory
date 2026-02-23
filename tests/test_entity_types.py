"""Tests for entity type definitions."""

from ai4u_memory.entity_types import (
    ENTITY_TYPES,
    Decision,
    Fact,
    Failure,
    Reflection,
)


class TestEntityTypes:
    """Verify entity type models and defaults."""

    def test_fact_defaults(self):
        fact = Fact()
        assert fact.salience == 5

    def test_decision_defaults(self):
        decision = Decision()
        assert decision.salience == 8
        assert decision.outcome == "pending"
        assert decision.reasoning == ""
        assert decision.alternatives == []

    def test_decision_with_data(self):
        decision = Decision(
            reasoning="Docker simplicity",
            alternatives=["Neo4j", "Memgraph"],
            outcome="success",
            salience=9,
        )
        assert decision.reasoning == "Docker simplicity"
        assert len(decision.alternatives) == 2
        assert decision.salience == 9

    def test_failure_defaults(self):
        failure = Failure()
        assert failure.salience == 9
        assert failure.severity == 7
        assert failure.root_cause == ""

    def test_failure_with_data(self):
        failure = Failure(
            root_cause="Race condition in session locking",
            prevention="Use sequential agent queue",
            severity=8,
        )
        assert failure.root_cause == "Race condition in session locking"
        assert failure.prevention == "Use sequential agent queue"

    def test_reflection_defaults(self):
        reflection = Reflection()
        assert reflection.salience == 9
        assert reflection.pattern == ""

    def test_entity_types_dict(self):
        assert len(ENTITY_TYPES) == 4
        assert "Fact" in ENTITY_TYPES
        assert "Decision" in ENTITY_TYPES
        assert "Failure" in ENTITY_TYPES
        assert "Reflection" in ENTITY_TYPES
        assert ENTITY_TYPES["Fact"] is Fact
        assert ENTITY_TYPES["Decision"] is Decision

    def test_salience_bounds(self):
        """Salience must be 1-10."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Fact(salience=0)
        with pytest.raises(ValidationError):
            Fact(salience=11)
