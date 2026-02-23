"""Custom entity types for the memory knowledge graph.

These Pydantic models define structured memory types that graphiti's
extraction pipeline uses to classify extracted knowledge automatically.

Each type carries a `salience` score (1-10) that drives progressive
disclosure during retrieval: high-salience memories (failures, decisions)
surface first, routine facts last.
"""

from pydantic import BaseModel, Field


class Fact(BaseModel):
    """Simple factual memory — preferences, attributes, observations.

    Examples:
        - "User prefers dark mode"
        - "CTO uses Docker Compose for deployments"
        - "API endpoint is api.ai4u.now"
    """

    salience: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Importance score. Facts default to medium salience.",
    )


class Decision(BaseModel):
    """A decision with reasoning trail — captures WHY, not just WHAT.

    Examples:
        - "Chose FalkorDB over Neo4j because Docker setup was simpler"
        - "Decided to use sequential queue instead of parallel processing"
    """

    reasoning: str = Field(
        default="",
        description="Why this decision was made.",
    )
    alternatives: list[str] = Field(
        default_factory=list,
        description="Other options that were considered.",
    )
    outcome: str = Field(
        default="pending",
        description="Result: pending | success | failure | revised",
    )
    salience: int = Field(
        default=8,
        ge=1,
        le=10,
        description="Decisions default to high salience.",
    )


class Failure(BaseModel):
    """Something that went wrong — for learning from mistakes.

    Examples:
        - "Parallel deploy failed due to race condition in session locking"
        - "Auth broke because Vertex AI credentials weren't configured"
    """

    root_cause: str = Field(
        default="",
        description="What caused the failure.",
    )
    prevention: str = Field(
        default="",
        description="How to prevent this in the future.",
    )
    severity: int = Field(
        default=7,
        ge=1,
        le=10,
        description="How bad was it. Minor inconvenience (1) to data loss (10).",
    )
    salience: int = Field(
        default=9,
        ge=1,
        le=10,
        description="Failures default to very high salience — learn from mistakes.",
    )


class Reflection(BaseModel):
    """A lesson learned or pattern recognized across experiences.

    Examples:
        - "Every time we rush deploys, auth breaks — always test auth first"
        - "Agents work better with sequential processing than parallel"
    """

    pattern: str = Field(
        default="",
        description="The recurring pattern or lesson.",
    )
    salience: int = Field(
        default=9,
        ge=1,
        le=10,
        description="Reflections are hard-won wisdom — high salience.",
    )


# All entity types to register with graphiti (name → model class)
# graphiti 0.28 expects dict[str, type[BaseModel]]
ENTITY_TYPES: dict[str, type[BaseModel]] = {
    "Fact": Fact,
    "Decision": Decision,
    "Failure": Failure,
    "Reflection": Reflection,
}
