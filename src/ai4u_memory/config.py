"""Application settings — loaded from environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the ai4u-memory service."""

    # FalkorDB
    falkordb_host: str = Field("localhost", alias="FALKORDB_HOST")
    falkordb_port: int = Field(6379, alias="FALKORDB_PORT")

    # LLM / Embeddings / Reranker (OpenAI-compatible via api.ai4u.now)
    llm_api_base: str = Field("https://api.ai4u.now/v1", alias="LLM_API_BASE")
    llm_api_key: str = Field("", alias="LLM_API_KEY")
    llm_model: str = Field("gemini-2.5-flash", alias="LLM_MODEL")
    embedding_model: str = Field("text-embedding-004", alias="EMBEDDING_MODEL")
    reranker_model: str = Field("gpt-4o-mini", alias="RERANKER_MODEL")

    # Service
    host: str = Field("0.0.0.0", alias="HOST")
    port: int = Field(8000, alias="PORT")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton
settings = Settings()
