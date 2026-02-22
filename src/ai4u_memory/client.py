"""Graphiti client initialization with FalkorDB.

Single function to create and configure a Graphiti instance
with FalkorDB as the graph+vector backend and OpenAI-compatible
LLM/embedder/reranker via api.ai4u.now.
"""

import logging

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient

from ai4u_memory.config import Settings

logger = logging.getLogger(__name__)


async def create_graphiti(settings: Settings) -> Graphiti:
    """Create and initialize a Graphiti instance.

    Args:
        settings: Application settings with FalkorDB + LLM config.

    Returns:
        Initialized Graphiti instance ready for use.
    """
    logger.info(
        f"Connecting to FalkorDB at {settings.falkordb_host}:{settings.falkordb_port}"
    )

    # Graph database driver
    driver = FalkorDriver(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
    )

    # LLM client for entity/relationship extraction
    llm_config = LLMConfig(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_api_base,
    )
    llm_client = OpenAIClient(config=llm_config)

    # Embedder for vector search
    embedder_config = OpenAIEmbedderConfig(
        embedding_model=settings.embedding_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_api_base,
    )
    embedder = OpenAIEmbedder(config=embedder_config)

    # Cross-encoder reranker for retrieval accuracy
    reranker_config = LLMConfig(
        model=settings.reranker_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_api_base,
    )
    cross_encoder = OpenAIRerankerClient(config=reranker_config)

    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
        store_raw_episode_content=True,
    )

    # Build required indexes and constraints in FalkorDB
    await graphiti.build_indices_and_constraints()

    logger.info("Graphiti initialized with FalkorDB")
    return graphiti
