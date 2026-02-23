"""Custom LLM client for OpenAI-compatible proxies.

graphiti 0.28 uses the OpenAI Responses API (POST /responses) for
structured output, which most OpenAI-compatible proxies don't support.

This module provides a drop-in replacement that uses Chat Completions
API (POST /chat/completions) with structured output instead.
"""

import json
import logging
import re

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig
from graphiti_core.llm_client.openai_base_client import (
    DEFAULT_REASONING,
    DEFAULT_VERBOSITY,
    BaseOpenAIClient,
)

logger = logging.getLogger(__name__)

# Regex to strip markdown code fences (```json ... ``` or ``` ... ```)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM JSON responses."""
    text = text.strip()
    m = _CODE_FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    # Also handle partial fences (start only)
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        return "\n".join(lines).strip()
    return text


class ChatCompletionsClient(BaseOpenAIClient):
    """LLM client that uses Chat Completions API instead of Responses API.

    Drop-in replacement for graphiti's OpenAIClient that works with
    OpenAI-compatible proxies (api.ai4u.now, LiteLLM, etc.) that
    don't support the Responses API.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client=None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning: str = DEFAULT_REASONING,
        verbosity: str = DEFAULT_VERBOSITY,
    ):
        super().__init__(config, cache, max_tokens, reasoning, verbosity)

        if config is None:
            config = LLMConfig()

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        """Create structured completion using Chat Completions with JSON mode.

        Instead of the Responses API (POST /responses), this uses the
        standard Chat Completions API with response_format for structured output.
        """
        # Build the JSON schema instruction
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        # Add schema instruction to the system message
        schema_messages = list(messages)
        schema_instruction = (
            f"\n\nRespond with ONLY raw JSON (no markdown, no code fences, no ```). "
            f"The JSON must match this schema:\n{schema_str}"
        )

        # Append to first system message or add a new one
        if schema_messages and schema_messages[0].get("role") == "system":
            schema_messages[0] = {
                **schema_messages[0],
                "content": str(schema_messages[0].get("content", "")) + schema_instruction,
            }
        else:
            schema_messages.insert(0, {
                "role": "system",
                "content": schema_instruction,
            })

        response = await self.client.chat.completions.create(
            model=model,
            messages=schema_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        # Parse the JSON response — strip code fences if the LLM adds them
        content = response.choices[0].message.content or "{}"
        content = _strip_code_fences(content)
        parsed = response_model.model_validate_json(content)

        # Return a mock response object that matches what graphiti expects
        return _StructuredResponse(parsed=parsed, usage=response.usage)

    async def _create_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel] | None = None,
        reasoning: str | None = None,
        verbosity: str | None = None,
    ):
        """Create a regular completion using Chat Completions API."""
        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )


class _StructuredResponse:
    """Mock response object for structured completions.

    graphiti's _handle_structured_response accesses response.output_text
    directly — it expects the JSON string of the parsed model.
    """

    def __init__(self, parsed: BaseModel, usage=None):
        # output_text is the JSON string that graphiti will json.loads()
        self.output_text = parsed.model_dump_json()
        self.usage = _UsageWrapper(usage)


class _UsageWrapper:
    """Wrap OpenAI usage to match graphiti's expected attribute names."""

    def __init__(self, usage=None):
        if usage:
            self.input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            self.output_tokens = getattr(usage, "completion_tokens", 0) or 0
        else:
            self.input_tokens = 0
            self.output_tokens = 0
