"""
OpenAI GPT LLM implementation.

Uses the OpenAI API to generate completions with GPT models.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .base import (
    LLM,
    LLMError,
    LLMResponse,
    Message,
    Role,
    Tool,
    ToolCall,
)


# Model configurations
# Models that use max_completion_tokens instead of max_tokens
_COMPLETION_TOKEN_MODELS = {"gpt-5", "gpt-4.1"}

OPENAI_MODELS = {
    # Latest flagship conversational models (use max_completion_tokens)
    "gpt-5.2": {"max_tokens": 16384},
    "gpt-5.1": {"max_tokens": 16384},
    "gpt-5": {"max_tokens": 16384},
    # Long-context family (use max_completion_tokens)
    "gpt-4.1": {"max_tokens": 32768},
    "gpt-4.1-mini": {"max_tokens": 32768},
    "gpt-4.1-nano": {"max_tokens": 32768},
    # Omni series / general multimodal models
    "gpt-4o": {"max_tokens": 16384},
    "gpt-4o-mini": {"max_tokens": 16384},
    "gpt-4o-realtime": {"max_tokens": 16384},
    "gpt-4o-mini-realtime": {"max_tokens": 16384},
    # Open-weight models
    "gpt-oss-120b": {"max_tokens": 131072},
    "gpt-oss-20b": {"max_tokens": 131072},
}


def _uses_completion_tokens(model: str) -> bool:
    """Check if model uses max_completion_tokens instead of max_tokens."""
    for prefix in _COMPLETION_TOKEN_MODELS:
        if model.startswith(prefix):
            return True
    return False


@dataclass
class OpenAILLM(LLM):
    """
    OpenAI GPT LLM provider.

    Uses the OpenAI API to generate completions. Requires openai package.

    Attributes:
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
        model: Model name. Defaults to gpt-4o.
        base_url: Optional custom base URL for API-compatible endpoints.

    Example:
        >>> llm = OpenAILLM()
        >>> response = llm.complete([Message.user("Hello!")])
        >>> print(response.content)
    """

    api_key: str | None = None
    model: str = "gpt-4o"
    base_url: str | None = None

    def __post_init__(self) -> None:
        """Initialize and validate."""
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise LLMError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )

        if self.model not in OPENAI_MODELS and self.base_url is None:
            raise LLMError(
                f"Unknown model: {self.model}. "
                f"Supported: {list(OPENAI_MODELS.keys())}"
            )

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self.model

    def _get_client(self):
        """Get OpenAI client (lazy import)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise LLMError("openai package required. Install with: pip install openai")

        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        return OpenAI(**kwargs)

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert messages to OpenAI format."""
        converted = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                converted.append({"role": "system", "content": msg.content})

            elif msg.role == Role.USER:
                converted.append({"role": "user", "content": msg.content})

            elif msg.role == Role.ASSISTANT:
                entry: dict[str, Any] = {"role": "assistant"}

                if msg.content:
                    entry["content"] = msg.content

                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]

                converted.append(entry)

            elif msg.role == Role.TOOL:
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )

        return converted

    def _convert_tools(self, tools: list[Tool] | None) -> list[dict[str, Any]] | None:
        """Convert tools to OpenAI format."""
        if not tools:
            return None

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _parse_response(self, response) -> LLMResponse:
        """Parse OpenAI response to LLMResponse."""
        choice = response.choices[0]
        message = choice.message

        content = message.content
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        return LLMResponse(
            content=content,
            tool_calls=tuple(tool_calls),
            stop_reason=choice.finish_reason or "stop",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=response.model,
        )

    def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Generate a completion using GPT.

        Args:
            messages: Conversation history.
            tools: Available tools (optional).
            temperature: Sampling temperature (0.0 - 2.0).
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with content and/or tool calls.

        Raises:
            LLMError: If completion fails.
        """
        try:
            client = self._get_client()

            converted_messages = self._convert_messages(messages)
            converted_tools = self._convert_tools(tools)

            # gpt-5.x and gpt-4.1 models use max_completion_tokens
            token_param = (
                "max_completion_tokens"
                if _uses_completion_tokens(self.model)
                else "max_tokens"
            )

            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": converted_messages,
                token_param: max_tokens,
                "temperature": temperature,
            }

            if converted_tools:
                kwargs["tools"] = converted_tools

            response = client.chat.completions.create(**kwargs)

            return self._parse_response(response)

        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"OpenAI completion failed: {e}", cause=e)


__all__ = ["OpenAILLM", "OPENAI_MODELS"]
