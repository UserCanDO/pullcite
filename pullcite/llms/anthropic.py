"""
Anthropic Claude LLM implementation.

Uses the Anthropic API to generate completions with Claude models.
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
ANTHROPIC_MODELS = {
    # Claude 4.5 (latest)
    "claude-opus-4-5-20251101": {"max_tokens": 8192},
    # Claude 4
    "claude-sonnet-4-20250514": {"max_tokens": 8192},
    # Claude 3.5
    "claude-3-5-sonnet-20241022": {"max_tokens": 8192},
    "claude-3-5-haiku-20241022": {"max_tokens": 8192},
    # Claude 3
    "claude-3-opus-20240229": {"max_tokens": 4096},
    "claude-3-sonnet-20240229": {"max_tokens": 4096},
    "claude-3-haiku-20240307": {"max_tokens": 4096},
}


@dataclass
class AnthropicLLM(LLM):
    """
    Anthropic Claude LLM provider.

    Uses the Anthropic API to generate completions. Requires anthropic package.

    Attributes:
        api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
        model: Model name. Defaults to claude-sonnet-4-20250514.

    Example:
        >>> llm = AnthropicLLM()
        >>> response = llm.complete([Message.user("Hello!")])
        >>> print(response.content)
    """

    api_key: str | None = None
    model: str = "claude-sonnet-4-20250514"

    def __post_init__(self) -> None:
        """Initialize and validate."""
        if self.api_key is None:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise LLMError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key."
            )

        if self.model not in ANTHROPIC_MODELS:
            raise LLMError(
                f"Unknown model: {self.model}. "
                f"Supported: {list(ANTHROPIC_MODELS.keys())}"
            )

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self.model

    def _get_client(self):
        """Get Anthropic client (lazy import)."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise LLMError(
                "anthropic package required. Install with: pip install anthropic"
            )

        return Anthropic(api_key=self.api_key)

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert messages to Anthropic format.

        Returns:
            Tuple of (system_prompt, messages_list)
        """
        system_prompt = None
        converted = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content
                continue

            if msg.role == Role.USER:
                converted.append({"role": "user", "content": msg.content})

            elif msg.role == Role.ASSISTANT:
                content: list[dict[str, Any]] = []

                if msg.content:
                    content.append({"type": "text", "text": msg.content})

                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )

                converted.append(
                    {"role": "assistant", "content": content or msg.content}
                )

            elif msg.role == Role.TOOL:
                # Anthropic uses tool_result in user messages
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )

        return system_prompt, converted

    def _convert_tools(self, tools: list[Tool] | None) -> list[dict[str, Any]] | None:
        """Convert tools to Anthropic format."""
        if not tools:
            return None

        return [tool.to_dict() for tool in tools]

    def _parse_response(self, response) -> LLMResponse:
        """Parse Anthropic response to LLMResponse."""
        content = None
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        return LLMResponse(
            content=content,
            tool_calls=tuple(tool_calls),
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
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
        Generate a completion using Claude.

        Args:
            messages: Conversation history.
            tools: Available tools (optional).
            temperature: Sampling temperature (0.0 - 1.0).
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with content and/or tool calls.

        Raises:
            LLMError: If completion fails.
        """
        try:
            client = self._get_client()

            system_prompt, converted_messages = self._convert_messages(messages)
            converted_tools = self._convert_tools(tools)

            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": converted_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            if converted_tools:
                kwargs["tools"] = converted_tools

            response = client.messages.create(**kwargs)

            return self._parse_response(response)

        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Anthropic completion failed: {e}", cause=e)


__all__ = ["AnthropicLLM", "ANTHROPIC_MODELS"]
