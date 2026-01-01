"""
Base LLM interface.

This module defines the abstract interface for LLM providers
and the tool calling protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    """
    A message in a conversation.

    Attributes:
        role: Who sent the message.
        content: Text content (may be None for tool calls).
        tool_calls: Tool calls made by assistant.
        tool_call_id: ID when this is a tool response.
        name: Tool name when role is TOOL.
    """

    role: Role
    content: str | None = None
    tool_calls: tuple["ToolCall", ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(
        cls,
        content: str | None = None,
        tool_calls: list["ToolCall"] | None = None,
    ) -> "Message":
        """Create an assistant message."""
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tuple(tool_calls or []),
        )

    @classmethod
    def tool_result(
        cls,
        tool_call_id: str,
        name: str,
        content: str,
    ) -> "Message":
        """Create a tool result message."""
        return cls(
            role=Role.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )


@dataclass(frozen=True)
class Tool:
    """
    A tool that the LLM can call.

    Attributes:
        name: Tool identifier.
        description: What the tool does.
        parameters: JSON Schema for parameters.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to API format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass(frozen=True)
class ToolCall:
    """
    A tool call made by the LLM.

    Attributes:
        id: Unique identifier for this call.
        name: Tool name.
        arguments: Parsed arguments dict.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    """
    Response from an LLM call.

    Attributes:
        content: Text response (may be None if only tool calls).
        tool_calls: Tool calls to execute.
        stop_reason: Why generation stopped.
        input_tokens: Tokens in prompt.
        output_tokens: Tokens generated.
        model: Model used.
    """

    content: str | None
    tool_calls: tuple[ToolCall, ...]
    stop_reason: str
    input_tokens: int
    output_tokens: int
    model: str

    @property
    def has_tool_calls(self) -> bool:
        """Check if response includes tool calls."""
        return len(self.tool_calls) > 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


class LLM(ABC):
    """
    Abstract base class for LLM providers.

    LLMs generate text and can use tools. Implementations handle
    API details for specific providers.

    Example:
        >>> llm = AnthropicLLM(api_key="...")
        >>> response = llm.complete([Message.user("Hello")])
        >>> print(response.content)
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Generate a completion.

        Args:
            messages: Conversation history.
            tools: Available tools (optional).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with content and/or tool calls.

        Raises:
            LLMError: If completion fails.
        """
        ...

    def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[Tool],
        tool_executor: "ToolExecutor",
        max_rounds: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> tuple[LLMResponse, list[Message]]:
        """
        Complete with automatic tool execution.

        Runs a loop: LLM generates -> execute tools -> feed results back
        until LLM stops calling tools or max_rounds reached.

        Args:
            messages: Initial conversation.
            tools: Available tools.
            tool_executor: Executes tool calls.
            max_rounds: Maximum tool call rounds.
            temperature: Sampling temperature.
            max_tokens: Max tokens per completion.

        Returns:
            Tuple of (final response, full message history).

        Raises:
            LLMError: If completion fails.
            MaxRoundsExceeded: If max_rounds reached.
        """
        history = list(messages)

        for _ in range(max_rounds):
            response = self.complete(
                messages=history,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Add assistant response to history
            history.append(
                Message.assistant(
                    content=response.content,
                    tool_calls=list(response.tool_calls),
                )
            )

            # If no tool calls, we're done
            if not response.has_tool_calls:
                return response, history

            # Execute tools and add results
            for tool_call in response.tool_calls:
                result = tool_executor.execute(tool_call)
                history.append(
                    Message.tool_result(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        content=result,
                    )
                )

        raise MaxRoundsExceeded(f"Exceeded {max_rounds} tool call rounds")


class ToolExecutor(ABC):
    """
    Executes tool calls.

    Implement this to define what happens when tools are called.
    """

    @abstractmethod
    def execute(self, tool_call: ToolCall) -> str:
        """
        Execute a tool call and return result.

        Args:
            tool_call: The tool call to execute.

        Returns:
            String result to send back to LLM.
        """
        ...


class LLMError(Exception):
    """Raised when LLM call fails."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause


class MaxRoundsExceeded(LLMError):
    """Raised when tool call loop exceeds max rounds."""

    pass


__all__ = [
    "LLM",
    "LLMResponse",
    "LLMError",
    "MaxRoundsExceeded",
    "Message",
    "Role",
    "Tool",
    "ToolCall",
    "ToolExecutor",
]
