"""
Tests for llms/anthropic.py - Anthropic Claude LLM implementation.
"""

from unittest.mock import MagicMock, patch

import pytest
from pullcite.llms.base import (
    LLMError,
    LLMResponse,
    Message,
    Role,
    Tool,
    ToolCall,
)
from pullcite.llms.anthropic import AnthropicLLM, ANTHROPIC_MODELS


class TestAnthropicModels:
    """Test model configurations."""

    def test_supported_models(self):
        assert "claude-opus-4-5-20251101" in ANTHROPIC_MODELS
        assert "claude-sonnet-4-20250514" in ANTHROPIC_MODELS
        assert "claude-3-5-sonnet-20241022" in ANTHROPIC_MODELS
        assert "claude-3-5-haiku-20241022" in ANTHROPIC_MODELS
        assert "claude-3-opus-20240229" in ANTHROPIC_MODELS
        assert "claude-3-sonnet-20240229" in ANTHROPIC_MODELS
        assert "claude-3-haiku-20240307" in ANTHROPIC_MODELS

    def test_model_max_tokens(self):
        assert ANTHROPIC_MODELS["claude-opus-4-5-20251101"]["max_tokens"] == 8192
        assert ANTHROPIC_MODELS["claude-sonnet-4-20250514"]["max_tokens"] == 8192
        assert ANTHROPIC_MODELS["claude-3-opus-20240229"]["max_tokens"] == 4096


class TestAnthropicLLMInit:
    """Test AnthropicLLM initialization."""

    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(LLMError) as exc:
                AnthropicLLM()
            assert "API key required" in str(exc.value)

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            llm = AnthropicLLM()
            assert llm.api_key == "test-key"

    def test_api_key_from_parameter(self):
        llm = AnthropicLLM(api_key="explicit-key")
        assert llm.api_key == "explicit-key"

    def test_unknown_model_raises(self):
        with pytest.raises(LLMError) as exc:
            AnthropicLLM(api_key="key", model="unknown-model")
        assert "Unknown model" in str(exc.value)
        assert "unknown-model" in str(exc.value)

    def test_default_model(self):
        llm = AnthropicLLM(api_key="key")
        assert llm.model == "claude-sonnet-4-20250514"

    def test_model_name_property(self):
        llm = AnthropicLLM(api_key="key", model="claude-3-opus-20240229")
        assert llm.model_name == "claude-3-opus-20240229"


class TestAnthropicMessageConversion:
    """Test message format conversion."""

    def test_convert_system_message(self):
        llm = AnthropicLLM(api_key="key")
        messages = [Message.system("You are helpful.")]
        system_prompt, converted = llm._convert_messages(messages)

        assert system_prompt == "You are helpful."
        assert converted == []

    def test_convert_user_message(self):
        llm = AnthropicLLM(api_key="key")
        messages = [Message.user("Hello!")]
        system_prompt, converted = llm._convert_messages(messages)

        assert system_prompt is None
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Hello!"

    def test_convert_assistant_text_only(self):
        llm = AnthropicLLM(api_key="key")
        messages = [Message.assistant(content="Hi there!")]
        system_prompt, converted = llm._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        # Anthropic uses content blocks
        assert converted[0]["content"] == [{"type": "text", "text": "Hi there!"}]

    def test_convert_assistant_with_tool_calls(self):
        llm = AnthropicLLM(api_key="key")
        tc = ToolCall(id="tc_1", name="search", arguments={"query": "test"})
        messages = [Message.assistant(content="Searching...", tool_calls=[tc])]
        system_prompt, converted = llm._convert_messages(messages)

        assert len(converted) == 1
        msg = converted[0]
        assert msg["role"] == "assistant"
        content = msg["content"]
        assert len(content) == 2  # text block + tool_use block
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Searching..."
        assert content[1]["type"] == "tool_use"
        assert content[1]["id"] == "tc_1"
        assert content[1]["name"] == "search"
        assert content[1]["input"] == {"query": "test"}

    def test_convert_assistant_tool_calls_only(self):
        llm = AnthropicLLM(api_key="key")
        tc = ToolCall(id="tc_1", name="search", arguments={})
        messages = [Message.assistant(tool_calls=[tc])]
        system_prompt, converted = llm._convert_messages(messages)

        msg = converted[0]
        assert msg["role"] == "assistant"
        content = msg["content"]
        assert len(content) == 1  # Only tool_use block
        assert content[0]["type"] == "tool_use"

    def test_convert_tool_result(self):
        llm = AnthropicLLM(api_key="key")
        messages = [
            Message.tool_result(
                tool_call_id="tc_1", name="search", content="Found 5 results"
            )
        ]
        system_prompt, converted = llm._convert_messages(messages)

        assert len(converted) == 1
        msg = converted[0]
        # Anthropic puts tool results in user messages
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["tool_use_id"] == "tc_1"
        assert msg["content"][0]["content"] == "Found 5 results"

    def test_convert_full_conversation(self):
        llm = AnthropicLLM(api_key="key")
        tc = ToolCall(id="tc_1", name="get_weather", arguments={"location": "NYC"})
        messages = [
            Message.system("You are a weather assistant."),
            Message.user("What's the weather in NYC?"),
            Message.assistant(tool_calls=[tc]),
            Message.tool_result(
                tool_call_id="tc_1",
                name="get_weather",
                content='{"temp": 72}',
            ),
            Message.assistant(content="It's 72 degrees in NYC."),
        ]
        system_prompt, converted = llm._convert_messages(messages)

        assert system_prompt == "You are a weather assistant."
        assert len(converted) == 4  # System is separate
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"
        assert converted[2]["role"] == "user"  # tool_result
        assert converted[3]["role"] == "assistant"


class TestAnthropicToolConversion:
    """Test tool format conversion."""

    def test_convert_none_tools(self):
        llm = AnthropicLLM(api_key="key")
        result = llm._convert_tools(None)
        assert result is None

    def test_convert_empty_tools(self):
        llm = AnthropicLLM(api_key="key")
        result = llm._convert_tools([])
        assert result is None

    def test_convert_single_tool(self):
        llm = AnthropicLLM(api_key="key")
        tools = [
            Tool(
                name="get_weather",
                description="Get weather for a location.",
                parameters={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            )
        ]
        converted = llm._convert_tools(tools)

        assert len(converted) == 1
        assert converted[0]["name"] == "get_weather"
        assert converted[0]["description"] == "Get weather for a location."
        assert converted[0]["input_schema"]["type"] == "object"


class TestAnthropicResponseParsing:
    """Test response parsing."""

    def test_parse_text_response(self):
        llm = AnthropicLLM(api_key="key")

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello!"

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.model = "claude-sonnet-4-20250514"

        result = llm._parse_response(mock_response)

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.tool_calls == ()
        assert result.stop_reason == "end_turn"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.model == "claude-sonnet-4-20250514"

    def test_parse_tool_call_response(self):
        llm = AnthropicLLM(api_key="key")

        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "toolu_123"
        mock_tool_use.name = "search"
        mock_tool_use.input = {"query": "test"}

        mock_response = MagicMock()
        mock_response.content = [mock_tool_use]
        mock_response.stop_reason = "tool_use"
        mock_response.usage.input_tokens = 20
        mock_response.usage.output_tokens = 15
        mock_response.model = "claude-sonnet-4-20250514"

        result = llm._parse_response(mock_response)

        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "toolu_123"
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"query": "test"}
        assert result.stop_reason == "tool_use"

    def test_parse_mixed_response(self):
        llm = AnthropicLLM(api_key="key")

        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Let me search."

        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "toolu_1"
        mock_tool_use.name = "search"
        mock_tool_use.input = {}

        mock_response = MagicMock()
        mock_response.content = [mock_text, mock_tool_use]
        mock_response.stop_reason = "tool_use"
        mock_response.usage.input_tokens = 30
        mock_response.usage.output_tokens = 20
        mock_response.model = "claude-sonnet-4-20250514"

        result = llm._parse_response(mock_response)

        assert result.content == "Let me search."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"


class TestAnthropicComplete:
    """Test the complete method."""

    def test_complete_basic(self):
        llm = AnthropicLLM(api_key="test-key")

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello!"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.model = "claude-sonnet-4-20250514"

        mock_client.messages.create.return_value = mock_response

        with patch.object(llm, "_get_client", return_value=mock_client):
            result = llm.complete([Message.user("Hello")])

        assert result.content == "Hello!"
        assert result.model == "claude-sonnet-4-20250514"

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 4096

    def test_complete_with_system(self):
        llm = AnthropicLLM(api_key="test-key")

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hi!"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 5
        mock_response.model = "claude-sonnet-4-20250514"

        mock_client.messages.create.return_value = mock_response

        with patch.object(llm, "_get_client", return_value=mock_client):
            llm.complete(
                [
                    Message.system("You are helpful."),
                    Message.user("Hello"),
                ]
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful."

    def test_complete_with_tools(self):
        llm = AnthropicLLM(api_key="test-key")

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Using tool"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 20
        mock_response.usage.output_tokens = 10
        mock_response.model = "claude-sonnet-4-20250514"

        mock_client.messages.create.return_value = mock_response

        tools = [Tool(name="search", description="Search for info")]

        with patch.object(llm, "_get_client", return_value=mock_client):
            llm.complete([Message.user("Search for X")], tools=tools)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1

    def test_complete_api_error(self):
        llm = AnthropicLLM(api_key="test-key")

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API Error")

        with patch.object(llm, "_get_client", return_value=mock_client):
            with pytest.raises(LLMError) as exc:
                llm.complete([Message.user("Hello")])
            assert "Anthropic completion failed" in str(exc.value)
            assert exc.value.cause is not None


class TestAnthropicIntegration:
    """Integration-style tests (still mocked)."""

    def test_full_conversation_flow(self):
        """Test a complete conversation with tool use."""
        llm = AnthropicLLM(api_key="test-key")

        mock_client = MagicMock()

        # First response: tool call
        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "toolu_1"
        mock_tool_use.name = "get_weather"
        mock_tool_use.input = {"location": "NYC"}

        mock_response1 = MagicMock()
        mock_response1.content = [mock_tool_use]
        mock_response1.stop_reason = "tool_use"
        mock_response1.usage.input_tokens = 20
        mock_response1.usage.output_tokens = 15
        mock_response1.model = "claude-sonnet-4-20250514"

        # Second response: final answer
        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "It's 72 degrees in NYC."

        mock_response2 = MagicMock()
        mock_response2.content = [mock_text]
        mock_response2.stop_reason = "end_turn"
        mock_response2.usage.input_tokens = 40
        mock_response2.usage.output_tokens = 10
        mock_response2.model = "claude-sonnet-4-20250514"

        mock_client.messages.create.side_effect = [mock_response1, mock_response2]

        with patch.object(llm, "_get_client", return_value=mock_client):
            # First call: get tool call
            response1 = llm.complete([Message.user("What's the weather in NYC?")])
            assert response1.has_tool_calls
            assert response1.tool_calls[0].name == "get_weather"

            # Build next messages with tool result
            messages = [
                Message.user("What's the weather in NYC?"),
                Message.assistant(tool_calls=list(response1.tool_calls)),
                Message.tool_result(
                    tool_call_id="toolu_1",
                    name="get_weather",
                    content='{"temp": 72}',
                ),
            ]

            # Second call: get final answer
            response2 = llm.complete(messages)
            assert response2.content == "It's 72 degrees in NYC."
            assert not response2.has_tool_calls
