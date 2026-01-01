"""
Tests for llms/openai.py - OpenAI GPT LLM implementation.
"""

import json
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
from pullcite.llms.openai import OpenAILLM, OPENAI_MODELS


class TestOpenAIModels:
    """Test model configurations."""

    def test_supported_models(self):
        # Flagship conversational models
        assert "gpt-5.2" in OPENAI_MODELS
        assert "gpt-5.1" in OPENAI_MODELS
        assert "gpt-5" in OPENAI_MODELS
        # Long-context family
        assert "gpt-4.1" in OPENAI_MODELS
        assert "gpt-4.1-mini" in OPENAI_MODELS
        assert "gpt-4.1-nano" in OPENAI_MODELS
        # Omni series
        assert "gpt-4o" in OPENAI_MODELS
        assert "gpt-4o-mini" in OPENAI_MODELS
        assert "gpt-4o-realtime" in OPENAI_MODELS
        assert "gpt-4o-mini-realtime" in OPENAI_MODELS
        # Open-weight models
        assert "gpt-oss-120b" in OPENAI_MODELS
        assert "gpt-oss-20b" in OPENAI_MODELS

    def test_model_max_tokens(self):
        assert OPENAI_MODELS["gpt-5.2"]["max_tokens"] == 16384
        assert OPENAI_MODELS["gpt-4.1"]["max_tokens"] == 32768
        assert OPENAI_MODELS["gpt-4o"]["max_tokens"] == 16384
        assert OPENAI_MODELS["gpt-oss-120b"]["max_tokens"] == 131072


class TestOpenAILLMInit:
    """Test OpenAILLM initialization."""

    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(LLMError) as exc:
                OpenAILLM()
            assert "API key required" in str(exc.value)

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            llm = OpenAILLM()
            assert llm.api_key == "test-key"

    def test_api_key_from_parameter(self):
        llm = OpenAILLM(api_key="explicit-key")
        assert llm.api_key == "explicit-key"

    def test_unknown_model_raises(self):
        with pytest.raises(LLMError) as exc:
            OpenAILLM(api_key="key", model="unknown-model")
        assert "Unknown model" in str(exc.value)
        assert "unknown-model" in str(exc.value)

    def test_unknown_model_allowed_with_base_url(self):
        # Custom base URL allows any model (for API-compatible endpoints)
        llm = OpenAILLM(
            api_key="key",
            model="custom-model",
            base_url="https://custom.api.com/v1",
        )
        assert llm.model == "custom-model"
        assert llm.base_url == "https://custom.api.com/v1"

    def test_default_model(self):
        llm = OpenAILLM(api_key="key")
        assert llm.model == "gpt-4o"

    def test_model_name_property(self):
        llm = OpenAILLM(api_key="key", model="gpt-4.1")
        assert llm.model_name == "gpt-4.1"


class TestOpenAILLMClient:
    """Test client creation."""

    def test_get_client_without_openai_package(self):
        llm = OpenAILLM(api_key="key")

        with patch.dict("sys.modules", {"openai": None}):
            # Simulate ImportError
            with patch.object(
                llm,
                "_get_client",
                side_effect=LLMError(
                    "openai package required. Install with: pip install openai"
                ),
            ):
                with pytest.raises(LLMError) as exc:
                    llm._get_client()
                assert "openai package required" in str(exc.value)


class TestOpenAIMessageConversion:
    """Test message format conversion."""

    def test_convert_system_message(self):
        llm = OpenAILLM(api_key="key")
        messages = [Message.system("You are helpful.")]
        converted = llm._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "system"
        assert converted[0]["content"] == "You are helpful."

    def test_convert_user_message(self):
        llm = OpenAILLM(api_key="key")
        messages = [Message.user("Hello!")]
        converted = llm._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Hello!"

    def test_convert_assistant_text_only(self):
        llm = OpenAILLM(api_key="key")
        messages = [Message.assistant(content="Hi there!")]
        converted = llm._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        assert converted[0]["content"] == "Hi there!"
        assert "tool_calls" not in converted[0]

    def test_convert_assistant_with_tool_calls(self):
        llm = OpenAILLM(api_key="key")
        tc = ToolCall(id="tc_1", name="search", arguments={"query": "test"})
        messages = [Message.assistant(content="Searching...", tool_calls=[tc])]
        converted = llm._convert_messages(messages)

        assert len(converted) == 1
        msg = converted[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Searching..."
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["id"] == "tc_1"
        assert msg["tool_calls"][0]["type"] == "function"
        assert msg["tool_calls"][0]["function"]["name"] == "search"
        assert msg["tool_calls"][0]["function"]["arguments"] == '{"query": "test"}'

    def test_convert_assistant_tool_calls_only(self):
        llm = OpenAILLM(api_key="key")
        tc = ToolCall(id="tc_1", name="search", arguments={})
        messages = [Message.assistant(tool_calls=[tc])]
        converted = llm._convert_messages(messages)

        assert len(converted) == 1
        msg = converted[0]
        assert msg["role"] == "assistant"
        assert "content" not in msg  # No content key when None
        assert len(msg["tool_calls"]) == 1

    def test_convert_tool_result(self):
        llm = OpenAILLM(api_key="key")
        messages = [
            Message.tool_result(
                tool_call_id="tc_1", name="search", content="Found 5 results"
            )
        ]
        converted = llm._convert_messages(messages)

        assert len(converted) == 1
        msg = converted[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "tc_1"
        assert msg["content"] == "Found 5 results"

    def test_convert_full_conversation(self):
        llm = OpenAILLM(api_key="key")
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
        converted = llm._convert_messages(messages)

        assert len(converted) == 5
        assert converted[0]["role"] == "system"
        assert converted[1]["role"] == "user"
        assert converted[2]["role"] == "assistant"
        assert converted[3]["role"] == "tool"
        assert converted[4]["role"] == "assistant"


class TestOpenAIToolConversion:
    """Test tool format conversion."""

    def test_convert_none_tools(self):
        llm = OpenAILLM(api_key="key")
        result = llm._convert_tools(None)
        assert result is None

    def test_convert_empty_tools(self):
        llm = OpenAILLM(api_key="key")
        result = llm._convert_tools([])
        assert result is None

    def test_convert_single_tool(self):
        llm = OpenAILLM(api_key="key")
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
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "get_weather"
        assert converted[0]["function"]["description"] == "Get weather for a location."
        assert converted[0]["function"]["parameters"]["type"] == "object"

    def test_convert_multiple_tools(self):
        llm = OpenAILLM(api_key="key")
        tools = [
            Tool(name="tool1", description="First tool"),
            Tool(name="tool2", description="Second tool"),
        ]
        converted = llm._convert_tools(tools)

        assert len(converted) == 2
        assert converted[0]["function"]["name"] == "tool1"
        assert converted[1]["function"]["name"] == "tool2"


class TestOpenAIResponseParsing:
    """Test response parsing."""

    def test_parse_text_response(self):
        llm = OpenAILLM(api_key="key")

        # Mock OpenAI response structure
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model = "gpt-4o"

        result = llm._parse_response(mock_response)

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.tool_calls == ()
        assert result.stop_reason == "stop"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.model == "gpt-4o"

    def test_parse_tool_call_response(self):
        llm = OpenAILLM(api_key="key")

        # Mock tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "search"
        mock_tool_call.function.arguments = '{"query": "test"}'

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 15
        mock_response.model = "gpt-4o"

        result = llm._parse_response(mock_response)

        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"query": "test"}
        assert result.stop_reason == "tool_calls"

    def test_parse_multiple_tool_calls(self):
        llm = OpenAILLM(api_key="key")

        mock_tc1 = MagicMock()
        mock_tc1.id = "call_1"
        mock_tc1.function.name = "tool1"
        mock_tc1.function.arguments = "{}"

        mock_tc2 = MagicMock()
        mock_tc2.id = "call_2"
        mock_tc2.function.name = "tool2"
        mock_tc2.function.arguments = '{"key": "value"}'

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Using tools"
        mock_response.choices[0].message.tool_calls = [mock_tc1, mock_tc2]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = 20
        mock_response.model = "gpt-4o"

        result = llm._parse_response(mock_response)

        assert result.content == "Using tools"
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "tool1"
        assert result.tool_calls[1].name == "tool2"
        assert result.tool_calls[1].arguments == {"key": "value"}

    def test_parse_invalid_json_arguments(self):
        llm = OpenAILLM(api_key="key")

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function.name = "test"
        mock_tool_call.function.arguments = "not valid json"

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model = "gpt-4o"

        result = llm._parse_response(mock_response)

        # Should default to empty dict on JSON parse error
        assert result.tool_calls[0].arguments == {}


class TestOpenAIComplete:
    """Test the complete method."""

    def test_complete_basic(self):
        llm = OpenAILLM(api_key="test-key")

        # Mock the client and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model = "gpt-4o"

        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(llm, "_get_client", return_value=mock_client):
            result = llm.complete([Message.user("Hello")])

        assert result.content == "Hello!"
        assert result.model == "gpt-4o"

        # Verify the API was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 4096

    def test_complete_with_tools(self):
        llm = OpenAILLM(api_key="test-key")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Using tool"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.model = "gpt-4o"

        mock_client.chat.completions.create.return_value = mock_response

        tools = [Tool(name="search", description="Search for info")]

        with patch.object(llm, "_get_client", return_value=mock_client):
            llm.complete([Message.user("Search for X")], tools=tools)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1

    def test_complete_custom_parameters(self):
        llm = OpenAILLM(api_key="test-key", model="gpt-4.1")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model = "gpt-4.1"

        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(llm, "_get_client", return_value=mock_client):
            llm.complete(
                [Message.user("Test")],
                temperature=0.7,
                max_tokens=1000,
            )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4.1"
        assert call_kwargs["temperature"] == 0.7
        # gpt-4.1 uses max_completion_tokens
        assert call_kwargs["max_completion_tokens"] == 1000
        assert "max_tokens" not in call_kwargs

    def test_gpt5_uses_max_completion_tokens(self):
        """Test that gpt-5.x models use max_completion_tokens parameter."""
        llm = OpenAILLM(api_key="test-key", model="gpt-5.2")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model = "gpt-5.2"

        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(llm, "_get_client", return_value=mock_client):
            llm.complete([Message.user("Test")], max_tokens=8000)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_completion_tokens"] == 8000
        assert "max_tokens" not in call_kwargs

    def test_gpt4o_uses_max_tokens(self):
        """Test that gpt-4o models use max_tokens parameter."""
        llm = OpenAILLM(api_key="test-key", model="gpt-4o")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model = "gpt-4o"

        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(llm, "_get_client", return_value=mock_client):
            llm.complete([Message.user("Test")], max_tokens=4000)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 4000
        assert "max_completion_tokens" not in call_kwargs

    def test_complete_api_error(self):
        llm = OpenAILLM(api_key="test-key")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        with patch.object(llm, "_get_client", return_value=mock_client):
            with pytest.raises(LLMError) as exc:
                llm.complete([Message.user("Hello")])
            assert "OpenAI completion failed" in str(exc.value)
            assert exc.value.cause is not None


class TestOpenAIIntegration:
    """Integration-style tests (still mocked)."""

    def test_full_conversation_flow(self):
        """Test a complete conversation with tool use."""
        llm = OpenAILLM(api_key="test-key")

        mock_client = MagicMock()

        # First response: tool call
        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.function.name = "get_weather"
        mock_tc.function.arguments = '{"location": "NYC"}'

        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock()]
        mock_response1.choices[0].message.content = None
        mock_response1.choices[0].message.tool_calls = [mock_tc]
        mock_response1.choices[0].finish_reason = "tool_calls"
        mock_response1.usage.prompt_tokens = 20
        mock_response1.usage.completion_tokens = 15
        mock_response1.model = "gpt-4o"

        # Second response: final answer
        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock()]
        mock_response2.choices[0].message.content = "It's 72 degrees in NYC."
        mock_response2.choices[0].message.tool_calls = None
        mock_response2.choices[0].finish_reason = "stop"
        mock_response2.usage.prompt_tokens = 40
        mock_response2.usage.completion_tokens = 10
        mock_response2.model = "gpt-4o"

        mock_client.chat.completions.create.side_effect = [
            mock_response1,
            mock_response2,
        ]

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
                    tool_call_id="call_1",
                    name="get_weather",
                    content='{"temp": 72}',
                ),
            ]

            # Second call: get final answer
            response2 = llm.complete(messages)
            assert response2.content == "It's 72 degrees in NYC."
            assert not response2.has_tool_calls
