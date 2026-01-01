"""
Tests for llms/base.py - LLM ABC and message/tool types.
"""

import pytest
from pullcite.llms.base import (
    LLM,
    LLMError,
    LLMResponse,
    MaxRoundsExceeded,
    Message,
    Role,
    Tool,
    ToolCall,
    ToolExecutor,
)


class TestRole:
    """Test Role enum."""

    def test_role_values(self):
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.TOOL == "tool"

    def test_role_is_string(self):
        # Role inherits from str
        assert isinstance(Role.USER, str)
        assert Role.USER == "user"


class TestMessage:
    """Test Message dataclass."""

    def test_system_message(self):
        msg = Message.system("You are a helpful assistant.")
        assert msg.role == Role.SYSTEM
        assert msg.content == "You are a helpful assistant."
        assert msg.tool_calls == ()
        assert msg.tool_call_id is None
        assert msg.name is None

    def test_user_message(self):
        msg = Message.user("Hello!")
        assert msg.role == Role.USER
        assert msg.content == "Hello!"

    def test_assistant_message_text_only(self):
        msg = Message.assistant(content="Hi there!")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Hi there!"
        assert msg.tool_calls == ()

    def test_assistant_message_with_tool_calls(self):
        tool_call = ToolCall(id="tc_1", name="search", arguments={"query": "test"})
        msg = Message.assistant(content="Let me search.", tool_calls=[tool_call])

        assert msg.role == Role.ASSISTANT
        assert msg.content == "Let me search."
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"

    def test_assistant_message_tool_calls_only(self):
        tool_call = ToolCall(id="tc_1", name="search", arguments={})
        msg = Message.assistant(tool_calls=[tool_call])

        assert msg.role == Role.ASSISTANT
        assert msg.content is None
        assert len(msg.tool_calls) == 1

    def test_tool_result_message(self):
        msg = Message.tool_result(
            tool_call_id="tc_1",
            name="search",
            content="Found 5 results.",
        )
        assert msg.role == Role.TOOL
        assert msg.content == "Found 5 results."
        assert msg.tool_call_id == "tc_1"
        assert msg.name == "search"

    def test_message_immutability(self):
        msg = Message.user("Hello")
        with pytest.raises(AttributeError):
            msg.content = "Modified"


class TestTool:
    """Test Tool dataclass."""

    def test_basic_creation(self):
        tool = Tool(
            name="get_weather",
            description="Get current weather for a location.",
        )
        assert tool.name == "get_weather"
        assert tool.description == "Get current weather for a location."
        assert tool.parameters == {}

    def test_with_parameters(self):
        tool = Tool(
            name="search",
            description="Search the database.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        )
        assert tool.parameters["type"] == "object"
        assert "query" in tool.parameters["properties"]

    def test_to_dict(self):
        tool = Tool(
            name="test",
            description="A test tool.",
            parameters={"type": "object"},
        )
        d = tool.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "A test tool."
        assert d["input_schema"] == {"type": "object"}

    def test_immutability(self):
        tool = Tool(name="test", description="Test")
        with pytest.raises(AttributeError):
            tool.name = "modified"


class TestToolCall:
    """Test ToolCall dataclass."""

    def test_basic_creation(self):
        tc = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"location": "NYC"},
        )
        assert tc.id == "call_123"
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "NYC"}

    def test_empty_arguments(self):
        tc = ToolCall(id="call_1", name="no_args", arguments={})
        assert tc.arguments == {}

    def test_immutability(self):
        tc = ToolCall(id="1", name="test", arguments={})
        with pytest.raises(AttributeError):
            tc.id = "modified"


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_text_response(self):
        response = LLMResponse(
            content="Hello!",
            tool_calls=(),
            stop_reason="stop",
            input_tokens=10,
            output_tokens=5,
            model="test-model",
        )
        assert response.content == "Hello!"
        assert response.tool_calls == ()
        assert response.stop_reason == "stop"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.model == "test-model"

    def test_has_tool_calls_false(self):
        response = LLMResponse(
            content="Hi",
            tool_calls=(),
            stop_reason="stop",
            input_tokens=10,
            output_tokens=5,
            model="test",
        )
        assert response.has_tool_calls is False

    def test_has_tool_calls_true(self):
        tc = ToolCall(id="1", name="test", arguments={})
        response = LLMResponse(
            content=None,
            tool_calls=(tc,),
            stop_reason="tool_use",
            input_tokens=10,
            output_tokens=5,
            model="test",
        )
        assert response.has_tool_calls is True

    def test_total_tokens(self):
        response = LLMResponse(
            content="Hi",
            tool_calls=(),
            stop_reason="stop",
            input_tokens=100,
            output_tokens=50,
            model="test",
        )
        assert response.total_tokens == 150

    def test_immutability(self):
        response = LLMResponse(
            content="Hi",
            tool_calls=(),
            stop_reason="stop",
            input_tokens=10,
            output_tokens=5,
            model="test",
        )
        with pytest.raises(AttributeError):
            response.content = "Modified"


class TestLLMError:
    """Test LLMError exception."""

    def test_basic_error(self):
        error = LLMError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.cause is None

    def test_error_with_cause(self):
        cause = ValueError("Original error")
        error = LLMError("Wrapper message", cause=cause)
        assert error.cause is cause


class TestMaxRoundsExceeded:
    """Test MaxRoundsExceeded exception."""

    def test_inherits_from_llm_error(self):
        error = MaxRoundsExceeded("Too many rounds")
        assert isinstance(error, LLMError)


class TestToolExecutor:
    """Test ToolExecutor ABC."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ToolExecutor()

    def test_concrete_implementation(self):
        class SimpleExecutor(ToolExecutor):
            def execute(self, tool_call: ToolCall) -> str:
                return f"Executed {tool_call.name}"

        executor = SimpleExecutor()
        tc = ToolCall(id="1", name="test", arguments={})
        result = executor.execute(tc)
        assert result == "Executed test"


class TestLLMABC:
    """Test LLM abstract base class."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            LLM()

    def test_concrete_implementation(self):
        """Test that a concrete implementation works."""

        class MockLLM(LLM):
            @property
            def model_name(self) -> str:
                return "mock-model"

            def complete(
                self,
                messages: list[Message],
                tools: list[Tool] | None = None,
                temperature: float = 0.0,
                max_tokens: int = 4096,
            ) -> LLMResponse:
                return LLMResponse(
                    content="Mock response",
                    tool_calls=(),
                    stop_reason="stop",
                    input_tokens=10,
                    output_tokens=5,
                    model=self.model_name,
                )

        llm = MockLLM()
        assert llm.model_name == "mock-model"

        response = llm.complete([Message.user("Hello")])
        assert response.content == "Mock response"

    def test_complete_with_tools_simple_case(self):
        """Test complete_with_tools when LLM doesn't call tools."""

        class NoToolsLLM(LLM):
            @property
            def model_name(self) -> str:
                return "no-tools"

            def complete(
                self,
                messages: list[Message],
                tools: list[Tool] | None = None,
                temperature: float = 0.0,
                max_tokens: int = 4096,
            ) -> LLMResponse:
                return LLMResponse(
                    content="I don't need tools",
                    tool_calls=(),
                    stop_reason="stop",
                    input_tokens=10,
                    output_tokens=5,
                    model=self.model_name,
                )

        class DummyExecutor(ToolExecutor):
            def execute(self, tool_call: ToolCall) -> str:
                raise AssertionError("Should not be called")

        llm = NoToolsLLM()
        tools = [Tool(name="test", description="Test tool")]
        executor = DummyExecutor()

        response, history = llm.complete_with_tools(
            messages=[Message.user("Hello")],
            tools=tools,
            tool_executor=executor,
        )

        assert response.content == "I don't need tools"
        assert len(history) == 2  # User message + assistant response

    def test_complete_with_tools_one_round(self):
        """Test complete_with_tools with one tool call round."""

        class OneToolLLM(LLM):
            def __init__(self):
                self.call_count = 0

            @property
            def model_name(self) -> str:
                return "one-tool"

            def complete(
                self,
                messages: list[Message],
                tools: list[Tool] | None = None,
                temperature: float = 0.0,
                max_tokens: int = 4096,
            ) -> LLMResponse:
                self.call_count += 1
                if self.call_count == 1:
                    # First call: make a tool call
                    return LLMResponse(
                        content=None,
                        tool_calls=(
                            ToolCall(
                                id="tc_1", name="get_data", arguments={"key": "value"}
                            ),
                        ),
                        stop_reason="tool_use",
                        input_tokens=10,
                        output_tokens=5,
                        model=self.model_name,
                    )
                else:
                    # Second call: give final answer
                    return LLMResponse(
                        content="Got the data: result",
                        tool_calls=(),
                        stop_reason="stop",
                        input_tokens=20,
                        output_tokens=10,
                        model=self.model_name,
                    )

        class SimpleExecutor(ToolExecutor):
            def execute(self, tool_call: ToolCall) -> str:
                return "result"

        llm = OneToolLLM()
        tools = [Tool(name="get_data", description="Get data")]
        executor = SimpleExecutor()

        response, history = llm.complete_with_tools(
            messages=[Message.user("Get the data")],
            tools=tools,
            tool_executor=executor,
        )

        assert response.content == "Got the data: result"
        assert llm.call_count == 2
        # History: user, assistant (tool call), tool result, assistant (final)
        assert len(history) == 4
        assert history[0].role == Role.USER
        assert history[1].role == Role.ASSISTANT
        assert history[2].role == Role.TOOL
        assert history[3].role == Role.ASSISTANT

    def test_complete_with_tools_max_rounds_exceeded(self):
        """Test that max_rounds is enforced."""

        class InfiniteToolLLM(LLM):
            @property
            def model_name(self) -> str:
                return "infinite"

            def complete(
                self,
                messages: list[Message],
                tools: list[Tool] | None = None,
                temperature: float = 0.0,
                max_tokens: int = 4096,
            ) -> LLMResponse:
                # Always make a tool call
                return LLMResponse(
                    content=None,
                    tool_calls=(ToolCall(id="tc", name="loop", arguments={}),),
                    stop_reason="tool_use",
                    input_tokens=10,
                    output_tokens=5,
                    model=self.model_name,
                )

        class LoopExecutor(ToolExecutor):
            def execute(self, tool_call: ToolCall) -> str:
                return "continue"

        llm = InfiniteToolLLM()
        tools = [Tool(name="loop", description="Loop")]
        executor = LoopExecutor()

        with pytest.raises(MaxRoundsExceeded) as exc:
            llm.complete_with_tools(
                messages=[Message.user("Loop")],
                tools=tools,
                tool_executor=executor,
                max_rounds=3,
            )
        assert "3" in str(exc.value)


class TestConversationFlow:
    """Test realistic conversation patterns."""

    def test_multi_turn_conversation(self):
        """Test building a multi-turn conversation."""
        messages = [
            Message.system("You are a helpful assistant."),
            Message.user("What is 2 + 2?"),
            Message.assistant(content="2 + 2 equals 4."),
            Message.user("What about 3 + 3?"),
            Message.assistant(content="3 + 3 equals 6."),
        ]

        assert len(messages) == 5
        assert messages[0].role == Role.SYSTEM
        assert messages[1].role == Role.USER
        assert messages[2].role == Role.ASSISTANT
        assert messages[3].role == Role.USER
        assert messages[4].role == Role.ASSISTANT

    def test_tool_use_conversation(self):
        """Test building a conversation with tool use."""
        messages = [
            Message.user("What's the weather in NYC?"),
            Message.assistant(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="tc_1", name="get_weather", arguments={"location": "NYC"}
                    )
                ],
            ),
            Message.tool_result(
                tool_call_id="tc_1",
                name="get_weather",
                content='{"temp": 72, "condition": "sunny"}',
            ),
            Message.assistant(content="It's 72 degrees and sunny in NYC."),
        ]

        assert len(messages) == 4
        assert messages[1].tool_calls[0].name == "get_weather"
        assert messages[2].tool_call_id == "tc_1"
