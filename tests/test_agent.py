import os

from dotenv import load_dotenv

from gemini_agent import Agent

load_dotenv()

# Load multiple API keys with fallback
api_keys = os.getenv("GEMINI_APIs", "dummy-key").split(",")
current_key_idx = 0


def get_current_api_key():
    return api_keys[current_key_idx]


def switch_to_next_api_key():
    global current_key_idx
    current_key_idx = (current_key_idx + 1) % len(api_keys)
    print(
        f"🔄 [DEBUG] Switching to API key index {current_key_idx}: {api_keys[current_key_idx][:10]}..."
    )


# Define your tools
@Agent.description("Multiplies two numbers.")
@Agent.parameters(
    {
        "a": {"type": int, "description": "The first number"},
        "b": {"type": int, "description": "The second number"},
    }
)
def multiply(a: int, b: int) -> int:
    return a * b


@Agent.description("Adds two numbers.")
@Agent.parameters(
    {
        "a": {"type": int, "description": "The first number"},
        "b": {"type": int, "description": "The second number"},
    }
)
def add(a: int, b: int) -> int:
    return a + b


# Create agent with the first API key
agent = Agent(api_key=get_current_api_key(), tools=[multiply, add])


# Define a wrapper to rotate API key AFTER the call
def agent_prompt_with_key_rotation(agent, *args, **kwargs):
    print(f"🚀 [DEBUG] Using API key index {current_key_idx}: {agent.api_key[:10]}...")
    response = agent.prompt(*args, **kwargs)
    switch_to_next_api_key()
    agent.api_key = get_current_api_key()
    return response


# Use the agent
response = agent_prompt_with_key_rotation(
    agent,
    user_prompt="multiply 3 and 7 then add 5 to the result",
    system_prompt="You are a helpful assistant. Give your response always with ❤️ at the start of the line. In your response you should mention the function you used.",
 
)

print(response)


def test_agent_initialization():
    """Test that an agent can be initialized with required parameters."""
    agent = Agent(api_key="test-key")
    assert agent is not None
    assert agent.api_key == "test-key"


def test_agent_with_tools():
    """Test that an agent can be initialized with tools."""

    @Agent.description("Test tool")
    @Agent.parameters({"x": {"type": int, "description": "Test parameter"}})
    def test_tool(x: int) -> int:
        return x * 2

    agent = Agent(api_key="test-key", tools=[test_tool])
    assert len(agent._registered_tools_json) == 1
    assert agent._registered_tools_json[0]["name"] == "test_tool"


def test_variable_management():
    """Test variable management functionality."""
    agent = Agent(api_key="test-key")

    # Test setting a variable
    var_name = agent.set_variable("test_var", 42, "Test variable", int)
    assert var_name == "test_var"

    # Test getting a variable
    value = agent.get_variable("test_var")
    assert value == 42

    # Test listing variables
    variables = agent.list_variables()
    assert "test_var" in variables
    assert variables["test_var"]["type"] == int
    assert variables["test_var"]["description"] == "Test variable"


def test_error_handling():
    """Test error handling in agent operations."""
    agent = Agent(api_key="test-key")

    # Test getting non-existent variable
    value = agent.get_variable("non_existent")
    assert value is None

    # Test setting variable with invalid type (should not raise an exception)
    var_name = agent.set_variable("test_var", "not_an_int", "Test variable", int)
    assert var_name == "test_var"
    value = agent.get_variable("test_var")
    assert value == "not_an_int"
