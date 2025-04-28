# Gemini Agent Framework

A powerful Python framework for building intelligent agents using Google's Gemini API. This framework simplifies the creation of agents that can break down complex tasks into sequential steps using available tools, with support for function calling, variable management, and structured responses.

## Features

- 🛠️ **Easy Tool Definition**: Define tools using simple decorators
- 🔄 **Sequential Task Breakdown**: Automatically breaks down complex tasks into manageable steps
- 📦 **Variable Management**: Store and manage variables with metadata
- 🎯 **Structured Responses**: Define response structures for consistent outputs
- 🔍 **Intermediate Results**: Access and manage intermediate results
- 🛡️ **Error Handling**: Built-in error handling and recovery mechanisms
- 🔌 **Extensible**: Easy to extend with custom tools and functionality

## Installation

```bash
pip install gemini-agent-framework
```

## Quick Start

```python
from gemini_agent import Agent
from dotenv import load_dotenv

load_dotenv()

# Define your tools
@Agent.description("Multiplies two numbers.")
@Agent.parameters({
    'a': {'type': int, 'description': 'The first number'},
    'b': {'type': int, 'description': 'The second number'}
})
def multiply(a: int, b: int) -> int:
    return a * b

# Create an agent instance
agent = Agent(api_key="your-api-key", tools=[multiply])

# Use the agent
response = agent.prompt("Multiply 3 and 7")
print(response)  # Should output 21
```

## Advanced Usage

### Variable Management

```python
# Store variables with metadata
agent.set_variable("user_name", "John", "The current user's name", str)
agent.set_variable("last_login", datetime.now(), "Last login timestamp", datetime)

# Retrieve variables
name = agent.get_variable("user_name")
```

### Structured Responses

```python
response_structure = {
    "result": {"type": "number"},
    "explanation": {"type": "string"}
}

response = agent.prompt(
    "Calculate 5 * 7 and explain the process",
    response_structure=response_structure
)
```

### Custom System Prompts

```python
system_prompt = """
You are a helpful assistant that specializes in mathematical calculations.
Always show your work and explain your reasoning.
"""

response = agent.prompt(
    "Solve 15 * 23",
    system_prompt=system_prompt
)
```

## Documentation

For detailed documentation, please visit our [documentation site](https://github.com/m7mdony/gemini-agent-framework).

### Key Topics

- [API Reference](https://m7mdony.github.io/gemini-agent-framework/api_reference/)
- [Tutorials](https://m7mdony.github.io/gemini-agent-framework/tutorials/)
- [Best Practices](https://m7mdony.github.io/gemini-agent-framework/best_practices/)
- [Architecture Overview](https://m7mdony.github.io/gemini-agent-framework/architecture/)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Run tests:
   ```bash
   pytest
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- [GitHub Issues](https://github.com/m7mdony/gemini-agent-framework/issues)

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{gemini_agent_framework,
  author = {Mohamed Baathman},
  title = {Gemini Agent Framework},
  year = {2025},
  url = {https://github.com/m7mdony/gemini-agent-framework}
}
``` 