# Welcome to Gemini Agent Framework

A powerful Python framework for building intelligent agents using Google's Gemini API. This framework simplifies the creation of agents that can break down complex tasks into sequential steps using available tools, with support for function calling, variable management, and structured responses.

## Features

- 🛠️ **Easy Tool Definition**: Define tools using simple decorators
- 🔄 **Sequential Task Breakdown**: Automatically breaks down complex tasks into manageable steps
- 📦 **Variable Management**: Store and manage variables with metadata
- 🎯 **Structured Responses**: Define response structures for consistent outputs
- 🔍 **Intermediate Results**: Access and manage intermediate results
- 🛡️ **Error Handling**: Built-in error handling and recovery mechanisms
- 🔌 **Extensible**: Easy to extend with custom tools and functionality

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

## Installation

```bash
pip install gemini-agent-framework
```

## Documentation

- [Getting Started](installation.md)
- [Tutorials](tutorials.md)
- [Architecture](architecture.md)
- [Best Practices](best_practices.md)
- [API Reference](api-reference.md)

## Support

- [GitHub Issues](https://github.com/m7mdony/gemini-agent-framework/issues)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/m7mdony/gemini-agent-framework/blob/main/LICENSE) file for details. 