# Installation

This guide will help you install and set up the Gemini Agent Framework.

## Requirements

- Python 3.8 or higher
- pip (Python package installer)
- A Google Generative AI API key

## Installation Methods

### Using pip

The simplest way to install the framework is using pip:

```bash
pip install gemini-agent-framework
```

### From Source

If you want to install from source:

1. Clone the repository:
   ```bash
   git clone https://github.com/m7mdony/gemini-agent-framework.git
   cd gemini-agent-framework
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

### Development Installation

For development, install with development dependencies:

```bash
pip install -e ".[dev]"
```

This will install additional tools for development:
- pytest for testing
- black for code formatting
- isort for import sorting
- mypy for type checking
- flake8 for linting

## API Key Setup

1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

2. Create a `.env` file in your project root:
   ```
   GEMINI_API_KEY=your-api-key-here
   ```

3. Load the API key in your code:
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

## Verification

To verify the installation, run:

```python
from gemini_agent import Agent

# Create an agent instance
agent = Agent(api_key="your-api-key")

# Test the agent
response = agent.prompt("Hello, world!")
print(response)
```

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Make sure your `.env` file exists and contains the correct API key
   - Check that you're loading the environment variables correctly

2. **Import Error**
   - Verify that the package is installed correctly
   - Check your Python version (must be 3.8 or higher)

3. **Dependency Issues**
   - Try reinstalling the package
   - Check for conflicting packages

### Getting Help

If you encounter any issues:
- Check the [GitHub Issues](https://github.com/m7mdony/gemini-agent-framework/issues)
- Create a new issue with details about your problem 