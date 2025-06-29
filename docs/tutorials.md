# Tutorials

This guide provides practical examples of using the Gemini Agent Framework for various use cases.

## Basic Usage

### Creating a Simple Calculator Agent

```python
from gemini_agent import Agent
from dotenv import load_dotenv

load_dotenv()

# Define calculator tools
@Agent.description("Adds two numbers.")
@Agent.parameters({
    'a': {'type': float, 'description': 'First number'},
    'b': {'type': float, 'description': 'Second number'}
})
def add(a: float, b: float) -> float:
    return a + b

@Agent.description("Subtracts second number from first.")
@Agent.parameters({
    'a': {'type': float, 'description': 'First number'},
    'b': {'type': float, 'description': 'Second number'}
})
def subtract(a: float, b: float) -> float:
    return a - b

# Create agent
agent = Agent(api_key="your-api-key", tools=[add, subtract])

# Use the agent
response = agent.prompt("What is 5 plus 3 minus 2?")
print(response)  # Should output 6
```

## Advanced Usage

### Building a Task Management Agent

```python
from datetime import datetime
from typing import List, Dict

@Agent.description("Creates a new task.")
@Agent.parameters({
    'title': {'type': str, 'description': 'Task title'},
    'description': {'type': str, 'description': 'Task description'},
    'due_date': {'type': str, 'description': 'Due date (YYYY-MM-DD)'}
})
def create_task(title: str, description: str, due_date: str) -> Dict:
    return {
        'title': title,
        'description': description,
        'due_date': due_date,
        'created_at': datetime.now().isoformat(),
        'status': 'pending'
    }

@Agent.description("Lists all tasks.")
@Agent.parameters({})
def list_tasks() -> List[Dict]:
    return agent.get_variable('tasks', [])

# Create agent with task management tools
agent = Agent(api_key="your-api-key", tools=[create_task, list_tasks])

# Use the agent
response = agent.prompt("Create a task to buy groceries due tomorrow")
print(response)
```

### Creating a Data Analysis Agent

```python
import pandas as pd
import numpy as np

@Agent.description("Calculates basic statistics for a numeric column.")
@Agent.parameters({
    'data': {'type': list, 'description': 'List of numeric values'},
    'column_name': {'type': str, 'description': 'Name of the column'}
})
def calculate_stats(data: list, column_name: str) -> Dict:
    df = pd.DataFrame(data)
    return {
        'mean': df[column_name].mean(),
        'median': df[column_name].median(),
        'std': df[column_name].std(),
        'min': df[column_name].min(),
        'max': df[column_name].max()
    }

@Agent.description("Creates a histogram of numeric data.")
@Agent.parameters({
    'data': {'type': list, 'description': 'List of numeric values'},
    'bins': {'type': int, 'description': 'Number of histogram bins'}
})
def create_histogram(data: list, bins: int) -> Dict:
    hist, bin_edges = np.histogram(data, bins=bins)
    return {
        'histogram': hist.tolist(),
        'bin_edges': bin_edges.tolist()
    }

# Create agent with data analysis tools
agent = Agent(api_key="your-api-key", tools=[calculate_stats, create_histogram])

# Use the agent
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
agent.set_variable('sample_data', data, 'Sample numeric data', list)
response = agent.prompt("Analyze the sample data and create a histogram")
print(response)
```

## Best Practices

### 1. Error Handling

```python
@Agent.description("Safely divides two numbers.")
@Agent.parameters({
    'a': {'type': float, 'description': 'Numerator'},
    'b': {'type': float, 'description': 'Denominator'}
})
def safe_divide(a: float, b: float) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        raise ValueError("Cannot divide by zero")
```

### 2. Using Response Structures

```python


response = agent.prompt(
    """Calculate 15 * 7 and show your work respond with {"answer": number}""",
    json_format=True
)
```

### 3. Managing Variables

```python
# Store user preferences
agent.set_variable(
    'user_preferences',
    {
        'language': 'en',
        'theme': 'dark',
        'notifications': True
    },
    'User preferences',
    dict
)

# Retrieve and use preferences
preferences = agent.get_variable('user_preferences')
```

## Common Patterns

### 1. Chaining Operations

```python
@Agent.description("Performs a series of calculations.")
@Agent.parameters({
    'operations': {'type': list, 'description': 'List of operations to perform'}
})
def chain_operations(operations: list) -> float:
    result = 0
    for op in operations:
        # Each operation is a dict with 'type' and 'value'
        if op['type'] == 'add':
            result += op['value']
        elif op['type'] == 'subtract':
            result -= op['value']
    return result
```

### 2. State Management

```python
@Agent.description("Updates the agent's state.")
@Agent.parameters({
    'key': {'type': str, 'description': 'State key'},
    'value': {'type': 'any', 'description': 'State value'}
})
def update_state(key: str, value: any) -> None:
    current_state = agent.get_variable('state', {})
    current_state[key] = value
    agent.set_variable('state', current_state, 'Agent state', dict)
```

## Next Steps

1. Explore the [API Reference](api_reference.md) for detailed information about all available features
2. Check out the [Best Practices](best_practices.md) guide for tips on building robust agents