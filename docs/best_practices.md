# Best Practices

This guide outlines best practices for using the Gemini Agent Framework effectively and building robust agents.

## Tool Design

### 1. Clear Descriptions

Always provide clear, concise descriptions for your tools and parameters:

```python
# Good
@Agent.description("Calculates the total price including tax.")
@Agent.parameters({
    'price': {'type': float, 'description': 'Base price before tax'},
    'tax_rate': {'type': float, 'description': 'Tax rate as a decimal (e.g., 0.1 for 10%)'}
})

# Bad
@Agent.description("Calculates price.")
@Agent.parameters({
    'price': {'type': float, 'description': 'Price'},
    'tax': {'type': float, 'description': 'Tax'}
})
```

### 2. Type Hints

Always use type hints in your tool functions:

```python
# Good
def calculate_total(price: float, tax_rate: float) -> float:
    return price * (1 + tax_rate)

# Bad
def calculate_total(price, tax_rate):
    return price * (1 + tax_rate)
```

### 3. Error Handling

Implement proper error handling in your tools:

```python
@Agent.description("Safely performs division.")
@Agent.parameters({
    'a': {'type': float, 'description': 'Numerator'},
    'b': {'type': float, 'description': 'Denominator'}
})
def safe_divide(a: float, b: float) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        raise ValueError("Cannot divide by zero")
    except TypeError:
        raise ValueError("Inputs must be numbers")
```

## Agent Configuration

### 1. API Key Management

Never hardcode your API key. Use environment variables:

```python
# Good
from dotenv import load_dotenv
load_dotenv()
agent = Agent(api_key=os.getenv('GEMINI_API_KEY'))

# Bad
agent = Agent(api_key="your-api-key-here")
```

### 2. Model Selection

Choose the appropriate model for your use case:

```python
# For general use
agent = Agent(api_key=api_key, model_name="gemini-1.5-flash")

# For more complex tasks
agent = Agent(api_key=api_key, model_name="gemini-1.5-pro")
```

## Variable Management

### 1. Meaningful Names

Use clear, descriptive names for variables:

```python
# Good
agent.set_variable(
    'user_preferences',
    {'theme': 'dark', 'language': 'en'},
    'User interface preferences',
    dict
)

# Bad
agent.set_variable('prefs', {'t': 'dark', 'l': 'en'}, 'prefs', dict)
```

### 2. Type Hints

Always specify type hints when storing variables:

```python
# Good
agent.set_variable('count', 5, 'Counter', int)

# Bad
agent.set_variable('count', 5, 'Counter')
```

## Response Handling

### 1. Structured Responses

Define clear response structures within the prompt then apply them:

```python
json_format = True
```

### 2. Error Responses

Handle errors gracefully:

```python
try:
    response = agent.prompt("Divide 10 by 0")
except ValueError as e:
    print(f"Error: {e}")
    # Handle error appropriately
```

## Performance Optimization

### 1. Tool Caching

Cache frequently used tools:

```python
# Cache tool results
@Agent.description("Cached calculation.")
@Agent.parameters({
    'input': {'type': float, 'description': 'Input value'}
})
def cached_calculation(input: float) -> float:
    cache_key = f"calc_{input}"
    cached_result = agent.get_variable(cache_key)
    if cached_result is not None:
        return cached_result
    
    result = expensive_calculation(input)
    agent.set_variable(cache_key, result, 'Cached calculation result', float)
    return result
```

### 2. Batch Processing

Process multiple items in batches:

```python
@Agent.description("Processes multiple items.")
@Agent.parameters({
    'items': {'type': list, 'description': 'List of items to process'}
})
def batch_process(items: list) -> list:
    results = []
    for item in items:
        result = process_item(item)
        results.append(result)
    return results
```

## Security

### 1. Input Validation

Always validate input parameters:

```python
@Agent.description("Processes user input.")
@Agent.parameters({
    'input': {'type': str, 'description': 'User input'}
})
def process_input(input: str) -> str:
    if not input or len(input) > 1000:
        raise ValueError("Invalid input length")
    return sanitize_input(input)
```

### 2. API Key Security

Never expose API keys in logs or error messages:

```python
# Good
try:
    response = agent.prompt("Calculate something")
except Exception as e:
    logger.error("Error occurred during calculation")
    # Don't log the actual error message if it might contain sensitive info

# Bad
try:
    response = agent.prompt("Calculate something")
except Exception as e:
    logger.error(f"Error: {e}")  # Might expose sensitive information
```

## Testing

### 1. Unit Tests

Write comprehensive unit tests for your tools:

```python
def test_safe_divide():
    assert safe_divide(10, 2) == 5
    with pytest.raises(ValueError):
        safe_divide(10, 0)
```

### 2. Integration Tests

Test the agent with real-world scenarios:

```python
def test_calculator_agent():
    agent = Agent(api_key=test_api_key, tools=[add, subtract])
    response = agent.prompt("What is 5 plus 3?")
    assert response == 8
```

## Documentation

### 1. Code Comments

Add clear comments to your code:

```python
# Calculate the total price including tax and discounts
@Agent.description("Calculates final price with tax and discounts.")
@Agent.parameters({
    'price': {'type': float, 'description': 'Base price'},
    'tax_rate': {'type': float, 'description': 'Tax rate'},
    'discount': {'type': float, 'description': 'Discount amount'}
})
def calculate_final_price(price: float, tax_rate: float, discount: float) -> float:
    # Apply discount first
    discounted_price = price - discount
    # Then apply tax
    return discounted_price * (1 + tax_rate)
```

### 2. README Updates

Keep your README up to date with:
- New features
- Breaking changes
- Usage examples
- Known issues
- Contributing guidelines 