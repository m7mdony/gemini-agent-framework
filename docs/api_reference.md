# API Reference

## Agent Class

The main class for creating and managing Gemini agents.

### Initialization

```python
Agent(api_key: str, tools: List[Callable] = None, model_name: str = "gemini-1.5-flash")
```

Parameters:
- `api_key` (str): Your Google Generative AI API key
- `tools` (List[Callable], optional): List of Python functions or class methods decorated as tools
- `model_name` (str, optional): Name of the Gemini model to use (default: "gemini-1.5-flash")

### Methods

#### prompt

```python
prompt(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    response_structure: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> Any
```

Sends a prompt to the agent and returns the response.

Parameters:
- `user_prompt` (str): The user's input prompt
- `system_prompt` (str, optional): Custom system prompt to override default
- `response_structure` (Dict[str, Any], optional): Structure for the response
- `conversation_history` (List[Dict[str, Any]], optional): Previous conversation messages

Returns:
- The agent's response, formatted according to response_structure if provided

#### set_variable

```python
set_variable(name: str, value: Any, description: str = "", type_hint: type = None) -> str
```

Stores a variable in the agent's memory with metadata.

Parameters:
- `name` (str): Variable name
- `value` (Any): Value to store
- `description` (str, optional): Description of the variable
- `type_hint` (type, optional): Type hint for the variable

Returns:
- The name of the stored variable (may be modified if name already exists)

#### get_variable

```python
get_variable(name: str) -> Any
```

Retrieves a stored variable's value.

Parameters:
- `name` (str): Name of the variable to retrieve

Returns:
- The stored value or None if not found

#### list_variables

```python
list_variables() -> Dict[str, Dict[str, Any]]
```

Returns information about all stored variables.

Returns:
- Dictionary mapping variable names to their metadata

### Decorators

#### @Agent.description

```python
@Agent.description(desc: str)
```

Decorator to add a description to a tool function.

Parameters:
- `desc` (str): Description of the tool's functionality

#### @Agent.parameters

```python
@Agent.parameters(params: Dict[str, Dict[str, Any]])
```

Decorator to define parameters for a tool function.

Parameters:
- `params` (Dict[str, Dict[str, Any]]): Dictionary mapping parameter names to their definitions

## Type Mapping

The framework automatically maps Python types to Gemini JSON schema types:

| Python Type | Gemini Type |
|-------------|-------------|
| str         | STRING      |
| int         | INTEGER     |
| float       | NUMBER      |
| bool        | BOOLEAN     |
| list        | ARRAY       |
| dict        | OBJECT      |

## Response Structure

The `json_format` parameter in the `prompt` method allows you to define the expected structure with in the prompt:

```python
Json_format = True
```

## Error Handling

The framework includes built-in error handling for:
- API errors
- Invalid tool definitions
- Type conversion errors
- Variable management errors

## Best Practices

1. Always provide clear descriptions for tools and parameters
2. Use type hints in your tool functions
3. Structure your responses for consistency
4. Handle errors appropriately in your tools
5. Use meaningful variable names and descriptions 