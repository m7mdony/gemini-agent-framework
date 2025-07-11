"""
Tool registry for managing tool decorators and metadata.
"""

import inspect
from functools import wraps
from typing import Any, Callable, Dict, Type


class ToolRegistry:
    """Manages tool registration and metadata storage."""
    
    PYTHON_TO_GEMINI_TYPE_MAP: Dict[Type, str] = {
        str: "STRING",
        int: "INTEGER",
        float: "NUMBER",
        bool: "BOOLEAN",
        list: "ARRAY",
        dict: "OBJECT",
    }
    
    _tools_registry: Dict[str, Dict[str, Any]] = {}  # Class-level registry
    
    @classmethod
    def get_gemini_type(cls, py_type: Type) -> str:
        """Maps Python types to Gemini JSON schema types."""
        return cls.PYTHON_TO_GEMINI_TYPE_MAP.get(py_type, "STRING")
    
    @staticmethod
    def description(desc: str) -> Callable:
        """Decorator to add a description to a tool function."""
        def decorator(func: Callable) -> Callable:
            if func.__name__ not in ToolRegistry._tools_registry:
                ToolRegistry._tools_registry[func.__name__] = {}
            ToolRegistry._tools_registry[func.__name__]["description"] = desc
            ToolRegistry._tools_registry[func.__name__]["signature"] = inspect.signature(func)
            ToolRegistry._tools_registry[func.__name__]["function_ref"] = func
            ToolRegistry._tools_registry[func.__name__]["is_method"] = inspect.ismethod(func)

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @staticmethod
    def parameters(params: Dict[str, Dict[str, Any]]) -> Callable:
        """Decorator to define parameters for a tool function."""
        def decorator(func: Callable) -> Callable:
            if func.__name__ not in ToolRegistry._tools_registry:
                ToolRegistry._tools_registry[func.__name__] = {}
            ToolRegistry._tools_registry[func.__name__]["parameters_def"] = params
            if "signature" not in ToolRegistry._tools_registry[func.__name__]:
                ToolRegistry._tools_registry[func.__name__]["signature"] = inspect.signature(func)
            if "function_ref" not in ToolRegistry._tools_registry[func.__name__]:
                ToolRegistry._tools_registry[func.__name__]["function_ref"] = func
            ToolRegistry._tools_registry[func.__name__]["is_method"] = inspect.ismethod(func)

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)
            return wrapper
        return decorator