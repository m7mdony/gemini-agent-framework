"""
Tool processor for handling tool execution and variable substitution.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional

from .tool_registry import ToolRegistry
from .variable_manager import VariableManager


class ToolProcessor:
    """Handles tool processing and execution."""
    
    def __init__(self, variable_manager: VariableManager):
        self.variable_manager = variable_manager
        self._registered_tools_json: List[Dict[str, Any]] = []
        self._tool_functions: Dict[str, Callable[..., Any]] = {}
        self._intermediate_results: Dict[str, Any] = {}
    
    def process_tools(self, tools: List[Callable[..., Any]]) -> None:
        """Converts decorated Python functions into the JSON format for the REST API."""
        for func in tools:
            tool_name = func.__name__
            if tool_name not in ToolRegistry._tools_registry:
                print(f"Warning: Function '{tool_name}' was passed but has no @Agent decorators. Skipping.")
                continue

            metadata = ToolRegistry._tools_registry[tool_name]
            if "description" not in metadata:
                print(f"Warning: Function '{tool_name}' is missing @Agent.description. Skipping.")
                continue

            # Store the bound method directly if it's a class method
            if inspect.ismethod(func):
                self._tool_functions[tool_name] = func
            else:
                self._tool_functions[tool_name] = metadata["function_ref"]

            # Build the parameters schema JSON
            gemini_params_schema = {"type": "OBJECT", "properties": {}, "required": []}
            params_def = metadata.get("parameters_def", {})
            signature = metadata.get("signature")

            if not params_def and signature:
                params_def = {}
                for name, param in signature.parameters.items():
                    # Skip 'self' parameter for class methods
                    if name == "self" and inspect.ismethod(func):
                        continue
                    py_type = param.annotation if param.annotation != inspect.Parameter.empty else str
                    params_def[name] = {"type": py_type, "description": f"Parameter {name}"}

            for name, definition in params_def.items():
                py_type = definition.get("type", str)
                gemini_type = ToolRegistry.get_gemini_type(py_type)
                gemini_params_schema["properties"][name] = {
                    "type": gemini_type,
                    "description": definition.get("description", ""),
                }
                if signature and signature.parameters[name].default == inspect.Parameter.empty:
                    gemini_params_schema["required"].append(name)

            # Create the Function Declaration JSON dictionary
            declaration_json = {
                "name": tool_name,
                "description": metadata["description"],
                "parameters": gemini_params_schema if gemini_params_schema["properties"] else None,
            }
            if declaration_json["parameters"] is None:
                del declaration_json["parameters"]

            self._registered_tools_json.append(declaration_json)
    
    def substitute_variables(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Substitutes variable references in arguments with their actual values."""
        result = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$"):
                # Handle $ prefixed variables
                var_name = value[1:]
                if var_name in self.variable_manager._stored_variables:
                    result[key] = self.variable_manager._stored_variables[var_name]["value"]
                else:
                    result[key] = value
            elif isinstance(value, dict) and "variable" in value:
                # Handle dictionary-style variable references
                var_name = value["variable"]
                if var_name in self.variable_manager._stored_variables:
                    result[key] = self.variable_manager._stored_variables[var_name]["value"]
                else:
                    result[key] = value
            else:
                result[key] = value
        return result
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any], debug_scope: Optional[str] = None) -> tuple[Any, str]:
        """Executes a tool function with the given arguments."""
        if tool_name not in self._tool_functions:
            raise ValueError(f"Unknown function '{tool_name}'")
        
        tool_function = self._tool_functions[tool_name]
        
        # Substitute both stored variables and intermediate results
        args = self.substitute_variables(args)
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$"):
                result_key = value[1:]
                if result_key in self._intermediate_results:
                    args[key] = self._intermediate_results[result_key]
        
        # Call the function directly - it's already bound if it's a method
        function_result = tool_function(**args)
        
        # Store intermediate result
        result_key = f"result_{len(self._intermediate_results)}"
        self._intermediate_results[result_key] = function_result
        
        # Store in variable manager
        variable_name = self.variable_manager.set_variable(
            result_key,
            function_result,
            f"the result of function call with name {tool_name} and arguments {args}",
        )
        
        return function_result, variable_name
    
    def get_tools_json(self) -> List[Dict[str, Any]]:
        """Returns the JSON representation of registered tools."""
        return self._registered_tools_json
    
    def has_tool(self, tool_name: str) -> bool:
        """Checks if a tool is registered."""
        return tool_name in self._tool_functions