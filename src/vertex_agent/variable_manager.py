"""
Variable manager for storing and retrieving variables with metadata.
"""

from datetime import datetime
from typing import Any, Dict, Optional, Type


class VariableManager:
    """Manages variable storage and retrieval."""
    
    def __init__(self):
        self._stored_variables: Dict[str, Dict[str, Any]] = {}
    
    def set_variable(self, name: str, value: Any, description: str = "", type_hint: Optional[Type] = None) -> str:
        """Stores a variable with metadata. Creates unique names if duplicates exist."""
        if name in self._stored_variables:
            existing_vars = [
                var_name for var_name in self._stored_variables.keys()
                if var_name.startswith(name + "_") or var_name == name
            ]
            
            max_counter = 0
            for var_name in existing_vars:
                if var_name == name:
                    max_counter = max(max_counter, 1)
                else:
                    try:
                        counter = int(var_name.split("_")[-1])
                        max_counter = max(max_counter, counter)
                    except ValueError:
                        continue
            
            new_name = f"{name}_{max_counter + 1}"
            print(f"Variable '{name}' already exists. Creating new variable '{new_name}'")
            name = new_name
        
        self._stored_variables[name] = {
            "value": value,
            "description": description,
            "type": type_hint or type(value).__name__,
            "created_at": datetime.now().isoformat(),
        }
        return name
    
    def get_variable(self, name: str) -> Any:
        """Retrieves a stored variable's value."""
        return self._stored_variables.get(name, {}).get("value")
    
    def list_variables(self) -> Dict[str, Dict[str, Any]]:
        """Returns information about all stored variables."""
        return {
            name: {k: v for k, v in data.items() if k != "value"}
            for name, data in self._stored_variables.items()
        }
    
    def get_variables_info(self) -> str:
        """Returns formatted string of variables for system prompt."""
        return "\n".join([
            f"- {name}: {data['description']} (Type: {data['type']})"
            for name, data in self._stored_variables.items()
        ])