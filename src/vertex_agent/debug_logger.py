"""
Debug logger for handling debug output.
"""

import json
from typing import Dict, Any, Optional


class DebugLogger:
    """Handles debug logging functionality."""
    
    @staticmethod
    def log_json(json_data: Dict[str, Any], file_name: str, debug_scope: Optional[str] = None) -> None:
        """Logs JSON data to a file if debug scope includes 'json'."""
        if debug_scope and "json" in debug_scope:
            with open(file_name, "w") as f:
                json.dump(json_data, f, indent=2)
    
    @staticmethod
    def log_text(text: str, debug_scope: Optional[str] = None) -> None:
        """Logs text to console if debug scope includes 'text'."""
        if debug_scope and "text" in debug_scope:
            print(text)