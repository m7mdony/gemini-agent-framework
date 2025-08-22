"""
Agent package for Gemini API interaction with tools and variable management.
"""

from .agent import Agent
from .tool_registry import ToolRegistry
from .variable_manager import VariableManager
from .tool_processor import ToolProcessor
from .api_client import APIClient
from .debug_logger import DebugLogger
from .vertext_routing.payload_operations import payloadOperations

__version__ = "0.3.3"
__all__ = [
    "Agent",
    "ToolRegistry", 
    "VariableManager",
    "ToolProcessor",
    "APIClient",
    "DebugLogger",
    "payloadOperations"
]