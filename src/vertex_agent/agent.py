"""
Main Agent class that orchestrates all components.
"""

import json
import requests
from typing import Any, Callable, Dict, List, Optional, Type

from .variable_manager import VariableManager
from .tool_processor import ToolProcessor
from .tool_registry import ToolRegistry
from .api_client import APIClient
from .debug_logger import DebugLogger


class Agent:
    """Main Agent class that orchestrates all components."""
    
    # Expose decorators at class level for backward compatibility
    @staticmethod
    def description(desc: str) -> Callable:
        return ToolRegistry.description(desc)
    
    @staticmethod
    def parameters(params: Dict[str, Dict[str, Any]]) -> Callable:
        return ToolRegistry.parameters(params)
    
    def __init__(
        self,
        tools: Optional[List[Callable[..., Any]]] = None,
        model_name: str = "gemini-1.5-flash",
        region: str = "us-central1",
        key_path: str = ""
    ):
        """Initialize the Agent with all its components."""
        self.variable_manager = VariableManager()
        self.tool_processor = ToolProcessor(self.variable_manager)
        self.api_client = APIClient(key_path, model_name, region)
        self.debug_logger = DebugLogger()
        
        if tools:
            self.tool_processor.process_tools(tools)
    
    def set_project(self, key_path: str) -> None:
        """Updates the project configuration."""
        self.api_client.set_project(key_path)
    
    def set_variable(self, name: str, value: Any, description: str = "", type_hint: Optional[Type] = None) -> str:
        """Stores a variable in the agent's memory."""
        return self.variable_manager.set_variable(name, value, description, type_hint)
    
    def get_variable(self, name: str) -> Any:
        """Retrieves a stored variable's value."""
        return self.variable_manager.get_variable(name)
    
    def list_variables(self) -> Dict[str, Dict[str, Any]]:
        """Returns information about all stored variables."""
        return self.variable_manager.list_variables()
    
    def _get_system_prompt(self) -> str:
        """Returns a system prompt that guides the model."""
        variables_info = self.variable_manager.get_variables_info()
        tools_list = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in self.tool_processor.get_tools_json()
        ])
        
        return f"""
        Available tools:
        {tools_list}
        
        Available variables:
        {variables_info}
        
        IMPORTANT - Variable Usage:
        When you need to use a stored variable in a function call, you MUST use the following syntax:
        - For function arguments: {{"variable": "variable_name"}}
        - For example, if you want to use the 'current_user' variable in a function call:
          {{"user_id": {{"variable": "current_user"}}}}
        
        Remember:
        - Always perform one operation at a time
        - Use intermediate results from previous steps
        - If a step requires multiple tools, execute them sequentially
        - If you're unsure about the next step, explain your reasoning
        - You can use both stored variables and values from the prompt
        - When using stored variables, ALWAYS use the {{"variable": "variable_name"}} syntax
        """
    
    def prompt(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        json_format: bool = False,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        debug_scope: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Sends a prompt to the Gemini model and processes the response."""
        if debug_scope is None:
            debug_scope = []
        if config is None:
            config = {}
        
        # Reset intermediate results for new conversation
        self.tool_processor._intermediate_results = {}
        
        current_contents = conversation_history if conversation_history else []
        
        # Build initial payload
        payload: Dict[str, Any] = {
            "system_instruction": {
                "parts": [
                    {"text": system_prompt if system_prompt else ""},
                    {"text": self._get_system_prompt()}
                ]
            },
            "contents": current_contents
        }
        
        # Add user prompt
        payload["contents"].append({"role": "user", "parts": [{"text": user_prompt}]})
        
        # Add tools if available
        if self.tool_processor.get_tools_json():
            payload["tools"] = [{"functionDeclarations": self.tool_processor.get_tools_json()}]
            payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
        
        # Handle JSON formatting
        apply_json_format_later = json_format and bool(self.tool_processor.get_tools_json())
        
        if json_format and not self.tool_processor.get_tools_json():
            payload["generationConfig"] = {"response_mime_type": "application/json"}
        
        # Main conversation loop
        count = 0
        while True:
            self.debug_logger.log_json(payload, f"payload_{count}.json", debug_scope)
            count += 1
            
            try:
                response_data = self.api_client.call_gemini_api(payload, config)
            except requests.exceptions.HTTPError as e:
                self.debug_logger.log_text(f"API call failed: {e}", debug_scope)
                return {"error": {"message": str(e)}}
            
            # Handle blocked requests
            if not response_data.get("candidates"):
                feedback = response_data.get("promptFeedback")
                block_reason = feedback.get("blockReason") if feedback else "Unknown"
                error_msg = f"Request blocked by API. Reason: {block_reason}."
                self.debug_logger.log_text(error_msg, debug_scope)
                return {"error": {"message": error_msg, "details": feedback}}
            
            try:
                candidate = response_data["candidates"][0]
                content = candidate["content"]
                
                # Process each part of the response
                for part in content["parts"]:
                    if "functionCall" in part:
                        # Handle function call
                        payload["contents"].append({"role": "model", "parts": [part]})
                        fc = part["functionCall"]
                        tool_name = fc["name"]
                        args = fc.get("args", {})
                        
                        if not self.tool_processor.has_tool(tool_name):
                            error_msg = f"Model attempted to call unknown function '{tool_name}'."
                            self.debug_logger.log_text(f"Error: {error_msg}", debug_scope)
                            error_response_part = {
                                "functionResponse": {
                                    "name": tool_name,
                                    "response": {"error": error_msg},
                                }
                            }
                            payload["contents"].append({"role": "user", "parts": [error_response_part]})
                            continue
                        
                        try:
                            self.debug_logger.log_text(f"--- Calling Function: {tool_name}({args}) ---", debug_scope)
                            
                            # Execute the tool
                            function_result, variable_name = self.tool_processor.execute_tool(
                                tool_name, args, debug_scope
                            )
                            
                            self.debug_logger.log_text(f"--- Function Result: {function_result} ---", debug_scope)
                            
                            # Prepare response
                            function_response_part = {
                                "functionResponse": {
                                    "name": tool_name,
                                    "response": {
                                        "content": function_result,
                                        "key": variable_name,
                                        "content_type": type(function_result).__name__,
                                    },
                                }
                            }
                            
                            payload["contents"].append({
                                "role": "user",
                                "parts": [{
                                    "text": f"the return value of the function stored in the variable {variable_name}"
                                }]
                            })
                            payload["contents"].append({"role": "user", "parts": [function_response_part]})
                            
                        except Exception as e:
                            self.debug_logger.log_text(f"Error executing function {tool_name}: {e}", debug_scope)
                            error_msg = f"Error during execution of tool '{tool_name}': {e}"
                            error_response_part = {
                                "functionResponse": {
                                    "name": tool_name,
                                    "response": {"error": error_msg},
                                }
                            }
                            payload["contents"].append({"role": "user", "parts": [error_response_part]})
                            continue
                    
                    elif "text" in part:
                        # Handle text response
                        final_text = part["text"]
                        
                        # Check if there are more function calls coming
                        has_more_function_calls = any(
                            "functionCall" in p
                            for p in content["parts"][content["parts"].index(part) + 1:]
                        )
                        
                        if not has_more_function_calls:
                            # Handle final response formatting
                            if apply_json_format_later:
                                return self._format_as_json(payload, final_text, system_prompt, debug_scope, config, count)
                            elif json_format:
                                try:
                                    return json.loads(final_text)
                                except json.JSONDecodeError as e:
                                    self.debug_logger.log_text(
                                        f"Warning: Failed to parse JSON response: {e}. Returning raw text.",
                                        debug_scope
                                    )
                                    return final_text
                            else:
                                return final_text
                
                continue
                
            except (KeyError, IndexError) as e:
                self.debug_logger.log_text(f"Error parsing API response structure: {e}", debug_scope)
                return {"error": {"message": f"Error parsing API response: {e}", "details": response_data}}
        
        return {"error": {"message": "Exited interaction loop unexpectedly."}}
    
    def _format_as_json(self, payload: Dict[str, Any], final_text: str, system_prompt: Optional[str], 
                       debug_scope: Optional[str], config: Optional[Dict[str, Any]], count: int) -> Any:
        """Formats the final response as JSON."""
        self.debug_logger.log_text("--- Making final JSON formatting call ---", debug_scope)
        
        formatting_payload = {
            "system_instruction": {
                "parts": [
                    {"text": system_prompt if system_prompt else ""},
                    {"text": self._get_system_prompt()}
                ]
            },
            "contents": payload["contents"] + [{
                "role": "user",
                "parts": [{
                    "text": f"Based on our conversation above, please format your response as JSON. Here is the current response: {final_text}"
                }]
            }],
            "generationConfig": {"response_mime_type": "application/json"},
        }
        
        self.debug_logger.log_json(formatting_payload, f"formatting_payload_{count}.json", debug_scope)
        
        try:
            structured_response_data = self.api_client.call_gemini_api(formatting_payload, config)
            structured_text = structured_response_data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(structured_text)
        except Exception as e:
            self.debug_logger.log_text(
                f"JSON formatting failed: {e}. Returning raw text.", debug_scope
            )
            return final_text