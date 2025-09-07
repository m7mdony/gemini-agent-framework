"""
Main Agent class that orchestrates all components with optional Vertex AI rotation.
"""
import json
import requests
from typing import Any, Callable, Dict, List, Optional, Type
from .variable_manager import VariableManager
from .tool_processor import ToolProcessor
from .tool_registry import ToolRegistry
from .debug_logger import DebugLogger
from .vertext_routing.payload_operations import payloadOperations
from .api_client import APIClient

class Agent:
    """Main Agent class that orchestrates all components with optional rotation.
    Args:
            tools: List of tool functions to register
            model_name: Gemini model to use
            region: Default region (used when router is disabled)
            key_path: Default key path 
            key_dict: Default key dictionary (used when connecting without a config file)
            use_router: Whether to use the Vertex AI rotation manager
            router_projects: List of project configurations for rotation
            use_redis: Whether to use Redis for token bucket management
            redis_url: Redis server URL (None if using host/port or use_redis is False)
            redis_host: Redis server host (None if using URL or use_redis is False)
            redis_port: Redis server port (None if using URL or use_redis is False)
            redis_db: Redis database number (None if using URL or use_redis is False)
            redis_password: Redis server password (None if using URL or use_redis is False)
            key_prefix: Redis key prefix for token buckets (None if use_redis is False)
            router_debug_mode: Whether to enable debug mode for the router
            router_debug_file_path: File path for router debug logs
    """

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
        model_name: str = "gemini-2.0-flash",
        region: str = "us-central1",
        key_path: str = "",
        key_dict: Optional[Dict[str, Any]] = None,  
        # Router configuration
        use_router: bool = False,
        router_projects: Optional[List[Dict[str, Any]]] = None,
        use_redis: bool = True,
        redis_url: Optional[str] = None,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        key_prefix: str = 'token_bucket',
        router_debug_mode: bool = False,
        router_debug_file_path: str = "debug_logs.json"
    ):
        self.key_path = key_path
        self.key_dict = key_dict
        self.router_debug_mode = router_debug_mode
        self.variable_manager = VariableManager()
        self.tool_processor = ToolProcessor(self.variable_manager)
        self.debug_logger = DebugLogger()
        self.api_client = None
        self.use_router = use_router
        self.router_projects = router_projects if router_projects else []
        self.router = payloadOperations(
            use_redis,
            redis_url,
            redis_host,
            redis_port,
            redis_db,
            redis_password,
            key_prefix,
            router_projects=self.router_projects,
            debug_mode=self.router_debug_mode,
            debug_file_path=router_debug_file_path
        )

        # Initialize API client or router
        if not self.use_router and not self.router_projects:
            self.api_client = APIClient(self.key_path, self.key_dict, model_name, region)
            self.router = None
        
        if tools:
            self.tool_processor.process_tools(tools)

        # Ensure all router projects have key_path and key_dict
        for project in self.router_projects:
            project.setdefault("key_path", "")
            project.setdefault("key_dict", {})

    def set_project(self, key_path: str) -> None:
        """Updates the project configuration (only works when router is disabled)."""
        if self.use_router:
            raise ValueError("Cannot set project when using router. Use add_project() instead.")
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
        
        if self.use_router and self.router_projects:
            # making a temporary API client for input token counting
            self.temp_api_client = APIClient(key_path=self.router_projects[0]["key_path"], key_dict=self.router_projects[0]["key_dict"], region="us-central1")
            input_tokens = self.temp_api_client.count_payload_tokens(payload, config)

        # Main conversation loop
        count = 0
        while True:
            self.debug_logger.log_json(payload, f"payload_{count}.json", debug_scope)
            count += 1
            
            try:
                if self.use_router and self.router_projects:
                    try:
                        # calling the vertex router for the initial step and processing the request.
                        project_name, region = self.router.main_router(input_tokens)
                        project_key_path = {item["project_id"]: item.get("key_path", None) for item in self.router_projects}[project_name]
                        project_key_dict = {item["project_id"]: item.get("key_dict", None) for item in self.router_projects}[project_name]
                        self.api_client = APIClient(key_path=project_key_path, key_dict=project_key_dict, region=region)
                        response_data = self.api_client.call_gemini_api(payload, config)
                        print(f"  ✓ Allocated: {project_name} -> {region}")

                        # counting the output tokens and updating the router
                        completion_tokens = response_data.get('usageMetadata', {}).get('candidatesTokenCount', 0)
                        print(f"Completion tokens used: {completion_tokens}")
                        success = self.router.output_calc(completion_tokens, project_name, region)
                        if success:
                            print(f"  ✓ Updated router with {completion_tokens} tokens for {project_name} in {region}")
                        else:
                            print(f"  ✗ Failed to update router for {project_name} in {region}")
                    
                    except requests.exceptions.HTTPError as e:
                        if e.response is not None and e.response.status_code == 429:
                            max_attempts = len(self.router_projects)
                            attempt = 0
                            while attempt < max_attempts:
                                attempt += 1
                                input_refund_success = self.router.input_refund(input_tokens, project_name, region)
                                if input_refund_success:
                                    try:
                                        self.router.retry_with_next_project(project_name, region)
                                        print(f"  ✓ Marked {region} in {project_name} as exhausted.")

                                        project_key_path = {item["project_id"]: item.get("key_path", None) for item in self.router_projects}[project_name]
                                        project_key_dict = {item["project_id"]: item.get("key_dict", None) for item in self.router_projects}[project_name]
                                        self.api_client = APIClient(key_path=project_key_path, key_dict=project_key_dict, region=region)
                                        response_data = self.api_client.call_gemini_api(payload, config)
                                        print(f"  ✓ Allocated: {project_name} -> {region}")

                                        # counting the output tokens and updating the router
                                        completion_tokens = response_data.get('usageMetadata', {}).get('candidatesTokenCount', 0)
                                        print(f"Completion tokens used: {completion_tokens}")
                                        success = self.router.output_calc(completion_tokens, project_name, region)
                                        if success:
                                            print(f"  ✓ Updated router with {completion_tokens} tokens for {project_name} in {region}")
                                        else:
                                            print(f"  ✗ Failed to update router for {project_name} in {region}")
                                        break  # Exit the retry loop on success
                                    except requests.exceptions.HTTPError as e:
                                        if e.response is not None and e.response.status_code == 429:
                                            print(f"  ✗ Retry {attempt}/{max_attempts} failed due to rate limit. Trying next project...")
                                            continue
                                        else:
                                            self.debug_logger.log_text(f"API call failed: {e}", debug_scope)
                                            return {"error": {"message": str(e)}}

                        else:
                            self.debug_logger.log_text(f"API call failed: {e}", debug_scope)
                            return {"error": {"message": str(e)}}
                        
                    if self.router_debug_mode:
                        # Print debug summary
                        print(f"\n=== Debug Summary ===")
                        summary = self.router.get_debug_summary()
                        for key, value in summary.items():
                            if key not in ["current_balances", "project_info"]:
                                print(f"{key}: {value}")

                        # This will automatically save the debug log
                        self.router.close()
                        print(f"\nDebug log saved. Check 'payload_operations_debug.json' for detailed logs.")

                    else: self.router.close()

                else:
                    # Direct API call without router
                    response_data = self.api_client.call_gemini_api(payload, config)
                    print("  ✓ Direct API call successful")

                print("\n Response Data:")
                    
            except requests.exceptions.HTTPError as e:
                self.debug_logger.log_text(f"API call failed: {e}", debug_scope)
                return {"error": {"message": str(e)}}
            except Exception as e:
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
            structured_response_data = self._call_api(formatting_payload, config)
            structured_text = structured_response_data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(structured_text)
        except Exception as e:
            self.debug_logger.log_text(
                f"JSON formatting failed: {e}. Returning raw text.", debug_scope
            )
            return final_text
    
