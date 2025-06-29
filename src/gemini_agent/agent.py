import inspect
import json
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union, Collection


import requests
from dotenv import load_dotenv



load_dotenv()


class Agent:
    PYTHON_TO_GEMINI_TYPE_MAP: Dict[Type, str] = {
        str: "STRING",
        int: "INTEGER",
        float: "NUMBER",
        bool: "BOOLEAN",
        list: "ARRAY",
        dict: "OBJECT",
    }
    _tools_registry: Dict[str, Dict[str, Any]] = {}  # Class-level registry

    def get_gemini_type(self, py_type: Type) -> str:
        """Maps Python types to Gemini JSON schema types."""
        return self.PYTHON_TO_GEMINI_TYPE_MAP.get(py_type, "STRING")  # Default to STRING if unknown

    @staticmethod
    def description(desc: str) -> Callable:
        """Decorator to add a description to a tool function."""

        def decorator(func: Callable) -> Callable:
            if func.__name__ not in Agent._tools_registry:
                Agent._tools_registry[func.__name__] = {}
            Agent._tools_registry[func.__name__]["description"] = desc
            Agent._tools_registry[func.__name__]["signature"] = inspect.signature(func)
            Agent._tools_registry[func.__name__]["function_ref"] = func
            Agent._tools_registry[func.__name__]["is_method"] = inspect.ismethod(func)

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def parameters(params: Dict[str, Dict[str, Any]]) -> Callable:
        """Decorator to define parameters for a tool function."""

        def decorator(func: Callable) -> Callable:
            if func.__name__ not in Agent._tools_registry:
                Agent._tools_registry[func.__name__] = {}
            Agent._tools_registry[func.__name__]["parameters_def"] = params
            if "signature" not in Agent._tools_registry[func.__name__]:
                Agent._tools_registry[func.__name__]["signature"] = inspect.signature(func)
            if "function_ref" not in Agent._tools_registry[func.__name__]:
                Agent._tools_registry[func.__name__]["function_ref"] = func
            Agent._tools_registry[func.__name__]["is_method"] = inspect.ismethod(func)

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def __init__(
        self, api_key: str, tools: Optional[List[Callable[..., Any]]] = None, model_name: str = "gemini-1.5-flash"
    ) -> None:
        """
        Initializes the Agent using REST API calls.

        Args:
            api_key: Your Google Generative AI API key.
            tools: A list of Python functions or class methods decorated as tools.
            model_name: The name of the Gemini model to use.
        """
        if not api_key:
            raise ValueError("API key is required.")
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}"
        self.headers = {"Content-Type": "application/json"}

        self._registered_tools_json: List[Dict[str, Any]] = []  # Store JSON representation
        self._tool_functions: Dict[str, Callable[..., Any]] = {}  # Map name to actual function
        self._tool_instances: Dict[str, Any] = {}  # Store instances for class methods
        self._intermediate_results: Dict[str, Any] = {}  # Store intermediate results
        self._stored_variables: Dict[str, Dict[str, Any]] = {}  # Store variables with metadata

        if tools:
            self._process_tools(tools)

    def _process_tools(self, tools: List[Callable[..., Any]]) -> None:
        """Converts decorated Python functions into the JSON format for the REST API."""
        for func in tools:
            tool_name = func.__name__
            if tool_name not in Agent._tools_registry:
                print(
                    f"Warning: Function '{tool_name}' was passed "
                    "but has no @Agent decorators. Skipping."
                )
                continue

            metadata = Agent._tools_registry[tool_name]
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
            signature = metadata.get("signature")  # inspect.signature object

            if not params_def and signature:
                params_def = {}
                for name, param in signature.parameters.items():
                    # Skip 'self' parameter for class methods
                    if name == "self" and inspect.ismethod(func):
                        continue
                    py_type = (
                        param.annotation if param.annotation != inspect.Parameter.empty else str
                    )
                    params_def[name] = {"type": py_type, "description": f"Parameter {name}"}

            for name, definition in params_def.items():
                py_type = definition.get("type", str)
                gemini_type = self.get_gemini_type(py_type)
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

    def set_variable(
        self, name: str, value: Any, description: str = "", type_hint: Optional[Type] = None
    ) -> str:
        """
        Stores a variable in the agent's memory with metadata.
        If a variable with the same name exists, creates a new variable with a counter suffix.

        Args:
            name: The name of the variable
            value: The actual value to store
            description: A description of what the variable represents
            type_hint: Optional type hint for the variable

        Returns:
            The name of the stored variable
        """
        # Check if the base name exists
        if name in self._stored_variables:
            # Find all variables that start with the base name
            existing_vars = [
                var_name
                for var_name in self._stored_variables.keys()
                if var_name.startswith(name + "_") or var_name == name
            ]

            # Find the highest counter used
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

            # Create new name with incremented counter
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
        """
        Retrieves a stored variable's value.

        Args:
            name: The name of the variable to retrieve

        Returns:
            The stored value or None if not found
        """
        return self._stored_variables.get(name, {}).get("value")

    def list_variables(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns information about all stored variables.

        Returns:
            Dictionary of variable names to their metadata
        """
        return {
            name: {k: v for k, v in data.items() if k != "value"}
            for name, data in self._stored_variables.items()
        }

    def _get_system_prompt(self) -> str:
        """Returns a system prompt that guides the model in breaking down complex operations."""
        variables_info = "\n".join(
            [
                f"- {name}: {data['description']} (Type: {data['type']})"
                for name, data in self._stored_variables.items()
            ]
        )

        return """

        Available variables:
        {variables_list}
        
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
        """.format(
            tools_list="\n".join(
                [
                    f"- {name}: {desc}"
                    for name, desc in [
                        (tool["name"], tool["description"]) for tool in self._registered_tools_json
                    ]
                ]
            ),
            variables_list=variables_info,
        )

    def _substitute_variables(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Substitutes variable references in arguments with their actual values."""
        result = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$"):
                # Handle $ prefixed variables
                var_name = value[1:]
                if var_name in self._stored_variables:

                    result[key] = self._stored_variables[var_name]["value"]
                else:
                    result[key] = value
            elif isinstance(value, dict) and "variable" in value:
                # Handle dictionary-style variable references
                var_name = value["variable"]
                if var_name in self._stored_variables:
                    result[key] = self._stored_variables[var_name]["value"]
                else:
                    result[key] = value
            else:
                result[key] = value
        return result

    def _call_gemini_api(self, payload: Dict[str, Any], debug_scope: Optional[str] = None) -> Dict[str, Any]:
        """Makes a call to the Gemini API."""
        response = requests.post(
            f"{self.base_url}:generateContent?key={self.api_key}",
            headers=self.headers,
            json=payload,
        )
        response_data = response.json()
        self._log_text(response_data, debug_scope)
        
        if not response.ok:
            error_details = response_data.get('error', {})
            error_message = f"Gemini API Error: {error_details.get('message', 'Unknown error')}"
            if 'details' in error_details:
                error_message += f"\nDetails: {error_details['details']}"
            raise requests.exceptions.HTTPError(error_message, response=response)
            
        return response_data
    
    def _log_json(self, json_data: Dict[str, Any], file_name: str, debug_scope: Optional[str] = None) -> None:
        """Logs the JSON data to a file."""
        if "json" not in debug_scope:
            return
        with open(file_name, "w") as f:
            json.dump(json_data, f)
    def _log_text(self, text: str, debug_scope: Optional[str] = None) -> None:
        """Logs the text to a file."""
        if "text" not in debug_scope:
            return
        print(text)

    def prompt(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        json_format: bool = False,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        debug_scope: Optional[str] = [],
    ) -> Any:
        """
        Sends a prompt to the Gemini model and processes the response.

        Args:
            user_prompt: The user's input prompt
            system_prompt: Optional system prompt to override the default
            json_format: If True, response will be formatted as JSON. Default is False (plain text)
            conversation_history: Optional list of previous conversation turns

        Returns:
            The model's response, formatted as JSON if json_format is True, otherwise plain text
        """
        self._intermediate_results = {}

        current_contents = conversation_history if conversation_history else []
        
        # Add system instruction to payload
        payload: Dict[str, Any] = {
            "system_instruction": {
                "parts": [{"text": system_prompt if system_prompt else ""},{"text": self._get_system_prompt()}]
            },
            "contents": current_contents
        }

        # Add user prompt to contents
        payload["contents"].append({"role": "user", "parts": [{"text": user_prompt}]})

        if self._registered_tools_json:
            payload["tools"] = [{"functionDeclarations": self._registered_tools_json}]
            payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

        # Don't set JSON formatting initially if tools are available
        # We'll apply it later after tool calls are completed
        apply_json_format_later = json_format and bool(self._registered_tools_json)
        
        # Set JSON formatting immediately if no tools are involved
        if json_format and not self._registered_tools_json:
            payload["generationConfig"] = {
                "response_mime_type": "application/json"
            }

        count = 0
        while True:
            self._log_json(payload, f"payload_{count}.json", debug_scope)
            count += 1
            response_data = self._call_gemini_api(payload, debug_scope)
            if "error" in response_data:
                self._log_text(
                    f"API call failed: {response_data['error'].get('message', 'Unknown API error')}"
                    , debug_scope
                )
                return response_data

            if not response_data.get("candidates"):
                feedback = response_data.get("promptFeedback")
                block_reason = feedback.get("blockReason") if feedback else "Unknown"
                safety_ratings = feedback.get("safetyRatings") if feedback else []
                error_msg = f"Request blocked by API. Reason: {block_reason}."
                if safety_ratings:
                    error_msg += f" Details: {json.dumps(safety_ratings)}"
                
                self._log_text(error_msg, debug_scope)
                return {"error": {"message": error_msg, "details": feedback}}
            
            try:
                candidate = response_data["candidates"][0]
                content = candidate["content"]

                for part in content["parts"]:
                    if "functionCall" in part:
                        payload["contents"].append({"role": "model", "parts": [part]})
                        fc = part["functionCall"]
                        tool_name = fc["name"]
                        args = fc.get("args", {})

                        if tool_name not in self._tool_functions:
                            error_msg = f"Model attempted to call unknown function '{tool_name}'."
                            self._log_text(f"Error: {error_msg}", debug_scope)
                            error_response_part = {
                                "functionResponse": {
                                    "name": tool_name,
                                    "response": {"error": error_msg},
                                }
                            }
                            payload["contents"].append(
                                {"role": "user", "parts": [error_response_part]}
                            )
                            continue

                        try:
                            tool_function = self._tool_functions[tool_name]
                            self._log_text(f"--- Calling Function: {tool_name}({args}) ---", debug_scope)

                            # Substitute both stored variables and intermediate results
                            args = self._substitute_variables(args)
                            for key, value in args.items():
                                if isinstance(value, str) and value.startswith("$"):
                                    result_key = value[1:]
                                    if result_key in self._intermediate_results:
                                        args[key] = self._intermediate_results[result_key]

                            # Call the function directly - it's already bound if it's a method
                            function_result = tool_function(**args)

                            self._log_text(f"--- Function Result: {function_result} ---", debug_scope)

                            result_key = f"result_{len(self._intermediate_results)}"
                            self._intermediate_results[result_key] = function_result

                            varaible_name = self.set_variable(
                                result_key,
                                function_result,
                                "the result of function call with name {tool_name} and arguments {args}",
                            )
                            function_response_part = {
                                "functionResponse": {
                                    "name": tool_name,
                                    "response": {
                                        "content": function_result,
                                        "key": varaible_name,
                                        "content_type": type(function_result).__name__,
                                    },
                                }
                            }

                            payload["contents"].append(
                                {
                                    "role": "user",
                                    "parts": [
                                        {
                                            "text": f"the return value of the function stored in the variable {varaible_name}"
                                        }
                                    ],
                                }
                            )

                            payload["contents"].append(
                                {"role": "user", "parts": [function_response_part]}
                            )

                        except Exception as e:
                            self._log_text(f"Error executing function {tool_name}: {e}", debug_scope)
                            error_msg = f"Error during execution of tool '{tool_name}': {e}"
                            error_response_part = {
                                "functionResponse": {
                                    "name": tool_name,
                                    "response": {"error": error_msg},
                                }
                            }
                            payload["contents"].append(
                                {"role": "user", "parts": [error_response_part]}
                            )
                            continue

                    elif "text" in part:
                        final_text = part["text"]

                        # Check if there are more function calls coming
                        has_more_function_calls = any(
                            "functionCall" in p
                            for p in content["parts"][content["parts"].index(part) + 1 :]
                        )

                        if not has_more_function_calls:
                            # If JSON format is requested and we have tools, make a final formatting call
                            if apply_json_format_later:
                                self._log_text("--- Making final JSON formatting call ---", debug_scope)
                                formatting_payload = {
                                    "system_instruction": {
                                        "parts": [{"text": system_prompt if system_prompt else ""},{"text": self._get_system_prompt()}]
                                    },
                                    "contents": payload["contents"] + [
                                        {
                                            "role": "user",
                                            "parts": [
                                                {
                                                    "text": f"Based on our conversation above, please format your response as JSON. Here is the current response: {final_text}"
                                                }
                                            ],
                                        }
                                    ],
                                    "generationConfig": {
                                        "response_mime_type": "application/json"
                                    },
                                }
                                self._log_json(formatting_payload, f"formatting_payload_{count}.json", debug_scope)
                                count += 1
                                structured_response_data = self._call_gemini_api(formatting_payload, debug_scope)

                                if "error" in structured_response_data:
                                    self._log_text(
                                        f"JSON formatting call failed: {structured_response_data['error']}. Returning raw text.",
                                        debug_scope
                                    )
                                    return final_text

                                try:
                                    structured_text = structured_response_data["candidates"][0][
                                        "content"
                                    ]["parts"][0]["text"]
                                    structured_output = json.loads(structured_text)
                                    return structured_output
                                except (KeyError, IndexError, json.JSONDecodeError) as e:
                                    self._log_text(
                                        f"Warning: Failed to parse JSON response after formatting call: {e}. Returning raw text.",
                                        debug_scope
                                    )
                                    return final_text
                            elif json_format:
                                # Direct JSON formatting (no tools involved)
                                try:
                                    structured_output = json.loads(final_text)
                                    return structured_output
                                except json.JSONDecodeError as e:
                                    self._log_text(
                                        f"Warning: Failed to parse JSON response: {e}. Returning raw text.",
                                        debug_scope
                                    )
                                    return final_text
                            else:
                                # Return plain text response
                                return final_text
                continue

            except (KeyError, IndexError) as e:
                self._log_text(f"Error parsing API response structure: {e}. Response: {response_data}", debug_scope)
                return {
                    "error": {
                        "message": f"Error parsing API response: {e}",
                        "details": response_data,
                    }
                }

        return {"error": {"message": "Exited interaction loop unexpectedly."}}

