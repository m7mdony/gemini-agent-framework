"""
API client for Google Cloud Gemini API interactions.
"""

import requests
from typing import Dict, Any, Optional
from google.oauth2 import service_account
from google.auth.transport.requests import Request


class APIClient:
    """Handles Google Cloud API authentication and requests."""
    
    def __init__(self, key_path: str, model_name: str = "gemini-1.5-flash", region: str = "us-central1"):
        self.key_path = key_path
        self.model_name = model_name
        self.region = region
        self._SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
        self.headers = {"Content-Type": "application/json"}
        
        if not key_path:
            raise ValueError("API key path is required.")
        
        self._setup_credentials()
    
    def _setup_credentials(self):
        """Sets up Google Cloud credentials and project info."""
        self.creds = service_account.Credentials.from_service_account_file(
            self.key_path, scopes=self._SCOPES
        )
        self.project_id = self.creds.project_id
        self.base_url = (
            f"https://{self.region}-aiplatform.googleapis.com/v1/projects/"
            f"{self.project_id}/locations/{self.region}/publishers/google/models/"
            f"{self.model_name}:generateContent"
        )
    
    def set_project(self, key_path: str):
        """Updates the project configuration."""
        if not key_path:
            raise Exception("please set a key path")
        self.key_path = key_path
        self._setup_credentials()
    
    def call_gemini_api(self, payload: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Makes a call to the Gemini API."""
        # Refresh credentials and get access token
        self.creds.refresh(Request())
        access_token = self.creds.token
        
        # Update headers with fresh token
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        # Handle config overrides
        url = self.base_url
        if config:
            model_name = config.get("model_name", self.model_name)
            region = config.get("region", self.region)
            url = (
                f"https://{region}-aiplatform.googleapis.com/v1/projects/"
                f"{self.project_id}/locations/{region}/publishers/google/models/"
                f"{model_name}:generateContent"
            )
        
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        
        if not response.ok:
            error_details = response_data.get('error', {})
            error_message = f"Gemini API Error: {error_details.get('message', 'Unknown error')}"
            if 'details' in error_details:
                error_message += f"\nDetails: {error_details['details']}"
            raise requests.exceptions.HTTPError(error_message, response=response)
        
        return response_data