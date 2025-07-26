"""
Main Vertex AI Rotation Manager for handling multiple projects and regions
to manage token limits and quota restrictions.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from ..api_client import APIClient
from .router_setup import ProjectConfig, RotationMetrics
from .router_enum import RouterConst
from .router_utils import (
    estimate_tokens, select_best_project, is_quota_error, 
    is_rate_limit_error, get_available_project_count, 
    reset_project_if_unblocked
)
from .router_errorHandling import ErrorHandler
from .router_stateSetup import StateManager


class VertexRotationManager:
    """Manages rotation across multiple Vertex AI projects and regions."""
    
    def __init__(self, 
                 projects_config: List[Dict[str, Any]], 
                 model_name: str = RouterConst.DEFAULT_MODEL_NAME.value,
                 rotation_strategy: str = "round_robin",
                 persistence_file: Optional[str] = None):
        """
        Initialize the rotation manager.
        
        Args:
            projects_config: List of project configurations
            model_name: Gemini model to use
            rotation_strategy: Strategy for rotation (round_robin, least_used, random)
            persistence_file: File to persist rotation state
        """
        self.model_name = model_name
        self.rotation_strategy = rotation_strategy
        self.metrics = RotationMetrics()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize helper classes
        self.error_handler = ErrorHandler(self.logger)
        self.state_manager = StateManager(persistence_file, self.logger)
        
        # Initialize projects
        self.projects = []
        for config in projects_config:
            project = ProjectConfig(
                project_id=config.get("project_id", ""),
                key_path=config["key_path"],
                regions=config.get("regions", RouterConst.DEFAULT_REGIONS.value[:14])  # 14 regions per project
            )
            self.projects.append(project)
        
        # Load persisted state
        self.current_project_index, self.metrics = self.state_manager.load_state(self.projects)
        
        # Initialize current client
        self.current_client = None
        self._initialize_current_client()
        
    def _initialize_current_client(self):
        """Initialize the current API client."""
        if not self.projects:
            raise ValueError("No projects configured")
        
        # Ensure valid project index
        if self.current_project_index >= len(self.projects):
            self.current_project_index = 0
        
        current_project = self.projects[self.current_project_index]
        current_region = current_project.regions[current_project.current_region_index]
        
        self.current_client = APIClient(
            key_path=current_project.key_path,
            model_name=self.model_name,
            region=current_region
        )
        
        self.logger.info(f"Initialized client for project {current_project.project_id} in region {current_region}")
    
    def _rotate_region(self, project: ProjectConfig) -> bool:
        """Rotate to next region within the same project."""
        if len(project.regions) <= 1:
            return False
        
        project.current_region_index = (project.current_region_index + 1) % len(project.regions)
        self.metrics.region_switches += 1
        
        current_region = project.regions[project.current_region_index]
        self.logger.info(f"Rotated to region {current_region} in project {project.project_id}")
        
        return True
    
    def _rotate_project(self) -> bool:
        """Rotate to next available project."""
        best_project_index = select_best_project(self.projects, self.rotation_strategy)
        
        if best_project_index is None:
            return False
        
        if best_project_index != self.current_project_index:
            self.current_project_index = best_project_index
            next_project = self.projects[self.current_project_index]
            reset_project_if_unblocked(next_project)
            
            self.metrics.project_switches += 1
            self._initialize_current_client()
            self.logger.info(f"Rotated to project {next_project.project_id}")
            
        return True
    
    def _update_project_metrics(self, project: ProjectConfig, token_count: int):
        """Update project usage metrics."""
        project.token_count += token_count
        project.daily_quota_used += token_count
        project.last_used = datetime.now()
        
        # Reset daily quota if it's a new day
        if project.last_used.date() < datetime.now().date():
            project.daily_quota_used = 0
    
    def call_gemini_api(self, payload: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """
        Make a call to Gemini API with automatic rotation on failures.
        
        Args:
            payload: Request payload
            max_retries: Maximum number of retry attempts
            
        Returns:
            API response
        """
        self.metrics.total_requests += 1
        estimated_tokens = estimate_tokens(payload)
        
        for attempt in range(max_retries):
            current_project = self.projects[self.current_project_index]
            
            try:
                # Check if current project is blocked
                if current_project.is_blocked and datetime.now() < current_project.block_until:
                    self.logger.info(f"Project {current_project.project_id} is blocked, rotating...")
                    if not self._rotate_project():
                        raise Exception("All projects are blocked or unavailable")
                    continue
                
                # Make the API call
                response = self.current_client.call_gemini_api(payload)
                
                # Update metrics on success
                self._update_project_metrics(current_project, estimated_tokens)
                self.metrics.successful_requests += 1
                
                # Save state periodically
                if self.metrics.total_requests % RouterConst.STATE_SAVE_INTERVAL.value == 0:
                    self.state_manager.save_state(self.current_project_index, self.projects, self.metrics)
                
                return response
                
            except Exception as e:
                error_str = str(e).lower()
                
                if is_quota_error(error_str):
                    if is_rate_limit_error(error_str):
                        self.error_handler.handle_rate_limit(current_project, self.metrics)
                    else:
                        self.error_handler.handle_quota_exceeded(current_project, self.metrics, {"message": str(e)})
                    
                    # Try rotating to next region first
                    if self._rotate_region(current_project):
                        self._initialize_current_client()
                        continue
                    
                    # If all regions exhausted, rotate project
                    if not self._rotate_project():
                        if attempt == max_retries - 1:
                            self.metrics.failed_requests += 1
                            raise Exception(f"All projects and regions exhausted: {e}")
                        continue
                
                elif attempt == max_retries - 1:
                    self.metrics.failed_requests += 1
                    raise e
                
                # For other errors, try next project if rotation is appropriate
                if self.error_handler.should_retry_with_rotation(e):
                    self.logger.warning(f"Error in project {current_project.project_id}: {e}")
                    if not self._rotate_project():
                        if attempt == max_retries - 1:
                            self.metrics.failed_requests += 1
                            raise e
                else:
                    # Don't retry for certain types of errors
                    self.metrics.failed_requests += 1
                    raise e
                
                # Small delay between retries
                time.sleep(1)
        
        self.metrics.failed_requests += 1
        raise Exception("Max retries exceeded")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rotation status and metrics."""
        current_project = self.projects[self.current_project_index]
        current_region = current_project.regions[current_project.current_region_index]
        
        return {
            "current_project": current_project.project_id,
            "current_region": current_region,
            "total_projects": len(self.projects),
            "active_projects": get_available_project_count(self.projects),
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_rate": self.metrics.success_rate,
                "project_switches": self.metrics.project_switches,
                "region_switches": self.metrics.region_switches,
                "quota_exceeded_count": self.metrics.quota_exceeded_count,
                "rate_limit_hits": self.metrics.rate_limit_hits
            },
            "projects_status": [
                {
                    "project_id": p.project_id,
                    "current_region": p.regions[p.current_region_index],
                    "token_count": p.token_count,
                    "daily_quota_used": p.daily_quota_used,
                    "is_blocked": p.is_blocked,
                    "block_until": p.block_until.isoformat() if p.is_blocked else None,
                    "last_used": p.last_used.isoformat()
                }
                for p in self.projects
            ]
        }
    
    def reset_daily_quotas(self):
        """Reset daily quotas for all projects (typically called at midnight)."""
        for project in self.projects:
            project.daily_quota_used = 0
            if project.is_blocked and "daily" in str(project.block_until):
                project.is_blocked = False
        
        self.logger.info("Reset daily quotas for all projects")
    
    def add_project(self, project_config: Dict[str, Any]):
        """Add a new project to the rotation."""
        project = ProjectConfig(
            project_id=project_config.get("project_id", ""),
            key_path=project_config["key_path"],
            regions=project_config.get("regions", RouterConst.DEFAULT_REGIONS.value[:14])  # 14 regions per project
        )
        self.projects.append(project)
        self.logger.info(f"Added project {project.project_id} to rotation")
    
    def remove_project(self, project_id: str):
        """Remove a project from rotation."""
        self.projects = [p for p in self.projects if p.project_id != project_id]
        if self.current_project_index >= len(self.projects):
            self.current_project_index = 0
            if self.projects:  # Only reinitialize if there are projects left
                self._initialize_current_client()
        self.logger.info(f"Removed project {project_id} from rotation")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save state."""
        self.state_manager.save_state(self.current_project_index, self.projects, self.metrics)