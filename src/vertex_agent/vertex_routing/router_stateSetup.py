"""
State (session file) persistence manager for the Vertex AI Rotation Manager.
"""

import json
import logging
from typing import List, Optional
from pathlib import Path
from datetime import datetime
from .router_setup import ProjectConfig, RotationMetrics


class StateManager:
    """Manages persistence and loading of rotation state."""
    
    def __init__(self, persistence_file: Optional[str], logger: logging.Logger):
        self.persistence_file = persistence_file
        self.logger = logger
    
    def save_state(self, current_project_index: int, projects: List[ProjectConfig], metrics: RotationMetrics) -> None:
        """Save current rotation state to file."""
        if not self.persistence_file:
            return
        
        state = {
            "current_project_index": current_project_index,
            "projects": [
                {
                    "project_id": p.project_id,
                    "current_region_index": p.current_region_index,
                    "token_count": p.token_count,
                    "daily_quota_used": p.daily_quota_used,
                    "is_blocked": p.is_blocked,
                    "block_until": p.block_until.isoformat() if p.block_until else None,
                    "last_used": p.last_used.isoformat()
                }
                for p in projects
            ],
            "metrics": {
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "project_switches": metrics.project_switches,
                "region_switches": metrics.region_switches,
                "quota_exceeded_count": metrics.quota_exceeded_count,
                "rate_limit_hits": metrics.rate_limit_hits
            }
        }
        
        try:
            with open(self.persistence_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    def load_state(self, projects: List[ProjectConfig]) -> tuple[int, RotationMetrics]:
        """Load rotation state from file."""
        current_project_index = 0
        metrics = RotationMetrics()
        
        if not self.persistence_file or not Path(self.persistence_file).exists():
            return current_project_index, metrics
        
        try:
            with open(self.persistence_file, 'r') as f:
                state = json.load(f)
            
            current_project_index = state.get("current_project_index", 0)
            
            # Load project states
            project_states = {p["project_id"]: p for p in state.get("projects", [])}
            for project in projects:
                if project.project_id in project_states:
                    saved_state = project_states[project.project_id]
                    project.current_region_index = saved_state.get("current_region_index", 0)
                    project.token_count = saved_state.get("token_count", 0)
                    project.daily_quota_used = saved_state.get("daily_quota_used", 0)
                    project.is_blocked = saved_state.get("is_blocked", False)
                    if saved_state.get("block_until"):
                        project.block_until = datetime.fromisoformat(saved_state["block_until"])
                    project.last_used = datetime.fromisoformat(saved_state["last_used"])
            
            # Load metrics
            metrics_data = state.get("metrics", {})
            metrics.total_requests = metrics_data.get("total_requests", 0)
            metrics.successful_requests = metrics_data.get("successful_requests", 0)
            metrics.failed_requests = metrics_data.get("failed_requests", 0)
            metrics.project_switches = metrics_data.get("project_switches", 0)
            metrics.region_switches = metrics_data.get("region_switches", 0)
            metrics.quota_exceeded_count = metrics_data.get("quota_exceeded_count", 0)
            metrics.rate_limit_hits = metrics_data.get("rate_limit_hits", 0)
            
            self.logger.info("Loaded rotation state from file")
            
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
        
        return current_project_index, metrics