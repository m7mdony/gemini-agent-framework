"""
Error handling logic for the Vertex AI Rotation Manager.
"""

import logging
from typing import Dict, Any
from datetime import datetime, timedelta
from .router_setup import ProjectConfig, RotationMetrics
from .router_enum import RouterConst
from .router_utils import is_daily_quota_error


class ErrorHandler:
    """Handles various types of API errors and applies appropriate responses."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def handle_quota_exceeded(self, project: ProjectConfig, metrics: RotationMetrics, error_details: Dict[str, Any]) -> None:
        """Handle quota exceeded errors."""
        metrics.quota_exceeded_count += 1
        error_str = str(error_details)
        
        if is_daily_quota_error(error_str):
            # Daily quota exceeded - block project for the day
            tomorrow = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            project.block_until = tomorrow
            project.is_blocked = True
            self.logger.warning(f"Daily quota exceeded for project {project.project_id}. Blocked until {tomorrow}")
        else:
            # Rate limit exceeded - short cooldown
            project.block_until = datetime.now() + timedelta(minutes=RouterConst.TOKEN_COOLDOWN_MINUTES.value)
            project.is_blocked = True
            self.logger.warning(f"Rate limit exceeded for project {project.project_id}. Blocked until {project.block_until}")
    
    def handle_rate_limit(self, project: ProjectConfig, metrics: RotationMetrics) -> None:
        """Handle rate limit errors."""
        metrics.rate_limit_hits += 1
        project.block_until = datetime.now() + timedelta(minutes=RouterConst.RATE_LIMIT_COOLDOWN_MINUTES.value)
        project.is_blocked = True
        self.logger.warning(f"Rate limit hit for project {project.project_id}. Blocked until {project.block_until}")
    
    def should_retry_with_rotation(self, error: Exception) -> bool:
        """Determine if error should trigger rotation and retry."""
        error_str = str(error).lower()
        
        # Quota and rate limit errors should trigger rotation
        if "quota" in error_str or "limit" in error_str or "429" in error_str:
            return True
        
        # Authentication errors should not trigger rotation
        if "auth" in error_str or "permission" in error_str:
            return False
        
        # Network errors might benefit from rotation
        if "network" in error_str or "connection" in error_str or "timeout" in error_str:
            return True
        
        # Unknown errors - default to retry with rotation
        return True