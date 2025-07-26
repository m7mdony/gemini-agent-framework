"""
Utility functions for the Vertex AI Rotation Manager.
"""

import random
from typing import Dict, Any, List, Optional
from datetime import datetime
from .router_setup import ProjectConfig


def estimate_tokens(payload: Dict[str, Any]) -> int:
    """Estimate token count for a payload."""
    # Simple estimation - count characters and divide by 4
    text_content = str(payload)
    return len(text_content) // 4


def select_best_project(projects: List[ProjectConfig], strategy: str) -> Optional[int]:
    """Select the best project based on the rotation strategy."""
    available_projects = []
    
    for i, project in enumerate(projects):
        if not project.is_blocked or datetime.now() > project.block_until:
            available_projects.append((i, project))
    
    if not available_projects:
        return None
    
    if strategy == "round_robin":
        return available_projects[0][0]
    elif strategy == "least_used":
        return min(available_projects, key=lambda x: x[1].token_count)[0]
    elif strategy == "random":
        return random.choice(available_projects)[0]
    else:
        return available_projects[0][0]


def is_quota_error(error_str: str) -> bool:
    """Check if error is related to quota limits."""
    return "quota" in error_str.lower() or "limit" in error_str.lower()


def is_rate_limit_error(error_str: str) -> bool:
    """Check if error is related to rate limits."""
    return "429" in error_str or "rate" in error_str.lower()


def is_daily_quota_error(error_str: str) -> bool:
    """Check if error is related to daily quota limits."""
    return "daily" in error_str.lower()


def get_available_project_count(projects: List[ProjectConfig]) -> int:
    """Get count of available (non-blocked) projects."""
    return len([p for p in projects if not p.is_blocked or datetime.now() > p.block_until])


def reset_project_if_unblocked(project: ProjectConfig) -> bool:
    """Reset project block status if block period has expired."""
    if project.is_blocked and datetime.now() > project.block_until:
        project.is_blocked = False
        return True
    return False