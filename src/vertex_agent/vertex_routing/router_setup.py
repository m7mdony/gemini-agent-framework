"""
Data models for the Vertex AI Rotation Manager.
"""

from typing import List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ProjectConfig:
    """Configuration for a single project."""
    project_id: str
    key_path: str
    regions: List[str] = field(default_factory=list)
    current_region_index: int = 0
    last_used: datetime = field(default_factory=datetime.now)
    token_count: int = 0
    daily_quota_used: int = 0
    rate_limit_reset: datetime = field(default_factory=datetime.now)
    is_blocked: bool = False
    block_until: datetime = field(default_factory=datetime.now)


@dataclass
class RotationMetrics:
    """Metrics for tracking rotation performance."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    project_switches: int = 0
    region_switches: int = 0
    quota_exceeded_count: int = 0
    rate_limit_hits: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100