"""
Configuration constants for Vertex AI Rotation Manager.
"""
from enum import Enum

class RouterConst(Enum):
    # Gemini 1.5 Flash quotas and limits
    DAILY_QUOTA_LIMIT = 1000000  # 1M tokens per day per project
    RATE_LIMIT_RPM = 300  # 300 requests per minute
    RATE_LIMIT_TPM = 32000  # 32K tokens per minute
    TOKEN_COOLDOWN_MINUTES = 60  # Cooldown period when hitting limits
    RATE_LIMIT_COOLDOWN_MINUTES = 5  # Cooldown for rate limit errors

    # Available regions for Vertex AI
    DEFAULT_REGIONS = [
        "us-central1", "us-east1", "us-east4", "us-west1", "us-west2", "us-west3", "us-west4",
        "europe-west1", "europe-west2", "europe-west3", "europe-west4", "europe-west6",
        "asia-east1", "asia-northeast1", "asia-southeast1"
    ]

    # Rotation strategies
    ROTATION_STRATEGIES = {
        "round_robin": "Round Robin",
        "least_used": "Least Used",
        "random": "Random"
    }

    # Default model name
    DEFAULT_MODEL_NAME = "gemini-1.5-flash"

    # State persistence settings
    STATE_SAVE_INTERVAL = 10  # Save state every N requests