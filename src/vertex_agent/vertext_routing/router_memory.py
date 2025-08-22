from enum import Enum
class routerMemory(Enum):
    GEMINI_FLASH = "gemini_flash"

    GEMINI_FLASH_TPM = {
        # --- US regions ---
        "us-central1": 3_400_000,
        "us-east1":    3_400_000,
        "us-east4":    3_400_000,
        "us-east5":    3_400_000,
        "us-south1":   3_400_000,
        "us-west1":    3_400_000,
        "us-west4":    3_400_000,

        # --- Europe regions ---
        "europe-central2":   3_400_000,
        "europe-north1":     3_400_000,
        "europe-southwest1": 3_400_000,
        "europe-west1":      3_400_000,
        "europe-west4":      3_400_000,
        "europe-west8":      3_400_000,
        "europe-west9":      3_400_000,
    }

    # Max tokens per request (model limits)
    GEMINI_FLASH_LIMITS = {
        "max_input_tokens": 1_040_384,
        "max_output_tokens": 8_192
    }

