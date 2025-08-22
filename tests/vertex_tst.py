from vertexai.preview.generative_models import GenerativeModel
from vertexai import init
from google.api_core.exceptions import PermissionDenied

MODEL_NAME = "gemini-2.0-flash"
PROJECT = "long-memory-465714-j2"
REGIONS = ["us-central1", "europe-west4", "me-central1"]

def try_regions_for_gemini(prompt):
    for region in REGIONS:
        try:
            print(f"Trying region: {region}")
            init(project=PROJECT, location=region)
            model = GenerativeModel(model_name=MODEL_NAME)  # ✅ No 'publisher'
            response = model.generate_content(prompt)
            return response.text, region
        except PermissionDenied as e:
            print(f"Permission denied in {region}: {e.message}")
            continue
        except Exception as e:
            print(f"Other error in {region}: {e}")
            continue
    return "All regions failed", None

# Example usage
response_text, used_region = try_regions_for_gemini("What's a cool fact about Jupiter?")
print(f"\n✅ Response from region {used_region}:\n{response_text}")
