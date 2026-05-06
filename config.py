"""Shared configuration for the RAG Engine API and scripts."""

import json
from google.cloud import storage

PROJECT_ID = "sjsu-it-genai-poc"
LOCATION = "us-west1"
BUCKET_NAME = "sjsu-rag-it-genai-poc-kb"
_DEPARTMENTS_FILE = "departments.json"

# Mutable dict — populated at startup by load_departments().
# All routers import this reference; updating it in-place is visible app-wide.
DEPARTMENTS: dict[str, dict] = {}

# Kept for backward-compat with legacy scripts
CORPUS_NAME = "projects/925509787316/locations/us-west1/ragCorpora/4035225266123964416"

_DEFAULT_DEPARTMENTS: dict[str, dict] = {
    "it": {
        "display_name": "Information Technology",
        "corpus_name": CORPUS_NAME,
        "fallback_message": "Please contact the IT Service Desk at (408) 924-1530 or visit sjsu.edu/it for help.",
    }
}


def load_departments() -> None:
    """Read departments.json from GCS into DEPARTMENTS.

    If the file doesn't exist yet, bootstraps it with the default IT department.
    """
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(_DEPARTMENTS_FILE)

    if not blob.exists():
        blob.upload_from_string(
            json.dumps(_DEFAULT_DEPARTMENTS, indent=2),
            content_type="application/json",
        )
        DEPARTMENTS.update(_DEFAULT_DEPARTMENTS)
    else:
        DEPARTMENTS.update(json.loads(blob.download_as_text()))


def save_departments() -> None:
    """Persist the current DEPARTMENTS dict back to GCS."""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(_DEPARTMENTS_FILE)
    blob.upload_from_string(
        json.dumps(DEPARTMENTS, indent=2),
        content_type="application/json",
    )
