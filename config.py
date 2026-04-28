"""Shared configuration for the RAG Engine API and scripts."""

PROJECT_ID = "sjsu-it-genai-poc"
LOCATION = "us-west1"
BUCKET_NAME = "sjsu-rag-it-genai-poc-kb"

DEPARTMENTS: dict[str, dict] = {
    "it": {
        "display_name": "Information Technology",
        "corpus_name": "projects/925509787316/locations/us-west1/ragCorpora/4035225266123964416",
    },
}

# Kept for backward-compat with legacy scripts
CORPUS_NAME = DEPARTMENTS["it"]["corpus_name"]
