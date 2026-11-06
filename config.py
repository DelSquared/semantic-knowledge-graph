"""Configuration for the semantic graph app."""

import os
from pathlib import Path

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent

# Directory containing note text files
NOTES_DIR = PROJECT_ROOT / "notes"

# Directory containing static frontend files
STATIC_DIR = PROJECT_ROOT / "static"

# Host and port for the server
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# Embedding model settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")

# Layout is now handled by PCA dimensionality reduction on embeddings (graph.py).
# No force-directed parameters needed.

# Similarity threshold for creating edges
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.15"))
