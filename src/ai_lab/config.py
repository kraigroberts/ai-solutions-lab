"""Simple configuration."""

import os
from pathlib import Path

# Basic settings
DATA_DIR = Path("./data")
DOCS_DIR = DATA_DIR / "docs"
INDEX_DIR = DATA_DIR / "index"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
