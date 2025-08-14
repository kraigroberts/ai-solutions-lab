"""
AI Solutions Lab - A production-minded lab demonstrating AI Solutions Engineering skills.

This package provides:
- LLM-powered features (chat, RAG, agent tool-calls)
- Clean service boundaries (FastAPI + CLI)
- Local-first vector store with optional cloud integration
- Comprehensive testing and CI/CD setup
"""

__version__ = "0.1.0"
__author__ = "AI Solutions Engineer"
__email__ = "your.email@example.com"

# Core modules
from . import api
from . import cli
from . import config
from . import llm
from . import rag
from . import tools

# Main exports for easy access
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "api",
    "cli", 
    "config",
    "llm",
    "rag",
    "tools",
]

# Version info for programmatic access
VERSION = __version__
