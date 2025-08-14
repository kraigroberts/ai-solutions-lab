"""
Documentation package for AI Solutions Lab.

This package contains seed documentation files that demonstrate
the RAG pipeline capabilities.
"""

__version__ = "0.1.0"
__author__ = "AI Solutions Engineer"

# List of available documentation files
AVAILABLE_DOCS = [
    "ai_basics.md",
    "machine_learning.md", 
    "rag_systems.md",
    "ai_agents.md",
    "system_design.md"
]

def get_doc_list():
    """Get list of available documentation files."""
    return AVAILABLE_DOCS.copy()

def get_doc_info():
    """Get information about available documentation."""
    return {
        "total_docs": len(AVAILABLE_DOCS),
        "documents": AVAILABLE_DOCS,
        "topics": [
            "Artificial Intelligence Basics",
            "Machine Learning Fundamentals", 
            "RAG Systems",
            "AI Agents",
            "Clean System Design"
        ]
    }
