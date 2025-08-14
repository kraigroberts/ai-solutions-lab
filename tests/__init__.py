"""
Test package for AI Solutions Lab.

This package contains comprehensive tests for all components.
"""

__version__ = "0.1.0"
__author__ = "AI Solutions Engineer"

# Test modules
TEST_MODULES = [
    "test_rag",
    "test_api", 
    "test_tools"
]

def get_test_info():
    """Get information about available tests."""
    return {
        "total_modules": len(TEST_MODULES),
        "modules": TEST_MODULES,
        "coverage_target": "90%",
        "framework": "pytest"
    }
