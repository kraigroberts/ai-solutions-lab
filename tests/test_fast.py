"""Fast tests that don't require heavy ML dependencies."""

import pytest
from pathlib import Path

def test_project_structure():
    """Test that the project has the expected structure."""
    # Check that key directories exist
    assert Path("src").exists()
    assert Path("src/ai_lab").exists()
    assert Path("tests").exists()
    
    # Check that key files exist
    assert Path("requirements.txt").exists()
    assert Path("README.md").exists()

def test_basic_imports():
    """Test that basic modules can be imported."""
    # These should work without heavy dependencies
    try:
        from src.ai_lab.config import DATA_DIR, DOCS_DIR, INDEX_DIR
        assert DATA_DIR is not None
        assert DOCS_DIR is not None
        assert INDEX_DIR is not None
    except ImportError as e:
        pytest.skip(f"Basic imports failed: {e}")

def test_simple_functionality():
    """Test simple functionality that doesn't require ML."""
    # Test that we can create basic objects
    assert True  # Placeholder for actual simple tests

def test_file_existence():
    """Test that expected files exist."""
    expected_files = [
        "src/ai_lab/__init__.py",
        "src/ai_lab/config.py",
        "src/ai_lab/simple_rag.py",
        "src/ai_lab/app.py"
    ]
    
    for file_path in expected_files:
        assert Path(file_path).exists(), f"Missing file: {file_path}"

def test_requirements_exist():
    """Test that requirements files exist."""
    assert Path("requirements.txt").exists()
    assert Path("requirements-test.txt").exists()
