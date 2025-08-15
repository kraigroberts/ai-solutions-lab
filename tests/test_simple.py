"""Simple tests."""

import pytest
from src.ai_lab.simple_rag import SimpleRAG

def test_simple_rag():
    """Test basic RAG functionality."""
    rag = SimpleRAG()
    
    # Test empty state
    assert rag.search("test") == []
    
    # Test with documents
    docs = ["Hello world", "Python programming", "Machine learning"]
    rag.add_documents(docs)
    
    results = rag.search("hello")
    assert len(results) > 0
    assert results[0]['document'] in docs

def test_rag_search():
    """Test search functionality."""
    rag = SimpleRAG()
    
    docs = [
        "Machine learning is a subset of AI",
        "Python is great for data science",
        "FastAPI is a modern web framework"
    ]
    
    rag.add_documents(docs)
    
    # Search for ML
    results = rag.search("machine learning")
    assert len(results) > 0
    
    # Check scores are reasonable
    for result in results:
        assert result['score'] > 0
        assert result['document'] in docs
