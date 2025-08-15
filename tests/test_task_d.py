"""Tests for Task D: Advanced Search & Filtering."""

import pytest
import sys
from pathlib import Path
import tempfile
import json
import math

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_lab.hybrid_search import (
    HybridSearchEngine, SearchQuery, SearchResult
)
from ai_lab.advanced_search import (
    AdvancedSearchInterface, AdvancedSearchRequest, AdvancedSearchResponse
)
from ai_lab.llm_config import LLMConfig


class TestSearchQuery:
    """Test SearchQuery dataclass."""
    
    def test_search_query_creation(self):
        """Test SearchQuery creation with default values."""
        query = SearchQuery(
            text="test query",
            filters={"file_type": ".md"}
        )
        
        assert query.text == "test query"
        assert query.filters == {"file_type": ".md"}
        assert query.search_type == "hybrid"
        assert query.boost_semantic == 1.0
        assert query.boost_keyword == 1.0
        assert query.boost_metadata == 1.0
    
    def test_search_query_custom_values(self):
        """Test SearchQuery creation with custom values."""
        query = SearchQuery(
            text="custom query",
            filters={"category": "tutorial"},
            search_type="semantic",
            boost_semantic=2.0,
            boost_keyword=0.5
        )
        
        assert query.text == "custom query"
        assert query.filters == {"category": "tutorial"}
        assert query.search_type == "semantic"
        assert query.boost_semantic == 2.0
        assert query.boost_keyword == 0.5


class TestSearchResult:
    """Test SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test SearchResult creation."""
        result = SearchResult(
            document="Test document content",
            source_file="test.md",
            chunk_id="test_chunk_0",
            metadata={"file_type": ".md"},
            semantic_score=0.8,
            keyword_score=0.6,
            metadata_score=0.9,
            combined_score=0.77,
            relevance_rank=1,
            search_highlights=["highlight 1", "highlight 2"]
        )
        
        assert result.document == "Test document content"
        assert result.source_file == "test.md"
        assert result.semantic_score == 0.8
        assert result.keyword_score == 0.6
        assert result.combined_score == 0.77
        assert result.relevance_rank == 1
        assert len(result.search_highlights) == 2


class TestHybridSearchEngine:
    """Test hybrid search engine functionality."""
    
    def test_hybrid_search_engine_initialization(self):
        """Test HybridSearchEngine initialization."""
        # Create mock vector store
        class MockVectorStore:
            def __init__(self):
                self.documents = []
                self.index = None
                self.model_name = "test-model"
            
            def search(self, query, k=5):
                return []
        
        mock_store = MockVectorStore()
        engine = HybridSearchEngine(mock_store)
        
        assert engine.vector_store == mock_store
        assert len(engine.search_history) == 0
    
    def test_keyword_extraction(self):
        """Test keyword extraction from text."""
        engine = HybridSearchEngine(None)
        
        # Test with stop words
        text = "The quick brown fox jumps over the lazy dog"
        keywords = engine._extract_keywords(text)
        
        assert "quick" in keywords
        assert "brown" in keywords
        assert "fox" in keywords
        assert "jumps" in keywords
        assert "lazy" in keywords
        assert "dog" in keywords
        
        # Stop words should be filtered out
        assert "the" not in keywords
        assert "over" not in keywords
    
    def test_keyword_scoring(self):
        """Test keyword relevance scoring."""
        engine = HybridSearchEngine(None)
        
        document = "Machine learning is a subset of artificial intelligence. Machine learning algorithms learn from data."
        keywords = ["machine", "learning", "algorithms"]
        
        score = engine._calculate_keyword_score(document, keywords)
        
        assert score > 0
        assert isinstance(score, float)
    
    def test_metadata_filtering(self):
        """Test metadata filtering functionality."""
        engine = HybridSearchEngine(None)
        
        # Test file type filter
        result = {
            'metadata': {'file_type': '.md'},
            'score': 0.8
        }
        
        filters = {'file_type': '.md'}
        assert engine._matches_filters(result, filters) is True
        
        filters = {'file_type': '.pdf'}
        assert engine._matches_filters(result, filters) is False
        
        # Test min confidence filter
        filters = {'min_confidence': 0.9}
        assert engine._matches_filters(result, filters) is False
        
        filters = {'min_confidence': 0.5}
        assert engine._matches_filters(result, filters) is True
    
    def test_score_combination(self):
        """Test score combination and boosting."""
        engine = HybridSearchEngine(None)
        
        results = [
            {
                'score': 0.8,
                'keyword_score': 0.6,
                'metadata_score': 0.9
            }
        ]
        
        combined = engine._combine_scores(
            results, 
            boost_semantic=2.0, 
            boost_keyword=1.0, 
            boost_metadata=1.0
        )
        
        assert len(combined) == 1
        assert 'combined_score' in combined[0]
        assert combined[0]['semantic_score'] == 0.8
        assert combined[0]['keyword_score'] == 0.6
        assert combined[0]['metadata_score'] == 0.9
    
    def test_search_highlights(self):
        """Test search highlight generation."""
        engine = HybridSearchEngine(None)
        
        # Create mock results
        results = [
            SearchResult(
                document="Machine learning is a subset of AI",
                source_file="test.md",
                chunk_id="chunk_0",
                metadata={},
                semantic_score=0.8,
                keyword_score=0.6,
                metadata_score=0.9,
                combined_score=0.77,
                relevance_rank=1,
                search_highlights=[]
            )
        ]
        
        query = "machine learning"
        highlighted_results = engine._add_search_highlights(results, query)
        
        assert len(highlighted_results[0].search_highlights) > 0
    
    def test_search_analytics(self):
        """Test search analytics functionality."""
        engine = HybridSearchEngine(None)
        
        # Add some search history
        engine.search_history = [
            {
                'query': 'test query',
                'filters': {'file_type': '.md'},
                'results_count': 5,
                'timestamp': '2024-01-01T00:00:00'
            }
        ]
        
        analytics = engine.get_search_analytics()
        
        assert analytics['total_searches'] == 1
        assert analytics['average_results_per_search'] == 5.0
        assert 'file_type' in analytics['most_common_filters']
    
    def test_query_suggestions(self):
        """Test query suggestion functionality."""
        engine = HybridSearchEngine(None)
        
        # Add search history
        engine.search_history = [
            {
                'query': 'machine learning algorithms',
                'filters': {},
                'results_count': 3,
                'timestamp': '2024-01-01T00:00:00'
            }
        ]
        
        suggestions = engine.suggest_queries('machine')
        
        assert len(suggestions) > 0
        assert 'machine learning algorithms' in suggestions


class TestAdvancedSearchRequest:
    """Test AdvancedSearchRequest dataclass."""
    
    def test_advanced_search_request_creation(self):
        """Test AdvancedSearchRequest creation with defaults."""
        request = AdvancedSearchRequest(query="test query")
        
        assert request.query == "test query"
        assert request.search_type == "hybrid"
        assert request.filters is None
        assert request.max_results == 10
        assert request.generate_answer is False
        assert request.include_highlights is True
    
    def test_advanced_search_request_custom_values(self):
        """Test AdvancedSearchRequest creation with custom values."""
        request = AdvancedSearchRequest(
            query="custom query",
            search_type="rag",
            filters={"file_type": ".md"},
            max_results=20,
            generate_answer=True
        )
        
        assert request.query == "custom query"
        assert request.search_type == "rag"
        assert request.filters == {"file_type": ".md"}
        assert request.max_results == 20
        assert request.generate_answer is True


class TestAdvancedSearchResponse:
    """Test AdvancedSearchResponse dataclass."""
    
    def test_advanced_search_response_creation(self):
        """Test AdvancedSearchResponse creation."""
        response = AdvancedSearchResponse(
            query="test query",
            search_type="hybrid",
            results=[],
            processing_time=0.1,
            total_results=0
        )
        
        assert response.query == "test query"
        assert response.search_type == "hybrid"
        assert len(response.results) == 0
        assert response.processing_time == 0.1
        assert response.total_results == 0


class TestAdvancedSearchInterface:
    """Test advanced search interface functionality."""
    
    def test_advanced_search_interface_initialization(self):
        """Test AdvancedSearchInterface initialization."""
        # Create mock components
        class MockVectorStore:
            def __init__(self):
                self.documents = []
                self.index = None
                self.model_name = "test-model"
            
            def get_document_stats(self):
                return {'total_documents': 0}
        
        class MockLLMConfig:
            def __init__(self):
                self.provider = "none"
        
        mock_store = MockVectorStore()
        mock_config = MockLLMConfig()
        
        interface = AdvancedSearchInterface(mock_store, mock_config)
        
        assert interface.vector_store == mock_store
        assert interface.llm_config == mock_config
        assert 'min_confidence' in interface.default_filters
    
    def test_filter_preparation(self):
        """Test filter preparation and merging."""
        # Create mock components
        class MockVectorStore:
            def __init__(self):
                self.documents = []
                self.index = None
                self.model_name = "test-model"
            
            def get_document_stats(self):
                return {'total_documents': 0}
        
        class MockLLMConfig:
            def __init__(self):
                self.provider = "none"
        
        mock_store = MockVectorStore()
        mock_config = MockLLMConfig()
        
        interface = AdvancedSearchInterface(mock_store, mock_config)
        
        # Test default filters
        filters = interface._prepare_filters(None)
        assert 'min_confidence' in filters
        assert 'include_file_types' in filters
        
        # Test user filters
        user_filters = {'custom_filter': 'value'}
        filters = interface._prepare_filters(user_filters)
        assert filters['custom_filter'] == 'value'
        assert 'min_confidence' in filters  # Defaults preserved
    
    def test_search_statistics(self):
        """Test search statistics retrieval."""
        # Create mock components
        class MockVectorStore:
            def __init__(self):
                self.documents = []
                self.index = None
                self.model_name = "test-model"
            
            def get_document_stats(self):
                return {'total_documents': 10}
        
        class MockLLMConfig:
            def __init__(self):
                self.provider = "none"
        
        mock_store = MockVectorStore()
        mock_config = MockLLMConfig()
        
        interface = AdvancedSearchInterface(mock_store, mock_config)
        
        stats = interface.get_search_statistics()
        
        assert 'search_analytics' in stats
        assert 'vector_store_stats' in stats
        assert 'rag_capabilities' in stats
        assert 'search_engine_info' in stats
    
    def test_export_functionality(self):
        """Test search result export functionality."""
        # Create mock components
        class MockVectorStore:
            def __init__(self):
                self.documents = []
                self.index = None
                self.model_name = "test-model"
            
            def get_document_stats(self):
                return {'total_documents': 0}
        
        class MockLLMConfig:
            def __init__(self):
                self.provider = "none"
        
        mock_store = MockVectorStore()
        mock_config = MockLLMConfig()
        
        interface = AdvancedSearchInterface(mock_store, mock_config)
        
        # Create mock response
        response = AdvancedSearchResponse(
            query="test query",
            search_type="hybrid",
            results=[],
            processing_time=0.1,
            total_results=0
        )
        
        # Test JSON export
        json_export = interface.export_search_results(response, "json")
        assert "test query" in json_export
        
        # Test CSV export
        csv_export = interface.export_search_results(response, "csv")
        assert "Rank" in csv_export
        assert "Document" in csv_export
        
        # Test unsupported format
        unsupported = interface.export_search_results(response, "xml")
        assert "Unsupported format" in unsupported
    
    def test_batch_search(self):
        """Test batch search functionality."""
        # Create mock components
        class MockVectorStore:
            def __init__(self):
                self.documents = []
                self.index = None
                self.model_name = "test-model"
            
            def get_document_stats(self):
                return {'total_documents': 0}
        
        class MockLLMConfig:
            def __init__(self):
                self.provider = "none"
        
        mock_store = MockVectorStore()
        mock_config = MockLLMConfig()
        
        interface = AdvancedSearchInterface(mock_store, mock_config)
        
        queries = ["query 1", "query 2", "query 3"]
        responses = interface.batch_search(queries)
        
        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert response.query == queries[i]


def test_integration_workflow():
    """Test the complete integration workflow."""
    # Create mock components
    class MockVectorStore:
        def __init__(self):
            self.documents = []
            self.index = None
            self.model_name = "test-model"
        
        def search(self, query, k=5):
            return [
                {
                    'document': 'Test document content',
                    'score': 0.8,
                    'source_file': 'test.md',
                    'chunk_id': 'chunk_0',
                    'metadata': {'file_type': '.md'}
                }
            ]
        
        def get_document_stats(self):
            return {'total_documents': 1}
    
    class MockLLMConfig:
        def __init__(self):
            self.provider = "none"
    
    # Test hybrid search engine
    mock_store = MockVectorStore()
    engine = HybridSearchEngine(mock_store)
    
    query = SearchQuery(
        text="test query",
        filters={"file_type": ".md"}
    )
    
    results = engine.search(query, k=5)
    assert len(results) > 0
    
    # Test advanced search interface
    mock_config = MockLLMConfig()
    interface = AdvancedSearchInterface(mock_store, mock_config)
    
    request = AdvancedSearchRequest(
        query="test query",
        search_type="hybrid",
        max_results=5
    )
    
    response = interface.search(request)
    assert response.query == "test query"
    assert response.search_type == "hybrid"
    assert response.total_results >= 0


if __name__ == "__main__":
    pytest.main([__file__])
