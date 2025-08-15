"""Tests for Task C: LLM Integration & RAG Enhancement."""

import pytest
import sys
from pathlib import Path
import tempfile
import json
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_lab.llm_config import LLMConfig, create_default_config
from ai_lab.llm_providers import (
    LLMProvider, OpenAIProvider, AnthropicProvider, 
    LocalProvider, MockProvider, create_llm_provider
)
from ai_lab.enhanced_rag import EnhancedRAG, RAGResponse


class TestLLMConfig:
    """Test LLM configuration system."""
    
    def test_llm_config_creation(self):
        """Test LLMConfig creation."""
        config = LLMConfig(provider="openai")
        assert config.provider == "openai"
        assert config.openai_model == "gpt-3.5-turbo"
        assert config.anthropic_model == "claude-3-haiku-20240307"
    
    def test_environment_variable_loading(self, monkeypatch):
        """Test loading configuration from environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key_123")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic_key")
        
        config = LLMConfig()
        assert config.openai_api_key == "test_key_123"
        assert config.anthropic_api_key == "test_anthropic_key"
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # OpenAI configured
        config = LLMConfig(provider="openai", openai_api_key="test_key")
        assert config.is_configured() is True
        
        # OpenAI not configured
        config = LLMConfig(provider="openai", openai_api_key=None)
        assert config.is_configured() is False
        
        # None provider always configured
        config = LLMConfig(provider="none")
        assert config.is_configured() is True
    
    def test_provider_info(self):
        """Test provider information retrieval."""
        config = LLMConfig(provider="openai", openai_api_key="test_key")
        info = config.get_provider_info()
        
        assert info["provider"] == "openai"
        assert info["model"] == "gpt-3.5-turbo"
        assert info["configured"] is True
        assert info["api_key_present"] is True
    
    def test_config_save_load(self, tmp_path):
        """Test configuration save and load."""
        config = LLMConfig(
            provider="openai",
            openai_model="gpt-4",
            openai_temperature=0.5
        )
        
        config_file = tmp_path / "test_config.json"
        config.save_to_file(str(config_file))
        
        # Verify file was created
        assert config_file.exists()
        
        # Load configuration
        loaded_config = LLMConfig.load_from_file(str(config_file))
        assert loaded_config.provider == "openai"
        assert loaded_config.openai_model == "gpt-4"
        assert loaded_config.openai_temperature == 0.5
    
    def test_create_default_config(self, monkeypatch):
        """Test default configuration creation."""
        # Test with OpenAI key
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        config = create_default_config()
        assert config.provider == "openai"
        
        # Test with no keys
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = create_default_config()
        assert config.provider == "none"


class TestLLMProviders:
    """Test LLM provider implementations."""
    
    def test_mock_provider(self):
        """Test mock provider functionality."""
        config = LLMConfig(provider="none")
        provider = MockProvider(config)
        
        assert provider.is_available() is True
        
        info = provider.get_model_info()
        assert info["provider"] == "mock"
        assert info["available"] is True
        
        response = provider.generate_response("test question")
        assert "Mock response" in response
        assert "test question" in response
    
    def test_mock_provider_with_context(self):
        """Test mock provider with context."""
        config = LLMConfig(provider="none")
        provider = MockProvider(config)
        
        response = provider.generate_response("test question", context="test context")
        assert "test context" in response
        assert "test question" in response
    
    def test_openai_provider_creation(self):
        """Test OpenAI provider creation."""
        config = LLMConfig(provider="openai", openai_api_key="test_key")
        provider = OpenAIProvider(config)
        
        # Should not be available without actual API key
        assert provider.is_available() is False
        
        info = provider.get_model_info()
        assert info["provider"] == "openai"
        assert info["model"] == "gpt-3.5-turbo"
    
    def test_anthropic_provider_creation(self):
        """Test Anthropic provider creation."""
        config = LLMConfig(provider="anthropic", anthropic_api_key="test_key")
        provider = AnthropicProvider(config)
        
        # Should not be available without actual API key
        assert provider.is_available() is False
        
        info = provider.get_model_info()
        assert info["provider"] == "anthropic"
        assert info["model"] == "claude-3-haiku-20240307"
    
    def test_local_provider_creation(self):
        """Test local provider creation."""
        config = LLMConfig(provider="local", local_model_path="/nonexistent/path")
        provider = LocalProvider(config)
        
        # Should not be available without actual model file
        assert provider.is_available() is False
        
        info = provider.get_model_info()
        assert info["provider"] == "local"
        assert info["model_path"] == "/nonexistent/path"
    
    def test_provider_factory(self):
        """Test provider factory function."""
        config = LLMConfig(provider="none")
        provider = create_llm_provider(config)
        
        assert isinstance(provider, MockProvider)
        
        # Test with OpenAI
        config.provider = "openai"
        provider = create_llm_provider(config)
        assert isinstance(provider, OpenAIProvider)


class TestEnhancedRAG:
    """Test enhanced RAG system."""
    
    def test_rag_response_creation(self):
        """Test RAGResponse dataclass."""
        response = RAGResponse(
            answer="Test answer",
            sources=[{"file": "test.md"}],
            search_results=[{"document": "test"}],
            llm_provider="mock",
            processing_time=0.1,
            confidence_score=0.8
        )
        
        assert response.answer == "Test answer"
        assert len(response.sources) == 1
        assert response.confidence_score == 0.8
        assert response.llm_provider == "mock"
    
    def test_enhanced_rag_initialization(self):
        """Test EnhancedRAG initialization."""
        # Create mock vector store
        class MockVectorStore:
            def __init__(self):
                self.documents = []
                self.index = None
                self.model_name = "test-model"
        
        mock_store = MockVectorStore()
        config = LLMConfig(provider="none")
        
        rag = EnhancedRAG(mock_store, config)
        
        assert rag.vector_store == mock_store
        assert rag.llm_config == config
        assert isinstance(rag.llm_provider, MockProvider)
    
    def test_rag_search_with_no_results(self):
        """Test RAG search with no search results."""
        class MockVectorStore:
            def __init__(self):
                self.documents = []
                self.index = None
                self.model_name = "test-model"
            
            def search(self, query, k=5):
                return []
        
        mock_store = MockVectorStore()
        config = LLMConfig(provider="none")
        rag = EnhancedRAG(mock_store, config)
        
        response = rag.search_and_answer("test query")
        
        assert "couldn't find any relevant information" in response.answer
        assert len(response.sources) == 0
        assert len(response.search_results) == 0
        assert response.confidence_score == 0.0
    
    def test_rag_search_with_results(self):
        """Test RAG search with search results."""
        class MockVectorStore:
            def __init__(self):
                self.documents = ["doc1", "doc2"]
                self.index = None
                self.model_name = "test-model"
            
            def search(self, query, k=5):
                return [
                    {
                        'document': 'Machine learning is a subset of AI',
                        'score': 0.8,
                        'source_file': 'test.md',
                        'chunk_id': 'chunk_0',
                        'metadata': {'file_type': '.md'}
                    }
                ]
        
        mock_store = MockVectorStore()
        config = LLMConfig(provider="none")
        rag = EnhancedRAG(mock_store, config)
        
        response = rag.search_and_answer("What is machine learning?")
        
        # Mock provider includes context in response, so check for the mock response pattern
        assert "Mock response based on context" in response.answer
        assert len(response.sources) == 1
        assert len(response.search_results) == 1
        assert response.confidence_score == 0.8
        assert response.sources[0]['file'] == 'test.md'
    
    def test_context_preparation(self):
        """Test context preparation from search results."""
        class MockVectorStore:
            def __init__(self):
                self.documents = []
                self.index = None
                self.model_name = "test-model"
            
            def search(self, query, k=5):
                return [
                    {
                        'document': 'Test document content',
                        'score': 0.9,
                        'source_file': 'test.md',
                        'chunk_id': 'chunk_0',
                        'metadata': {'file_type': '.md'}
                    }
                ]
        
        mock_store = MockVectorStore()
        config = LLMConfig(provider="none")
        rag = EnhancedRAG(mock_store, config)
        
        response = rag.search_and_answer("test query")
        
        # Check that context was prepared
        assert "Source 1 (test.md, relevance: 0.900)" in response.answer
        # Mock provider includes context in response
        assert "Mock response based on context" in response.answer
    
    def test_conversational_search(self):
        """Test conversational search with history."""
        class MockVectorStore:
            def __init__(self):
                self.documents = []
                self.index = None
                self.model_name = "test-model"
            
            def search(self, query, k=5):
                return [
                    {
                        'document': 'Follow-up answer',
                        'score': 0.8,
                        'source_file': 'test.md',
                        'chunk_id': 'chunk_0',
                        'metadata': {'file_type': '.md'}
                    }
                ]
        
        mock_store = MockVectorStore()
        config = LLMConfig(provider="none")
        rag = EnhancedRAG(mock_store, config)
        
        conversation_history = [
            {'user': 'What is AI?', 'assistant': 'AI is artificial intelligence'},
            {'user': 'Tell me more', 'assistant': 'AI includes machine learning'}
        ]
        
        response = rag.conversational_search("What about machine learning?", conversation_history)
        
        # Should include conversation context
        assert "Mock response based on context" in response.answer
    
    def test_system_status(self):
        """Test system status retrieval."""
        class MockVectorStore:
            def __init__(self):
                self.documents = ["doc1", "doc2"]
                self.index = None
                self.model_name = "test-model"
        
        mock_store = MockVectorStore()
        config = LLMConfig(provider="none")
        rag = EnhancedRAG(mock_store, config)
        
        status = rag.get_system_status()
        
        assert status['vector_store']['total_documents'] == 2
        assert status['vector_store']['embedding_model'] == "test-model"
        assert status['llm_provider']['provider'] == "mock"
        assert status['capabilities']['search'] is True
        assert status['capabilities']['answer_generation'] is True
    
    def test_batch_processing(self):
        """Test batch query processing."""
        class MockVectorStore:
            def __init__(self):
                self.documents = []
                self.index = None
                self.model_name = "test-model"
            
            def search(self, query, k=5):
                return [
                    {
                        'document': f'Answer to: {query}',
                        'score': 0.8,
                        'source_file': 'test.md',
                        'chunk_id': 'chunk_0',
                        'metadata': {'file_type': '.md'}
                    }
                ]
        
        mock_store = MockVectorStore()
        config = LLMConfig(provider="none")
        rag = EnhancedRAG(mock_store, config)
        
        queries = ["What is AI?", "What is ML?", "What is DL?"]
        responses = rag.batch_process_queries(queries)
        
        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert queries[i] in response.answer
            assert response.confidence_score == 0.8


def test_integration_workflow():
    """Test the complete integration workflow."""
    # Create configuration
    config = LLMConfig(provider="none")
    
    # Create mock vector store
    class MockVectorStore:
        def __init__(self):
            self.documents = ["doc1", "doc2"]
            self.index = None
            self.model_name = "test-model"
        
        def search(self, query, k=5):
            return [
                {
                    'document': 'Machine learning is a subset of artificial intelligence.',
                    'score': 0.9,
                    'source_file': 'ml.md',
                    'chunk_id': 'ml_chunk_0',
                    'metadata': {'file_type': '.md'}
                }
            ]
    
    mock_store = MockVectorStore()
    
    # Create enhanced RAG
    rag = EnhancedRAG(mock_store, config)
    
    # Test search and answer
    response = rag.search_and_answer("What is machine learning?")
    
    # Mock provider includes context in response
    assert "Mock response based on context" in response.answer
    assert response.confidence_score == 0.9
    assert response.llm_provider == "mock"
    assert len(response.sources) == 1


if __name__ == "__main__":
    pytest.main([__file__])
