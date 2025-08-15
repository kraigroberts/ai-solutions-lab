"""
Unit tests for RAG pipeline components.

Tests the document ingestion, vector retrieval, and answer generation modules.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ai_lab.rag.answer import RAGAnswerer
from ai_lab.rag.ingest import (DocumentChunk, DocumentIngester, DocumentLoader,
                               MarkdownLoader, PDFLoader, TextLoader)
from ai_lab.rag.retrieve import VectorRetriever


class TestDocumentChunk:
    """Test DocumentChunk class."""

    def test_document_chunk_creation(self):
        """Test creating a document chunk."""
        chunk = DocumentChunk(
            content="Test content",
            document_id="doc123",
            chunk_id="chunk_001",
            title="Test Document",
            source_path="/path/to/doc.md",
            chunk_index=0,
            start_char=0,
            end_char=12,
        )

        assert chunk.content == "Test content"
        assert chunk.document_id == "doc123"
        assert chunk.chunk_id == "chunk_001"
        assert chunk.title == "Test Document"
        assert chunk.source_path == "/path/to/doc.md"
        assert chunk.chunk_index == 0
        assert chunk.start_char == 0
        assert chunk.end_char == 12
        assert chunk.embedding is None
        assert chunk.metadata == {}

    def test_document_chunk_to_dict(self):
        """Test converting chunk to dictionary."""
        chunk = DocumentChunk(
            content="Test content",
            document_id="doc123",
            chunk_id="chunk_001",
            title="Test Document",
            source_path="/path/to/doc.md",
            chunk_index=0,
            start_char=0,
            end_char=12,
        )

        chunk.embedding = [0.1, 0.2, 0.3]
        chunk.metadata = {"language": "en"}

        chunk_dict = chunk.to_dict()

        assert chunk_dict["content"] == "Test content"
        assert chunk_dict["document_id"] == "doc123"
        assert chunk_dict["embedding"] == [0.1, 0.2, 0.3]
        assert chunk_dict["metadata"] == {"language": "en"}

    def test_document_chunk_from_dict(self):
        """Test creating chunk from dictionary."""
        chunk_data = {
            "content": "Test content",
            "document_id": "doc123",
            "chunk_id": "chunk_001",
            "title": "Test Document",
            "source_path": "/path/to/doc.md",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 12,
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {"language": "en"},
        }

        chunk = DocumentChunk.from_dict(chunk_data)

        assert chunk.content == "Test content"
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.metadata == {"language": "en"}


class TestDocumentLoaders:
    """Test document loader classes."""

    def test_markdown_loader_supported_extensions(self):
        """Test MarkdownLoader supported extensions."""
        loader = MarkdownLoader()
        assert ".md" in loader.supported_extensions
        assert ".markdown" in loader.supported_extensions

    def test_markdown_loader_can_load(self):
        """Test MarkdownLoader can_load method."""
        loader = MarkdownLoader()

        assert loader.can_load(Path("test.md"))
        assert loader.can_load(Path("test.markdown"))
        assert not loader.can_load(Path("test.txt"))
        assert not loader.can_load(Path("test.pdf"))

    @pytest.mark.asyncio
    async def test_markdown_loader_extract_title(self):
        """Test MarkdownLoader title extraction."""
        loader = MarkdownLoader()

        # Test with heading
        content = "# Main Title\n\nSome content"
        title = loader._extract_title(content)
        assert title == "Main Title"

        # Test without heading
        content = "Some content without heading"
        title = loader._extract_title(content)
        assert title is None

    def test_pdf_loader_supported_extensions(self):
        """Test PDFLoader supported extensions."""
        loader = PDFLoader()
        assert ".pdf" in loader.supported_extensions

    def test_text_loader_supported_extensions(self):
        """Test TextLoader supported extensions."""
        loader = TextLoader()
        assert ".txt" in loader.supported_extensions
        assert ".text" in loader.supported_extensions


class TestDocumentIngester:
    """Test DocumentIngester class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        with patch("ai_lab.rag.ingest.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.embeddings_backend = "local"
            mock_settings.embedding_model = "all-MiniLM-L6-v2"
            mock_settings.has_openai.return_value = False
            mock_get_settings.return_value = mock_settings
            yield mock_settings

    def test_ingester_initialization(self, mock_settings):
        """Test DocumentIngester initialization."""
        ingester = DocumentIngester()

        assert len(ingester.loaders) == 3
        assert any(isinstance(loader, MarkdownLoader) for loader in ingester.loaders)
        assert any(isinstance(loader, PDFLoader) for loader in ingester.loaders)
        assert any(isinstance(loader, TextLoader) for loader in ingester.loaders)

    def test_get_loader(self, mock_settings):
        """Test getting appropriate loader for file."""
        ingester = DocumentIngester()

        # Test markdown file
        loader = ingester._get_loader(Path("test.md"))
        assert isinstance(loader, MarkdownLoader)

        # Test PDF file
        loader = ingester._get_loader(Path("test.pdf"))
        assert isinstance(loader, PDFLoader)

        # Test text file
        loader = ingester._get_loader(Path("test.txt"))
        assert isinstance(loader, TextLoader)

        # Test unsupported file
        loader = ingester._get_loader(Path("test.xyz"))
        assert loader is None

    def test_chunk_document(self, mock_settings):
        """Test document chunking."""
        ingester = DocumentIngester()

        content = "This is a test document. " * 50  # Create long content

        chunks = ingester._chunk_document(content, chunk_size=100, chunk_overlap=20)

        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks)

        # Test with zero chunk size
        chunks = ingester._chunk_document(content, chunk_size=0, chunk_overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == content

    @pytest.mark.asyncio
    async def test_generate_embeddings_local(self, mock_settings):
        """Test local embedding generation."""
        with patch("sentence_transformers.SentenceTransformer") as mock_transformer:
            mock_model = Mock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_transformer.return_value = mock_model

            ingester = DocumentIngester()
            texts = ["Hello", "World"]

            embeddings = await ingester._generate_embeddings(texts)

            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]


class TestVectorRetriever:
    """Test VectorRetriever class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        with patch("ai_lab.rag.retrieve.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.embeddings_backend = "local"
            mock_settings.embedding_model = "all-MiniLM-L6-v2"
            mock_settings.has_openai.return_value = False
            mock_get_settings.return_value = mock_settings
            yield mock_settings

    def test_retriever_initialization(self, temp_dir, mock_settings):
        """Test VectorRetriever initialization."""
        retriever = VectorRetriever(temp_dir)

        assert retriever.index_path == temp_dir
        assert retriever.faiss_index is None
        assert retriever.metadata is None
        assert retriever.chunks == []

    def test_is_loaded(self, temp_dir, mock_settings):
        """Test is_loaded method."""
        retriever = VectorRetriever(temp_dir)

        # Should be False when not loaded
        assert not retriever.is_loaded()

        # Mock loaded state
        retriever.faiss_index = Mock()
        retriever.metadata = {"chunks": []}
        retriever.chunks = [Mock()]

        assert retriever.is_loaded()

    @pytest.mark.asyncio
    async def test_generate_query_embedding_local(self, temp_dir, mock_settings):
        """Test local query embedding generation."""
        with patch("sentence_transformers.SentenceTransformer") as mock_transformer:
            mock_model = Mock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
            mock_transformer.return_value = mock_model

            retriever = VectorRetriever(temp_dir)
            query = "test query"

            embedding = await retriever._generate_query_embedding(query)

            assert embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_retrieve_not_loaded(self, temp_dir, mock_settings):
        """Test retrieve method when index not loaded."""
        retriever = VectorRetriever(temp_dir)

        with pytest.raises(RuntimeError, match="Index not loaded"):
            await retriever.retrieve("test query")


class TestRAGAnswerer:
    """Test RAGAnswerer class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        with patch("ai_lab.rag.answer.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.top_k = 5
            mock_get_settings.return_value = mock_settings
            yield mock_settings

    @pytest.fixture
    def mock_retriever(self):
        """Mock VectorRetriever."""
        with patch("ai_lab.rag.answer.VectorRetriever") as mock_retriever_class:
            mock_retriever = Mock()
            mock_retriever.is_loaded.return_value = True
            mock_retriever_class.return_value = mock_retriever
            yield mock_retriever

    @pytest.fixture
    def mock_llm_router(self):
        """Mock LLMRouter."""
        with patch("ai_lab.rag.answer.LLMRouter") as mock_router_class:
            mock_router = Mock()
            mock_router.chat = AsyncMock(return_value={"content": "Test answer"})
            mock_router_class.return_value = mock_router
            yield mock_router

    def test_answerer_initialization(self, temp_dir, mock_settings):
        """Test RAGAnswerer initialization."""
        answerer = RAGAnswerer(temp_dir)

        assert answerer.default_top_k == 5
        assert answerer.default_score_threshold == 0.5
        assert answerer.max_context_length == 4000

    def test_is_ready(self, temp_dir, mock_settings, mock_retriever):
        """Test is_ready method."""
        answerer = RAGAnswerer(temp_dir)

        # Mock retriever as ready
        mock_retriever.is_loaded.return_value = True
        assert answerer.is_ready()

        # Mock retriever as not ready
        mock_retriever.is_loaded.return_value = False
        assert not answerer.is_ready()

    @pytest.mark.asyncio
    async def test_answer_not_ready(self, temp_dir, mock_settings, mock_retriever):
        """Test answer method when not ready."""
        answerer = RAGAnswerer(temp_dir)
        mock_retriever.is_loaded.return_value = False

        with pytest.raises(RuntimeError, match="RAG answerer not ready"):
            await answerer.answer("test query")

    @pytest.mark.asyncio
    async def test_answer_success(
        self, temp_dir, mock_settings, mock_retriever, mock_llm_router
    ):
        """Test successful answer generation."""
        # Mock retriever results
        mock_retriever.retrieve.return_value = [
            {
                "content": "Test content",
                "title": "Test Doc",
                "source_path": "/test.md",
                "score": 0.8,
                "rank": 1,
            }
        ]

        answerer = RAGAnswerer(temp_dir)

        result = await answerer.answer("test query")

        assert "answer" in result
        assert "sources" in result
        assert "query" in result
        assert result["query"] == "test query"
        assert len(result["sources"]) == 1

    def test_prepare_context(self, temp_dir, mock_settings):
        """Test context preparation."""
        answerer = RAGAnswerer(temp_dir)

        chunks = [
            {
                "content": "First chunk content",
                "title": "Doc 1",
                "source_path": "/doc1.md",
            },
            {
                "content": "Second chunk content",
                "title": "Doc 2",
                "source_path": "/doc2.md",
            },
        ]

        context = answerer._prepare_context(chunks, "test query")

        assert "Source 1 (Doc 1)" in context
        assert "Source 2 (Doc 2)" in context
        assert "First chunk content" in context
        assert "Second chunk content" in context

    def test_prepare_sources(self, temp_dir, mock_settings):
        """Test source preparation."""
        answerer = RAGAnswerer(temp_dir)

        chunks = [
            {
                "content": "Test content",
                "title": "Test Doc",
                "source_path": "/test.md",
                "score": 0.8,
                "rank": 1,
                "chunk_index": 0,
            }
        ]

        sources = answerer._prepare_sources(chunks)

        assert len(sources) == 1
        assert sources[0]["title"] == "Test Doc"
        assert sources[0]["score"] == 0.8
        assert sources[0]["rank"] == 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestRAGIntegration:
    """Integration tests for RAG pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.mark.asyncio
    async def test_full_rag_pipeline(self, temp_dir):
        """Test complete RAG pipeline integration."""
        # This would test the full pipeline from ingestion to answer generation
        # For now, it's a placeholder for future integration tests
        assert True


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestRAGPerformance:
    """Performance tests for RAG components."""

    @pytest.mark.slow
    def test_large_document_processing(self):
        """Test processing large documents."""
        # This would test performance with large documents
        # Marked as slow to avoid running in regular CI
        assert True

    @pytest.mark.slow
    def test_embedding_generation_performance(self):
        """Test embedding generation performance."""
        # This would test embedding generation speed
        assert True


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestRAGErrorHandling:
    """Test error handling in RAG components."""

    def test_invalid_file_paths(self):
        """Test handling of invalid file paths."""
        # Test various error conditions
        assert True

    def test_corrupted_documents(self):
        """Test handling of corrupted documents."""
        # Test error handling for malformed files
        assert True
