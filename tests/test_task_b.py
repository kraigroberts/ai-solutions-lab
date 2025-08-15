"""Tests for Task B: Core Infrastructure & Data."""

import pytest
from pathlib import Path
import tempfile
import shutil

from src.ai_lab.document_ingestion import DocumentIngester, DocumentChunk
from src.ai_lab.vector_store import VectorStore


class TestDocumentIngestion:
    """Test document ingestion functionality."""
    
    def test_document_chunk_creation(self):
        """Test DocumentChunk dataclass."""
        chunk = DocumentChunk(
            text="Test text",
            source_file="test.md",
            chunk_id="test_chunk_0",
            start_char=0,
            end_char=9,
            metadata={"file_type": ".md"}
        )
        
        assert chunk.text == "Test text"
        assert chunk.source_file == "test.md"
        assert chunk.chunk_id == "test_chunk_0"
        assert chunk.start_char == 0
        assert chunk.end_char == 9
        assert chunk.metadata["file_type"] == ".md"
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        ingester = DocumentIngester()
        
        dirty_text = "  This   is   dirty   text  \n\nwith\n\n\nnewlines  "
        cleaned = ingester.clean_text(dirty_text)
        
        assert cleaned == "This is dirty text with newlines"
        assert "  " not in cleaned  # No double spaces
        assert "\n" not in cleaned  # No newlines
    
    def test_chunking_small_text(self):
        """Test chunking text smaller than chunk size."""
        ingester = DocumentIngester(chunk_size=100, chunk_overlap=20)
        
        text = "This is a short text that should fit in one chunk."
        chunks = ingester.chunk_text(text, "test.md")
        
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].source_file == "test.md"
    
    def test_chunking_large_text(self):
        """Test chunking text larger than chunk size."""
        ingester = DocumentIngester(chunk_size=50, chunk_overlap=10)
        
        # Create text longer than chunk size
        text = "This is a longer text that should be split into multiple chunks. " * 3
        chunks = ingester.chunk_text(text, "test.md")
        
        assert len(chunks) > 1
        assert all(len(chunk.text) <= 50 for chunk in chunks)
        
        # Check overlap
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            curr_chunk = chunks[i]
            # Should have some overlap
            assert prev_chunk.end_char > curr_chunk.start_char
    
    def test_markdown_loading(self, tmp_path):
        """Test markdown file loading."""
        ingester = DocumentIngester()
        
        # Create test markdown file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test\nThis is test content.")
        
        content = ingester.load_markdown(test_file)
        assert "# Test" in content
        assert "This is test content." in content
    
    def test_pdf_loading_placeholder(self):
        """Test PDF loading (placeholder - would need actual PDF)."""
        ingester = DocumentIngester()
        
        # This test would require a real PDF file
        # For now, just test the method exists
        assert hasattr(ingester, 'load_pdf')
        assert callable(ingester.load_pdf)


class TestVectorStore:
    """Test vector store functionality."""
    
    def test_vector_store_initialization(self):
        """Test vector store initialization."""
        vector_store = VectorStore()
        
        assert vector_store.model_name == 'all-MiniLM-L6-v2'
        assert vector_store.index_dir == Path("./data/index")
        assert vector_store.index is None
        assert len(vector_store.documents) == 0
    
    def test_adding_documents(self):
        """Test adding documents to vector store."""
        vector_store = VectorStore()
        
        # Create test chunks
        chunks = [
            DocumentChunk(
                text="First document",
                source_file="test1.md",
                chunk_id="test1_chunk_0",
                start_char=0,
                end_char=15,
                metadata={"file_type": ".md"}
            ),
            DocumentChunk(
                text="Second document",
                source_file="test2.md",
                chunk_id="test2_chunk_0",
                start_char=0,
                end_char=16,
                metadata={"file_type": ".md"}
            )
        ]
        
        vector_store.add_documents(chunks)
        
        assert len(vector_store.documents) == 2
        assert vector_store.index is not None
        assert vector_store.index.ntotal == 2
    
    def test_search_functionality(self):
        """Test search functionality."""
        vector_store = VectorStore()
        
        # Add test documents
        chunks = [
            DocumentChunk(
                text="Machine learning is a subset of AI",
                source_file="ml.md",
                chunk_id="ml_chunk_0",
                start_char=0,
                end_char=35,
                metadata={"file_type": ".md"}
            ),
            DocumentChunk(
                text="Python is a programming language",
                source_file="python.md",
                chunk_id="python_chunk_0",
                start_char=0,
                end_char=32,
                metadata={"file_type": ".md"}
            )
        ]
        
        vector_store.add_documents(chunks)
        
        # Search for ML-related content
        results = vector_store.search("machine learning", k=2)
        
        assert len(results) > 0
        assert any("Machine learning" in result['document'] for result in results)
        
        # Check result structure
        for result in results:
            assert 'document' in result
            assert 'score' in result
            assert 'source_file' in result
            assert 'chunk_id' in result
            assert 'metadata' in result
    
    def test_metadata_filtering(self):
        """Test metadata-based filtering."""
        vector_store = VectorStore()
        
        # Add documents with different file types
        chunks = [
            DocumentChunk(
                text="Markdown content",
                source_file="doc1.md",
                chunk_id="md_chunk_0",
                start_char=0,
                end_char=17,
                metadata={"file_type": ".md", "category": "tutorial"}
            ),
            DocumentChunk(
                text="PDF content",
                source_file="doc2.pdf",
                chunk_id="pdf_chunk_0",
                start_char=0,
                end_char=12,
                metadata={"file_type": ".pdf", "category": "reference"}
            )
        ]
        
        vector_store.add_documents(chunks)
        
        # Filter by file type
        md_results = vector_store.search_by_metadata(
            "content", 
            {"file_type": ".md"}, 
            k=5
        )
        
        assert len(md_results) > 0
        assert all(result['metadata']['file_type'] == '.md' for result in md_results)
    
    def test_document_stats(self):
        """Test document statistics."""
        vector_store = VectorStore()
        
        # Add some documents
        chunks = [
            DocumentChunk(
                text="Test content",
                source_file="test.md",
                chunk_id="test_chunk_0",
                start_char=0,
                end_char=12,
                metadata={"file_type": ".md"}
            )
        ]
        
        vector_store.add_documents(chunks)
        
        stats = vector_store.get_document_stats()
        
        assert stats['total_documents'] == 1
        assert stats['index_size'] == 1
        assert stats['model_name'] == 'all-MiniLM-L6-v2'
        assert '.md' in stats['file_types']
        assert stats['file_types']['.md'] == 1


def test_integration_workflow():
    """Test the complete workflow from ingestion to search."""
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test document
        test_doc = temp_path / "test.md"
        test_doc.write_text("""
        Machine learning is a subset of artificial intelligence.
        It enables computers to learn from data without explicit programming.
        Common applications include image recognition and natural language processing.
        """)
        
        # Test document ingestion
        ingester = DocumentIngester(chunk_size=100, chunk_overlap=20)
        chunks = ingester.process_file(test_doc)
        
        assert len(chunks) > 0
        
        # Test vector store
        vector_store = VectorStore(index_dir=str(temp_path / "index"))
        vector_store.add_documents(chunks)
        
        # Test search
        results = vector_store.search("machine learning", k=3)
        assert len(results) > 0
        
        # Test saving and loading
        vector_store.save_index()
        
        # Create new vector store and load
        new_vector_store = VectorStore(index_dir=str(temp_path / "index"))
        loaded = new_vector_store.load_index()
        
        assert loaded
        assert len(new_vector_store.documents) == len(chunks)


if __name__ == "__main__":
    pytest.main([__file__])
