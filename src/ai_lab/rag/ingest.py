"""
Document Ingestion Pipeline for AI Solutions Lab.

Provides functionality for:
- Loading documents from various formats
- Chunking documents into manageable pieces
- Generating embeddings for chunks
- Building and saving vector indexes
"""

import asyncio
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import get_settings
from ..llm.router import LLMRouter


class DocumentChunk:
    """Represents a chunk of a document with metadata."""

    def __init__(
        self,
        content: str,
        document_id: str,
        chunk_id: str,
        title: str,
        source_path: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
    ):
        self.content = content
        self.document_id = document_id
        self.chunk_id = chunk_id
        self.title = title
        self.source_path = source_path
        self.chunk_index = chunk_index
        self.start_char = start_char
        self.end_char = end_char
        self.embedding: Optional[List[float]] = None
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return {
            "content": self.content,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "source_path": self.source_path,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        """Create chunk from dictionary."""
        chunk = cls(
            content=data["content"],
            document_id=data["document_id"],
            chunk_id=data["chunk_id"],
            title=data["title"],
            source_path=data["source_path"],
            chunk_index=data["chunk_index"],
            start_char=data["start_char"],
            end_char=data["end_char"],
        )
        chunk.embedding = data.get("embedding")
        chunk.metadata = data.get("metadata", {})
        return chunk


class DocumentLoader:
    """Base class for document loaders."""

    def __init__(self):
        """Initialize document loader."""
        self.supported_extensions: List[str] = []

    def can_load(self, file_path: Path) -> bool:
        """Check if this loader can handle the given file."""
        return file_path.suffix.lower() in self.supported_extensions

    async def load(self, file_path: Path) -> Dict[str, Any]:
        """Load document from file path."""
        raise NotImplementedError("Subclasses must implement load method")


class MarkdownLoader(DocumentLoader):
    """Loader for Markdown documents."""

    def __init__(self):
        """Initialize Markdown loader."""
        super().__init__()
        self.supported_extensions = [".md", ".markdown"]

    async def load(self, file_path: Path) -> Dict[str, Any]:
        """Load Markdown document."""
        try:
            content = file_path.read_text(encoding="utf-8")

            # Extract title from first heading or filename
            title = self._extract_title(content) or file_path.stem

            return {
                "content": content,
                "title": title,
                "file_path": str(file_path),
                "file_type": "markdown",
                "file_size": len(content),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load Markdown file {file_path}: {e}")

    def _extract_title(self, content: str) -> Optional[str]:
        """Extract title from first heading in markdown."""
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("#"):
                # Remove markdown heading markers
                title = line.lstrip("#").strip()
                if title:
                    return title
        return None


class PDFLoader(DocumentLoader):
    """Loader for PDF documents."""

    def __init__(self):
        """Initialize PDF loader."""
        super().__init__()
        self.supported_extensions = [".pdf"]

    async def load(self, file_path: Path) -> Dict[str, Any]:
        """Load PDF document."""
        try:
            import pypdf

            with open(file_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)

                content_parts = []
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        content_parts.append(f"Page {page_num + 1}:\n{page_text}")

                content = "\n\n".join(content_parts)

                # Extract title from metadata or filename
                title = self._extract_title(pdf_reader.metadata) or file_path.stem

                return {
                    "content": content,
                    "title": title,
                    "file_path": str(file_path),
                    "file_type": "pdf",
                    "file_size": len(content),
                    "page_count": len(pdf_reader.pages),
                }

        except ImportError:
            raise RuntimeError(
                "pypdf is required for PDF loading. Install with: pip install pypdf"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF file {file_path}: {e}")

    def _extract_title(self, metadata: Any) -> Optional[str]:
        """Extract title from PDF metadata."""
        if metadata and hasattr(metadata, "title"):
            title = metadata.title
            if title and title.strip():
                return str(title.strip())
        return None


class TextLoader(DocumentLoader):
    """Loader for plain text documents."""

    def __init__(self):
        """Initialize text loader."""
        super().__init__()
        self.supported_extensions = [".txt", ".text"]

    async def load(self, file_path: Path) -> Dict[str, Any]:
        """Load text document."""
        try:
            content = file_path.read_text(encoding="utf-8")

            return {
                "content": content,
                "title": file_path.stem,
                "file_path": str(file_path),
                "file_type": "text",
                "file_size": len(content),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load text file {file_path}: {e}")


class DocumentIngester:
    """Main document ingestion pipeline."""

    def __init__(self):
        """Initialize document ingester."""
        self.settings = get_settings()
        self.loaders = [MarkdownLoader(), PDFLoader(), TextLoader()]

        # Initialize embedding model
        self.embedding_model = None
        self._initialize_embedding_model()

    def _initialize_embedding_model(self) -> None:
        """Initialize the embedding model."""
        try:
            if self.settings.embeddings_backend == "local":
                self.embedding_model = SentenceTransformer(
                    self.settings.embedding_model
                )
                print(f"Local embedding model loaded: {self.settings.embedding_model}")
            else:
                # Will use OpenAI embeddings if configured
                print("Using OpenAI embeddings backend")
        except Exception as e:
            print(f"Warning: Failed to load embedding model: {e}")
            self.embedding_model = None

    def _get_loader(self, file_path: Path) -> Optional[DocumentLoader]:
        """Get appropriate loader for file."""
        for loader in self.loaders:
            if loader.can_load(file_path):
                return loader
        return None

    def _chunk_document(
        self, content: str, chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        """Split document content into chunks."""
        if chunk_size <= 0:
            return [content]

        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size

            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(content):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if content[i] in ".!?":
                        end = i + 1
                        break

                # If no sentence boundary found, look for paragraph breaks
                if end == start + chunk_size:
                    for i in range(end, max(start + chunk_size - 200, start), -1):
                        if content[i] == "\n" and (
                            i + 1 >= len(content) or content[i + 1] == "\n"
                        ):
                            end = i + 1
                            break

            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position for next chunk
            start = end - chunk_overlap
            if start >= len(content):
                break

        return chunks

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks."""
        if self.settings.embeddings_backend == "local" and self.embedding_model:
            # Use local sentence-transformers
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            result = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
            if isinstance(result, list) and all(isinstance(x, list) and all(isinstance(y, (int, float)) for y in x) for x in result):
                return result
            else:
                # Convert to proper format if needed
                return [[float(y) for y in x] if isinstance(x, list) else [float(x)] for x in result]
        elif self.settings.has_openai():
            # Use OpenAI embeddings
            from ..llm.openai import OpenAIProvider

            provider = OpenAIProvider()
            return await provider.generate_embeddings(texts)
        else:
            raise RuntimeError("No embedding backend available")

    async def build_index(
        self,
        source_dir: Path,
        output_dir: Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Build vector index from documents in source directory.

        Args:
            source_dir: Directory containing source documents
            output_dir: Directory to save the index
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            force: Force rebuild existing index

        Returns:
            Dictionary with build results and statistics
        """
        start_time = time.time()

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if index already exists
        index_file = output_dir / "index.faiss"
        metadata_file = output_dir / "metadata.pkl"

        if not force and index_file.exists() and metadata_file.exists():
            print(f"Index already exists at {output_dir}. Use --force to rebuild.")
            return await self._load_existing_index(output_dir)

        # Find all documents
        documents: List[Path] = []
        for doc_loader in self.loaders:
            for ext in doc_loader.supported_extensions:
                documents.extend(source_dir.glob(f"**/*{ext}"))

        if not documents:
            raise RuntimeError(f"No supported documents found in {source_dir}")

        print(f"Found {len(documents)} documents to process")

        # Process documents
        all_chunks = []
        document_stats = []

        for doc_path in documents:
            try:
                # Load document
                loader: Optional[DocumentLoader] = self._get_loader(doc_path)
                if not loader:
                    continue

                doc_data = await loader.load(doc_path)

                # Generate document ID
                doc_id = hashlib.md5(doc_data["file_path"].encode()).hexdigest()[:8]

                # Chunk document
                chunks = self._chunk_document(
                    doc_data["content"], chunk_size, chunk_overlap
                )

                # Create chunk objects
                for i, chunk_content in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i:03d}"

                    chunk = DocumentChunk(
                        content=chunk_content,
                        document_id=doc_id,
                        chunk_id=chunk_id,
                        title=doc_data["title"],
                        source_path=doc_data["file_path"],
                        chunk_index=i,
                        start_char=i * chunk_size,
                        end_char=min((i + 1) * chunk_size, len(doc_data["content"])),
                    )

                    all_chunks.append(chunk)

                document_stats.append(
                    {
                        "file_path": str(doc_path),
                        "title": doc_data["title"],
                        "file_type": doc_data["file_type"],
                        "chunks_created": len(chunks),
                        "file_size": doc_data["file_size"],
                    }
                )

                print(f"Processed {doc_path.name}: {len(chunks)} chunks")

            except Exception as e:
                print(f"Warning: Failed to process {doc_path}: {e}")
                continue

        if not all_chunks:
            raise RuntimeError("No chunks created from documents")

        print(f"Generated {len(all_chunks)} chunks total")

        # Generate embeddings
        print("Generating embeddings...")
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = await self._generate_embeddings(chunk_texts)

        # Assign embeddings to chunks
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk.embedding = embedding

        # Build FAISS index
        print("Building FAISS index...")
        embedding_dim = len(embeddings[0])

        # Use IndexFlatIP for inner product (cosine similarity when normalized)
        index = faiss.IndexFlatIP(embedding_dim)

        # Normalize embeddings for cosine similarity
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)

        # Add vectors to index
        index.add(embeddings_array)

        # Save index and metadata
        print("Saving index...")
        faiss.write_index(index, str(index_file))

        # Save metadata
        metadata = {
            "chunks": [chunk.to_dict() for chunk in all_chunks],
            "document_stats": document_stats,
            "index_info": {
                "total_chunks": len(all_chunks),
                "embedding_dimension": embedding_dim,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "build_timestamp": time.time(),
            },
        }

        with open(metadata_file, "wb") as f:
            pickle.dump(metadata, f)

        build_time = time.time() - start_time

        # Calculate index size
        index_size_mb = (index_file.stat().st_size + metadata_file.stat().st_size) / (
            1024 * 1024
        )

        print(f"Index built successfully in {build_time:.2f}s")
        print(f"Index size: {index_size_mb:.2f} MB")

        return {
            "documents_processed": len(document_stats),
            "chunks_created": len(all_chunks),
            "index_size_mb": index_size_mb,
            "build_time": build_time,
            "output_directory": str(output_dir),
            "embedding_dimension": embedding_dim,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

    async def _load_existing_index(self, index_dir: Path) -> Dict[str, Any]:
        """Load existing index metadata."""
        metadata_file = index_dir / "metadata.pkl"

        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)

        index_info = metadata["index_info"]

        return {
            "documents_processed": len(metadata["document_stats"]),
            "chunks_created": index_info["total_chunks"],
            "index_size_mb": 0,  # Would need to calculate
            "build_time": 0,
            "output_directory": str(index_dir),
            "embedding_dimension": index_info["embedding_dimension"],
            "chunk_size": index_info["chunk_size"],
            "chunk_overlap": index_info["chunk_overlap"],
            "status": "loaded_existing",
        }

    async def list_indexes(
        self, index_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """List available indexes."""
        if index_dir is None:
            index_dir = self.settings.index_dir

        indexes = []

        if index_dir.exists():
            for item in index_dir.iterdir():
                if item.is_dir():
                    # Check if it's a valid index
                    index_file = item / "index.faiss"
                    metadata_file = item / "metadata.pkl"

                    if index_file.exists() and metadata_file.exists():
                        try:
                            with open(metadata_file, "rb") as f:
                                metadata = pickle.load(f)

                            index_info = metadata["index_info"]

                            indexes.append(
                                {
                                    "name": item.name,
                                    "path": str(item),
                                    "total_chunks": index_info["total_chunks"],
                                    "embedding_dimension": index_info[
                                        "embedding_dimension"
                                    ],
                                    "chunk_size": index_info["chunk_size"],
                                    "chunk_overlap": index_info["chunk_overlap"],
                                    "build_timestamp": index_info["build_timestamp"],
                                }
                            )
                        except Exception as e:
                            print(f"Warning: Failed to read index {item}: {e}")

        return indexes
