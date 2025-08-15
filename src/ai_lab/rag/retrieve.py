"""
Vector Retrieval Module for AI Solutions Lab.

Provides functionality for:
- Loading FAISS indexes
- Performing similarity search
- Retrieving relevant document chunks
- Score-based ranking and filtering
"""

import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import get_settings
from .ingest import DocumentChunk


class VectorRetriever:
    """Vector retriever for finding relevant document chunks."""

    def __init__(self, index_path: Optional[Path] = None):
        """Initialize vector retriever."""
        self.settings = get_settings()
        self.index_path = Path(index_path) if index_path else self.settings.index_dir

        # Index components
        self.faiss_index = None
        self.metadata = None
        self.chunks: List[DocumentChunk] = []

        # Embedding model for queries
        self.embedding_model = None
        self._initialize_embedding_model()

        # Load index if available
        self._load_index()

    def _initialize_embedding_model(self) -> None:
        """Initialize the embedding model for queries."""
        try:
            if self.settings.embeddings_backend == "local":
                self.embedding_model = SentenceTransformer(
                    self.settings.embedding_model
                )
                print(
                    f"Local embedding model loaded for retrieval: {self.settings.embedding_model}"
                )
            else:
                print("Using OpenAI embeddings backend for retrieval")
        except Exception as e:
            print(f"Warning: Failed to load embedding model for retrieval: {e}")
            self.embedding_model = None

    def _load_index(self) -> None:
        """Load FAISS index and metadata."""
        try:
            index_file = self.index_path / "index.faiss"
            metadata_file = self.index_path / "metadata.pkl"

            if not index_file.exists() or not metadata_file.exists():
                print(f"Index not found at {self.index_path}")
                return

            # Load FAISS index
            self.faiss_index = faiss.read_index(str(index_file))
            if self.faiss_index is not None:
                print(f"FAISS index loaded: {self.faiss_index.ntotal} vectors")

            # Load metadata
            with open(metadata_file, "rb") as f:
                self.metadata = pickle.load(f)

            # Reconstruct chunk objects
            self.chunks = []
            for chunk_data in self.metadata["chunks"]:
                chunk = DocumentChunk.from_dict(chunk_data)
                self.chunks.append(chunk)

            print(f"Loaded {len(self.chunks)} document chunks")

        except Exception as e:
            print(f"Error loading index: {e}")
            self.faiss_index = None
            self.metadata = None
            self.chunks = []

    def is_loaded(self) -> bool:
        """Check if index is loaded."""
        return (
            self.faiss_index is not None
            and self.metadata is not None
            and len(self.chunks) > 0
        )

    async def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query text."""
        if self.settings.embeddings_backend == "local" and self.embedding_model:
            # Use local sentence-transformers
            embedding = self.embedding_model.encode([query], convert_to_tensor=False)
            return embedding[0].tolist()
        elif self.settings.has_openai():
            # Use OpenAI embeddings
            from ..llm.openai import OpenAIProvider

            provider = OpenAIProvider()
            embeddings = await provider.generate_embeddings([query])
            return embeddings[0]
        else:
            raise RuntimeError("No embedding backend available for queries")

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: Search query text
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            include_metadata: Whether to include chunk metadata

        Returns:
            List of relevant chunks with scores and metadata
        """
        if not self.is_loaded():
            raise RuntimeError("Index not loaded. Cannot perform retrieval.")

        start_time = time.time()

        try:
            # Generate query embedding
            query_embedding = await self._generate_query_embedding(query)

            # Normalize query embedding for cosine similarity
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)

            # Perform similarity search
            if self.faiss_index is None:
                raise RuntimeError("FAISS index is not loaded")
            scores, indices = self.faiss_index.search(
                query_vector, min(top_k, len(self.chunks))
            )

            # Process results
            results = []
            for i, (score, chunk_idx) in enumerate(zip(scores[0], indices[0])):
                if chunk_idx == -1:  # FAISS returns -1 for invalid indices
                    continue

                # Convert score to similarity (0-1 range)
                similarity = float(score)

                # Apply score threshold
                if similarity < score_threshold:
                    continue

                chunk = self.chunks[chunk_idx]

                result = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "content": chunk.content,
                    "source_path": chunk.source_path,
                    "chunk_index": chunk.chunk_index,
                    "score": similarity,
                    "rank": i + 1,
                }

                if include_metadata:
                    result.update(
                        {
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                            "metadata": chunk.metadata,
                        }
                    )

                results.append(result)

            retrieval_time = time.time() - start_time

            # Add timing information
            for result in results:
                result["retrieval_time"] = retrieval_time

            return results

        except Exception as e:
            raise RuntimeError(f"Retrieval error: {str(e)}")

    async def retrieve_by_document(
        self, document_id: str, query: str = "", top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks from a specific document.

        Args:
            document_id: ID of the document to search within
            query: Optional query to filter chunks
            top_k: Number of top results to return

        Returns:
            List of chunks from the specified document
        """
        if not self.is_loaded():
            raise RuntimeError("Index not loaded. Cannot perform retrieval.")

        # Filter chunks by document ID
        document_chunks = [
            chunk for chunk in self.chunks if chunk.document_id == document_id
        ]

        if not document_chunks:
            return []

        if not query:
            # Return chunks without similarity scoring
            results = []
            for i, chunk in enumerate(document_chunks[:top_k]):
                results.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "content": chunk.content,
                        "source_path": chunk.source_path,
                        "chunk_index": chunk.chunk_index,
                        "rank": i + 1,
                    }
                )
            return results

        # If query provided, perform similarity search within document
        try:
            # Generate query embedding
            query_embedding = await self._generate_query_embedding(query)

            # Get embeddings for document chunks
            chunk_embeddings = []
            valid_chunks = []

            for chunk in document_chunks:
                if chunk.embedding:
                    chunk_embeddings.append(chunk.embedding)
                    valid_chunks.append(chunk)

            if not chunk_embeddings:
                return []

            # Convert to numpy array and normalize
            embeddings_array = np.array(chunk_embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)

            # Create temporary index for this document
            temp_index = faiss.IndexFlatIP(embeddings_array.shape[1])
            temp_index.add(embeddings_array)

            # Search
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)

            scores, indices = temp_index.search(
                query_vector, min(top_k, len(valid_chunks))
            )

            # Process results
            results = []
            for i, (score, chunk_idx) in enumerate(zip(scores[0], indices[0])):
                if chunk_idx == -1:
                    continue

                chunk = valid_chunks[chunk_idx]
                similarity = float(score)

                results.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "content": chunk.content,
                        "source_path": chunk.source_path,
                        "chunk_index": chunk.chunk_index,
                        "score": similarity,
                        "rank": i + 1,
                    }
                )

            return results

        except Exception as e:
            raise RuntimeError(f"Document retrieval error: {str(e)}")

    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a specific chunk by its ID."""
        if not self.is_loaded():
            return None

        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks from a specific document."""
        if not self.is_loaded():
            return []

        return [chunk for chunk in self.chunks if chunk.document_id == document_id]

    def get_index_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the loaded index."""
        if not self.is_loaded():
            return None

        if self.faiss_index is None:
            return None
        return {
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.faiss_index.d,
            "index_type": "faiss",
            "index_path": str(self.index_path),
            "build_info": self.metadata.get("index_info", {}),
            "document_stats": self.metadata.get("document_stats", []),
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the index."""
        if not self.is_loaded():
            return []

        documents = {}

        for chunk in self.chunks:
            doc_id = chunk.document_id
            if doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "title": chunk.title,
                    "source_path": chunk.source_path,
                    "chunk_count": 0,
                    "total_content_length": 0,
                }

            documents[doc_id]["chunk_count"] = int(documents[doc_id]["chunk_count"]) + 1
            documents[doc_id]["total_content_length"] = int(
                documents[doc_id]["total_content_length"]
            ) + len(chunk.content)

        return list(documents.values())

    def reload_index(self) -> bool:
        """Reload the index from disk."""
        try:
            self._load_index()
            return True
        except Exception as e:
            print(f"Failed to reload index: {e}")
            return False

    async def batch_retrieve(
        self, queries: List[str], top_k: int = 5, score_threshold: float = 0.5
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform batch retrieval for multiple queries.

        Args:
            queries: List of query strings
            top_k: Number of top results per query
            score_threshold: Minimum similarity score threshold

        Returns:
            List of results for each query
        """
        if not self.is_loaded():
            raise RuntimeError("Index not loaded. Cannot perform retrieval.")

        results = []

        for query in queries:
            try:
                query_results = await self.retrieve(
                    query, top_k=top_k, score_threshold=score_threshold
                )
                results.append(query_results)
            except Exception as e:
                print(f"Warning: Failed to retrieve for query '{query}': {e}")
                results.append([])

        return results
