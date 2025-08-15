"""Enhanced vector store with FAISS and persistence."""

import os
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install: pip install faiss-cpu sentence-transformers")
    raise

from src.ai_lab.document_ingestion import DocumentChunk

class VectorStore:
    """Enhanced vector store with FAISS and persistence."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', index_dir: str = "./data/index"):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the embedding model
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # FAISS index
        self.index = None
        self.documents: List[DocumentChunk] = []
        self.document_map: Dict[str, DocumentChunk] = {}
        
        # Index metadata
        self.metadata_file = self.index_dir / "metadata.json"
        self.index_file = self.index_dir / "faiss_index.bin"
        
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector store."""
        if not chunks:
            return
        
        # Extract text from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Create embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Initialize or extend FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store document chunks and metadata
        start_idx = len(self.documents)
        for i, chunk in enumerate(chunks):
            # Use the original chunk_id from the DocumentChunk
            self.documents.append(chunk)
            self.document_map[chunk.chunk_id] = chunk
        
        print(f"Added {len(chunks)} documents to vector store. Total: {len(self.documents)}")
    
    def search(self, query: str, k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents) and score >= threshold:
                chunk = self.documents[idx]
                results.append({
                    'document': chunk.text,
                    'score': float(score),
                    'source_file': chunk.source_file,
                    'chunk_id': chunk.chunk_id,
                    'metadata': chunk.metadata
                })
        
        return results
    
    def search_by_metadata(self, query: str, metadata_filter: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Search with metadata filtering."""
        # First get all results with no threshold
        all_results = self.search(query, k=len(self.documents), threshold=-1.0)
        
        # Filter by metadata
        filtered_results = []
        for result in all_results:
            chunk = self.document_map.get(result['chunk_id'])
            if chunk and self._matches_metadata(chunk.metadata, metadata_filter):
                filtered_results.append(result)
                if len(filtered_results) >= k:
                    break
        
        return filtered_results
    
    def _matches_metadata(self, chunk_metadata: Dict[str, Any], filter_metadata: Dict[str, Any]) -> bool:
        """Check if chunk metadata matches filter criteria."""
        for key, value in filter_metadata.items():
            if key not in chunk_metadata:
                return False
            if chunk_metadata[key] != value:
                return False
        return True
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        stats = {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.dimension,
            'model_name': self.model_name
        }
        
        # File type distribution
        file_types = {}
        for doc in self.documents:
            file_type = doc.metadata.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        stats['file_types'] = file_types
        
        return stats
    
    def save_index(self) -> None:
        """Save the FAISS index and metadata."""
        if self.index is None:
            print("No index to save")
            return
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_file))
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'total_documents': len(self.documents),
            'documents': [
                {
                    'text': doc.text,
                    'source_file': doc.source_file,
                    'chunk_id': doc.chunk_id,
                    'start_char': doc.start_char,
                    'end_char': doc.end_char,
                    'metadata': doc.metadata
                }
                for doc in self.documents
            ]
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Index saved to {self.index_dir}")
    
    def load_index(self) -> bool:
        """Load the FAISS index and metadata."""
        if not self.index_file.exists() or not self.metadata_file.exists():
            print("No saved index found")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_file))
            
            # Load metadata
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Restore documents
            self.documents = []
            self.document_map = {}
            
            for doc_data in metadata['documents']:
                chunk = DocumentChunk(
                    text=doc_data['text'],
                    source_file=doc_data['source_file'],
                    chunk_id=doc_data['chunk_id'],
                    start_char=doc_data['start_char'],
                    end_char=doc_data['end_char'],
                    metadata=doc_data['metadata']
                )
                self.documents.append(chunk)
                self.document_map[chunk.chunk_id] = chunk
            
            print(f"Index loaded: {len(self.documents)} documents")
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def clear_index(self) -> None:
        """Clear the current index."""
        self.index = None
        self.documents = []
        self.document_map = {}
        print("Index cleared")
    
    def delete_index_files(self) -> None:
        """Delete saved index files."""
        if self.index_file.exists():
            self.index_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        print("Index files deleted")

def main():
    """Demo the enhanced vector store."""
    from src.ai_lab.document_ingestion import DocumentIngester
    
    # Initialize components
    ingester = DocumentIngester(chunk_size=500, chunk_overlap=100)
    vector_store = VectorStore()
    
    # Process sample document
    sample_file = Path("data/docs/sample.md")
    if sample_file.exists():
        chunks = ingester.process_file(sample_file)
        
        if chunks:
            # Add to vector store
            vector_store.add_documents(chunks)
            
            # Search
            query = "What is machine learning?"
            results = vector_store.search(query, k=3)
            
            print(f"\nSearch results for: {query}")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result['score']:.3f}")
                print(f"   Source: {result['source_file']}")
                print(f"   Text: {result['document'][:100]}...")
            
            # Save index
            vector_store.save_index()
            
            # Show stats
            stats = vector_store.get_document_stats()
            print(f"\nVector store stats: {stats}")
        else:
            print("No chunks to process")
    else:
        print("Sample document not found. Create data/docs/sample.md first.")

if __name__ == "__main__":
    main()
