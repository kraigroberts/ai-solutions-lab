"""Simple RAG implementation."""

import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("Please install: pip install sentence-transformers faiss-cpu")
    raise

class SimpleRAG:
    """Simple RAG system that actually works."""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = []
        
    def add_documents(self, texts: List[str]):
        """Add documents to the index."""
        if not texts:
            return
            
        # Create embeddings
        embeddings = self.model.encode(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        
        self.documents = texts
        print(f"Added {len(texts)} documents to index")
        
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if self.index is None:
            return []
            
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'score': float(score)
                })
                
        return results

def main():
    """Simple demo."""
    rag = SimpleRAG()
    
    # Add some sample documents
    docs = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "FastAPI is a modern web framework for building APIs.",
        "Vector databases store and search high-dimensional data."
    ]
    
    rag.add_documents(docs)
    
    # Search
    query = "What is machine learning?"
    results = rag.search(query)
    
    print(f"\nQuery: {query}")
    print("Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.3f}")
        print(f"   {result['document']}")
        print()

if __name__ == "__main__":
    main()
