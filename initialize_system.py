#!/usr/bin/env python3
"""Initialize AI Solutions Lab system with documents and indexing."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Initialize the system."""
    print("🚀 Initializing AI Solutions Lab...")
    
    try:
        # Import components
        from ai_lab.vector_store import VectorStore
        from ai_lab.document_ingestion import DocumentIngester
        from ai_lab.enhanced_rag import EnhancedRAG
        from ai_lab.advanced_search import AdvancedSearchInterface
        
        print("✅ Components imported successfully")
        
        # Initialize vector store
        print("📚 Initializing vector store...")
        vector_store = VectorStore()
        
        # Process documents
        print("📄 Processing documents...")
        ingestion = DocumentIngester()
        
        # Get documents from data/docs
        docs_dir = Path("data/docs")
        if docs_dir.exists():
            for doc_file in docs_dir.glob("*.md"):
                print(f"Processing {doc_file.name}...")
                chunks = ingestion.process_file(doc_file)
                print(f"  - Created {len(chunks)} chunks")
                
                # Add to vector store
                vector_store.add_documents(chunks)
        
        print("✅ System initialized successfully!")
        print("\n🎯 Now you can:")
        print("1. Open http://localhost:8000 in your browser")
        print("2. Search for 'machine learning', 'neural networks', etc.")
        print("3. Use RAG to ask questions like 'What is AI?'")
        print("4. Explore analytics and system status")
        
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
