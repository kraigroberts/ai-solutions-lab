#!/usr/bin/env python3
"""Test search functionality with initialized data."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Test search functionality."""
    print("🔍 Testing search functionality...")
    
    try:
        from ai_lab.vector_store import VectorStore
        from ai_lab.advanced_search import AdvancedSearchInterface
        from ai_lab.llm_config import create_default_config
        
        print("✅ Components imported successfully")
        
        # Create components
        vector_store = VectorStore()
        llm_config = create_default_config()
        search_interface = AdvancedSearchInterface(vector_store, llm_config)
        
        print("✅ Search interface created")
        
        # Test search
        print("\n🔍 Testing search for 'machine learning'...")
        
        # Create a simple search request
        class SimpleSearchRequest:
            def __init__(self, query, search_type="hybrid", max_results=5):
                self.query = query
                self.search_type = search_type
                self.max_results = max_results
                self.filters = {}
                self.boost_semantic = 1.0
                self.boost_keyword = 1.0
                self.boost_metadata = 1.0
                self.generate_answer = False
                self.include_highlights = True
        
        search_request = SimpleSearchRequest("machine learning")
        
        # Perform search
        response = search_interface.search(search_request)
        
        print(f"✅ Search completed!")
        print(f"📊 Total results: {response.total_results}")
        print(f"⏱️  Processing time: {response.processing_time:.4f}s")
        
        if response.results:
            print(f"\n📄 Top results:")
            for i, result in enumerate(response.results[:3], 1):
                print(f"{i}. Score: {result.combined_score:.3f}")
                print(f"   Document: {result.document[:100]}...")
                print(f"   Source: {result.source_file}")
                print()
        else:
            print("❌ No results found")
            
        # Check vector store stats
        print(f"📚 Vector store has {len(vector_store.documents)} documents")
        
    except Exception as e:
        print(f"❌ Error testing search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
