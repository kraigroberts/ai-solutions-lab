"""Advanced search interface integrating hybrid search with enhanced RAG."""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import json
from datetime import datetime

from .hybrid_search import HybridSearchEngine, SearchQuery, SearchResult
from .enhanced_rag import EnhancedRAG, RAGResponse
from .vector_store import VectorStore
from .llm_config import LLMConfig

@dataclass
class AdvancedSearchRequest:
    """Request for advanced search with multiple options."""
    query: str
    search_type: str = "hybrid"  # "semantic", "keyword", "hybrid", "rag"
    filters: Dict[str, Any] = None
    boost_semantic: float = 1.0
    boost_keyword: float = 1.0
    boost_metadata: float = 1.0
    max_results: int = 10
    generate_answer: bool = False
    include_highlights: bool = True
    include_metadata: bool = True

@dataclass
class AdvancedSearchResponse:
    """Comprehensive search response with multiple result types."""
    query: str
    search_type: str
    results: List[SearchResult]
    rag_response: Optional[RAGResponse] = None
    search_analytics: Dict[str, Any] = None
    query_suggestions: List[str] = None
    processing_time: float = 0.0
    total_results: int = 0
    filters_applied: Dict[str, Any] = None

class AdvancedSearchInterface:
    """Advanced search interface combining multiple search strategies."""
    
    def __init__(self, vector_store: VectorStore, llm_config: LLMConfig):
        self.vector_store = vector_store
        self.llm_config = llm_config
        
        # Initialize search components
        self.hybrid_engine = HybridSearchEngine(vector_store)
        self.enhanced_rag = EnhancedRAG(vector_store, llm_config)
        
        # Search configuration
        self.default_filters = {
            'min_confidence': 0.1,
            'include_file_types': ['.md', '.pdf', '.txt']
        }
        
    def search(self, request: AdvancedSearchRequest) -> AdvancedSearchResponse:
        """Perform advanced search based on request parameters."""
        import time
        start_time = time.time()
        
        # Prepare filters
        filters = self._prepare_filters(request.filters)
        
        # Create search query
        search_query = SearchQuery(
            text=request.query,
            filters=filters,
            search_type=request.search_type,
            boost_semantic=request.boost_semantic,
            boost_keyword=request.boost_keyword,
            boost_metadata=request.boost_metadata
        )
        
        # Perform search based on type
        if request.search_type == "rag":
            results, rag_response = self._rag_search(request.query, filters, request.max_results)
        else:
            results = self._hybrid_search(search_query, request.max_results)
            rag_response = None
        
        # Generate RAG answer if requested
        if request.generate_answer and not rag_response:
            rag_response = self._generate_rag_answer(request.query, results)
        
        # Add highlights if requested
        if request.include_highlights:
            results = self._add_highlights(results, request.query)
        
        # Get search analytics
        search_analytics = self.hybrid_engine.get_search_analytics()
        
        # Get query suggestions
        query_suggestions = self.hybrid_engine.suggest_queries(request.query)
        
        processing_time = time.time() - start_time
        
        return AdvancedSearchResponse(
            query=request.query,
            search_type=request.search_type,
            results=results,
            rag_response=rag_response,
            search_analytics=search_analytics,
            query_suggestions=query_suggestions,
            processing_time=processing_time,
            total_results=len(results),
            filters_applied=filters
        )
    
    def _prepare_filters(self, user_filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare and merge filters with defaults."""
        filters = self.default_filters.copy()
        
        if user_filters:
            filters.update(user_filters)
        
        return filters
    
    def _hybrid_search(self, search_query: SearchQuery, max_results: int) -> List[SearchResult]:
        """Perform hybrid search."""
        try:
            return self.hybrid_engine.search(search_query, k=max_results)
        except Exception as e:
            print(f"Hybrid search error: {e}")
            return []
    
    def _rag_search(self, query: str, filters: Dict[str, Any], max_results: int) -> tuple[List[SearchResult], RAGResponse]:
        """Perform RAG-based search."""
        try:
            # Get RAG response
            rag_response = self.enhanced_rag.search_and_answer(query, k=max_results)
            
            # Convert to SearchResult format
            results = []
            for i, source in enumerate(rag_response.sources):
                result = SearchResult(
                    document=source['content_preview'],
                    source_file=source['file'],
                    chunk_id=source['chunk_id'],
                    metadata=source['metadata'],
                    semantic_score=rag_response.search_results[i]['score'] if i < len(rag_response.search_results) else 0.0,
                    keyword_score=0.0,
                    metadata_score=0.0,
                    combined_score=rag_response.search_results[i]['score'] if i < len(rag_response.search_results) else 0.0,
                    relevance_rank=i + 1,
                    search_highlights=[]
                )
                results.append(result)
            
            return results, rag_response
            
        except Exception as e:
            print(f"RAG search error: {e}")
            return [], None
    
    def _generate_rag_answer(self, query: str, results: List[SearchResult]) -> Optional[RAGResponse]:
        """Generate RAG answer from search results."""
        try:
            # Convert SearchResults back to the format expected by RAG
            search_results = []
            for result in results:
                search_result = {
                    'document': result.document,
                    'source_file': result.source_file,
                    'chunk_id': result.chunk_id,
                    'metadata': result.metadata,
                    'score': result.combined_score
                }
                search_results.append(search_result)
            
            # Create a mock response structure for RAG
            # In a real implementation, this would integrate more seamlessly
            return RAGResponse(
                answer=f"Based on the search results, here's what I found for: {query}",
                sources=[{
                    'file': result.source_file,
                    'chunk_id': result.chunk_id,
                    'relevance_score': result.combined_score,
                    'content_preview': result.document[:200] + '...',
                    'metadata': result.metadata
                } for result in results],
                search_results=search_results,
                llm_provider=self.enhanced_rag.llm_provider.get_model_info()["provider"],
                processing_time=0.0,
                confidence_score=results[0].combined_score if results else 0.0
            )
            
        except Exception as e:
            print(f"RAG answer generation error: {e}")
            return None
    
    def _add_highlights(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Add search highlights to results."""
        try:
            return self.hybrid_engine._add_search_highlights(results, query)
        except Exception as e:
            print(f"Highlight generation error: {e}")
            return results
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        analytics = self.hybrid_engine.get_search_analytics()
        
        # Add vector store statistics
        vector_stats = self.vector_store.get_document_stats()
        
        # Add RAG capabilities
        rag_capabilities = self.enhanced_rag.get_system_status()
        
        return {
            'search_analytics': analytics,
            'vector_store_stats': vector_stats,
            'rag_capabilities': rag_capabilities,
            'search_engine_info': {
                'hybrid_search': True,
                'keyword_search': True,
                'semantic_search': True,
                'metadata_filtering': True,
                'search_highlights': True,
                'query_suggestions': True,
                'search_analytics': True
            }
        }
    
    def export_search_results(self, response: AdvancedSearchResponse, 
                            format: str = "json") -> str:
        """Export search results in various formats."""
        if format.lower() == "json":
            return json.dumps(asdict(response), indent=2, default=str)
        elif format.lower() == "csv":
            return self._export_to_csv(response)
        else:
            return f"Unsupported format: {format}"
    
    def _export_to_csv(self, response: AdvancedSearchResponse) -> str:
        """Export results to CSV format."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Rank', 'Document', 'Source File', 'Semantic Score', 
            'Keyword Score', 'Combined Score', 'Metadata'
        ])
        
        # Write data
        for result in response.results:
            writer.writerow([
                result.relevance_rank,
                result.document[:100] + '...' if len(result.document) > 100 else result.document,
                result.source_file,
                f"{result.semantic_score:.3f}",
                f"{result.keyword_score:.3f}",
                f"{result.combined_score:.3f}",
                json.dumps(result.metadata)
            ])
        
        return output.getvalue()
    
    def batch_search(self, queries: List[str], 
                    search_type: str = "hybrid",
                    **kwargs) -> List[AdvancedSearchResponse]:
        """Perform batch search on multiple queries."""
        responses = []
        
        for query in queries:
            request = AdvancedSearchRequest(
                query=query,
                search_type=search_type,
                **kwargs
            )
            
            response = self.search(request)
            responses.append(response)
        
        return responses
    
    def search_with_feedback(self, query: str, 
                           user_feedback: Dict[str, Any],
                           **kwargs) -> AdvancedSearchResponse:
        """Search with user feedback to improve results."""
        # Adjust search parameters based on feedback
        if user_feedback.get('prefer_semantic'):
            kwargs['boost_semantic'] = 2.0
            kwargs['boost_keyword'] = 0.5
        
        if user_feedback.get('prefer_keyword'):
            kwargs['boost_keyword'] = 2.0
            kwargs['boost_semantic'] = 0.5
        
        if user_feedback.get('min_confidence'):
            kwargs['filters'] = kwargs.get('filters', {})
            kwargs['filters']['min_confidence'] = user_feedback['min_confidence']
        
        request = AdvancedSearchRequest(query=query, **kwargs)
        return self.search(request)

def main():
    """Demo the advanced search interface."""
    print("Advanced Search Interface Demo")
    print("=" * 40)
    
    print("Advanced Search Features:")
    print("- Multiple search strategies (hybrid, semantic, keyword, RAG)")
    print("- Advanced filtering and boosting")
    print("- Search highlights and analytics")
    print("- Query suggestions and feedback")
    print("- Export capabilities (JSON, CSV)")
    print("- Batch search operations")
    
    print("\nReady for integration with vector store and LLM!")

if __name__ == "__main__":
    main()
