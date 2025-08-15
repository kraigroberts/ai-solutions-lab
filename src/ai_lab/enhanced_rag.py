"""Enhanced RAG system with LLM integration."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

from .vector_store import VectorStore
from .llm_providers import create_llm_provider, LLMProvider
from .llm_config import LLMConfig

@dataclass
class RAGResponse:
    """Response from the enhanced RAG system."""
    answer: str
    sources: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]]
    llm_provider: str
    processing_time: float
    confidence_score: Optional[float] = None

class EnhancedRAG:
    """Enhanced RAG system with LLM integration."""
    
    def __init__(self, vector_store: VectorStore, llm_config: LLMConfig):
        self.vector_store = vector_store
        self.llm_config = llm_config
        self.llm_provider = create_llm_provider(llm_config)
        
    def search_and_answer(self, query: str, k: int = 5, 
                         generate_answer: bool = True,
                         **kwargs) -> RAGResponse:
        """Search for relevant documents and optionally generate an answer."""
        start_time = time.time()
        
        # Step 1: Search for relevant documents
        search_results = self.vector_store.search(query, k=k)
        
        if not search_results:
                    return RAGResponse(
            answer="I couldn't find any relevant information to answer your question.",
            sources=[],
            search_results=[],
            llm_provider=self.llm_provider.get_model_info()["provider"],
            processing_time=time.time() - start_time,
            confidence_score=0.0
        )
        
        # Step 2: Prepare context from search results
        context = self._prepare_context(search_results)
        sources = self._extract_sources(search_results)
        
        # Step 3: Generate answer if requested
        if generate_answer and self.llm_provider.is_available():
            try:
                answer = self.llm_provider.generate_response(
                    query, 
                    context=context,
                    **kwargs
                )
            except Exception as e:
                answer = f"I found some relevant information but couldn't generate a complete answer due to an error: {e}"
        else:
            answer = self._generate_summary_answer(query, search_results)
        
        # Step 4: Calculate confidence score
        confidence_score = self._calculate_confidence(search_results)
        
        processing_time = time.time() - start_time
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            search_results=search_results,
            llm_provider=self.llm_provider.get_model_info()["provider"],
            processing_time=processing_time,
            confidence_score=confidence_score
        )
    
    def _prepare_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Prepare context from search results for LLM."""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            source_file = result.get('source_file', 'Unknown')
            content = result.get('document', '')
            score = result.get('score', 0.0)
            
            context_parts.append(f"Source {i} ({source_file}, relevance: {score:.3f}):\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _extract_sources(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from search results."""
        sources = []
        
        for result in search_results:
            source = {
                'file': result.get('source_file', 'Unknown'),
                'chunk_id': result.get('chunk_id', 'Unknown'),
                'relevance_score': result.get('score', 0.0),
                'content_preview': result.get('document', '')[:200] + '...' if len(result.get('document', '')) > 200 else result.get('document', ''),
                'metadata': result.get('metadata', {})
            }
            sources.append(source)
        
        return sources
    
    def _generate_summary_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Generate a summary answer when LLM is not available."""
        if not search_results:
            return "No relevant information found."
        
        # Create a simple summary based on search results
        top_result = search_results[0]
        top_content = top_result.get('document', '')
        
        # Simple heuristic: if query terms appear in top result, use it
        query_terms = query.lower().split()
        content_lower = top_content.lower()
        
        matching_terms = [term for term in query_terms if term in content_lower]
        
        if matching_terms:
            # Find the most relevant sentence
            sentences = top_content.split('.')
            best_sentence = max(sentences, key=lambda s: sum(term in s.lower() for term in query_terms))
            
            return f"Based on the search results, here's what I found: {best_sentence.strip()}."
        else:
            return f"I found some relevant information: {top_content[:200]}..."
    
    def _calculate_confidence(self, search_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on search results."""
        if not search_results:
            return 0.0
        
        # Use the top result's score as confidence
        top_score = search_results[0].get('score', 0.0)
        
        # Normalize to 0-1 range (assuming FAISS scores are typically 0-1)
        confidence = min(max(top_score, 0.0), 1.0)
        
        return confidence
    
    def conversational_search(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None,
                            k: int = 5, **kwargs) -> RAGResponse:
        """Enhanced search with conversation context."""
        # Enhance query with conversation history
        enhanced_query = self._enhance_query_with_history(query, conversation_history)
        
        # Perform search and answer generation
        return self.search_and_answer(enhanced_query, k=k, **kwargs)
    
    def _enhance_query_with_history(self, query: str, 
                                   conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Enhance query with conversation history context."""
        if not conversation_history:
            return query
        
        # Take last few exchanges for context
        recent_history = conversation_history[-4:]  # Last 2 exchanges
        
        context_parts = []
        for exchange in recent_history:
            if 'user' in exchange and 'assistant' in exchange:
                context_parts.append(f"User: {exchange['user']}")
                context_parts.append(f"Assistant: {exchange['assistant']}")
        
        if context_parts:
            enhanced_query = f"Context from recent conversation:\n" + "\n".join(context_parts) + f"\n\nCurrent question: {query}"
            return enhanced_query
        
        return query
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and capabilities."""
        return {
            'vector_store': {
                'total_documents': len(self.vector_store.documents),
                'index_size': self.vector_store.index.ntotal if self.vector_store.index else 0,
                'embedding_model': self.vector_store.model_name
            },
            'llm_provider': self.llm_provider.get_model_info(),
            'capabilities': {
                'search': True,
                'answer_generation': self.llm_provider.is_available(),
                'conversational_search': True,
                'metadata_filtering': True
            }
        }
    
    def batch_process_queries(self, queries: List[str], k: int = 5, 
                             generate_answers: bool = True, **kwargs) -> List[RAGResponse]:
        """Process multiple queries in batch."""
        responses = []
        
        for query in queries:
            try:
                response = self.search_and_answer(
                    query, 
                    k=k, 
                    generate_answer=generate_answers,
                    **kwargs
                )
                responses.append(response)
            except Exception as e:
                # Create error response
                error_response = RAGResponse(
                    answer=f"Error processing query: {e}",
                    sources=[],
                    search_results=[],
                    llm_provider=self.llm_provider.get_model_info()["provider"],
                    processing_time=0.0,
                    confidence_score=0.0
                )
                responses.append(error_response)
        
        return responses

def main():
    """Demo the enhanced RAG system."""
    print("Enhanced RAG System Demo")
    print("=" * 40)
    
    # This would require a populated vector store
    # For demo purposes, we'll show the configuration
    
    from .llm_config import create_default_config
    
    config = create_default_config()
    print(f"LLM Config: {config.get_provider_info()}")
    
    # Create mock vector store for demo
    class MockVectorStore:
        def __init__(self):
            self.documents = []
            self.index = None
            self.model_name = "demo-model"
        
        def search(self, query, k=5):
            return [
                {
                    'document': 'Machine learning is a subset of artificial intelligence.',
                    'score': 0.8,
                    'source_file': 'demo.md',
                    'chunk_id': 'demo_chunk_0',
                    'metadata': {'file_type': '.md'}
                }
            ]
    
    mock_store = MockVectorStore()
    
    # Create enhanced RAG
    rag = EnhancedRAG(mock_store, config)
    
    # Test search and answer
    query = "What is machine learning?"
    response = rag.search_and_answer(query, generate_answer=True)
    
    print(f"\nQuery: {query}")
    print(f"Answer: {response.answer}")
    print(f"Sources: {len(response.sources)}")
    print(f"Processing time: {response.processing_time:.3f}s")
    print(f"LLM Provider: {response.llm_provider}")
    
    # Show system status
    status = rag.get_system_status()
    print(f"\nSystem Status: {status}")

if __name__ == "__main__":
    main()
