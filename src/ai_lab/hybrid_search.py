"""Hybrid search engine combining semantic and keyword search."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import math

from .vector_store import VectorStore
from .document_ingestion import DocumentChunk

@dataclass
class SearchQuery:
    """Represents a search query with various components."""
    text: str
    filters: Dict[str, Any]
    search_type: str = "hybrid"  # "semantic", "keyword", "hybrid"
    boost_semantic: float = 1.0
    boost_keyword: float = 1.0
    boost_metadata: float = 1.0

@dataclass
class SearchResult:
    """Enhanced search result with multiple scoring factors."""
    document: str
    source_file: str
    chunk_id: str
    metadata: Dict[str, Any]
    semantic_score: float
    keyword_score: float
    metadata_score: float
    combined_score: float
    relevance_rank: int
    search_highlights: List[str]

class HybridSearchEngine:
    """Advanced search engine with multiple search strategies."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.search_history = []
        
    def search(self, query: SearchQuery, k: int = 10) -> List[SearchResult]:
        """Perform hybrid search combining multiple strategies."""
        results = []
        
        # Step 1: Semantic search using vector store
        semantic_results = self._semantic_search(query.text, k * 2)
        
        # Step 2: Keyword search
        keyword_results = self._keyword_search(query.text, k * 2)
        
        # Step 3: Metadata filtering
        filtered_results = self._apply_metadata_filters(
            semantic_results + keyword_results, 
            query.filters
        )
        
        # Step 4: Score combination and ranking
        combined_results = self._combine_scores(
            filtered_results, 
            query.boost_semantic,
            query.boost_keyword,
            query.boost_metadata
        )
        
        # Step 5: Final ranking and result preparation
        ranked_results = self._rank_results(combined_results, k)
        
        # Step 6: Add search highlights
        final_results = self._add_search_highlights(ranked_results, query.text)
        
        # Store search history
        self.search_history.append({
            'query': query.text,
            'filters': query.filters,
            'results_count': len(final_results),
            'timestamp': self._get_timestamp()
        })
        
        return final_results
    
    def _semantic_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Perform semantic search using vector store."""
        try:
            return self.vector_store.search(query, k=k)
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []
    
    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Perform keyword-based search."""
        if not query.strip():
            return []
        
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        if not keywords:
            return []
        
        results = []
        documents = self.vector_store.documents
        
        for doc in documents:
            keyword_score = self._calculate_keyword_score(doc.text, keywords)
            if keyword_score > 0:
                result = {
                    'document': doc.text,
                    'source_file': doc.source_file,
                    'chunk_id': doc.chunk_id,
                    'metadata': doc.metadata,
                    'score': keyword_score
                }
                results.append(result)
        
        # Sort by keyword score and limit results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'over', 'under', 'above', 'below', 'between', 'among'
        }
        
        # Clean and tokenize
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _calculate_keyword_score(self, document_text: str, keywords: List[str]) -> float:
        """Calculate keyword relevance score for a document."""
        if not keywords:
            return 0.0
        
        doc_lower = document_text.lower()
        total_score = 0.0
        
        for keyword in keywords:
            # Count occurrences
            count = doc_lower.count(keyword)
            
            # Calculate TF-IDF inspired score
            if count > 0:
                # Term frequency (normalized by document length)
                tf = count / len(document_text.split())
                
                # Inverse document frequency (simplified)
                # In a real system, this would be calculated across all documents
                idf = math.log(1000 / (count + 1))  # Simplified IDF
                
                score = tf * idf
                total_score += score
        
        return total_score / len(keywords) if keywords else 0.0
    
    def _apply_metadata_filters(self, results: List[Dict[str, Any]], 
                                filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply metadata filters to search results."""
        if not filters:
            return results
        
        filtered_results = []
        
        for result in results:
            if self._matches_filters(result, filters):
                filtered_results.append(result)
        
        return filtered_results
    
    def _matches_filters(self, result: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if a result matches the given filters."""
        metadata = result.get('metadata', {})
        
        for key, value in filters.items():
            if key == 'file_type':
                if metadata.get('file_type') != value:
                    return False
            elif key == 'category':
                if metadata.get('category') != value:
                    return False
            elif key == 'date_range':
                # Date range filtering (if date metadata exists)
                doc_date = metadata.get('date')
                if doc_date and not self._date_in_range(doc_date, value):
                    return False
            elif key == 'min_confidence':
                if result.get('score', 0) < value:
                    return False
            elif key == 'source_file':
                if not value.lower() in result.get('source_file', '').lower():
                    return False
        
        return True
    
    def _date_in_range(self, doc_date: str, date_range: Tuple[str, str]) -> bool:
        """Check if document date is within specified range."""
        try:
            # Simplified date comparison - in production would use proper date parsing
            return True  # Placeholder
        except:
            return True
    
    def _combine_scores(self, results: List[Dict[str, Any]], 
                        boost_semantic: float, boost_keyword: float, 
                        boost_metadata: float) -> List[Dict[str, Any]]:
        """Combine scores from different search strategies."""
        combined_results = []
        
        for result in results:
            # Extract scores
            semantic_score = result.get('score', 0.0)
            keyword_score = result.get('keyword_score', 0.0)
            metadata_score = result.get('metadata_score', 0.0)
            
            # Apply boosts
            boosted_semantic = semantic_score * boost_semantic
            boosted_keyword = keyword_score * boost_keyword
            boosted_metadata = metadata_score * boost_metadata
            
            # Combine scores (weighted average)
            combined_score = (
                boosted_semantic + boosted_keyword + boosted_metadata
            ) / 3.0
            
            # Create enhanced result
            enhanced_result = {
                **result,
                'semantic_score': semantic_score,
                'keyword_score': keyword_score,
                'metadata_score': metadata_score,
                'combined_score': combined_score
            }
            
            combined_results.append(enhanced_result)
        
        return combined_results
    
    def _rank_results(self, results: List[Dict[str, Any]], k: int) -> List[SearchResult]:
        """Rank results by combined score and create SearchResult objects."""
        # Sort by combined score
        sorted_results = sorted(results, key=lambda x: x['combined_score'], reverse=True)
        
        # Take top k results
        top_results = sorted_results[:k]
        
        # Convert to SearchResult objects
        search_results = []
        for i, result in enumerate(top_results):
            search_result = SearchResult(
                document=result['document'],
                source_file=result['source_file'],
                chunk_id=result['chunk_id'],
                metadata=result['metadata'],
                semantic_score=result.get('semantic_score', 0.0),
                keyword_score=result.get('keyword_score', 0.0),
                metadata_score=result.get('metadata_score', 0.0),
                combined_score=result['combined_score'],
                relevance_rank=i + 1,
                search_highlights=[]
            )
            search_results.append(search_result)
        
        return search_results
    
    def _add_search_highlights(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Add search highlights to results."""
        keywords = self._extract_keywords(query)
        
        for result in results:
            highlights = self._find_highlights(result.document, keywords)
            result.search_highlights = highlights
        
        return results
    
    def _find_highlights(self, text: str, keywords: List[str]) -> List[str]:
        """Find text snippets that highlight keywords."""
        highlights = []
        
        for keyword in keywords:
            # Find keyword occurrences
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            matches = pattern.finditer(text)
            
            for match in matches:
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                
                snippet = text[start:end]
                if snippet not in highlights:
                    highlights.append(snippet)
        
        return highlights[:5]  # Limit to 5 highlights
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get search analytics and insights."""
        if not self.search_history:
            return {"message": "No search history available"}
        
        # Analyze search patterns
        total_searches = len(self.search_history)
        avg_results = sum(h['results_count'] for h in self.search_history) / total_searches
        
        # Most common filters
        filter_counts = defaultdict(int)
        for history in self.search_history:
            for filter_key in history['filters']:
                filter_counts[filter_key] += 1
        
        # Recent searches
        recent_searches = self.search_history[-10:] if len(self.search_history) >= 10 else self.search_history
        
        return {
            'total_searches': total_searches,
            'average_results_per_search': round(avg_results, 2),
            'most_common_filters': dict(filter_counts),
            'recent_searches': recent_searches,
            'search_performance': {
                'total_queries': total_searches,
                'successful_searches': len([h for h in self.search_history if h['results_count'] > 0])
            }
        }
    
    def suggest_queries(self, partial_query: str, limit: int = 5) -> List[str]:
        """Suggest queries based on partial input."""
        if not partial_query.strip():
            return []
        
        suggestions = []
        
        # Look for similar queries in search history
        for history in self.search_history:
            query = history['query']
            if partial_query.lower() in query.lower() and query not in suggestions:
                suggestions.append(query)
                if len(suggestions) >= limit:
                    break
        
        # Add common ML/AI terms if we have room
        common_terms = [
            "machine learning", "artificial intelligence", "deep learning",
            "neural networks", "data science", "algorithm", "model training"
        ]
        
        for term in common_terms:
            if partial_query.lower() in term.lower() and term not in suggestions:
                suggestions.append(term)
                if len(suggestions) >= limit:
                    break
        
        return suggestions[:limit]

def main():
    """Demo the hybrid search engine."""
    print("Hybrid Search Engine Demo")
    print("=" * 40)
    
    # This would require a populated vector store
    # For demo purposes, we'll show the interface
    
    print("Hybrid Search Engine Features:")
    print("- Semantic search using embeddings")
    print("- Keyword-based search with TF-IDF scoring")
    print("- Metadata filtering (file type, category, etc.)")
    print("- Score combination and ranking")
    print("- Search highlights and analytics")
    print("- Query suggestions")
    
    print("\nReady for integration with vector store!")

if __name__ == "__main__":
    main()
