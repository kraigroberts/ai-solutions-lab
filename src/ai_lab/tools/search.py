"""
Search Tool for AI Solutions Lab.

Provides functionality for:
- Searching the document index
- Retrieving relevant chunks
- Formatting search results for agents
- Supporting different search strategies
"""

import asyncio
from typing import Any, Dict, List, Optional

from ..rag.retrieve import VectorRetriever


class SearchTool:
    """Search tool for querying the document index."""
    
    def __init__(self):
        """Initialize search tool."""
        self.retriever = VectorRetriever()
        self.default_top_k = 5
        self.default_score_threshold = 0.5
    
    async def search(self, input_data: str) -> str:
        """
        Search the document index for relevant information.
        
        Args:
            input_data: Search query string
            
        Returns:
            Formatted search results as string
        """
        try:
            # Check if index is loaded
            if not self.retriever.is_loaded():
                return "Search tool is not available. No document index has been loaded."
            
            # Perform search
            results = await self.retriever.retrieve(
                query=input_data,
                top_k=self.default_top_k,
                score_threshold=self.default_score_threshold,
                include_metadata=True
            )
            
            if not results:
                return f"No relevant information found for query: '{input_data}'"
            
            # Format results
            formatted_results = self._format_search_results(results, input_data)
            
            return formatted_results
            
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def _format_search_results(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> str:
        """Format search results into a readable string."""
        if not results:
            return "No results found."
        
        # Header
        output = [f"Search results for: '{query}'"]
        output.append(f"Found {len(results)} relevant document chunks:")
        output.append("")
        
        # Results
        for i, result in enumerate(results, 1):
            output.append(f"{i}. {result['title']}")
            output.append(f"   Source: {result['source_path']}")
            output.append(f"   Relevance Score: {result['score']:.3f}")
            output.append(f"   Content Preview: {result['content'][:150]}...")
            output.append("")
        
        # Summary
        output.append(f"Total chunks retrieved: {len(results)}")
        output.append(f"Query: '{query}'")
        
        return "\n".join(output)
    
    async def search_with_options(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5,
        include_content: bool = True
    ) -> Dict[str, Any]:
        """
        Search with additional options for more control.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            include_content: Whether to include full content
            
        Returns:
            Dictionary with search results and metadata
        """
        try:
            if not self.retriever.is_loaded():
                return {
                    "success": False,
                    "error": "Search tool is not available. No document index has been loaded."
                }
            
            # Perform search
            results = await self.retriever.retrieve(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                include_metadata=True
            )
            
            # Process results
            processed_results = []
            for result in results:
                processed_result = {
                    "title": result["title"],
                    "source_path": result["source_path"],
                    "score": result["score"],
                    "rank": result["rank"]
                }
                
                if include_content:
                    processed_result["content"] = result["content"]
                else:
                    # Include just a preview
                    processed_result["content_preview"] = result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
                
                processed_results.append(processed_result)
            
            return {
                "success": True,
                "query": query,
                "results": processed_results,
                "total_results": len(processed_results),
                "search_parameters": {
                    "top_k": top_k,
                    "score_threshold": score_threshold,
                    "include_content": include_content
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def search_by_document(
        self,
        document_id: str,
        query: str = "",
        top_k: int = 5
    ) -> str:
        """
        Search within a specific document.
        
        Args:
            document_id: ID of the document to search within
            query: Optional query to filter results
            top_k: Number of results to return
            
        Returns:
            Formatted search results as string
        """
        try:
            if not self.retriever.is_loaded():
                return "Search tool is not available. No document index has been loaded."
            
            # Get document chunks
            results = await self.retriever.retrieve_by_document(
                document_id=document_id,
                query=query,
                top_k=top_k
            )
            
            if not results:
                if query:
                    return f"No results found for query '{query}' in document {document_id}"
                else:
                    return f"No chunks found in document {document_id}"
            
            # Format results
            output = [f"Document search results for document: {document_id}"]
            if query:
                output.append(f"Query: '{query}'")
            output.append(f"Found {len(results)} chunks:")
            output.append("")
            
            for i, result in enumerate(results, 1):
                output.append(f"{i}. Chunk {result['chunk_index']}")
                if query and 'score' in result:
                    output.append(f"   Relevance Score: {result['score']:.3f}")
                output.append(f"   Content: {result['content'][:150]}...")
                output.append("")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Document search error: {str(e)}"
    
    async def get_document_info(self) -> str:
        """Get information about available documents."""
        try:
            if not self.retriever.is_loaded():
                return "Search tool is not available. No document index has been loaded."
            
            documents = self.retriever.list_documents()
            
            if not documents:
                return "No documents found in the index."
            
            output = ["Available documents in the index:"]
            output.append("")
            
            for doc in documents:
                output.append(f"Document ID: {doc['document_id']}")
                output.append(f"Title: {doc['title']}")
                output.append(f"Source: {doc['source_path']}")
                output.append(f"Chunks: {doc['chunk_count']}")
                output.append(f"Total Content Length: {doc['total_content_length']} characters")
                output.append("")
            
            output.append(f"Total documents: {len(documents)}")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error getting document info: {str(e)}"
    
    async def get_index_stats(self) -> str:
        """Get statistics about the document index."""
        try:
            if not self.retriever.is_loaded():
                return "Search tool is not available. No document index has been loaded."
            
            index_info = self.retriever.get_index_info()
            
            if not index_info:
                return "Could not retrieve index information."
            
            output = ["Document Index Statistics:"]
            output.append("")
            output.append(f"Total chunks: {index_info['total_chunks']}")
            output.append(f"Embedding dimension: {index_info['embedding_dimension']}")
            output.append(f"Index type: {index_info['index_type']}")
            output.append(f"Index path: {index_info['index_path']}")
            
            # Build info
            build_info = index_info.get('build_info', {})
            if build_info:
                output.append("")
                output.append("Build Information:")
                output.append(f"  Chunk size: {build_info.get('chunk_size', 'N/A')}")
                output.append(f"  Chunk overlap: {build_info.get('chunk_overlap', 'N/A')}")
                output.append(f"  Build timestamp: {build_info.get('build_timestamp', 'N/A')}")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error getting index stats: {str(e)}"
    
    async def search_similar(
        self,
        chunk_id: str,
        top_k: int = 5
    ) -> str:
        """
        Find chunks similar to a specific chunk.
        
        Args:
            chunk_id: ID of the chunk to find similar ones for
            top_k: Number of similar chunks to return
            
        Returns:
            Formatted results as string
        """
        try:
            if not self.retriever.is_loaded():
                return "Search tool is not available. No document index has been loaded."
            
            # Get the target chunk
            target_chunk = self.retriever.get_chunk_by_id(chunk_id)
            if not target_chunk:
                return f"Chunk with ID '{chunk_id}' not found."
            
            # Use the chunk's content as a query to find similar chunks
            results = await self.retriever.retrieve(
                query=target_chunk.content,
                top_k=top_k + 1,  # +1 to account for the target chunk itself
                score_threshold=0.3,  # Lower threshold for similarity search
                include_metadata=True
            )
            
            # Filter out the target chunk itself
            similar_chunks = [r for r in results if r['chunk_id'] != chunk_id]
            
            if not similar_chunks:
                return f"No similar chunks found for chunk '{chunk_id}'."
            
            # Format results
            output = [f"Chunks similar to '{chunk_id}':"]
            output.append(f"Target chunk title: {target_chunk.title}")
            output.append("")
            
            for i, result in enumerate(similar_chunks, 1):
                output.append(f"{i}. {result['title']}")
                output.append(f"   Chunk ID: {result['chunk_id']}")
                output.append(f"   Similarity Score: {result['score']:.3f}")
                output.append(f"   Content Preview: {result['content'][:150]}...")
                output.append("")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Similarity search error: {str(e)}"
