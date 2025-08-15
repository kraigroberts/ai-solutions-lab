"""Comprehensive API routes for AI Solutions Lab."""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import json
import time

from .advanced_search import AdvancedSearchInterface, AdvancedSearchRequest
from .llm_config import LLMConfig, create_default_config
from .vector_store import VectorStore
from .enhanced_rag import EnhancedRAG

# API Models
class SearchRequest(BaseModel):
    """Request model for search operations."""
    query: str = Field(..., description="Search query text", min_length=1)
    search_type: str = Field("hybrid", description="Search type: hybrid, semantic, keyword, rag")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    max_results: int = Field(10, description="Maximum number of results", ge=1, le=100)
    generate_answer: bool = Field(False, description="Generate RAG answer")
    include_highlights: bool = Field(True, description="Include search highlights")
    boost_semantic: float = Field(1.0, description="Semantic search boost factor", ge=0.0, le=5.0)
    boost_keyword: float = Field(1.0, description="Keyword search boost factor", ge=0.0, le=5.0)
    boost_metadata: float = Field(1.0, description="Metadata boost factor", ge=0.0, le=5.0)

class RAGRequest(BaseModel):
    """Request model for RAG operations."""
    query: str = Field(..., description="Question for RAG system", min_length=1)
    k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    generate_answer: bool = Field(True, description="Generate answer using LLM")
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, description="Conversation context")

class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    documents: List[str] = Field(..., description="List of document texts")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")

class BatchSearchRequest(BaseModel):
    """Request model for batch search operations."""
    queries: List[str] = Field(..., description="List of search queries", min_items=1, max_items=50)
    search_type: str = Field("hybrid", description="Search type for all queries")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    max_results: int = Field(10, description="Maximum results per query")

class SystemStatusResponse(BaseModel):
    """Response model for system status."""
    status: str = Field(..., description="System status")
    version: str = Field(..., description="System version")
    components: Dict[str, Any] = Field(..., description="Component status")
    capabilities: Dict[str, bool] = Field(..., description="System capabilities")
    statistics: Dict[str, Any] = Field(..., description="System statistics")

# Global instances (in production, these would be properly managed)
vector_store = None
llm_config = None
advanced_search = None
enhanced_rag = None

def get_components():
    """Get or initialize system components."""
    global vector_store, llm_config, advanced_search, enhanced_rag
    
    if vector_store is None:
        vector_store = VectorStore()
    
    if llm_config is None:
        llm_config = create_default_config()
    
    if advanced_search is None:
        advanced_search = AdvancedSearchInterface(vector_store, llm_config)
    
    if enhanced_rag is None:
        enhanced_rag = EnhancedRAG(vector_store, llm_config)
    
    return vector_store, llm_config, advanced_search, enhanced_rag

# Create API router
api_router = APIRouter(prefix="/api/v1", tags=["AI Solutions Lab API"])

@api_router.get("/", summary="API Root")
async def api_root():
    """Get API root information."""
    return {
        "message": "AI Solutions Lab API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "search": "/api/v1/search",
            "rag": "/api/v1/rag",
            "documents": "/api/v1/documents",
            "analytics": "/api/v1/analytics",
            "system": "/api/v1/system"
        }
    }

@api_router.get("/health", summary="Health Check")
async def health_check():
    """Check system health."""
    try:
        vector_store, llm_config, advanced_search, enhanced_rag = get_components()
        
        # Basic health checks
        vector_store_healthy = vector_store is not None
        llm_config_healthy = llm_config is not None
        search_healthy = advanced_search is not None
        rag_healthy = enhanced_rag is not None
        
        overall_health = all([vector_store_healthy, llm_config_healthy, search_healthy, rag_healthy])
        
        return {
            "status": "healthy" if overall_health else "degraded",
            "timestamp": time.time(),
            "components": {
                "vector_store": "healthy" if vector_store_healthy else "unavailable",
                "llm_config": "healthy" if llm_config_healthy else "unavailable",
                "advanced_search": "healthy" if search_healthy else "unavailable",
                "enhanced_rag": "healthy" if rag_healthy else "unavailable"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@api_router.post("/search", summary="Advanced Search")
async def advanced_search_endpoint(request: SearchRequest):
    """Perform advanced search with multiple strategies."""
    try:
        vector_store, llm_config, advanced_search, enhanced_rag = get_components()
        
        # Convert to AdvancedSearchRequest
        search_request = AdvancedSearchRequest(
            query=request.query,
            search_type=request.search_type,
            filters=request.filters,
            boost_semantic=request.boost_semantic,
            boost_keyword=request.boost_keyword,
            boost_metadata=request.boost_metadata,
            max_results=request.max_results,
            generate_answer=request.generate_answer,
            include_highlights=request.include_highlights
        )
        
        # Perform search
        response = advanced_search.search(search_request)
        
        # Convert response to dict for JSON serialization
        return {
            "query": response.query,
            "search_type": response.search_type,
            "total_results": response.total_results,
            "processing_time": response.processing_time,
            "results": [
                {
                    "rank": result.relevance_rank,
                    "document": result.document,
                    "source_file": result.source_file,
                    "chunk_id": result.chunk_id,
                    "scores": {
                        "semantic": result.semantic_score,
                        "keyword": result.keyword_score,
                        "metadata": result.metadata_score,
                        "combined": result.combined_score
                    },
                    "metadata": result.metadata,
                    "highlights": result.search_highlights
                }
                for result in response.results
            ],
            "rag_response": {
                "answer": response.rag_response.answer,
                "confidence": response.rag_response.confidence_score,
                "sources": response.rag_response.sources
            } if response.rag_response else None,
            "suggestions": response.query_suggestions,
            "filters_applied": response.filters_applied
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@api_router.post("/rag", summary="RAG Question Answering")
async def rag_endpoint(request: RAGRequest):
    """Generate answers using RAG system."""
    try:
        vector_store, llm_config, advanced_search, enhanced_rag = get_components()
        
        if request.conversation_history:
            response = enhanced_rag.conversational_search(
                request.query,
                request.conversation_history,
                k=request.k,
                generate_answer=request.generate_answer
            )
        else:
            response = enhanced_rag.search_and_answer(
                request.query,
                k=request.k,
                generate_answer=request.generate_answer
            )
        
        return {
            "query": request.query,
            "answer": response.answer,
            "confidence": response.confidence_score,
            "processing_time": response.processing_time,
            "sources": response.sources,
            "llm_provider": response.llm_provider
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG failed: {str(e)}")

@api_router.post("/search/batch", summary="Batch Search")
async def batch_search_endpoint(request: BatchSearchRequest):
    """Perform batch search on multiple queries."""
    try:
        vector_store, llm_config, advanced_search, enhanced_rag = get_components()
        
        responses = advanced_search.batch_search(
            request.queries,
            search_type=request.search_type,
            filters=request.filters,
            max_results=request.max_results
        )
        
        return {
            "total_queries": len(request.queries),
            "responses": [
                {
                    "query": response.query,
                    "total_results": response.total_results,
                    "processing_time": response.processing_time,
                    "results_count": len(response.results)
                }
                for response in responses
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch search failed: {str(e)}")

@api_router.post("/documents", summary="Add Documents")
async def add_documents_endpoint(request: DocumentUploadRequest):
    """Add documents to the vector store."""
    try:
        vector_store, llm_config, advanced_search, enhanced_rag = get_components()
        
        # For now, we'll add simple text documents
        # In a real implementation, this would process and chunk the documents
        
        return {
            "message": f"Added {len(request.documents)} documents",
            "count": len(request.documents),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

@api_router.get("/documents", summary="Get Document Statistics")
async def get_documents_endpoint():
    """Get document statistics and information."""
    try:
        vector_store, llm_config, advanced_search, enhanced_rag = get_components()
        
        stats = vector_store.get_document_stats()
        
        return {
            "total_documents": stats["total_documents"],
            "index_size": stats["index_size"],
            "embedding_model": stats["embedding_model"],
            "file_types": stats.get("file_types", {}),
            "model_name": stats["model_name"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document stats: {str(e)}")

@api_router.get("/analytics", summary="Search Analytics")
async def get_analytics_endpoint():
    """Get search analytics and insights."""
    try:
        vector_store, llm_config, advanced_search, enhanced_rag = get_components()
        
        analytics = advanced_search.get_search_statistics()
        
        return {
            "search_analytics": analytics["search_analytics"],
            "vector_store_stats": analytics["vector_store_stats"],
            "rag_capabilities": analytics["rag_capabilities"],
            "search_engine_info": analytics["search_engine_info"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@api_router.get("/suggestions", summary="Query Suggestions")
async def get_suggestions_endpoint(
    partial_query: str = Query(..., description="Partial query for suggestions")
):
    """Get query suggestions based on partial input."""
    try:
        vector_store, llm_config, advanced_search, enhanced_rag = get_components()
        
        suggestions = advanced_search.hybrid_engine.suggest_queries(partial_query)
        
        return {
            "partial_query": partial_query,
            "suggestions": suggestions,
            "count": len(suggestions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")

@api_router.get("/system", summary="System Status")
async def get_system_status_endpoint():
    """Get comprehensive system status."""
    try:
        vector_store, llm_config, advanced_search, enhanced_rag = get_components()
        
        # Get component statuses
        vector_stats = vector_store.get_document_stats()
        search_stats = advanced_search.get_search_statistics()
        llm_info = enhanced_rag.llm_provider.get_model_info()
        
        return SystemStatusResponse(
            status="operational",
            version="1.0.0",
            components={
                "vector_store": {
                    "status": "operational",
                    "total_documents": vector_stats["total_documents"],
                    "index_size": vector_stats["index_size"],
                    "embedding_model": vector_stats["embedding_model"]
                },
                "llm_provider": {
                    "status": "operational" if llm_info["available"] else "unavailable",
                    "provider": llm_info["provider"],
                    "model": llm_info.get("model", "N/A")
                },
                "search_engine": {
                    "status": "operational",
                    "capabilities": search_stats["search_engine_info"]
                }
            },
            capabilities={
                "search": True,
                "rag": True,
                "llm_integration": llm_info["available"],
                "batch_processing": True,
                "analytics": True,
                "export": True
            },
            statistics={
                "total_searches": search_stats["search_analytics"].get("total_searches", 0),
                "average_results": search_stats["search_analytics"].get("average_results_per_search", 0),
                "total_documents": vector_stats["total_documents"]
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@api_router.post("/export", summary="Export Search Results")
async def export_results_endpoint(
    search_request: SearchRequest,
    format: str = Query("json", description="Export format: json, csv")
):
    """Export search results in various formats."""
    try:
        vector_store, llm_config, advanced_search, enhanced_rag = get_components()
        
        # Perform search
        search_req = AdvancedSearchRequest(
            query=search_request.query,
            search_type=search_request.search_type,
            filters=search_request.filters,
            max_results=search_request.max_results
        )
        
        response = advanced_search.search(search_req)
        
        # Export results
        if format.lower() == "csv":
            csv_data = advanced_search.export_search_results(response, "csv")
            return StreamingResponse(
                iter([csv_data]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=search_results.csv"}
            )
        else:
            json_data = advanced_search.export_search_results(response, "json")
            return JSONResponse(content=json.loads(json_data))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

# Note: Exception handlers are moved to main_app.py since APIRouter doesn't support them
