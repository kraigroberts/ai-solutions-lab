"""
FastAPI application for AI Solutions Lab.

Provides REST API endpoints for:
- Chat with LLM backends
- RAG document querying
- Agent tool execution
- Health and status information
"""

import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import get_settings
from .llm.router import LLMRouter
from .rag.answer import RAGAnswerer
from .tools.registry import ToolRegistry


# Pydantic models for request/response
class ChatMessage(BaseModel):
    """Individual chat message."""

    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request model."""

    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    backend: Optional[str] = Field(None, description="Override default LLM backend")


class ChatResponse(BaseModel):
    """Chat response model."""

    content: str = Field(..., description="Generated response content")
    backend: str = Field(..., description="LLM backend used")
    response_time: float = Field(..., description="Response time in seconds")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model information")


class RAGRequest(BaseModel):
    """RAG query request model."""

    query: str = Field(..., description="Query to search for in documents")
    top_k: int = Field(default=5, description="Number of top chunks to retrieve")
    index_path: Optional[str] = Field(None, description="Custom index path")


class RAGResponse(BaseModel):
    """RAG response model."""

    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents used")
    query_time: float = Field(..., description="Query processing time in seconds")
    total_chunks: int = Field(..., description="Total chunks retrieved")


class AgentRequest(BaseModel):
    """Agent execution request model."""

    goal: str = Field(..., description="Goal for the agent to accomplish")
    tools: Optional[List[str]] = Field(None, description="Specific tools to use")
    max_steps: int = Field(default=10, description="Maximum execution steps")


class AgentResponse(BaseModel):
    """Agent execution response model."""

    result: str = Field(..., description="Final result of agent execution")
    steps: List[Dict[str, Any]] = Field(..., description="Execution steps taken")
    execution_time: float = Field(..., description="Total execution time in seconds")
    tools_used: List[str] = Field(..., description="Tools that were used")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: float = Field(..., description="Current timestamp")
    backends: Dict[str, bool] = Field(..., description="Backend availability status")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: float = Field(..., description="Error timestamp")


# Initialize FastAPI app
app = FastAPI(
    title="AI Solutions Lab API",
    description="A production-minded lab demonstrating AI Solutions Engineering skills",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global instances
settings = get_settings()
llm_router = LLMRouter()
tool_registry = ToolRegistry()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error", detail=str(exc), timestamp=time.time()
        ).dict(),
    )


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "AI Solutions Lab API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Check backend availability
    backends = {
        "local": True,  # Local is always available
        "openai": settings.has_openai(),
        "anthropic": settings.has_anthropic(),
        "pinecone": settings.has_pinecone(),
    }

    return HealthResponse(
        status="healthy", version=__version__, timestamp=time.time(), backends=backends
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for LLM interactions."""
    try:
        start_time = time.time()

        # Override backend if specified
        if request.backend:
            # Ensure the backend is one of the allowed values
            if request.backend in ["local", "openai", "anthropic"]:
                settings.model_backend = request.backend

        # Convert messages to format expected by router
        messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        # Get response from LLM router
        response = await llm_router.chat(
            message=messages[-1]["content"],  # Use last message as current
            system_prompt=request.system_prompt,
            conversation_history=messages[:-1] if len(messages) > 1 else [],
        )

        response_time = time.time() - start_time

        return ChatResponse(
            content=response["content"],
            backend=settings.model_backend,
            response_time=response_time,
            model_info=response.get("model_info"),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat error: {str(e)}",
        )


@app.post("/rag/answer", response_model=RAGResponse)
async def rag_answer(request: RAGRequest):
    """RAG endpoint for document querying."""
    try:
        start_time = time.time()

        # Initialize RAG answerer
        index_path = request.index_path or str(settings.index_dir)
        answerer = RAGAnswerer(index_path=index_path)

        # Get answer
        result = await answerer.answer(query=request.query, top_k=request.top_k)

        query_time = time.time() - start_time

        return RAGResponse(
            answer=result["answer"],
            sources=result["sources"],
            query_time=query_time,
            total_chunks=len(result["sources"]),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG error: {str(e)}",
        )


@app.post("/agent/run", response_model=AgentResponse)
async def agent_run(request: AgentRequest):
    """Agent endpoint for tool execution."""
    try:
        start_time = time.time()

        # Get available tools
        available_tools = request.tools or list(tool_registry.list_tools().keys())

        # Run agent
        result = await llm_router.run_agent(
            goal=request.goal, tools=available_tools, max_steps=request.max_steps
        )

        execution_time = time.time() - start_time

        return AgentResponse(
            result=result["result"],
            steps=result["steps"],
            execution_time=execution_time,
            tools_used=list(set(step["tool"] for step in result["steps"])),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {str(e)}",
        )


@app.get("/tools", response_model=Dict[str, Any])
async def list_tools():
    """List available tools."""
    try:
        tools = tool_registry.list_tools()
        return {"tools": tools, "total_count": len(tools)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing tools: {str(e)}",
        )


@app.get("/tools/{tool_name}", response_model=Dict[str, Any])
async def get_tool_info(tool_name: str):
    """Get information about a specific tool."""
    try:
        tool_info = tool_registry.get_tool_info(tool_name)
        if not tool_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool '{tool_name}' not found",
            )
        return tool_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting tool info: {str(e)}",
        )


@app.get("/config", response_model=Dict[str, Any])
async def get_config():
    """Get current configuration (non-sensitive)."""
    return {
        "model_backend": settings.model_backend,
        "embeddings_backend": settings.embeddings_backend,
        "vector_store": settings.vector_store,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "local_mode": settings.is_local_mode(),
        "has_openai": settings.has_openai(),
        "has_anthropic": settings.has_anthropic(),
        "has_pinecone": settings.has_pinecone(),
    }


@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """Get system statistics."""
    try:
        # Basic stats for now - can be enhanced with actual metrics
        return {
            "version": __version__,
            "uptime": time.time(),  # Would need to track actual start time
            "backends": {
                "local": True,
                "openai": settings.has_openai(),
                "anthropic": settings.has_anthropic(),
                "pinecone": settings.has_pinecone(),
            },
            "data_paths": {
                "docs_dir": str(settings.docs_dir),
                "index_dir": str(settings.index_dir),
                "models_dir": str(settings.models_dir),
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting stats: {str(e)}",
        )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    # Ensure data directories exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    # Any cleanup needed
    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.ai_lab.api:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
