"""FastAPI application for AI Solutions Lab."""

import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import get_settings

# Initialize FastAPI app
app = FastAPI(
    title="AI Solutions Lab",
    description="Clean System Design with LLM Integrations and RAG Pipelines",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API requests/responses
class ChatMessage(BaseModel):
    """Chat message model."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat request model."""

    messages: List[ChatMessage]
    system_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model."""

    content: str
    response_time: float


class RAGRequest(BaseModel):
    """RAG request model."""

    query: str
    top_k: Optional[int] = None


class RAGResponse(BaseModel):
    """RAG response model."""

    answer: str
    sources: List[Dict[str, Any]]
    query_time: float
    total_chunks: int


class AgentRequest(BaseModel):
    """Agent request model."""

    goal: str
    tools: Optional[List[str]] = None
    max_steps: int = 5


class AgentResponse(BaseModel):
    """Agent response model."""

    result: str
    steps: List[Dict[str, Any]]
    execution_time: float
    tools_used: List[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: float
    version: str
    backends: Dict[str, bool]


class ToolInfo(BaseModel):
    """Tool information model."""

    name: str
    description: str
    async_support: bool
    examples: List[Dict[str, str]]


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return {"error": str(exc), "type": type(exc).__name__}


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    print("AI Solutions Lab starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    print("AI Solutions Lab shutting down...")


# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "AI Solutions Lab API", "version": "0.1.0", "status": "running"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    settings = get_settings()

    # Mock backend availability
    backends = {
        "local": True,
        "openai": False,  # Mock - no API key
        "anthropic": False,  # Mock - no API key
    }

    return HealthResponse(
        status="healthy", timestamp=time.time(), version="0.1.0", backends=backends
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with mock LLM responses."""
    try:
        start_time = time.time()
        settings = get_settings()

        # Mock LLM response
        if settings.mock_llm_enabled:
            # Simulate processing delay
            time.sleep(settings.mock_response_delay)

            # Generate mock response based on last message
            last_message = request.messages[-1].content if request.messages else "Hello"
            system_context = (
                f"System: {request.system_prompt}\n" if request.system_prompt else ""
            )

            mock_response = f"{system_context}Mock LLM Response: I understand you said '{last_message}'. This is a simulated response from the AI Solutions Lab mock LLM."
        else:
            mock_response = "Mock LLM is disabled. Please configure a real LLM backend."

        response_time = time.time() - start_time

        return ChatResponse(content=mock_response, response_time=response_time)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat error: {str(e)}",
        )


@app.post("/rag/answer", response_model=RAGResponse)
async def rag_answer(request: RAGRequest):
    """RAG endpoint with mock document retrieval."""
    try:
        start_time = time.time()
        settings = get_settings()

        # Mock RAG response
        top_k = request.top_k or settings.top_k

        # Simulate document retrieval
        mock_sources = [
            {
                "title": f"Document {i}",
                "content": f"Mock content about {request.query}",
                "score": 0.9 - (i * 0.1),
                "source_path": f"/mock/doc_{i}.md",
            }
            for i in range(min(top_k, 3))
        ]

        # Generate mock answer
        mock_answer = f"Based on the retrieved documents, here's what I found about '{request.query}': This is a simulated RAG response demonstrating the retrieval-augmented generation pipeline."

        query_time = time.time() - start_time

        return RAGResponse(
            answer=mock_answer,
            sources=mock_sources,
            query_time=query_time,
            total_chunks=len(mock_sources),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG error: {str(e)}",
        )


@app.post("/agent/run", response_model=AgentResponse)
async def agent_run(request: AgentRequest):
    """Agent endpoint with mock tool execution."""
    try:
        start_time = time.time()

        # Mock agent execution
        mock_steps = [
            {
                "tool": "search",
                "input": f"Searching for information about: {request.goal}",
                "output": "Found relevant information",
                "step": 1,
            },
            {
                "tool": "process",
                "input": "Processing search results",
                "output": "Information processed successfully",
                "step": 2,
            },
        ]

        mock_result = f"Goal accomplished: {request.goal}. This was achieved through simulated tool execution."

        execution_time = time.time() - start_time

        return AgentResponse(
            result=mock_result,
            steps=mock_steps,
            execution_time=execution_time,
            tools_used=["search", "process"],
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
        tools = {
            "search": {
                "name": "search",
                "description": "Search for information in documents",
                "async_support": True,
                "examples": [
                    {"input": "AI basics", "output": "Search results about AI"}
                ],
            },
            "math": {
                "name": "math",
                "description": "Perform mathematical calculations",
                "async_support": True,
                "examples": [{"input": "2 + 2", "output": "4"}],
            },
            "web": {
                "name": "web",
                "description": "Fetch information from web sources",
                "async_support": True,
                "examples": [{"input": "https://example.com", "output": "Web content"}],
            },
        }

        return {"tools": tools, "total_count": len(tools)}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tools error: {str(e)}",
        )


@app.get("/tools/{tool_name}", response_model=ToolInfo)
async def get_tool_info(tool_name: str):
    """Get information about a specific tool."""
    try:
        tools = {
            "search": ToolInfo(
                name="search",
                description="Search for information in the document index",
                async_support=True,
                examples=[{"input": "AI basics", "output": "Search results about AI"}],
            ),
            "math": ToolInfo(
                name="math",
                description="Evaluate mathematical expressions and perform calculations",
                async_support=True,
                examples=[{"input": "2 + 2", "output": "4"}],
            ),
            "web": ToolInfo(
                name="web",
                description="Fetch information from web sources (stub implementation)",
                async_support=True,
                examples=[{"input": "https://example.com", "output": "Web content"}],
            ),
        }

        if tool_name not in tools:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool '{tool_name}' not found",
            )

        return tools[tool_name]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool info error: {str(e)}",
        )


@app.get("/config", response_model=Dict[str, Any])
async def get_config():
    """Get current configuration."""
    try:
        settings = get_settings()
        return {
            "host": settings.host,
            "port": settings.port,
            "debug": settings.debug,
            "mock_llm_enabled": settings.mock_llm_enabled,
            "top_k": settings.top_k,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Config error: {str(e)}",
        )


@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """Get application statistics."""
    try:
        return {
            "uptime": time.time(),
            "version": "0.1.0",
            "endpoints": [
                "/",
                "/health",
                "/chat",
                "/rag/answer",
                "/agent/run",
                "/tools",
                "/config",
                "/stats",
            ],
            "status": "operational",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stats error: {str(e)}",
        )
