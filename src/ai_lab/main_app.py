"""Main FastAPI application for AI Solutions Lab."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import os
from pathlib import Path

from .api_routes import api_router

# Create FastAPI app
app = FastAPI(
    title="AI Solutions Lab",
    description="Advanced AI-powered search and RAG system with multiple LLM providers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Mount static files
static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Templates for web interface
templates_dir = Path(__file__).parent / "templates"
if not templates_dir.exists():
    templates_dir.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(templates_dir))

# Root endpoint with web interface
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check."""
    return {"status": "healthy", "service": "AI Solutions Lab"}

# API documentation redirect
@app.get("/api")
async def api_docs_redirect():
    """Redirect to API documentation."""
    return {"message": "API Documentation", "docs_url": "/docs", "redoc_url": "/redoc"}

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors."""
    return templates.TemplateResponse(
        "404.html", 
        {"request": request, "message": "Page not found"}, 
        status_code=404
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors."""
    return templates.TemplateResponse(
        "500.html", 
        {"request": request, "message": "Internal server error"}, 
        status_code=500
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    print("🚀 AI Solutions Lab starting up...")
    print("📚 Advanced Search & RAG System")
    print("🤖 Multiple LLM Provider Support")
    print("🔍 Hybrid Search Engine")
    print("📊 Analytics & Export Capabilities")
    print("🌐 Web Interface & API")
    print("✅ Ready for production use!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("🛑 AI Solutions Lab shutting down...")

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "main_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
