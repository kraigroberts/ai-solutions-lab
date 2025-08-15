"""Tests for Task E: API Endpoints & Web Interface."""

import pytest
import sys
from pathlib import Path
import json
from unittest.mock import Mock, patch
from fastapi import HTTPException

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_lab.api_routes import (
    api_router, SearchRequest, RAGRequest, DocumentUploadRequest,
    BatchSearchRequest, SystemStatusResponse
)
from ai_lab.main_app import app


class TestAPIModels:
    """Test API request/response models."""
    
    def test_search_request_creation(self):
        """Test SearchRequest model creation."""
        request = SearchRequest(
            query="test query",
            search_type="hybrid",
            max_results=20,
            generate_answer=True
        )
        
        assert request.query == "test query"
        assert request.search_type == "hybrid"
        assert request.max_results == 20
        assert request.generate_answer is True
        assert request.boost_semantic == 1.0  # Default value
    
    def test_search_request_validation(self):
        """Test SearchRequest validation."""
        # Test min length validation
        with pytest.raises(ValueError):
            SearchRequest(query="", search_type="hybrid")
        
        # Test max results validation
        with pytest.raises(ValueError):
            SearchRequest(query="test", max_results=0)
        
        with pytest.raises(ValueError):
            SearchRequest(query="test", max_results=101)
    
    def test_rag_request_creation(self):
        """Test RAGRequest model creation."""
        request = RAGRequest(
            query="What is machine learning?",
            k=10,
            generate_answer=True
        )
        
        assert request.query == "What is machine learning?"
        assert request.k == 10
        assert request.generate_answer is True
    
    def test_rag_request_validation(self):
        """Test RAGRequest validation."""
        # Test min length validation
        with pytest.raises(ValueError):
            RAGRequest(query="", k=5)
        
        # Test k validation
        with pytest.raises(ValueError):
            RAGRequest(query="test", k=0)
        
        with pytest.raises(ValueError):
            RAGRequest(query="test", k=21)
    
    def test_document_upload_request(self):
        """Test DocumentUploadRequest model creation."""
        request = DocumentUploadRequest(
            documents=["doc1", "doc2"],
            metadata={"source": "test"}
        )
        
        assert len(request.documents) == 2
        assert request.metadata["source"] == "test"
    
    def test_batch_search_request(self):
        """Test BatchSearchRequest model creation."""
        request = BatchSearchRequest(
            queries=["query1", "query2"],
            search_type="hybrid",
            max_results=15
        )
        
        assert len(request.queries) == 2
        assert request.search_type == "hybrid"
        assert request.max_results == 15
    
    def test_batch_search_request_validation(self):
        """Test BatchSearchRequest validation."""
        # Test empty queries
        with pytest.raises(ValueError):
            BatchSearchRequest(queries=[], search_type="hybrid")
        
        # Test too many queries
        with pytest.raises(ValueError):
            BatchSearchRequest(queries=["q"] * 51, search_type="hybrid")
    
    def test_system_status_response(self):
        """Test SystemStatusResponse model creation."""
        response = SystemStatusResponse(
            status="operational",
            version="1.0.0",
            components={"test": "component"},
            capabilities={"search": True},
            statistics={"total": 100}
        )
        
        assert response.status == "operational"
        assert response.version == "1.0.0"
        assert "test" in response.components
        assert response.capabilities["search"] is True


class TestAPIRoutes:
    """Test API route functionality."""
    
    def test_api_root_endpoint(self):
        """Test API root endpoint."""
        # Test that the endpoint exists in the router
        root_route = None
        for route in api_router.routes:
            if hasattr(route, 'path') and route.path == "/api/v1/" and hasattr(route, 'methods') and "GET" in route.methods:
                root_route = route
                break
        
        assert root_route is not None
        assert root_route.summary == "API Root"
    
    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        # Test that the endpoint exists in the router
        health_route = None
        for route in api_router.routes:
            if hasattr(route, 'path') and route.path == "/api/v1/health" and hasattr(route, 'methods') and "GET" in route.methods:
                health_route = route
                break
        
        assert health_route is not None
        assert health_route.summary == "Health Check"
    
    def test_search_endpoint_structure(self):
        """Test search endpoint structure."""
        # Test that the endpoint exists in the router
        search_route = None
        for route in api_router.routes:
            if hasattr(route, 'path') and route.path == "/api/v1/search" and hasattr(route, 'methods') and "POST" in route.methods:
                search_route = route
                break
        
        assert search_route is not None
        assert search_route.summary == "Advanced Search"
    
    def test_rag_endpoint_structure(self):
        """Test RAG endpoint structure."""
        # Test that the endpoint exists in the router
        rag_route = None
        for route in api_router.routes:
            if hasattr(route, 'path') and route.path == "/api/v1/rag" and hasattr(route, 'methods') and "POST" in route.methods:
                rag_route = route
                break
        
        assert rag_route is not None
        assert rag_route.summary == "RAG Question Answering"
    
    def test_batch_search_endpoint_structure(self):
        """Test batch search endpoint structure."""
        # Test that the endpoint exists in the router
        batch_route = None
        for route in api_router.routes:
            if hasattr(route, 'path') and route.path == "/api/v1/search/batch" and hasattr(route, 'methods') and "POST" in route.methods:
                batch_route = route
                break
        
        assert batch_route is not None
        assert batch_route.summary == "Batch Search"
    
    def test_documents_endpoint_structure(self):
        """Test documents endpoint structure."""
        # Test that the endpoint exists in the router
        docs_route = None
        for route in api_router.routes:
            if hasattr(route, 'path') and route.path == "/api/v1/documents" and hasattr(route, 'methods') and "GET" in route.methods:
                docs_route = route
                break
        
        assert docs_route is not None
        assert docs_route.summary == "Get Document Statistics"
    
    def test_analytics_endpoint_structure(self):
        """Test analytics endpoint structure."""
        # Test that the endpoint exists in the router
        analytics_route = None
        for route in api_router.routes:
            if hasattr(route, 'path') and route.path == "/api/v1/analytics" and hasattr(route, 'methods') and "GET" in route.methods:
                analytics_route = route
                break
        
        assert analytics_route is not None
        assert analytics_route.summary == "Search Analytics"
    
    def test_suggestions_endpoint_structure(self):
        """Test suggestions endpoint structure."""
        # Test that the endpoint exists in the router
        suggestions_route = None
        for route in api_router.routes:
            if hasattr(route, 'path') and route.path == "/api/v1/suggestions" and hasattr(route, 'methods') and "GET" in route.methods:
                suggestions_route = route
                break
        
        assert suggestions_route is not None
        assert suggestions_route.summary == "Query Suggestions"
    
    def test_system_endpoint_structure(self):
        """Test system endpoint structure."""
        # Test that the endpoint exists in the router
        system_route = None
        for route in api_router.routes:
            if hasattr(route, 'path') and route.path == "/api/v1/system" and hasattr(route, 'methods') and "GET" in route.methods:
                system_route = route
                break
        
        assert system_route is not None
        assert system_route.summary == "System Status"
    
    def test_export_endpoint_structure(self):
        """Test export endpoint structure."""
        # Test that the endpoint exists in the router
        export_route = None
        for route in api_router.routes:
            if hasattr(route, 'path') and route.path == "/api/v1/export" and hasattr(route, 'methods') and "POST" in route.methods:
                export_route = route
                break
        
        assert export_route is not None
        assert export_route.summary == "Export Search Results"


class TestMainApp:
    """Test main FastAPI application."""
    
    def test_app_creation(self):
        """Test that the FastAPI app is created correctly."""
        assert app.title == "AI Solutions Lab"
        assert app.version == "1.0.0"
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"
    
    def test_cors_middleware(self):
        """Test that CORS middleware is configured."""
        cors_middleware = None
        for middleware in app.user_middleware:
            if "CORSMiddleware" in str(middleware.cls):
                cors_middleware = middleware
                break
        
        assert cors_middleware is not None
    
    def test_api_router_inclusion(self):
        """Test that API router is included."""
        # Check if any of our API routes are registered
        api_routes = []
        for route in app.routes:
            if hasattr(route, 'path') and route.path.startswith('/api/v1'):
                api_routes.append(route)
        
        assert len(api_routes) > 0
    
    def test_static_files_mount(self):
        """Test that static files are mounted."""
        static_mount = None
        for route in app.routes:
            if hasattr(route, 'path') and route.path == '/static':
                static_mount = route
                break
        
        assert static_mount is not None
    
    def test_templates_mount(self):
        """Test that templates are configured."""
        # The templates directory should be created
        templates_dir = Path(__file__).parent.parent / "src" / "ai_lab" / "templates"
        assert templates_dir.exists()
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        # Test that the endpoint exists
        root_route = None
        for route in app.routes:
            if hasattr(route, 'path') and route.path == "/api/v1/" and hasattr(route, 'methods') and "GET" in route.methods:
                root_route = route
                break
        
        assert root_route is not None
    
    def test_health_endpoint(self):
        """Test health endpoint."""
        # Test that the endpoint exists
        health_route = None
        for route in app.routes:
            if hasattr(route, 'path') and route.path == "/api/v1/health" and hasattr(route, 'methods') and "GET" in route.methods:
                health_route = route
                break
        
        assert health_route is not None
    
    def test_api_docs_redirect(self):
        """Test API docs redirect endpoint."""
        # Test that the endpoint exists
        api_route = None
        for route in app.routes:
            if hasattr(route, 'path') and route.path == "/api" and hasattr(route, 'methods') and "GET" in route.methods:
                api_route = route
                break
        
        assert api_route is not None


class TestErrorHandling:
    """Test error handling functionality."""
    
    def test_404_handler(self):
        """Test 404 error handler."""
        # Test that the handler exists
        handlers = app.exception_handlers
        assert 404 in handlers
    
    def test_http_exception_handler(self):
        """Test HTTP exception handler."""
        # This would require testing with actual API endpoints
        # For now, we'll test the handler exists
        handlers = app.exception_handlers
        # FastAPI uses starlette.exceptions.HTTPException, not fastapi.HTTPException
        assert any("HTTPException" in str(handler) for handler in handlers.keys())


class TestComponentIntegration:
    """Test component integration."""
    
    @patch('ai_lab.api_routes.VectorStore')
    @patch('ai_lab.api_routes.create_default_config')
    @patch('ai_lab.api_routes.AdvancedSearchInterface')
    @patch('ai_lab.api_routes.EnhancedRAG')
    def test_component_initialization(self, mock_rag, mock_search, mock_config, mock_store):
        """Test that components are initialized correctly."""
        # Mock the components
        mock_store_instance = Mock()
        mock_store.return_value = mock_store_instance
        
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        mock_search_instance = Mock()
        mock_search.return_value = mock_search_instance
        
        mock_rag_instance = Mock()
        mock_rag.return_value = mock_rag_instance
        
        # Import and test the function
        from ai_lab.api_routes import get_components
        
        # Reset global variables
        import ai_lab.api_routes
        ai_lab.api_routes.vector_store = None
        ai_lab.api_routes.llm_config = None
        ai_lab.api_routes.advanced_search = None
        ai_lab.api_routes.enhanced_rag = None
        
        # Test component initialization
        vs, lc, as_, er = get_components()
        
        assert vs == mock_store_instance
        assert lc == mock_config_instance
        assert as_ == mock_search_instance
        assert er == mock_rag_instance


def test_api_completeness():
    """Test that all expected API endpoints are present."""
    expected_endpoints = [
        ("/api/v1/", "GET", "API Root"),
        ("/api/v1/health", "GET", "Health Check"),
        ("/api/v1/search", "POST", "Advanced Search"),
        ("/api/v1/rag", "POST", "RAG Question Answering"),
        ("/api/v1/search/batch", "POST", "Batch Search"),
        ("/api/v1/documents", "POST", "Add Documents"),
        ("/api/v1/documents", "GET", "Get Document Statistics"),
        ("/api/v1/analytics", "GET", "Search Analytics"),
        ("/api/v1/suggestions", "GET", "Query Suggestions"),
        ("/api/v1/system", "GET", "System Status"),
        ("/api/v1/export", "POST", "Export Search Results")
    ]
    
    # Get all registered routes
    registered_routes = []
    for route in api_router.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            for method in route.methods:
                registered_routes.append((route.path, method, getattr(route, 'summary', 'No Summary')))
    
    # Check that all expected endpoints are present
    for expected_path, expected_method, expected_summary in expected_endpoints:
        found = False
        for actual_path, actual_method, actual_summary in registered_routes:
            if actual_path == expected_path and actual_method == expected_method:
                found = True
                break
        
        if not found:
            print(f"Missing endpoint: {expected_method} {expected_path} - {expected_summary}")
            print(f"Available routes: {registered_routes}")
        
        assert found, f"Missing endpoint: {expected_method} {expected_path} - {expected_summary}"


def test_web_interface_files():
    """Test that web interface files exist."""
    base_dir = Path(__file__).parent.parent / "src" / "ai_lab"
    
    # Check HTML templates
    assert (base_dir / "templates" / "index.html").exists()
    assert (base_dir / "templates" / "404.html").exists()
    assert (base_dir / "templates" / "500.html").exists()
    
    # Check main app file
    assert (base_dir / "main_app.py").exists()
    
    # Check API routes file
    assert (base_dir / "main_app.py").exists()


if __name__ == "__main__":
    pytest.main([__file__])
