"""Tests for the AI Solutions Lab API."""

import pytest
from fastapi.testclient import TestClient

from ai_lab.api import app

client = TestClient(app)


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_endpoint(self):
        """Test root endpoint returns correct information."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["message"] == "AI Solutions Lab API"
        assert data["version"] == "0.1.0"
        assert data["status"] == "running"


class TestHealthEndpoint:
    """Test health endpoint."""

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "backends" in data
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert isinstance(data["backends"], dict)


class TestChatEndpoint:
    """Test chat endpoint."""

    def test_chat_endpoint_success(self):
        """Test successful chat request."""
        request_data = {"messages": [{"role": "user", "content": "Hello"}]}

        response = client.post("/chat", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "content" in data
        assert "response_time" in data
        assert "Mock LLM Response" in data["content"]

    def test_chat_endpoint_with_system_prompt(self):
        """Test chat with system prompt."""
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "system_prompt": "You are a helpful assistant.",
        }

        response = client.post("/chat", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "System: You are a helpful assistant." in data["content"]


class TestRAGEndpoint:
    """Test RAG endpoint."""

    def test_rag_answer_success(self):
        """Test successful RAG request."""
        request_data = {"query": "What is AI?"}

        response = client.post("/rag/answer", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "query_time" in data
        assert "total_chunks" in data
        assert "What is AI?" in data["answer"]
        assert len(data["sources"]) > 0


class TestAgentEndpoint:
    """Test agent endpoint."""

    def test_agent_run_success(self):
        """Test successful agent execution."""
        request_data = {"goal": "Find information about Python"}

        response = client.post("/agent/run", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "result" in data
        assert "steps" in data
        assert "execution_time" in data
        assert "tools_used" in data
        assert "Python" in data["result"]


class TestToolsEndpoints:
    """Test tools endpoints."""

    def test_list_tools(self):
        """Test listing available tools."""
        response = client.get("/tools")
        assert response.status_code == 200

        data = response.json()
        assert "tools" in data
        assert "total_count" in data
        assert data["total_count"] == 3  # search, math, web
        assert "search" in data["tools"]
        assert "math" in data["tools"]
        assert "web" in data["tools"]

    def test_get_tool_info(self):
        """Test getting specific tool information."""
        response = client.get("/tools/search")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "description" in data
        assert "async_support" in data
        assert "examples" in data
        assert data["name"] == "search"


class TestConfigEndpoint:
    """Test config endpoint."""

    def test_get_config(self):
        """Test getting configuration."""
        response = client.get("/config")
        assert response.status_code == 200

        data = response.json()
        assert "host" in data
        assert "port" in data
        assert "debug" in data
        assert "mock_llm_enabled" in data
        assert "top_k" in data


class TestStatsEndpoint:
    """Test stats endpoint."""

    def test_get_stats(self):
        """Test getting statistics."""
        response = client.get("/stats")
        assert response.status_code == 200

        data = response.json()
        assert "uptime" in data
        assert "version" in data
        assert "endpoints" in data
        assert "status" in data
        assert data["version"] == "0.1.0"
        assert data["status"] == "operational"
