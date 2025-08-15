"""
Unit tests for agent tools.

Tests the search, math, and web stub tools used by the agent system.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from ai_lab.tools.registry import Tool, ToolRegistry
from ai_lab.tools.search import SearchTool
from ai_lab.tools.math import MathTool
from ai_lab.tools.webstub import WebStubTool


class TestTool:
    """Test Tool class."""

    def test_tool_creation(self):
        """Test creating a tool."""
        def test_function(input_data: str) -> str:
            return f"Processed: {input_data}"

        tool = Tool(
            name="test_tool",
            description="A test tool",
            function=test_function,
            input_schema={"type": "string", "description": "Input string"},
            examples=[{"input": "hello", "output": "Processed: hello"}]
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.function == test_function
        assert tool.input_schema["type"] == "string"
        assert len(tool.examples) == 1

    def test_tool_execute(self):
        """Test tool execution."""
        def test_function(input_data: str) -> str:
            return f"Processed: {input_data}"

        tool = Tool(
            name="test_tool",
            description="A test tool",
            function=test_function,
            input_schema={"type": "string", "description": "Input string"},
            examples=[{"input": "hello", "output": "Processed: hello"}]
        )

        result = tool.execute("test input")
        assert result == "Processed: test input"

    def test_tool_to_dict(self):
        """Test tool dictionary representation."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            function=lambda x: x,
            input_schema={"type": "string"},
            examples=[{"input": "test", "output": "test"}]
        )

        tool_dict = tool.to_dict()
        assert tool_dict["name"] == "test_tool"
        assert tool_dict["description"] == "A test tool"
        assert "function" in tool_dict
        assert tool_dict["input_schema"]["type"] == "string"


class TestToolRegistry:
    """Test ToolRegistry class."""

    def test_registry_creation(self):
        """Test creating a tool registry."""
        registry = ToolRegistry()
        assert isinstance(registry.tools, dict)
        assert len(registry.tools) > 0  # Should have default tools

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        initial_count = len(registry.tools)

        def test_function(input_data: str) -> str:
            return f"Test: {input_data}"

        tool = Tool(
            name="custom_tool",
            description="Custom tool",
            function=test_function,
            input_schema={"type": "string"},
            examples=[]
        )

        registry.register_tool(tool)
        assert len(registry.tools) == initial_count + 1
        assert "custom_tool" in registry.tools

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        initial_count = len(registry.tools)

        # Register a custom tool first
        def test_function(input_data: str) -> str:
            return f"Test: {input_data}"

        tool = Tool(
            name="temp_tool",
            description="Temporary tool",
            function=test_function,
            input_schema={"type": "string"},
            examples=[]
        )

        registry.register_tool(tool)
        assert len(registry.tools) == initial_count + 1

        # Now unregister it
        registry.unregister_tool("temp_tool")
        assert len(registry.tools) == initial_count
        assert "temp_tool" not in registry.tools

    def test_get_tool(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()
        
        # Test getting existing tool
        search_tool = registry.get_tool("search")
        assert search_tool is not None
        assert search_tool.name == "search"

        # Test getting non-existent tool
        non_existent = registry.get_tool("non_existent")
        assert non_existent is None

    def test_list_tools(self):
        """Test listing all tools."""
        registry = ToolRegistry()
        tools = registry.list_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert all(isinstance(tool, Tool) for tool in tools)

    def test_get_tool_descriptions(self):
        """Test getting tool descriptions."""
        registry = ToolRegistry()
        descriptions = registry.get_tool_descriptions()
        
        assert isinstance(descriptions, list)
        assert len(descriptions) > 0
        assert all("name" in desc for desc in descriptions)
        assert all("description" in desc for desc in descriptions)

    def test_execute_tool(self):
        """Test executing a tool."""
        registry = ToolRegistry()
        
        # Test executing search tool
        result = registry.execute_tool("search", "test query")
        assert isinstance(result, str)
        assert len(result) > 0

        # Test executing non-existent tool
        with pytest.raises(ValueError):
            registry.execute_tool("non_existent", "input")


class TestSearchTool:
    """Test SearchTool class."""

    @pytest.fixture
    def mock_retriever(self):
        """Mock VectorRetriever."""
        with patch('ai_lab.tools.search.VectorRetriever') as mock_retriever_class:
            mock_retriever = Mock()
            mock_retriever.is_loaded.return_value = True
            mock_retriever.retrieve = AsyncMock(return_value=[
                {
                    "content": "Test content about AI",
                    "title": "AI Basics",
                    "source_path": "/test.md",
                    "score": 0.85,
                    "rank": 1
                }
            ])
            mock_retriever_class.return_value = mock_retriever
            yield mock_retriever

    @pytest.mark.asyncio
    async def test_search_success(self, mock_retriever):
        """Test successful search."""
        search_tool = SearchTool()
        result = await search_tool.search("AI basics")

        assert "AI Basics" in result
        assert "Test content about AI" in result
        assert "/test.md" in result

    @pytest.mark.asyncio
    async def test_search_no_index(self, mock_retriever):
        """Test search when index is not loaded."""
        mock_retriever.is_loaded.return_value = False
        search_tool = SearchTool()
        result = await search_tool.search("test query")

        assert "not available" in result
        assert "No document index has been loaded" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self, mock_retriever):
        """Test search with no results."""
        mock_retriever.retrieve.return_value = []
        search_tool = SearchTool()
        result = await search_tool.search("nonexistent topic")

        assert "No relevant information found" in result
        assert "nonexistent topic" in result

    @pytest.mark.asyncio
    async def test_search_with_options(self, mock_retriever):
        """Test search with custom options."""
        search_tool = SearchTool()
        result = await search_tool.search_with_options(
            "test query",
            top_k=3,
            score_threshold=0.7
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_search_results(self):
        """Test formatting search results."""
        search_tool = SearchTool()
        results = [
            {
                "content": "Content 1",
                "title": "Doc 1",
                "source_path": "/doc1.md",
                "score": 0.9,
                "rank": 1
            },
            {
                "content": "Content 2",
                "title": "Doc 2",
                "source_path": "/doc2.md",
                "score": 0.8,
                "rank": 2
            }
        ]

        formatted = search_tool._format_search_results(results, "test query")
        assert "Doc 1" in formatted
        assert "Doc 2" in formatted
        assert "/doc1.md" in formatted
        assert "/doc2.md" in formatted


class TestMathTool:
    """Test MathTool class."""

    def test_math_tool_creation(self):
        """Test creating math tool."""
        math_tool = MathTool()
        assert len(math_tool.allowed_functions) > 0
        assert len(math_tool.allowed_operators) > 0
        assert math_tool.max_expression_length > 0

    @pytest.mark.asyncio
    async def test_basic_math_operations(self):
        """Test basic mathematical operations."""
        math_tool = MathTool()

        # Addition
        result = await math_tool.evaluate("2 + 2")
        assert "4" in result

        # Subtraction
        result = await math_tool.evaluate("10 - 5")
        assert "5" in result

        # Multiplication
        result = await math_tool.evaluate("3 * 4")
        assert "12" in result

        # Division
        result = await math_tool.evaluate("15 / 3")
        assert "5" in result

    @pytest.mark.asyncio
    async def test_advanced_math_operations(self):
        """Test advanced mathematical operations."""
        math_tool = MathTool()

        # Power
        result = await math_tool.evaluate("2^3")
        assert "8" in result

        # Square root
        result = await math_tool.evaluate("sqrt(16)")
        assert "4" in result

        # Parentheses
        result = await math_tool.evaluate("(2 + 3) * 4")
        assert "20" in result

    @pytest.mark.asyncio
    async def test_unsafe_expressions(self):
        """Test rejection of unsafe expressions."""
        math_tool = MathTool()

        # Import statements
        result = await math_tool.evaluate("import os")
        assert "unsafe" in result.lower()

        # Function calls
        result = await math_tool.evaluate("os.system('rm -rf /')")
        assert "unsafe" in result.lower()

        # File operations
        result = await math_tool.evaluate("open('file.txt')")
        assert "unsafe" in result.lower()

    @pytest.mark.asyncio
    async def test_invalid_expressions(self):
        """Test handling of invalid expressions."""
        math_tool = MathTool()

        # Empty expression
        result = await math_tool.evaluate("")
        assert "Empty" in result

        # Unbalanced parentheses
        result = await math_tool.evaluate("(2 + 3")
        assert "unbalanced" in result.lower()

        # Invalid syntax
        result = await math_tool.evaluate("2 + + 3")
        assert "error" in result.lower()

    def test_clean_expression(self):
        """Test expression cleaning."""
        math_tool = MathTool()

        # Remove extra whitespace
        cleaned = math_tool._clean_expression("  2  +  3  ")
        assert cleaned == "2 + 3"

        # Remove newlines
        cleaned = math_tool._clean_expression("2\n+\n3")
        assert cleaned == "2+3"

    def test_safe_expression_check(self):
        """Test safety checking."""
        math_tool = MathTool()

        # Safe expressions
        assert math_tool._is_safe_expression("2 + 3")
        assert math_tool._is_safe_expression("sqrt(16)")
        assert math_tool._is_safe_expression("(2 + 3) * 4")

        # Unsafe expressions
        assert not math_tool._is_safe_expression("import os")
        assert not math_tool._is_safe_expression("__import__('os')")
        assert not math_tool._is_safe_expression("eval('2 + 2')")

    def test_balanced_parentheses(self):
        """Test parentheses balancing check."""
        math_tool = MathTool()

        # Balanced
        assert math_tool._check_balanced_parentheses("(2 + 3)")
        assert math_tool._check_balanced_parentheses("((2 + 3) * 4)")
        assert math_tool._check_balanced_parentheses("2 + 3")

        # Unbalanced
        assert not math_tool._check_balanced_parentheses("(2 + 3")
        assert not math_tool._check_balanced_parentheses("2 + 3)")
        assert not math_tool._check_balanced_parentheses("((2 + 3)")

    @pytest.mark.asyncio
    async def test_percentage_calculations(self):
        """Test percentage calculations."""
        math_tool = MathTool()

        result = await math_tool.calculate_percentage("25% of 80")
        assert "20" in result

        result = await math_tool.calculate_percentage("15% increase on 100")
        assert "115" in result

    @pytest.mark.asyncio
    async def test_equation_solving(self):
        """Test equation solving."""
        math_tool = MathTool()

        result = await math_tool.solve_equation("x + 5 = 10")
        assert "5" in result

        result = await math_tool.solve_equation("2x = 8")
        assert "4" in result


class TestWebStubTool:
    """Test WebStubTool class."""

    def test_web_tool_creation(self):
        """Test creating web stub tool."""
        web_tool = WebStubTool()
        assert len(web_tool.mock_responses) > 0
        assert web_tool.max_url_length > 0

    @pytest.mark.asyncio
    async def test_fetch_url(self):
        """Test fetching from URL."""
        web_tool = WebStubTool()

        result = await web_tool._fetch_url("https://example.com")
        assert "example.com" in result
        assert "stub" in result.lower()

    @pytest.mark.asyncio
    async def test_search_topic(self):
        """Test searching for a topic."""
        web_tool = WebStubTool()

        result = await web_tool._search_topic("artificial intelligence")
        assert "artificial intelligence" in result
        assert "search results" in result.lower()

    @pytest.mark.asyncio
    async def test_fetch_main_method(self):
        """Test main fetch method."""
        web_tool = WebStubTool()

        # URL input
        result = await web_tool.fetch("https://test.com")
        assert "test.com" in result

        # Topic input
        result = await web_tool.fetch("machine learning")
        assert "machine learning" in result

    def test_url_validation(self):
        """Test URL validation."""
        web_tool = WebStubTool()

        # Valid URLs
        assert web_tool._is_valid_url("https://example.com")
        assert web_tool._is_valid_url("http://test.org/path")
        assert web_tool._is_valid_url("https://sub.domain.co.uk")

        # Invalid URLs
        assert not web_tool._is_valid_url("not-a-url")
        assert not web_tool._is_valid_url("ftp://invalid")
        assert not web_tool._is_valid_url("")

    def test_generate_stub_response(self):
        """Test stub response generation."""
        web_tool = WebStubTool()

        response = web_tool._generate_stub_response("https://example.com")
        assert "example.com" in response
        assert "stub" in response.lower()
        assert "content" in response.lower()

    def test_generate_search_results(self):
        """Test search results generation."""
        web_tool = WebStubTool()

        results = web_tool._generate_search_results("test topic")
        assert "test topic" in results
        assert "search results" in results.lower()
        assert "relevant" in results.lower()

    def test_format_web_content(self):
        """Test web content formatting."""
        web_tool = WebStubTool()

        content = web_tool._format_web_content("https://example.com", "Test content")
        assert "example.com" in content
        assert "Test content" in content
        assert "URL" in content

    def test_format_search_results(self):
        """Test search results formatting."""
        web_tool = WebStubTool()

        results = web_tool._format_search_results("test query", ["result1", "result2"])
        assert "test query" in results
        assert "result1" in results
        assert "result2" in results

    @pytest.mark.asyncio
    async def test_web_help(self):
        """Test web help method."""
        web_tool = WebStubTool()

        help_text = await web_tool.get_web_help()
        assert "web tool" in help_text.lower()
        assert "fetch" in help_text.lower()
        assert "search" in help_text.lower()

    def test_url_validation_method(self):
        """Test URL validation method."""
        web_tool = WebStubTool()

        # Valid URL
        result = web_tool.validate_url("https://example.com")
        assert "valid" in result.lower()

        # Invalid URL
        result = web_tool.validate_url("invalid-url")
        assert "invalid" in result.lower()

    def test_domain_extraction(self):
        """Test domain information extraction."""
        web_tool = WebStubTool()

        info = web_tool.extract_domain_info("https://sub.example.com/path")
        assert "example.com" in info
        assert "sub" in info
        assert "path" in info

    @pytest.mark.asyncio
    async def test_simulate_web_request(self):
        """Test web request simulation."""
        web_tool = WebStubTool()

        result = await web_tool.simulate_web_request("https://example.com")
        assert "example.com" in result
        assert "simulated" in result.lower()
        assert "response" in result.lower()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestToolIntegration:
    """Integration tests for tool interactions."""

    @pytest.mark.asyncio
    async def test_tool_registry_with_all_tools(self):
        """Test that all tools work together in the registry."""
        registry = ToolRegistry()
        
        # Test search tool
        search_result = registry.execute_tool("search", "test query")
        assert isinstance(search_result, str)
        
        # Test math tool
        math_result = registry.execute_tool("math", "2 + 2")
        assert "4" in math_result
        
        # Test web tool
        web_result = registry.execute_tool("web", "https://example.com")
        assert "example.com" in web_result

    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        """Test error handling across tools."""
        registry = ToolRegistry()
        
        # Test invalid tool name
        with pytest.raises(ValueError):
            registry.execute_tool("invalid_tool", "input")
        
        # Test invalid input to math tool
        math_result = registry.execute_tool("math", "invalid expression")
        assert "error" in math_result.lower()


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestToolPerformance:
    """Performance tests for tools."""

    @pytest.mark.asyncio
    async def test_math_tool_performance(self):
        """Test math tool performance with complex expressions."""
        math_tool = MathTool()
        
        import time
        start_time = time.time()
        
        # Complex expression
        result = await math_tool.evaluate("(2^10 + sqrt(144)) * (5 - 2)")
        
        execution_time = time.time() - start_time
        assert execution_time < 1.0  # Should complete within 1 second
        assert "1028" in result  # (1024 + 12) * 3

    @pytest.mark.asyncio
    async def test_search_tool_performance(self):
        """Test search tool performance."""
        with patch('ai_lab.tools.search.VectorRetriever') as mock_retriever_class:
            mock_retriever = Mock()
            mock_retriever.is_loaded.return_value = True
            mock_retriever.retrieve = AsyncMock(return_value=[
                {"content": "Test content", "title": "Test", "source_path": "/test.md", "score": 0.9, "rank": 1}
            ])
            mock_retriever_class.return_value = mock_retriever
            
            search_tool = SearchTool()
            
            import time
            start_time = time.time()
            
            result = await search_tool.search("test query")
            
            execution_time = time.time() - start_time
            assert execution_time < 2.0  # Should complete within 2 seconds
            assert "Test content" in result


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestToolErrorHandling:
    """Test error handling in tools."""

    @pytest.mark.asyncio
    async def test_math_tool_edge_cases(self):
        """Test math tool with edge cases."""
        math_tool = MathTool()
        
        # Very long expression
        long_expr = "2 + " * 100 + "1"
        result = await math_tool.evaluate(long_expr)
        assert "too long" in result.lower()
        
        # Division by zero (should be handled gracefully)
        result = await math_tool.evaluate("1 / 0")
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_search_tool_error_handling(self):
        """Test search tool error handling."""
        with patch('ai_lab.tools.search.VectorRetriever') as mock_retriever_class:
            mock_retriever = Mock()
            mock_retriever.is_loaded.return_value = True
            mock_retriever.retrieve = AsyncMock(side_effect=Exception("Database error"))
            mock_retriever_class.return_value = mock_retriever
            
            search_tool = SearchTool()
            result = await search_tool.search("test query")
            
            assert "error" in result.lower()
            assert "Database error" in result

    def test_tool_registry_error_handling(self):
        """Test tool registry error handling."""
        registry = ToolRegistry()
        
        # Test registering tool with duplicate name
        def test_function(input_data: str) -> str:
            return input_data
        
        tool1 = Tool(
            name="duplicate_tool",
            description="First tool",
            function=test_function,
            input_schema={"type": "string"},
            examples=[]
        )
        
        tool2 = Tool(
            name="duplicate_tool",
            description="Second tool",
            function=test_function,
            input_schema={"type": "string"},
            examples=[]
        )
        
        # First registration should work
        registry.register_tool(tool1)
        assert "duplicate_tool" in registry.tools
        
        # Second registration should overwrite
        registry.register_tool(tool2)
        assert registry.tools["duplicate_tool"].description == "Second tool"
