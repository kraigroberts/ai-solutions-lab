"""
Tool Registry for AI Solutions Lab.

Provides functionality for:
- Registering and managing agent tools
- Executing tools with proper error handling
- Tool discovery and information retrieval
- Tool execution logging and monitoring
"""

import asyncio
import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Union

from ..config import get_settings


class Tool:
    """Base class for agent tools."""

    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, str]]] = None,
    ):
        self.name = name
        self.description = description
        self.function = function
        self.input_schema = input_schema or {}
        self.output_schema = output_schema or {}
        self.examples = examples or []

        # Validate function signature
        self._validate_function()

    def _validate_function(self) -> None:
        """Validate that the function has the correct signature."""
        sig = inspect.signature(self.function)
        params = list(sig.parameters.keys())

        # Check if function is async
        if not asyncio.iscoroutinefunction(self.function):
            raise ValueError(f"Tool function {self.name} must be async")

        # Check if function takes input parameter
        if len(params) != 1 or params[0] != "input_data":
            raise ValueError(
                f"Tool function {self.name} must take exactly one parameter named 'input_data'"
            )

    async def execute(self, input_data: str) -> Any:
        """Execute the tool with input data."""
        try:
            start_time = time.time()
            result = await self.function(input_data)
            execution_time = time.time() - start_time

            return {"result": result, "execution_time": execution_time, "success": True}
        except Exception as e:
            return {
                "result": f"Error: {str(e)}",
                "execution_time": 0,
                "success": False,
                "error": str(e),
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "examples": self.examples,
            "async": True,
        }


class ToolRegistry:
    """Registry for managing agent tools."""

    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def register_tool(self, tool: Tool) -> None:
        """Register a new tool."""
        if tool.name in self.tools:
            raise ValueError(f"Tool with name '{tool.name}' already registered")

        self.tools[tool.name] = tool

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool by name."""
        if name in self.tools:
            del self.tools[name]
            return True
        return False

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """List all registered tools with their information."""
        return {name: tool.to_dict() for name, tool in self.tools.items()}

    def get_tool_descriptions(
        self, tool_names: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Get descriptions for specified tools or all tools."""
        if tool_names is None:
            tool_names = list(self.tools.keys())

        descriptions = {}
        for name in tool_names:
            if name in self.tools:
                descriptions[name] = self.tools[name].description

        return descriptions

    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific tool."""
        tool = self.get_tool(name)
        if not tool:
            return None

        return tool.to_dict()

    async def execute_tool(self, name: str, input_data: str) -> Any:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")

        result = await tool.execute(input_data)

        if not result["success"]:
            raise RuntimeError(f"Tool execution failed: {result['result']}")

        return result["result"]

    def _register_default_tools(self) -> None:
        """Register default tools."""
        from .math import MathTool
        from .search import SearchTool
        from .webstub import WebStubTool

        # Register search tool
        search_tool = Tool(
            name="search",
            description="Search for information in the document index. Use this to find relevant documents and information.",
            function=SearchTool().search,
            input_schema={
                "type": "string",
                "description": "Search query to find relevant documents",
            },
            examples=[
                {
                    "input": "machine learning basics",
                    "output": "Information about machine learning fundamentals",
                },
                {
                    "input": "Python programming",
                    "output": "Python programming concepts and examples",
                },
            ],
        )
        self.register_tool(search_tool)

        # Register math tool
        math_tool = Tool(
            name="math",
            description="Evaluate mathematical expressions and perform calculations.",
            function=MathTool().evaluate,
            input_schema={
                "type": "string",
                "description": "Mathematical expression to evaluate",
            },
            examples=[
                {"input": "2 + 2", "output": "4"},
                {"input": "15 * 23", "output": "345"},
                {"input": "sqrt(16)", "output": "4.0"},
            ],
        )
        self.register_tool(math_tool)

        # Register web stub tool
        web_tool = Tool(
            name="web",
            description="Fetch information from web sources (stub implementation).",
            function=WebStubTool().fetch,
            input_schema={
                "type": "string",
                "description": "URL or topic to fetch information about",
            },
            examples=[
                {
                    "input": "https://example.com",
                    "output": "Information about the webpage",
                },
                {"input": "current weather", "output": "Current weather information"},
            ],
        )
        self.register_tool(web_tool)

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())

    def get_tool_count(self) -> int:
        """Get total number of registered tools."""
        return len(self.tools)

    async def test_tool(self, name: str, test_input: str = "test") -> Dict[str, Any]:
        """Test if a tool is working properly."""
        tool = self.get_tool(name)
        if not tool:
            return {"name": name, "status": "not_found", "error": "Tool not registered"}

        try:
            result = await tool.execute(test_input)

            if result["success"]:
                return {
                    "name": name,
                    "status": "working",
                    "execution_time": result["execution_time"],
                    "test_input": test_input,
                }
            else:
                return {
                    "name": name,
                    "status": "error",
                    "error": result["result"],
                    "test_input": test_input,
                }

        except Exception as e:
            return {
                "name": name,
                "status": "error",
                "error": str(e),
                "test_input": test_input,
            }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all tools."""
        results = {}

        for name in self.tools.keys():
            results[name] = await self.test_tool(name)

        return {"total_tools": len(self.tools), "tool_status": results}

    def get_tool_examples(self, name: str) -> List[Dict[str, str]]:
        """Get examples for a specific tool."""
        tool = self.get_tool(name)
        if not tool:
            return []

        return tool.examples

    def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """Search for tools based on description or name."""
        query_lower = query.lower()
        results = []

        for name, tool in self.tools.items():
            # Check name
            if query_lower in name.lower():
                results.append(
                    {"name": name, "match_type": "name", "tool_info": tool.to_dict()}
                )
                continue

            # Check description
            if query_lower in tool.description.lower():
                results.append(
                    {
                        "name": name,
                        "match_type": "description",
                        "tool_info": tool.to_dict(),
                    }
                )

        return results


# Convenience function for quick access
def get_tool_registry() -> ToolRegistry:
    """Get a configured tool registry instance."""
    return ToolRegistry()
