"""
LLM Router for AI Solutions Lab.

Provides unified interfaces for:
- Chat with different LLM backends
- Agent execution with tool calling
- Automatic fallback and routing logic
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union

from ..config import get_settings
from .anthropic import AnthropicProvider
from .llama_cpp import LocalLLMProvider
from .openai import OpenAIProvider


class LLMRouter:
    """Router for different LLM backends with fallback logic."""

    def __init__(self):
        """Initialize the LLM router with available providers."""
        self.settings = get_settings()
        self.providers = {}
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize available LLM providers based on configuration."""
        # Local provider is always available
        try:
            self.providers["local"] = LocalLLMProvider()
        except Exception as e:
            print(f"Warning: Local LLM provider not available: {e}")

        # OpenAI provider if key is available
        if self.settings.has_openai():
            try:
                self.providers["openai"] = OpenAIProvider()
            except Exception as e:
                print(f"Warning: OpenAI provider not available: {e}")

        # Anthropic provider if key is available
        if self.settings.has_anthropic():
            try:
                self.providers["anthropic"] = AnthropicProvider()
            except Exception as e:
                print(f"Warning: Anthropic provider not available: {e}")

        # Ensure we have at least one provider
        if not self.providers:
            raise RuntimeError("No LLM providers available")

    def get_provider(self, backend: Optional[str] = None) -> Any:
        """Get the specified or default LLM provider."""
        backend = backend or self.settings.model_backend

        if backend not in self.providers:
            # Fallback to available providers
            available = list(self.providers.keys())
            if available:
                fallback = available[0]
                print(f"Warning: Backend '{backend}' not available, using '{fallback}'")
                backend = fallback
            else:
                raise RuntimeError("No LLM providers available")

        return self.providers[backend]

    async def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        backend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a chat message and get response.

        Args:
            message: The user's message
            system_prompt: Optional system prompt
            conversation_history: Previous conversation messages
            backend: Specific backend to use

        Returns:
            Dictionary with response content and metadata
        """
        start_time = time.time()

        try:
            provider = self.get_provider(backend)

            # Prepare messages for the provider
            messages = self._prepare_messages(
                message, system_prompt, conversation_history
            )

            # Get response from provider
            response = await provider.chat(messages)

            response_time = time.time() - start_time

            return {
                "content": response["content"],
                "backend": backend or self.settings.model_backend,
                "response_time": response_time,
                "model_info": response.get("model_info", {}),
                "usage": response.get("usage", {}),
            }

        except Exception as e:
            # Try fallback to another provider
            if backend and backend != self.settings.model_backend:
                print(
                    f"Fallback: Trying default backend '{self.settings.model_backend}'"
                )
                return await self.chat(
                    message,
                    system_prompt,
                    conversation_history,
                    backend=self.settings.model_backend,
                )
            raise e

    async def run_agent(
        self,
        goal: str,
        tools: Optional[List[str]] = None,
        max_steps: int = 10,
        backend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run an agent to accomplish a goal using available tools.

        Args:
            goal: The goal to accomplish
            tools: List of tool names to use
            max_steps: Maximum execution steps
            backend: Specific LLM backend to use

        Returns:
            Dictionary with execution result and steps
        """
        start_time = time.time()

        try:
            provider = self.get_provider(backend)

            # Run agent execution
            result = await provider.run_agent(goal, tools, max_steps)

            execution_time = time.time() - start_time

            return {
                "result": result["result"],
                "steps": result["steps"],
                "execution_time": execution_time,
                "backend": backend or self.settings.model_backend,
                "tools_used": result.get("tools_used", []),
            }

        except Exception as e:
            # Try fallback to another provider
            if backend and backend != self.settings.model_backend:
                print(
                    f"Fallback: Trying default backend '{self.settings.model_backend}'"
                )
                return await self.run_agent(
                    goal, tools, max_steps, backend=self.settings.model_backend
                )
            raise e

    def _prepare_messages(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Prepare messages in the format expected by providers."""
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Add current message
        messages.append({"role": "user", "content": message})

        return messages

    def list_available_backends(self) -> List[str]:
        """List available LLM backends."""
        return list(self.providers.keys())

    def get_backend_info(self, backend: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific backend."""
        if backend not in self.providers:
            return None

        provider = self.providers[backend]
        return {
            "name": backend,
            "available": True,
            "model_info": getattr(provider, "model_info", {}),
            "capabilities": getattr(provider, "capabilities", []),
        }

    async def test_backend(self, backend: str) -> Dict[str, Any]:
        """Test if a backend is working properly."""
        try:
            provider = self.providers.get(backend)
            if not provider:
                return {
                    "backend": backend,
                    "status": "unavailable",
                    "error": "Provider not initialized",
                }

            # Test with a simple message
            test_message = "Hello, this is a test message."
            response = await provider.chat([{"role": "user", "content": test_message}])

            return {
                "backend": backend,
                "status": "working",
                "response_time": response.get("response_time", 0),
                "model_info": response.get("model_info", {}),
            }

        except Exception as e:
            return {"backend": backend, "status": "error", "error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all available backends."""
        results = {}

        for backend in self.providers.keys():
            results[backend] = await self.test_backend(backend)

        return {
            "total_backends": len(self.providers),
            "available_backends": self.list_available_backends(),
            "backend_status": results,
        }


# Convenience function for quick access
def get_llm_router() -> LLMRouter:
    """Get a configured LLM router instance."""
    return LLMRouter()
