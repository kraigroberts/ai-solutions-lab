"""
OpenAI Provider for AI Solutions Lab.

Provides integration with OpenAI's GPT models for:
- Chat completion
- Agent execution with tool calling
- Embeddings generation
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from ..config import get_settings
from ..tools.registry import ToolRegistry


class OpenAIProvider:
    """OpenAI provider for GPT models and embeddings."""

    def __init__(self):
        """Initialize OpenAI provider."""
        self.settings = get_settings()

        if not self.settings.has_openai():
            raise ValueError("OpenAI API key not configured")

        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.tool_registry = ToolRegistry()

        # Model configuration
        self.default_model = "gpt-3.5-turbo"
        self.embedding_model = "text-embedding-ada-002"

        # Capabilities
        self.capabilities = ["chat", "agent", "embeddings", "function_calling"]

        # Model information
        self.model_info = {
            "provider": "openai",
            "default_model": self.default_model,
            "embedding_model": self.embedding_model,
            "max_tokens": 4096,
            "supports_tools": True,
        }

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send chat messages to OpenAI and get response.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: OpenAI model to use (default: gpt-3.5-turbo)
            temperature: Response randomness (0.0 to 2.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with response content and metadata
        """
        start_time = time.time()

        try:
            # Use default model if none specified
            model = model or self.default_model

            # Prepare messages for OpenAI
            openai_messages = []
            for msg in messages:
                if msg["role"] in ["user", "assistant", "system"]:
                    openai_messages.append(
                        {"role": msg["role"], "content": msg["content"]}
                    )

            # Make API call
            response = await self.client.chat.completions.create(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens or self.model_info["max_tokens"],
            )

            response_time = time.time() - start_time

            # Extract response content
            content = response.choices[0].message.content or ""

            # Extract usage information
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return {
                "content": content,
                "model": model,
                "response_time": response_time,
                "usage": usage,
                "model_info": {
                    "provider": "openai",
                    "model": model,
                    "finish_reason": response.choices[0].finish_reason,
                },
            }

        except Exception as e:
            raise RuntimeError(f"OpenAI chat error: {str(e)}")

    async def run_agent(
        self, goal: str, tools: Optional[List[str]] = None, max_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Run an agent to accomplish a goal using available tools.

        Args:
            goal: The goal to accomplish
            tools: List of tool names to use
            max_steps: Maximum execution steps

        Returns:
            Dictionary with execution result and steps
        """
        start_time = time.time()

        try:
            # Get available tools
            available_tools = tools or list(self.tool_registry.list_tools().keys())
            tool_descriptions = self.tool_registry.get_tool_descriptions(
                available_tools
            )

            # Prepare system prompt for agent
            system_prompt = self._create_agent_system_prompt(goal, tool_descriptions)

            # Initialize conversation
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Goal: {goal}\n\nPlease accomplish this goal step by step.",
                },
            ]

            steps = []
            current_step = 1

            while current_step <= max_steps:
                # Get LLM response
                response = await self.chat(messages, temperature=0.1)
                content = response["content"]

                # Check if goal is accomplished
                if (
                    "GOAL ACCOMPLISHED" in content.upper()
                    or "TASK COMPLETE" in content.upper()
                ):
                    steps.append(
                        {
                            "step": current_step,
                            "tool": "reasoning",
                            "input": "Final reasoning",
                            "output": content,
                            "timestamp": time.time(),
                        }
                    )
                    break

                # Parse tool usage from response
                tool_usage = self._parse_tool_usage(content)

                if tool_usage:
                    # Execute tool
                    tool_name = tool_usage["tool"]
                    tool_input = tool_usage["input"]

                    try:
                        tool_result = await self.tool_registry.execute_tool(
                            tool_name, tool_input
                        )

                        steps.append(
                            {
                                "step": current_step,
                                "tool": tool_name,
                                "input": tool_input,
                                "output": str(tool_result),
                                "timestamp": time.time(),
                            }
                        )

                        # Add tool result to conversation
                        messages.append({"role": "assistant", "content": content})
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Tool result for {tool_name}: {tool_result}\n\nContinue working toward the goal.",
                            }
                        )

                    except Exception as e:
                        steps.append(
                            {
                                "step": current_step,
                                "tool": tool_name,
                                "input": tool_input,
                                "output": f"Error: {str(e)}",
                                "timestamp": time.time(),
                            }
                        )

                        # Add error to conversation
                        messages.append({"role": "assistant", "content": content})
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Tool {tool_name} failed with error: {str(e)}\n\nPlease try a different approach.",
                            }
                        )
                else:
                    # No tool usage, just reasoning
                    steps.append(
                        {
                            "step": current_step,
                            "tool": "reasoning",
                            "input": "Reasoning step",
                            "output": content,
                            "timestamp": time.time(),
                        }
                    )

                    # Add to conversation
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": "Please continue working toward the goal.",
                        }
                    )

                current_step += 1

            execution_time = time.time() - start_time

            # Extract final result from last step
            final_result = steps[-1]["output"] if steps else "No result generated"

            return {
                "result": final_result,
                "steps": steps,
                "execution_time": execution_time,
                "total_steps": len(steps),
                "tools_used": list(
                    set(step["tool"] for step in steps if step["tool"] != "reasoning")
                ),
            }

        except Exception as e:
            raise RuntimeError(f"OpenAI agent error: {str(e)}")

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        try:
            # OpenAI embeddings API
            response = await self.client.embeddings.create(
                model=self.embedding_model, input=texts
            )

            # Extract embeddings
            embeddings = [data.embedding for data in response.data]
            return embeddings

        except Exception as e:
            raise RuntimeError(f"OpenAI embeddings error: {str(e)}")

    def _create_agent_system_prompt(
        self, goal: str, tool_descriptions: Dict[str, str]
    ) -> str:
        """Create system prompt for agent execution."""
        tools_text = "\n".join(
            [
                f"- {name}: {description}"
                for name, description in tool_descriptions.items()
            ]
        )

        return f"""You are an AI agent tasked with accomplishing the following goal: {goal}

Available tools:
{tools_text}

Instructions:
1. Think step by step about how to accomplish the goal
2. Use tools when appropriate to gather information or perform actions
3. Format tool usage as: TOOL: <tool_name> | INPUT: <input_data>
4. Continue until the goal is accomplished or max steps reached
5. When finished, clearly state "GOAL ACCOMPLISHED" or "TASK COMPLETE"

Example tool usage:
TOOL: search | INPUT: machine learning basics
TOOL: math | INPUT: 15 * 23

Be efficient and use tools strategically to accomplish the goal."""

    def _parse_tool_usage(self, content: str) -> Optional[Dict[str, str]]:
        """Parse tool usage from LLM response."""
        lines = content.split("\n")

        for line in lines:
            if line.strip().startswith("TOOL:"):
                try:
                    # Parse "TOOL: tool_name | INPUT: input_data"
                    parts = line.split("|")
                    if len(parts) == 2:
                        tool_part = parts[0].replace("TOOL:", "").strip()
                        input_part = parts[1].replace("INPUT:", "").strip()

                        return {"tool": tool_part, "input": input_part}
                except:
                    continue

        return None

    async def test_connection(self) -> Dict[str, Any]:
        """Test OpenAI API connection."""
        try:
            start_time = time.time()

            # Simple test call
            response = await self.chat(
                [{"role": "user", "content": "Hello, this is a test message."}]
            )

            response_time = time.time() - start_time

            return {
                "status": "connected",
                "response_time": response_time,
                "model": response.get("model", "unknown"),
                "api_key_valid": True,
            }

        except Exception as e:
            return {"status": "error", "error": str(e), "api_key_valid": False}
