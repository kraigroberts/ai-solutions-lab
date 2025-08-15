"""
Anthropic Provider for AI Solutions Lab.

Provides integration with Anthropic's Claude models for:
- Chat completion
- Agent execution with tool calling
- Structured output handling
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from ..config import get_settings
from ..tools.registry import ToolRegistry


class AnthropicProvider:
    """Anthropic provider for Claude models."""

    def __init__(self):
        """Initialize Anthropic provider."""
        self.settings = get_settings()

        if not self.settings.has_anthropic():
            raise ValueError("Anthropic API key not configured")

        self.client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        self.tool_registry = ToolRegistry()

        # Model configuration
        self.default_model = "claude-3-sonnet-20240229"
        self.fast_model = "claude-3-haiku-20240307"

        # Capabilities
        self.capabilities = ["chat", "agent", "structured_output", "vision"]

        # Model information
        self.model_info = {
            "provider": "anthropic",
            "default_model": self.default_model,
            "fast_model": self.fast_model,
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
        Send chat messages to Anthropic and get response.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Anthropic model to use (default: claude-3-sonnet)
            temperature: Response randomness (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with response content and metadata
        """
        start_time = time.time()

        try:
            # Use default model if none specified
            model = model or self.default_model

            # Prepare messages for Anthropic (convert OpenAI format to Anthropic format)
            anthropic_messages = self._convert_messages_format(messages)

            # Make API call
            response = await self.client.messages.create(
                model=model,
                messages=anthropic_messages,
                temperature=temperature,
                max_tokens=max_tokens or self.model_info["max_tokens"],
            )

            response_time = time.time() - start_time

            # Extract response content
            content = response.content[0].text if response.content else ""

            # Extract usage information
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }

            return {
                "content": content,
                "model": model,
                "response_time": response_time,
                "usage": usage,
                "model_info": {
                    "provider": "anthropic",
                    "model": model,
                    "finish_reason": getattr(response, "stop_reason", "unknown"),
                },
            }

        except Exception as e:
            raise RuntimeError(f"Anthropic chat error: {str(e)}")

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
                {
                    "role": "user",
                    "content": f"Goal: {goal}\n\nPlease accomplish this goal step by step.",
                }
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
            raise RuntimeError(f"Anthropic agent error: {str(e)}")

    def _convert_messages_format(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI message format to Anthropic format."""
        anthropic_messages: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Convert role names
            if role == "system":
                # Anthropic doesn't have system messages, prepend to first user message
                if anthropic_messages and anthropic_messages[0]["role"] == "user":
                    anthropic_messages[0][
                        "content"
                    ] = f"{content}\n\n{anthropic_messages[0]['content']}"
                else:
                    # If no user message yet, create one
                    anthropic_messages.append({"role": "user", "content": content})
            elif role == "user":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": content})

        return anthropic_messages

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
        """Test Anthropic API connection."""
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

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        try:
            # Anthropic doesn't have a models endpoint like OpenAI
            # Return static information
            return {
                "models": [
                    {
                        "id": self.default_model,
                        "name": "Claude 3 Sonnet",
                        "provider": "anthropic",
                        "capabilities": self.capabilities,
                    },
                    {
                        "id": self.fast_model,
                        "name": "Claude 3 Haiku",
                        "provider": "anthropic",
                        "capabilities": self.capabilities,
                    },
                ]
            }
        except Exception as e:
            return {"error": f"Failed to get model info: {str(e)}"}
