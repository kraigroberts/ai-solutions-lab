"""
Local LLM Provider for AI Solutions Lab.

Provides local LLM inference using llama.cpp for:
- Chat completion with GGUF models
- Agent execution with tool calling
- Offline operation without API keys
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_cpp import Llama

from ..config import get_settings
from ..tools.registry import ToolRegistry


class LocalLLMProvider:
    """Local LLM provider using llama.cpp for GGUF models."""

    def __init__(self):
        """Initialize local LLM provider."""
        self.settings = get_settings()
        self.tool_registry = ToolRegistry()

        # Model configuration
        self.model_path = self.settings.local_model_path
        self.context_size = self.settings.local_model_context_size
        self.max_tokens = self.settings.local_model_max_tokens

        # Initialize model
        self.model = None
        self._load_model()

        # Capabilities
        self.capabilities = ["chat", "agent", "offline", "customizable"]

        # Model information
        self.model_info = {
            "provider": "local",
            "model_path": str(self.model_path),
            "context_size": self.context_size,
            "max_tokens": self.max_tokens,
            "supports_tools": True,
        }

    def _load_model(self) -> None:
        """Load the GGUF model using llama.cpp."""
        try:
            if not self.model_path or not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            # Initialize llama.cpp model
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.context_size,
                n_threads=4,  # Adjust based on available CPU cores
                n_gpu_layers=0,  # Set to >0 if GPU acceleration is available
                verbose=False,
            )

            print(f"Local model loaded successfully: {self.model_path}")

        except Exception as e:
            print(f"Warning: Failed to load local model: {e}")
            print("Local LLM provider will not be available")
            self.model = None

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send chat messages to local LLM and get response.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Response randomness (0.0 to 2.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with response content and metadata
        """
        if not self.model:
            raise RuntimeError("Local model not loaded")

        start_time = time.time()

        try:
            # Prepare prompt for local model
            prompt = self._prepare_prompt(messages)

            # Generate response
            response = self.model(
                prompt,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature,
                stop=["</s>", "\n\n", "User:", "Assistant:"],
                echo=False,
            )

            response_time = time.time() - start_time

            # Extract generated text
            content = response["choices"][0]["text"].strip()

            # Clean up response (remove prompt if it was included)
            if prompt in content:
                content = content.replace(prompt, "").strip()

            return {
                "content": content,
                "model": "local",
                "response_time": response_time,
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(content.split()),
                    "total_tokens": len(prompt.split()) + len(content.split()),
                },
                "model_info": {
                    "provider": "local",
                    "model_path": str(self.model_path),
                    "context_size": self.context_size,
                },
            }

        except Exception as e:
            raise RuntimeError(f"Local LLM chat error: {str(e)}")

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
        if not self.model:
            raise RuntimeError("Local model not loaded")

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
            raise RuntimeError(f"Local LLM agent error: {str(e)}")

    def _prepare_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Prepare prompt for local LLM from messages."""
        prompt_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        # Add final prompt for response
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

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

    def is_available(self) -> bool:
        """Check if local model is available."""
        return self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model:
            return {"error": "No model loaded"}

        return {
            "model_path": str(self.model_path),
            "context_size": self.context_size,
            "max_tokens": self.max_tokens,
            "provider": "local",
            "capabilities": self.capabilities,
        }

    async def test_model(self) -> Dict[str, Any]:
        """Test if the local model is working properly."""
        if not self.model:
            return {"status": "unavailable", "error": "No model loaded"}

        try:
            start_time = time.time()

            # Simple test prompt
            test_prompt = "User: Hello, this is a test message.\nAssistant:"
            response = self.model(
                test_prompt,
                max_tokens=50,
                temperature=0.1,
                stop=["\n", "User:"],
                echo=False,
            )

            response_time = time.time() - start_time

            return {
                "status": "working",
                "response_time": response_time,
                "model_path": str(self.model_path),
                "context_size": self.context_size,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def reload_model(self) -> bool:
        """Reload the model (useful after configuration changes)."""
        try:
            self._load_model()
            return True
        except Exception as e:
            print(f"Failed to reload model: {e}")
            return False
