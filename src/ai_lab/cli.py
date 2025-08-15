"""Command-line interface for AI Solutions Lab."""

import argparse
import asyncio
import sys
from pathlib import Path

from .config import get_settings


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Solutions Lab - Clean System Design with LLM Integrations and RAG Pipelines"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with mock LLM")
    chat_parser.add_argument("message", help="Message to send")
    chat_parser.add_argument("--system-prompt", help="System prompt")
    chat_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    # RAG command
    rag_parser = subparsers.add_parser("rag", help="RAG document querying")
    rag_parser.add_argument("query", help="Query to search for")
    rag_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top results"
    )
    rag_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    # Agent command
    agent_parser = subparsers.add_parser("agent", help="Run agent with tools")
    agent_parser.add_argument("goal", help="Goal for the agent")
    agent_parser.add_argument("--tools", nargs="+", help="Tools to use")
    agent_parser.add_argument("--max-steps", type=int, default=5, help="Maximum steps")
    agent_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    # Health command
    health_parser = subparsers.add_parser("health", help="Check system health")
    health_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration")
    config_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    return parser


async def run_chat(args: argparse.Namespace) -> None:
    """Run chat command."""
    try:
        if args.verbose:
            print(f"Chat: {args.message}")
            if args.system_prompt:
                print(f"System: {args.system_prompt}")
            print("-" * 50)

        # Mock LLM response
        settings = get_settings()
        if settings.mock_llm_enabled:
            # Simulate processing delay
            await asyncio.sleep(settings.mock_response_delay)

            system_context = (
                f"System: {args.system_prompt}\n" if args.system_prompt else ""
            )
            mock_response = f"{system_context}Mock LLM Response: I understand you said '{args.message}'. This is a simulated response from the AI Solutions Lab mock LLM."
        else:
            mock_response = "Mock LLM is disabled. Please configure a real LLM backend."

        print(f"Response: {mock_response}")

    except Exception as e:
        print(f"Error in chat: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


async def run_rag(args: argparse.Namespace) -> None:
    """Run RAG command."""
    try:
        if args.verbose:
            print(f"RAG Query: {args.query}")
            print(f"Top K: {args.top_k}")
            print("-" * 50)

        # Mock RAG response
        settings = get_settings()
        top_k = args.top_k or settings.top_k

        # Simulate document retrieval
        mock_sources = [
            {
                "title": f"Document {i}",
                "content": f"Mock content about {args.query}",
                "score": 0.9 - (i * 0.1),
                "source_path": f"/mock/doc_{i}.md",
            }
            for i in range(min(top_k, 3))
        ]

        # Generate mock answer
        mock_answer = f"Based on the retrieved documents, here's what I found about '{args.query}': This is a simulated RAG response demonstrating the retrieval-augmented generation pipeline."

        print(f"Answer: {mock_answer}")

        if args.verbose:
            print(f"\nSources ({len(mock_sources)}):")
            for i, source in enumerate(mock_sources, 1):
                print(f"{i}. {source['title']} (score: {source['score']:.3f})")
                print(f"   Content: {source['content'][:100]}...")
                print()

    except Exception as e:
        print(f"Error in RAG: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


async def run_agent(args: argparse.Namespace) -> None:
    """Run agent command."""
    try:
        if args.verbose:
            print(f"Goal: {args.goal}")
            print(f"Tools: {args.tools or 'All available'}")
            print(f"Max steps: {args.max_steps}")
            print("-" * 50)

        # Mock agent execution
        mock_steps = [
            {
                "tool": "search",
                "input": f"Searching for information about: {args.goal}",
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

        mock_result = f"Goal accomplished: {args.goal}. This was achieved through simulated tool execution."

        print(f"Result: {mock_result}")

        if args.verbose:
            print(f"\nExecution steps ({len(mock_steps)}):")
            for step in mock_steps:
                print(f"{step['step']}. Tool: {step['tool']}")
                print(f"   Input: {step['input']}")
                print(f"   Output: {step['output']}")
                print()

    except Exception as e:
        print(f"Error in agent: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


async def run_health(args: argparse.Namespace) -> None:
    """Run health command."""
    try:
        if args.verbose:
            print("Checking system health...")
            print("-" * 50)

        # Mock health check
        import time

        backends = {
            "local": True,
            "openai": False,  # Mock - no API key
            "anthropic": False,  # Mock - no API key
        }

        print(f"Status: healthy")
        print(f"Timestamp: {time.time()}")
        print(f"Version: 0.1.0")
        print(f"Backends:")
        for name, status in backends.items():
            print(f"  {name}: {'✓' if status else '✗'}")

    except Exception as e:
        print(f"Error in health check: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


async def run_config(args: argparse.Namespace) -> None:
    """Run config command."""
    try:
        if args.verbose:
            print("Current configuration:")
            print("-" * 50)

        settings = get_settings()

        print(f"Host: {settings.host}")
        print(f"Port: {settings.port}")
        print(f"Debug: {settings.debug}")
        print(f"Mock LLM: {settings.mock_llm_enabled}")
        print(f"Top K: {settings.top_k}")
        print(f"Chunk Size: {settings.chunk_size}")
        print(f"Chunk Overlap: {settings.chunk_overlap}")

    except Exception as e:
        print(f"Error in config: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


async def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate command
    if args.command == "chat":
        await run_chat(args)
    elif args.command == "rag":
        await run_rag(args)
    elif args.command == "agent":
        await run_agent(args)
    elif args.command == "health":
        await run_health(args)
    elif args.command == "config":
        await run_config(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
