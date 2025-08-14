"""
Command-line interface for AI Solutions Lab.

Provides subcommands for:
- chat: Interactive chat with LLM backends
- rag: Query documents using RAG pipeline
- agent: Run agent with tool calling
- ingest: Build vector index from documents
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from .config import get_settings
from .rag.ingest import DocumentIngester
from .rag.answer import RAGAnswerer
from .llm.router import LLMRouter
from .tools.registry import ToolRegistry


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="ai-lab",
        description="AI Solutions Lab - LLM-powered features with RAG and agent capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ai-lab chat --message "Hello, how are you?"
  ai-lab rag --query "What is machine learning?"
  ai-lab agent --goal "Calculate 15 * 23 and search for Python info"
  ai-lab ingest build --src ./docs --out ./index
        """
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"AI Solutions Lab {__version__}"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Create subparsers for each command
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND"
    )
    
    # Chat command
    chat_parser = subparsers.add_parser(
        "chat",
        help="Chat with LLM backends"
    )
    chat_parser.add_argument(
        "--message", "-m",
        required=True,
        help="Message to send to the LLM"
    )
    chat_parser.add_argument(
        "--backend",
        choices=["local", "openai", "anthropic"],
        help="Override default LLM backend"
    )
    chat_parser.add_argument(
        "--system-prompt",
        help="System prompt to use for the conversation"
    )
    
    # RAG command
    rag_parser = subparsers.add_parser(
        "rag",
        help="Query documents using RAG pipeline"
    )
    rag_parser.add_argument(
        "--query", "-q",
        required=True,
        help="Query to search for in documents"
    )
    rag_parser.add_argument(
        "--index",
        help="Path to vector index (default: ./data/index)"
    )
    rag_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top chunks to retrieve (default: 5)"
    )
    rag_parser.add_argument(
        "--include-sources",
        action="store_true",
        help="Include source documents in output"
    )
    
    # Agent command
    agent_parser = subparsers.add_parser(
        "agent",
        help="Run agent with tool calling"
    )
    agent_parser.add_argument(
        "--goal", "-g",
        required=True,
        help="Goal for the agent to accomplish"
    )
    agent_parser.add_argument(
        "--tools",
        nargs="+",
        help="Specific tools to use (default: all available)"
    )
    agent_parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum number of steps for agent (default: 10)"
    )
    agent_parser.add_argument(
        "--verbose-execution",
        action="store_true",
        help="Show detailed execution steps"
    )
    
    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Build vector index from documents"
    )
    ingest_subparsers = ingest_parser.add_subparsers(
        dest="ingest_command",
        help="Ingest subcommands"
    )
    
    # Ingest build command
    build_parser = ingest_subparsers.add_parser(
        "build",
        help="Build vector index from source documents"
    )
    build_parser.add_argument(
        "--src", "-s",
        required=True,
        type=Path,
        help="Source directory containing documents"
    )
    build_parser.add_argument(
        "--out", "-o",
        required=True,
        type=Path,
        help="Output directory for vector index"
    )
    build_parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Document chunk size (default: 1000)"
    )
    build_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap (default: 200)"
    )
    build_parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild existing index"
    )
    
    # Ingest list command
    list_parser = ingest_subparsers.add_parser(
        "list",
        help="List available indexes"
    )
    list_parser.add_argument(
        "--index-dir",
        type=Path,
        help="Index directory to list (default: ./data/index)"
    )
    
    return parser


async def run_chat(args: argparse.Namespace) -> None:
    """Run chat command."""
    settings = get_settings()
    
    # Override backend if specified
    if args.backend:
        settings.model_backend = args.backend
    
    try:
        router = LLMRouter()
        response = await router.chat(
            message=args.message,
            system_prompt=args.system_prompt
        )
        
        if args.verbose:
            print(f"Backend: {settings.model_backend}")
            print(f"Response time: {response.get('response_time', 'N/A')}s")
        
        print(f"\n{response['content']}")
        
    except Exception as e:
        print(f"Error in chat: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


async def run_rag(args: argparse.Namespace) -> None:
    """Run RAG command."""
    settings = get_settings()
    
    try:
        # Initialize RAG components
        index_path = Path(args.index) if args.index else settings.index_dir
        answerer = RAGAnswerer(index_path=index_path)
        
        # Get answer
        result = await answerer.answer(
            query=args.query,
            top_k=args.top_k
        )
        
        print(f"\nAnswer: {result['answer']}")
        
        if args.include_sources or args.verbose:
            print(f"\nSources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['title']} (score: {source['score']:.3f})")
                if args.verbose:
                    print(f"   Content: {source['content'][:200]}...")
                print()
        
        if args.verbose:
            print(f"Query time: {result.get('query_time', 'N/A')}s")
            print(f"Total chunks retrieved: {len(result['sources'])}")
            
    except Exception as e:
        print(f"Error in RAG: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


async def run_agent(args: argparse.Namespace) -> None:
    """Run agent command."""
    try:
        # Initialize agent components
        tool_registry = ToolRegistry()
        router = LLMRouter()
        
        # Get available tools
        available_tools = args.tools if args.tools else list(tool_registry.list_tools().keys())
        
        if args.verbose:
            print(f"Available tools: {', '.join(available_tools)}")
            print(f"Goal: {args.goal}")
            print(f"Max steps: {args.max_steps}")
            print("-" * 50)
        
        # Run agent
        result = await router.run_agent(
            goal=args.goal,
            tools=available_tools,
            max_steps=args.max_steps
        )
        
        print(f"\nResult: {result['result']}")
        
        if args.verbose_execution or args.verbose:
            print(f"\nExecution steps ({len(result['steps'])}):")
            for i, step in enumerate(result['steps'], 1):
                print(f"{i}. Tool: {step['tool']}")
                print(f"   Input: {step['input']}")
                print(f"   Output: {step['output']}")
                print()
        
        if args.verbose:
            print(f"Total execution time: {result.get('execution_time', 'N/A')}s")
            print(f"Tools used: {', '.join(set(step['tool'] for step in result['steps']))}")
            
    except Exception as e:
        print(f"Error in agent: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


async def run_ingest(args: argparse.Namespace) -> None:
    """Run ingest command."""
    if args.ingest_command == "build":
        await run_ingest_build(args)
    elif args.ingest_command == "list":
        await run_ingest_list(args)
    else:
        print("Error: Must specify ingest subcommand (build or list)", file=sys.stderr)
        sys.exit(1)


async def run_ingest_build(args: argparse.Namespace) -> None:
    """Run ingest build command."""
    try:
        ingester = DocumentIngester()
        
        if args.verbose:
            print(f"Building index from: {args.src}")
            print(f"Output directory: {args.out}")
            print(f"Chunk size: {args.chunk_size}")
            print(f"Chunk overlap: {args.chunk_overlap}")
            print("-" * 50)
        
        # Build index
        result = await ingester.build_index(
            source_dir=args.src,
            output_dir=args.out,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            force=args.force
        )
        
        print(f"Index built successfully!")
        print(f"Documents processed: {result['documents_processed']}")
        print(f"Chunks created: {result['chunks_created']}")
        print(f"Index size: {result['index_size_mb']:.2f} MB")
        print(f"Build time: {result['build_time']:.2f}s")
        
    except Exception as e:
        print(f"Error building index: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


async def run_ingest_list(args: argparse.Namespace) -> None:
    """Run ingest list command."""
    settings = get_settings()
    index_dir = args.index_dir if args.index_dir else settings.index_dir
    
    try:
        if not index_dir.exists():
            print(f"No index directory found at: {index_dir}")
            return
        
        print(f"Available indexes in: {index_dir}")
        print("-" * 50)
        
        for item in index_dir.iterdir():
            if item.is_dir():
                # Check if it's a valid index
                index_files = list(item.glob("*.faiss")) + list(item.glob("*.pkl"))
                if index_files:
                    size_mb = sum(f.stat().st_size for f in index_files) / (1024 * 1024)
                    print(f"📁 {item.name} ({size_mb:.2f} MB)")
                else:
                    print(f"📁 {item.name} (empty)")
            elif item.is_file() and item.suffix in ['.faiss', '.pkl']:
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"📄 {item.name} ({size_mb:.2f} MB)")
        
    except Exception as e:
        print(f"Error listing indexes: {e}", file=sys.stderr)
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
    
    # Set up logging if verbose
    if args.verbose:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    try:
        if args.command == "chat":
            await run_chat(args)
        elif args.command == "rag":
            await run_rag(args)
        elif args.command == "agent":
            await run_agent(args)
        elif args.command == "ingest":
            await run_ingest(args)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def cli_main() -> None:
    """Entry point for console script."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
