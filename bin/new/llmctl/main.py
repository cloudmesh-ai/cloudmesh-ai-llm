#!/usr/bin/env python3
"""Main CLI entry point for llmctl."""

import argparse
import sys
from typing import List, Optional

from llmctl import __version__
from llmctl.commands import launch, proxy, check, tunnel
from llmctl.utils import logger


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="llmctl",
        description="Cloudmesh AI LLM - Unified CLI for managing LLM infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llmctl launch                    # Launch a model interactively
  llmctl proxy start               # Start LiteLLM proxy
  llmctl proxy stop                # Stop LiteLLM proxy
  llmctl proxy probe               # Check all model endpoints
  llmctl tunnel start              # Start SSH tunnel for UVA Kimi
  llmctl tunnel stop               # Stop SSH tunnel
  llmctl tunnel status             # Check tunnel status
  llmctl check                     # Validate environment setup
  llmctl --version                 # Show version
        """,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Add launch subcommand
    launch_parser = subparsers.add_parser(
        "launch",
        help="Launch models on remote GPU hosts",
        description="Interactive model launcher for vLLM containers",
    )
    # Launch has no sub-subcommands currently
    
    # Add proxy subcommand
    proxy_parser = subparsers.add_parser(
        "proxy",
        help="Manage LiteLLM proxy",
        description="Start, stop, and manage the LiteLLM proxy container",
    )
    proxy.add_subparser(proxy_parser)
    
    # Add check subcommand
    check_parser = subparsers.add_parser(
        "check",
        help="Validate environment setup",
        description="Check all prerequisites and configuration",
    )
    # Check has no sub-subcommands currently
    
    # Add tunnel subcommand
    tunnel_parser = subparsers.add_parser(
        "tunnel",
        help="Manage SSH tunnel for UVA Kimi",
        description="Start, stop, and check SSH tunnel for UVA GENAI access",
    )
    tunnel.add_subparser(tunnel_parser)
    
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point.
    
    Args:
        argv: Command line arguments (defaults to sys.argv[1:]).
        
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Set verbose logging if requested
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)
    
    # Route to appropriate command handler
    if args.command == "launch":
        return launch.handle(args)
    elif args.command == "proxy":
        return proxy.handle(args)
    elif args.command == "check":
        return check.handle(args)
    elif args.command == "tunnel":
        return tunnel.handle(args)
    else:
        # No command specified - show help
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())