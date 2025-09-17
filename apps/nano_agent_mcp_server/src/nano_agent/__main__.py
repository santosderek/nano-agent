#!/usr/bin/env python
"""Nano Agent MCP Server - Main entry point."""

# Apply typing fixes FIRST before any other imports that might use OpenAI SDK
from .modules import typing_fix

import logging
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables from .env file
load_dotenv()

# Import our nano agent tools
from .modules.nano_agent import prompt_nano_agent, get_reliability_report

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the MCP server instance
mcp = FastMCP(
    name="nano-agent",
    instructions="""
    A powerful MCP server with RELIABLE ARCHITECTURE that bridges Model Context Protocol
    with OpenAI's Agent SDK and comprehensive verification systems.

    This server enables autonomous agent execution with guaranteed reliability through:
    - Mandatory file operation verification
    - Preflight checks for file system access
    - Environment detection and risk assessment
    - Fail-fast mechanisms to prevent phantom operations

    The agent uses verified tools that ensure all operations are actually performed
    and independently verified before reporting success.

    Available tools:
    - prompt_nano_agent: Execute a reliable agent with verified file operations
    - get_reliability_report: Get comprehensive system reliability analysis
    """
)

# Register the nano agent tools
mcp.tool()(prompt_nano_agent)
mcp.tool()(get_reliability_report)


def run():
    """Entry point for the nano-agent command."""
    try:
        logger.info("Starting Nano Agent MCP Server...")
        # FastMCP.run() handles its own async context with anyio
        # Don't wrap it in asyncio.run()
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    run()