# mcp_server/mcp_http_client.py

import asyncio
import logging
from typing import Dict, Any

from mcp_server.mcp_client import MCPClientWrapper, get_mcp_client

logger = logging.getLogger(__name__)


def run_async(coro):
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        # If already in an event loop (rare in your case), just schedule
        return asyncio.create_task(coro)
    return loop.run_until_complete(coro)


def get_sync_mcp_client(server_url: str = "http://127.0.0.1:8080") -> MCPClientWrapper:
    """
    Get a connected MCPClientWrapper synchronously.

    NOTE:
    - server_url points to the FastMCP base (http://127.0.0.1:8080)
    - MCPClientWrapper itself will call /health on this base URL.
    - Do NOT append /mcp here, because /health is defined at root in sales_mcp_server.py.
    """
    client = run_async(get_mcp_client(server_url))
    return client


def read_resource_sync(uri: str) -> Dict[str, Any]:
    """
    Synchronous wrapper to read an MCP resource.

    Example:
        res = read_resource_sync("sales://data")
    """
    client = get_sync_mcp_client()
    return run_async(client.read_resource(uri))


def call_tool_sync(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous wrapper to call an MCP tool.

    Example:
        res = call_tool_sync("load_sales_data", {"connection_params": {...}})
    """
    client = get_sync_mcp_client()
    return run_async(client.call_tool(tool_name, arguments))
