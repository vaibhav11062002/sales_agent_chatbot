# mcp_server/mcp_client.py

import httpx
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class MCPClientWrapper:
    """HTTP-based MCP client to connect to FastMCP HTTP server."""

    def __init__(self, server_url: str = "http://127.0.0.1:8080"):
        """
        server_url: base URL where FastMCP is running (without /mcp).
        Example: "http://127.0.0.1:8080"
        """
        self.server_url = server_url.rstrip("/")
        self.mcp_base = f"{self.server_url}/mcp"  # FastMCP HTTP manager base
        self.client: Optional[httpx.AsyncClient] = None
        self.connected = False

    async def connect(self):
        """Connect to MCP HTTP server and verify with /health."""
        try:
            self.client = httpx.AsyncClient(timeout=30.0)

            # Use your custom health route defined in sales_mcp_server.py
            health_url = f"{self.server_url}/health"
            logger.info(f"🔎 Checking MCP health at {health_url}")
            response = await self.client.get(health_url)

            if response.status_code == 200:
                self.connected = True
                logger.info(f"✅ Connected to MCP server at {self.server_url}")
            else:
                raise Exception(f"Server returned status {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"❌ Failed to connect to MCP server: {str(e)}")
            raise

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read a resource from MCP server via HTTP.

        For now, map known URIs to custom REST endpoints instead of
        using the FastMCP HTTP manager /mcp/resources/read, which
        returns 404 in your setup.
        """
        try:
            if not self.connected:
                await self.connect()

            # Map URI to our custom REST API
            if uri == "sales://data":
                url = f"{self.server_url}/api/sales-data"
                logger.info(f"📡 Reading sales data via {url}")
                response = await self.client.get(url)
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported URI for REST read: {uri}"
                }

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "error",
                    "message": f"HTTP {response.status_code}: {response.text}"
                }

        except Exception as e:
            logger.error(f"Error reading resource {uri}: {str(e)}")
            return {"status": "error", "message": str(e)}


    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on MCP server via HTTP.
        
        For now, map known tools to custom REST endpoints.
        """
        try:
            if not self.connected:
                await self.connect()

            # Map tool to custom REST API
            if tool_name == "load_sales_data":
                url = f"{self.server_url}/api/load-sales-data"
                logger.info(f"🛠️ Calling tool {tool_name} via {url}")
                response = await self.client.post(url, json=arguments)
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported tool for REST call: {tool_name}"
                }

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "error",
                    "message": f"HTTP {response.status_code}: {response.text}"
                }

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {str(e)}")
            return {"status": "error", "message": str(e)}


    async def list_resources(self) -> Dict[str, Any]:
        """List all available resources via FastMCP HTTP manager."""
        try:
            if not self.connected:
                await self.connect()

            url = f"{self.mcp_base}/resources/list"
            logger.info(f"📡 Listing resources via {url}")
            response = await self.client.get(url)

            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}: {response.text}"}

        except Exception as e:
            logger.error(f"Error listing resources: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def list_tools(self) -> Dict[str, Any]:
        """List all available tools via FastMCP HTTP manager."""
        try:
            if not self.connected:
                await self.connect()

            url = f"{self.mcp_base}/tools/list"
            logger.info(f"📡 Listing tools via {url}")
            response = await self.client.get(url)

            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}: {response.text}"}

        except Exception as e:
            logger.error(f"Error listing tools: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def disconnect(self):
        """Disconnect from MCP server."""
        if self.client:
            await self.client.aclose()
            self.connected = False
            logger.info("🔌 Disconnected from MCP server")


# Singleton instance
_mcp_client: Optional[MCPClientWrapper] = None


async def get_mcp_client(server_url: str = "http://127.0.0.1:8080") -> MCPClientWrapper:
    """Get or create MCP client singleton."""
    global _mcp_client

    if _mcp_client is None:
        _mcp_client = MCPClientWrapper(server_url)
        await _mcp_client.connect()

    return _mcp_client
