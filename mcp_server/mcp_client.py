import httpx
import logging
from typing import Optional, Dict, Any
import json

logger = logging.getLogger(__name__)

class MCPClientWrapper:
    """HTTP-based MCP client to connect to FastMCP HTTP server"""
    
    def __init__(self, server_url: str = "http://127.0.0.1:8080"):
        self.server_url = server_url
        self.client: Optional[httpx.AsyncClient] = None
        self.connected = False
        
    async def connect(self):
        """Connect to MCP HTTP server"""
        try:
            self.client = httpx.AsyncClient(timeout=30.0)
            
            # Test connection with health check
            response = await self.client.get(f"{self.server_url}/health")
            
            if response.status_code == 200:
                self.connected = True
                logger.info(f"✅ Connected to MCP server at {self.server_url}")
            else:
                raise Exception(f"Server returned status {response.status_code}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MCP server: {str(e)}")
            raise
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read a resource from MCP server via HTTP
        
        Args:
            uri: Resource URI (e.g., "sales://data")
        """
        try:
            if not self.connected:
                await self.connect()
            
            # Call resource endpoint
            response = await self.client.post(
                f"{self.server_url}/resources/read",
                json={"uri": uri}
            )
            
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
        Call a tool on MCP server via HTTP
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
        """
        try:
            if not self.connected:
                await self.connect()
            
            # Call tool endpoint
            response = await self.client.post(
                f"{self.server_url}/tools/call",
                json={
                    "name": tool_name,
                    "arguments": arguments
                }
            )
            
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
        """List all available resources"""
        try:
            if not self.connected:
                await self.connect()
            
            response = await self.client.get(f"{self.server_url}/resources/list")
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error listing resources: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def list_tools(self) -> Dict[str, Any]:
        """List all available tools"""
        try:
            if not self.connected:
                await self.connect()
            
            response = await self.client.get(f"{self.server_url}/tools/list")
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error listing tools: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.client:
            await self.client.aclose()
            self.connected = False
            logger.info("Disconnected from MCP server")

# Singleton instance
_mcp_client: Optional[MCPClientWrapper] = None

async def get_mcp_client(server_url: str = "http://127.0.0.1:8080") -> MCPClientWrapper:
    """Get or create MCP client singleton"""
    global _mcp_client
    
    if _mcp_client is None:
        _mcp_client = MCPClientWrapper(server_url)
        await _mcp_client.connect()
    
    return _mcp_client
