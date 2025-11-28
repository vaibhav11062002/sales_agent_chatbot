from mcp_server.mcp_http_client import call_tool_sync
from config import DB_CONFIG

resp = call_tool_sync("load_sales_data", {"connection_params": DB_CONFIG})
print(resp)
