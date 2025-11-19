import logging
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global context store
class ContextStore:
    """Centralized context storage accessible by all agents"""
    
    def __init__(self):
        self.sales_df: Optional[pd.DataFrame] = None
        self.data_timestamp: Optional[datetime] = None
        self.agent_contexts: Dict[str, Any] = {}
        self.conversation_history: list = []
        self.query_cache: Dict[str, Any] = {}
        
    def set_sales_data(self, df: pd.DataFrame):
        """Store sales DataFrame with timestamp"""
        self.sales_df = df
        self.data_timestamp = datetime.now()
        logger.info(f"Sales data stored: {len(df)} records at {self.data_timestamp}")
    
    def get_sales_data(self) -> Optional[pd.DataFrame]:
        """Retrieve sales DataFrame"""
        return self.sales_df
    
    def update_agent_context(self, agent_name: str, context: Dict[str, Any]):
        """Store agent execution results"""
        self.agent_contexts[agent_name] = {
            "data": context,
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name
        }
        logger.info(f"Context updated for {agent_name}")
    
    def get_agent_context(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific agent's context"""
        return self.agent_contexts.get(agent_name)
    
    def get_all_contexts(self) -> Dict[str, Any]:
        """Get all agent contexts for cross-agent sharing"""
        return self.agent_contexts
    
    def add_to_history(self, role: str, content: str):
        """Add conversation turn to history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def cache_query_result(self, query: str, result: Any):
        """Cache query results"""
        self.query_cache[query.lower()] = {
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

# Initialize MCP server
mcp = FastMCP(name="SalesAnalyticsMCP")

# Initialize context store
context_store = ContextStore()

# ============== CUSTOM ROUTES ==============

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint for monitoring"""
    return JSONResponse({
        "status": "healthy",
        "service": "SalesAnalyticsMCP",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": context_store.get_sales_data() is not None
    })

@mcp.custom_route("/status", methods=["GET"])
async def status_check(request: Request) -> JSONResponse:
    """Detailed status endpoint"""
    df = context_store.get_sales_data()
    
    return JSONResponse({
        "status": "online",
        "service": "SalesAnalyticsMCP",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "loaded": df is not None,
            "records": len(df) if df is not None else 0,
            "last_loaded": context_store.data_timestamp.isoformat() if context_store.data_timestamp else None
        },
        "agents": {
            "active_contexts": len(context_store.agent_contexts),
            "conversation_turns": len(context_store.conversation_history)
        }
    })

# ============== RESOURCES (Read-only data access) ==============

@mcp.resource("sales://data")
def get_sales_data() -> dict:
    """
    Provides access to the main sales DataFrame.
    All agents can read this shared resource.
    """
    df = context_store.get_sales_data()
    
    if df is None:
        return {
            "status": "empty",
            "message": "No data loaded yet",
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "status": "success",
        "data": df.to_dict('records'),
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "timestamp": context_store.data_timestamp.isoformat() if context_store.data_timestamp else None
    }

@mcp.resource("sales://data/summary")
def get_data_summary() -> dict:
    """
    Provides quick summary statistics of sales data.
    Lightweight alternative to full data access.
    """
    df = context_store.get_sales_data()
    
    if df is None:
        return {"status": "empty", "message": "No data loaded"}
    
    return {
        "status": "success",
        "summary": {
            "total_records": len(df),
            "date_range": {
                "start": df['CreationDate'].min().isoformat() if 'CreationDate' in df.columns and pd.notna(df['CreationDate'].min()) else None,
                "end": df['CreationDate'].max().isoformat() if 'CreationDate' in df.columns and pd.notna(df['CreationDate'].max()) else None
            },
            "total_sales": float(df['NetAmount'].sum()) if 'NetAmount' in df.columns else 0,
            "unique_customers": int(df['SoldToParty'].nunique()) if 'SoldToParty' in df.columns else 0,
            "unique_products": int(df['Product'].nunique()) if 'Product' in df.columns else 0
        },
        "timestamp": datetime.now().isoformat()
    }

@mcp.resource("agent://context/{agent_name}")
def get_agent_context(agent_name: str) -> dict:
    """
    Provides access to specific agent's execution context.
    Enables agents to read results from other agents.
    """
    context = context_store.get_agent_context(agent_name)
    
    if context is None:
        return {
            "status": "not_found",
            "message": f"No context found for {agent_name}",
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "status": "success",
        "agent": agent_name,
        "context": context
    }

@mcp.resource("agent://context/all")
def get_all_agent_contexts() -> dict:
    """
    Provides access to all agent contexts.
    Useful for orchestrator to aggregate results.
    """
    all_contexts = context_store.get_all_contexts()
    
    return {
        "status": "success",
        "contexts": all_contexts,
        "agent_count": len(all_contexts),
        "timestamp": datetime.now().isoformat()
    }

@mcp.resource("conversation://history")
def get_conversation_history() -> dict:
    """
    Provides access to conversation history.
    Enables context-aware responses.
    """
    return {
        "status": "success",
        "history": context_store.conversation_history,
        "turn_count": len(context_store.conversation_history),
        "timestamp": datetime.now().isoformat()
    }

# ============== TOOLS (Actions that modify state) ==============

@mcp.tool()
def load_sales_data(connection_params: dict) -> dict:
    """
    Tool to load sales data from SAP Datasphere.
    Only called once or on refresh.
    
    Args:
        connection_params: Database connection parameters
    """
    try:
        from hdbcli import dbapi
        
        conn = dbapi.connect(**connection_params)
        df = pd.read_sql("SELECT * FROM DSP_CUST_CONTENT.SAP_SALES_CUSTOMER", conn)
        conn.close()
        
        # Preprocess data
        date_columns = ['CreationDate', 'SalesDocumentDate', 'BillingDocumentDate', 'LastChangeDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['NetAmount', 'OrderQuantity', 'TaxAmount', 'CostAmount']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.fillna({'NetAmount': 0, 'OrderQuantity': 0})
        
        context_store.set_sales_data(df)
        
        return {
            "status": "success",
            "message": f"Loaded {len(df)} records",
            "rows": len(df),
            "columns": len(df.columns)
        }
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@mcp.tool()
def update_agent_context(agent_name: str, context_data: dict) -> dict:
    """
    Tool for agents to write their execution results to shared context.
    
    Args:
        agent_name: Name of the agent
        context_data: Agent's execution results and state
    """
    try:
        context_store.update_agent_context(agent_name, context_data)
        
        return {
            "status": "success",
            "message": f"Context updated for {agent_name}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating context: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@mcp.tool()
def add_conversation_turn(role: str, content: str) -> dict:
    """
    Tool to add conversation turns to history.
    
    Args:
        role: 'user' or 'assistant'
        content: Message content
    """
    try:
        context_store.add_to_history(role, content)
        
        return {
            "status": "success",
            "message": "Added to conversation history",
            "turn_count": len(context_store.conversation_history)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@mcp.tool()
def filter_sales_data(filters: dict) -> dict:
    """
    Tool to filter sales data based on criteria.
    
    Args:
        filters: Dictionary with filter criteria
                 {'year': 2024, 'product': 'X', 'customer': 'Y'}
    """
    try:
        df = context_store.get_sales_data()
        
        if df is None:
            return {"status": "error", "message": "No data loaded"}
        
        filtered_df = df.copy()
        
        # Apply filters
        if 'year' in filters and 'CreationDate' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['CreationDate'].dt.year == filters['year']]
        
        if 'month' in filters and 'CreationDate' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['CreationDate'].dt.month == filters['month']]
        
        if 'product' in filters and 'Product' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Product'] == filters['product']]
        
        # Store filtered result
        context_store.update_agent_context("DataFetchingAgent", {
            "filtered_data": filtered_df.to_dict('records')[:100],  # Limit to 100 records
            "row_count": len(filtered_df),
            "filters_applied": filters
        })
        
        return {
            "status": "success",
            "message": f"Filtered to {len(filtered_df)} records",
            "row_count": len(filtered_df),
            "filters": filters
        }
        
    except Exception as e:
        logger.error(f"Error filtering data: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@mcp.tool()
def cache_query_result(query: str, result: dict) -> dict:
    """
    Tool to cache query results for performance.
    
    Args:
        query: User query string
        result: Computation result to cache
    """
    try:
        context_store.cache_query_result(query, result)
        
        return {
            "status": "success",
            "message": "Result cached successfully"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# Run server - port is now passed here
if __name__ == "__main__":
    port = int(os.getenv('MCP_SERVER_PORT', 8080))
    logger.info("🚀 Starting Sales Analytics MCP Server...")
    logger.info(f"📡 Server will listen on port {port}")
    
    # Run with HTTP transport and port
    mcp.run(transport="http", host="127.0.0.1", port=port)
