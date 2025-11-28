import logging
from datetime import datetime, time, date
from typing import Optional, Dict, Any
import os

import pandas as pd
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===========================
# GLOBAL CONTEXT STORE
# ===========================

class ContextStore:
    """Centralized context storage accessible by all agents and tools."""

    def __init__(self):
        self.sales_df: Optional[pd.DataFrame] = None
        self.data_timestamp: Optional[datetime] = None
        self.agent_contexts: Dict[str, Any] = {}
        self.conversation_history: list = []
        self.query_cache: Dict[str, Any] = {}

    def set_sales_data(self, df: pd.DataFrame):
        """Store sales DataFrame with timestamp - keeps ALL columns."""
        self.sales_df = df
        self.data_timestamp = datetime.now()
        logger.info(f"✅ Sales data stored: {len(df)} records, {len(df.columns)} columns at {self.data_timestamp}")
        logger.info(f"📋 Columns: {list(df.columns)}")

    def get_sales_data(self) -> Optional[pd.DataFrame]:
        """Retrieve sales DataFrame with ALL columns."""
        return self.sales_df

    def update_agent_context(self, agent_name: str, context: Dict[str, Any]):
        """Store agent execution results."""
        self.agent_contexts[agent_name] = {
            "data": context,
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
        }
        logger.info(f"Context updated for {agent_name}")

    def get_agent_context(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific agent's context."""
        return self.agent_contexts.get(agent_name)

    def get_all_contexts(self) -> Dict[str, Any]:
        """Get all agent contexts for cross-agent sharing."""
        return self.agent_contexts

    def add_to_history(self, role: str, content: str):
        """Add conversation turn to history."""
        self.conversation_history.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def cache_query_result(self, query: str, result: Any):
        """Cache query results."""
        self.query_cache[query.lower()] = {
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }


# Initialize MCP server and context store
mcp = FastMCP(name="SalesAnalyticsMCP")
context_store = ContextStore()


# ===========================
# CUSTOM ROUTES
# ===========================

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint for monitoring."""
    return JSONResponse(
        {
            "status": "healthy",
            "service": "SalesAnalyticsMCP",
            "timestamp": datetime.now().isoformat(),
            "data_loaded": context_store.get_sales_data() is not None,
        }
    )


@mcp.custom_route("/status", methods=["GET"])
async def status_check(request: Request) -> JSONResponse:
    """Detailed status endpoint."""
    df = context_store.get_sales_data()

    return JSONResponse(
        {
            "status": "online",
            "service": "SalesAnalyticsMCP",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "loaded": df is not None,
                "records": len(df) if df is not None else 0,
                "columns": len(df.columns) if df is not None else 0,
                "column_names": list(df.columns) if df is not None else [],
                "last_loaded": context_store.data_timestamp.isoformat()
                if context_store.data_timestamp
                else None,
            },
            "agents": {
                "active_contexts": len(context_store.agent_contexts),
                "conversation_turns": len(context_store.conversation_history),
            },
        }
    )


@mcp.custom_route("/api/sales-data", methods=["GET"])
async def api_sales_data(request: Request) -> JSONResponse:
    """Return ALL sales data with ALL columns."""
    df = context_store.get_sales_data()

    if df is None:
        return JSONResponse({
            "status": "empty",
            "message": "No data loaded yet",
            "timestamp": datetime.now().isoformat()
        })

    df_json = df.copy()

    # Convert any datetime/date/time objects to ISO strings in ALL columns
    def to_json_safe(v):
        if isinstance(v, (datetime, date, time)):
            return v.isoformat()
        if pd.isna(v):
            return None
        return v

    for col in df_json.columns:
        df_json[col] = df_json[col].apply(to_json_safe)

    return JSONResponse({
        "status": "success",
        "data": df_json.to_dict("records"),
        "shape": {"rows": len(df_json), "columns": len(df_json.columns)},
        "columns": list(df_json.columns),
        "timestamp": context_store.data_timestamp.isoformat()
        if context_store.data_timestamp else None
    })


@mcp.custom_route("/api/load-sales-data", methods=["POST"])
async def api_load_sales_data(request: Request) -> JSONResponse:
    """
    Load ALL columns from SAP Datasphere.
    No column filtering - keeps everything.
    """
    try:
        body = await request.json()
        connection_params = body.get("connection_params", {})
        
        if not connection_params:
            return JSONResponse({
                "status": "error",
                "message": "Missing connection_params in request body"
            })
        
        from hdbcli import dbapi
        
        logger.info("📡 Connecting to SAP Datasphere...")
        conn = dbapi.connect(**connection_params)
        
        logger.info("🔄 Loading ALL data from SALES_DATA_VIEW...")
        df = pd.read_sql("SELECT * FROM DSP_CUST_CONTENT.SALES_DATA_VIEW", conn)
        conn.close()
        
        logger.info(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"📋 Columns: {list(df.columns)}")
        
        # Basic type conversion for known date/numeric columns
        # But keep ALL columns
        date_columns = ["Date", "CreationDate", "SalesDocumentDate", "BillingDocumentDate", "LastChangeDate"]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                logger.info(f"  ✓ Converted {col} to datetime")
        
        numeric_columns = ["Revenue", "Volume", "ASP", "COGS", "COGS_SP", "NetAmount", "OrderQuantity", "TaxAmount", "CostAmount"]
        for col in numeric_columns:
            if col in df.columns:
                # Clean numeric columns (remove commas, etc.)
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(",", "").str.replace(r"[^\d\.\-]", "", regex=True)
                df[col] = pd.to_numeric(df[col], errors="coerce")
                logger.info(f"  ✓ Converted {col} to numeric")
        
        # Fill NaN in numeric columns with 0
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Store with ALL columns
        context_store.set_sales_data(df)
        
        return JSONResponse({
            "status": "success",
            "message": f"Loaded {len(df)} records with {len(df.columns)} columns",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns)
        })
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {str(e)}", exc_info=True)
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })


# ===========================
# RESOURCES (READ-ONLY)
# ===========================

@mcp.resource("sales://data")
def get_sales_data() -> dict:
    """
    Provides access to the main sales DataFrame with ALL columns.
    """
    df = context_store.get_sales_data()

    if df is None:
        return {
            "status": "empty",
            "message": "No data loaded yet",
            "timestamp": datetime.now().isoformat(),
        }

    return {
        "status": "success",
        "data": df.to_dict("records"),
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "timestamp": context_store.data_timestamp.isoformat()
        if context_store.data_timestamp
        else None,
    }


@mcp.resource("sales://data/summary")
def get_data_summary() -> dict:
    """
    Provides quick summary statistics of sales data.
    """
    df = context_store.get_sales_data()

    if df is None:
        return {"status": "empty", "message": "No data loaded"}

    # Try to find revenue column (could be 'Revenue' or 'NetAmount')
    revenue_col = None
    if 'Revenue' in df.columns:
        revenue_col = 'Revenue'
    elif 'NetAmount' in df.columns:
        revenue_col = 'NetAmount'
    
    # Try to find date column
    date_col = None
    if 'Date' in df.columns:
        date_col = 'Date'
    elif 'CreationDate' in df.columns:
        date_col = 'CreationDate'

    summary = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
    }

    if date_col:
        summary["date_range"] = {
            "start": df[date_col].min().isoformat() if pd.notna(df[date_col].min()) else None,
            "end": df[date_col].max().isoformat() if pd.notna(df[date_col].max()) else None,
        }
    
    if revenue_col:
        summary["total_revenue"] = float(df[revenue_col].sum())

    return {
        "status": "success",
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }


@mcp.resource("agent://context/{agent_name}")
def get_agent_context(agent_name: str) -> dict:
    """
    Provides access to specific agent's execution context.
    """
    context = context_store.get_agent_context(agent_name)

    if context is None:
        return {
            "status": "not_found",
            "message": f"No context found for {agent_name}",
            "timestamp": datetime.now().isoformat(),
        }

    return {
        "status": "success",
        "agent": agent_name,
        "context": context,
    }


@mcp.resource("agent://context/all")
def get_all_agent_contexts() -> dict:
    """
    Provides access to all agent contexts.
    """
    all_contexts = context_store.get_all_contexts()

    return {
        "status": "success",
        "contexts": all_contexts,
        "agent_count": len(all_contexts),
        "timestamp": datetime.now().isoformat(),
    }


@mcp.resource("conversation://history")
def get_conversation_history() -> dict:
    """
    Provides access to conversation history.
    """
    return {
        "status": "success",
        "history": context_store.conversation_history,
        "turn_count": len(context_store.conversation_history),
        "timestamp": datetime.now().isoformat(),
    }


# ===========================
# TOOLS (STATE-MUTATING)
# ===========================

@mcp.tool()
def load_sales_data(connection_params: dict) -> dict:
    """
    Tool to load ALL sales data from SAP Datasphere.
    Keeps all columns - no filtering.

    Args:
        connection_params: Database connection parameters
    """
    try:
        from hdbcli import dbapi

        logger.info("📡 Connecting to SAP Datasphere...")
        conn = dbapi.connect(**connection_params)
        
        logger.info("🔄 Loading ALL data...")
        df = pd.read_sql("SELECT * FROM DSP_CUST_CONTENT.SALES_DATA_VIEW", conn)
        conn.close()

        logger.info(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

        # Basic preprocessing - keep ALL columns
        date_columns = ["Date", "CreationDate", "SalesDocumentDate", "BillingDocumentDate", "LastChangeDate"]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        numeric_columns = ["Revenue", "Volume", "ASP", "COGS", "COGS_SP", "NetAmount", "OrderQuantity", "TaxAmount", "CostAmount"]
        for col in numeric_columns:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(",", "").str.replace(r"[^\d\.\-]", "", regex=True)
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        context_store.set_sales_data(df)

        return {
            "status": "success",
            "message": f"Loaded {len(df)} records with {len(df.columns)} columns",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns)
        }

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
        }


@mcp.tool()
def update_agent_context(agent_name: str, context_data: dict) -> dict:
    """Tool for agents to write their execution results to shared context."""
    try:
        context_store.update_agent_context(agent_name, context_data)
        return {
            "status": "success",
            "message": f"Context updated for {agent_name}",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error updating context: {str(e)}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
def add_conversation_turn(role: str, content: str) -> dict:
    """Tool to add conversation turns to history."""
    try:
        context_store.add_to_history(role, content)
        return {
            "status": "success",
            "message": "Added to conversation history",
            "turn_count": len(context_store.conversation_history),
        }
    except Exception as e:
        logger.error(f"Error adding conversation turn: {str(e)}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
def filter_sales_data(filters: dict) -> dict:
    """Tool to filter sales data based on criteria."""
    try:
        df = context_store.get_sales_data()

        if df is None:
            return {"status": "error", "message": "No data loaded"}

        filtered_df = df.copy()

        # Apply filters
        date_col = 'Date' if 'Date' in df.columns else 'CreationDate'
        
        if "year" in filters and date_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[date_col].dt.year == filters["year"]]

        if "month" in filters and date_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[date_col].dt.month == filters["month"]]

        if "product" in filters and "Product" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Product"] == filters["product"]]

        if "customer" in filters and "Customer" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Customer"] == filters["customer"]]

        context_store.update_agent_context(
            "DataFetchingAgent",
            {
                "filtered_data": filtered_df.to_dict("records")[:100],
                "row_count": len(filtered_df),
                "filters_applied": filters,
            },
        )

        return {
            "status": "success",
            "message": f"Filtered to {len(filtered_df)} records",
            "row_count": len(filtered_df),
            "filters": filters,
        }

    except Exception as e:
        logger.error(f"Error filtering data: {str(e)}")
        return {"status": "error", "message": str(e)}


@mcp.tool()
def cache_query_result(query: str, result: dict) -> dict:
    """Tool to cache query results for performance."""
    try:
        context_store.cache_query_result(query, result)
        return {"status": "success", "message": "Result cached successfully"}
    except Exception as e:
        logger.error(f"Error caching query result: {str(e)}")
        return {"status": "error", "message": str(e)}


# ===========================
# SERVER ENTRYPOINT
# ===========================

if __name__ == "__main__":
    port = int(os.getenv("MCP_SERVER_PORT", 8080))
    logger.info("🚀 Starting Sales Analytics MCP Server...")
    logger.info(f"📡 Server will listen on port {port}")
    logger.info("📋 Mode: ALL COLUMNS (no filtering)")

    # Run with HTTP transport and port
    mcp.run(transport="http", host="127.0.0.1", port=port)
