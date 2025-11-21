from datetime import datetime
import pandas as pd
import logging
from config import DB_CONFIG

logger = logging.getLogger(__name__)

class MCPBackedDataStore:
    STANDARD_FIELDS = [
        "CreationDate", "NetAmount", "OrderQuantity", "TaxAmount",
        "CostAmount", "SoldToParty", "Product", "SalesDocument", "SalesOrganization"
    ]

    def __init__(self):
        self.sales_df = None
        self.agent_contexts = {}
        self.conversation_history = []
        self.data_timestamp = None

    def load_sales_data(self, schema="DSP_CUST_CONTENT", table="SALES_DATA_VIEW"):
        if self.sales_df is not None:
            logger.info("Data already loaded.")
            return

        from hdbcli import dbapi
        conn = dbapi.connect(**DB_CONFIG)
        full_table = f'"{schema}"."{table}"'
        logger.info(f"Querying SAP: {full_table}")
        df = pd.read_sql(f'SELECT * FROM {full_table}', conn)
        conn.close()
        logger.info("Rows loaded: %d", len(df))

        # Clean up numeric columns as per your clustering code
        for col in ["Revenue", "Volume", "ASP", "COGS", "COGS_SP"]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace(r"[^\d\.\-]", "", regex=True)
                )
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        # Stringify key fields
        for col in ["Customer", "Product", "Sales Org"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # Synthesize all MCP fields, using your rules and the provided mapping:
        sales_df = pd.DataFrame()
        # Dates
        sales_df["CreationDate"] = pd.to_datetime(df["Date"], errors="coerce") if "Date" in df else pd.NaT
        # Revenue or fallback
        if "Revenue" in df:
            sales_df["NetAmount"] = df["Revenue"]
        elif "Volume" in df and "ASP" in df:
            sales_df["NetAmount"] = df["Volume"] * df["ASP"]
        else:
            sales_df["NetAmount"] = 0.0
        # Volume is quantity
        sales_df["OrderQuantity"] = df["Volume"] if "Volume" in df else 0.0
        # No taxes available in your view
        sales_df["TaxAmount"] = 0.0
        # COGS
        if "COGS" in df:
            sales_df["CostAmount"] = df["COGS"]
        elif "Volume" in df and "COGS_SP" in df:
            sales_df["CostAmount"] = df["Volume"] * df["COGS_SP"]
        else:
            sales_df["CostAmount"] = 0.0
        # Map IDs
        sales_df["SoldToParty"] = df["Customer"] if "Customer" in df else None
        sales_df["Product"] = df["Product"] if "Product" in df else None
        sales_df["SalesOrganization"] = df["Sales Org"] if "Sales Org" in df else None
        # SalesDocument required by agent code, not present in schema, so use None or blank
        sales_df["SalesDocument"] = None

        # Order columns (always all present)
        self.sales_df = sales_df[self.STANDARD_FIELDS]
        self.data_timestamp = datetime.now()
        logger.info("Standardized dataframe head:\n%s", repr(self.sales_df.head()))

    def get_sales_data(self):
        if self.sales_df is None:
            raise ValueError("No data loaded.")
        return self.sales_df

    def update_agent_context(self, agent_name: str, context: dict):
        self.agent_contexts[agent_name] = {
            "data": context,
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name
        }
        logger.info(f"Context updated for {agent_name}")

    def get_agent_context(self, agent_name: str):
        return self.agent_contexts.get(agent_name)

    def get_all_contexts(self):
        return self.agent_contexts

    def add_conversation_turn(self, role: str, content: str):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

# Singleton
mcp_store = MCPBackedDataStore()
mcp_store.load_sales_data() 
