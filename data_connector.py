from datetime import datetime
from typing import Optional
import pandas as pd
from config import DB_CONFIG
import logging

logger = logging.getLogger(__name__)

class MCPBackedDataStore:
    """Centralized data store accessible by all agents (MCP-style)"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.sales_df: Optional[pd.DataFrame] = None
        self.data_timestamp: Optional[datetime] = None
        self.agent_contexts: dict = {}
        self.conversation_history: list = []
        self._initialized = True
    
    def load_sales_data(self):
        """Load data from SAP Datasphere"""
        if self.sales_df is not None:
            logger.info("Data already loaded")
            return
        
        try:
            from hdbcli import dbapi
            
            logger.info("Connecting to SAP Datasphere...")
            conn = dbapi.connect(**DB_CONFIG)
            
            logger.info("Fetching data...")
            self.sales_df = pd.read_sql(
                "SELECT * FROM DSP_CUST_CONTENT.SAP_SALES_CUSTOMER", 
                conn
            )
            conn.close()
            
            # Preprocess
            date_columns = ['CreationDate', 'SalesDocumentDate', 'BillingDocumentDate', 'LastChangeDate']
            for col in date_columns:
                if col in self.sales_df.columns:
                    self.sales_df[col] = pd.to_datetime(self.sales_df[col], errors='coerce')
            
            numeric_columns = ['NetAmount', 'OrderQuantity', 'TaxAmount', 'CostAmount']
            for col in numeric_columns:
                if col in self.sales_df.columns:
                    self.sales_df[col] = pd.to_numeric(self.sales_df[col], errors='coerce')
            
            self.sales_df = self.sales_df.fillna({'NetAmount': 0, 'OrderQuantity': 0})
            self.data_timestamp = datetime.now()
            
            logger.info(f"✅ Data loaded: {len(self.sales_df)} records")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_sales_data(self) -> pd.DataFrame:
        """Get sales DataFrame"""
        if self.sales_df is None:
            raise ValueError("Data not loaded. Call load_sales_data() first.")
        return self.sales_df
    
    def update_agent_context(self, agent_name: str, context: dict):
        """Store agent execution results"""
        self.agent_contexts[agent_name] = {
            "data": context,
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name
        }
        logger.info(f"Context updated for {agent_name}")
    
    def get_agent_context(self, agent_name: str) -> Optional[dict]:
        """Get specific agent context"""
        return self.agent_contexts.get(agent_name)
    
    def get_all_contexts(self) -> dict:
        """Get all agent contexts"""
        return self.agent_contexts
    
    def add_conversation_turn(self, role: str, content: str):
        """Add to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

# Singleton instance
mcp_store = MCPBackedDataStore()
