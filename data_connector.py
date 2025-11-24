from datetime import datetime
import pandas as pd
import logging
from config import DB_CONFIG
from typing import Any, Dict, List, Optional
import hashlib

logger = logging.getLogger(__name__)


class MCPBackedDataStore:
    """
    Multi-Context Protocol (MCP) backed data store with:
    - SAP HANA data loading with singleton pattern
    - Dynamic dialogue state management
    - Context-aware query caching
    - Agent context tracking
    """
    
    STANDARD_FIELDS = [
        "CreationDate", "NetAmount", "OrderQuantity", "TaxAmount",
        "CostAmount", "SoldToParty", "Product", "SalesDocument", "SalesOrganization"
    ]

    def __init__(self):
        self.sales_df = None
        self.agent_contexts = {}
        self.conversation_history = []
        self.data_timestamp = None
        
        # Dynamic dialogue state (no hardcoded fields)
        self.dialogue_state = {
            "entities": {},  # Generic key-value store: {entity_type: {value, timestamp, query}}
            "context_stack": [],  # Stack of recent contexts with full metadata
            "query_results_cache": {}  # Cache with query fingerprints
        }
        
        logger.info("✅ MCPBackedDataStore initialized")

    # ===========================
    # DATA LOADING
    # ===========================
    
    def load_sales_data(self, schema="DSP_CUST_CONTENT", table="SALES_DATA_VIEW"):
        """Load sales data from SAP HANA (only if not already loaded)"""
        
        # ✅ Simple check - if already loaded, skip
        if self.sales_df is not None:
            logger.info("📦 Data already loaded, using cached DataFrame")
            return

        from hdbcli import dbapi
        
        conn = dbapi.connect(**DB_CONFIG)
        full_table = f'"{schema}"."{table}"'
        logger.info(f"🔄 Querying SAP HANA: {full_table}")
        
        df = pd.read_sql(f'SELECT * FROM {full_table}', conn)
        conn.close()
        logger.info(f"✅ Rows loaded: {len(df)}")

        # Clean up numeric columns
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

        # Synthesize MCP standard fields
        sales_df = pd.DataFrame()
        
        # Date field
        sales_df["CreationDate"] = pd.to_datetime(
            df["Date"], errors="coerce"
        ) if "Date" in df else pd.NaT
        
        # NetAmount (Revenue)
        if "Revenue" in df:
            sales_df["NetAmount"] = df["Revenue"]
        elif "Volume" in df and "ASP" in df:
            sales_df["NetAmount"] = df["Volume"] * df["ASP"]
        else:
            sales_df["NetAmount"] = 0.0
        
        # OrderQuantity (Volume)
        sales_df["OrderQuantity"] = df["Volume"] if "Volume" in df else 0.0
        
        # TaxAmount
        sales_df["TaxAmount"] = 0.0
        
        # CostAmount (COGS)
        if "COGS" in df:
            sales_df["CostAmount"] = df["COGS"]
        elif "Volume" in df and "COGS_SP" in df:
            sales_df["CostAmount"] = df["Volume"] * df["COGS_SP"]
        else:
            sales_df["CostAmount"] = 0.0
        
        # Entity fields
        sales_df["SoldToParty"] = df["Customer"] if "Customer" in df else None
        sales_df["Product"] = df["Product"] if "Product" in df else None
        sales_df["SalesOrganization"] = df["Sales Org"] if "Sales Org" in df else None
        sales_df["SalesDocument"] = None

        # Store standardized DataFrame
        self.sales_df = sales_df[self.STANDARD_FIELDS]
        self.data_timestamp = datetime.now()
        
        logger.info(f"✅ Data loaded and processed: {len(self.sales_df)} records")

    def get_sales_data(self):
        """Get loaded sales data"""
        if self.sales_df is None:
            raise ValueError("No data loaded. Call load_sales_data() first.")
        return self.sales_df

    # ===========================
    # AGENT CONTEXT MANAGEMENT
    # ===========================
    
    def update_agent_context(self, agent_name: str, context: dict):
        """Update context for a specific agent"""
        self.agent_contexts[agent_name] = {
            "data": context,
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name
        }
        logger.info(f"Context updated for {agent_name}")

    def get_agent_context(self, agent_name: str):
        """Get context for a specific agent"""
        return self.agent_contexts.get(agent_name)

    def get_all_contexts(self):
        """Get all agent contexts"""
        return self.agent_contexts

    # ===========================
    # CONVERSATION HISTORY
    # ===========================
    
    def add_conversation_turn(self, role: str, content: str):
        """Add a turn to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    # ===========================
    # DYNAMIC DIALOGUE STATE MANAGEMENT
    # ===========================
    
    def update_dialogue_state(self, entities: Dict[str, Any], query: str, response: str):
        """
        Update dialogue state with ANY entities - fully dynamic
        
        Args:
            entities: Dictionary of entities {entity_type: entity_value}
            query: The user's query
            response: The system's response
        """
        if not entities:
            logger.debug("No entities to update in dialogue state")
            return
        
        timestamp = datetime.now().isoformat()
        
        # Update current entities (generic - supports any entity type)
        for entity_type, entity_value in entities.items():
            self.dialogue_state['entities'][entity_type] = {
                'value': entity_value,
                'timestamp': timestamp,
                'query': query
            }
            logger.info(f"🔄 Updated entity: {entity_type} = {entity_value}")
        
        # Add to context stack with full metadata
        context_entry = {
            'query': query,
            'response': response[:500] if response else "",  # Store first 500 chars
            'entities': entities.copy(),
            'timestamp': timestamp,
            'query_type': self._infer_query_type(query)
        }
        
        self.dialogue_state['context_stack'].insert(0, context_entry)
        self.dialogue_state['context_stack'] = self.dialogue_state['context_stack'][:20]  # Keep last 20
        
        # Store in query results cache with fingerprint
        query_fingerprint = self._generate_query_fingerprint(query, entities)
        self.dialogue_state['query_results_cache'][query_fingerprint] = {
            'response': response,
            'entities': entities,
            'timestamp': timestamp
        }
        
        logger.info(f"💾 Dialogue state updated: {len(entities)} entities, stack size: {len(self.dialogue_state['context_stack'])}")
    
    def _generate_query_fingerprint(self, query: str, entities: Dict[str, Any]) -> str:
        """Generate unique fingerprint for query + entities combination"""
        query_normalized = query.lower().strip()
        entities_str = str(sorted(entities.items()))
        fingerprint_input = f"{query_normalized}|{entities_str}"
        return hashlib.md5(fingerprint_input.encode()).hexdigest()[:16]
    
    def _infer_query_type(self, query: str) -> str:
        """Infer query type from query text"""
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ['predict', 'forecast', 'next', 'future', 'will be']):
            return 'prediction'
        elif any(kw in query_lower for kw in ['anomaly', 'unusual', 'outlier', 'detect']):
            return 'anomaly'
        elif any(kw in query_lower for kw in ['compare', 'difference', 'vs', 'versus']):
            return 'comparison'
        elif any(kw in query_lower for kw in ['trend', 'pattern', 'over time']):
            return 'trend'
        elif any(kw in query_lower for kw in ['dashboard', 'chart', 'graph', 'visualization']):
            return 'dashboard'
        else:
            return 'analysis'
    
    # ===========================
    # DIALOGUE STATE QUERIES
    # ===========================
    
    def get_entity(self, entity_type: str) -> Optional[Any]:
        """Get current value of any entity type"""
        entity_info = self.dialogue_state['entities'].get(entity_type)
        return entity_info['value'] if entity_info else None
    
    def get_all_active_entities(self) -> Dict[str, Any]:
        """Get all currently active entities (just values)"""
        return {k: v['value'] for k, v in self.dialogue_state['entities'].items()}
    
    def get_context_stack(self) -> List[Dict]:
        """Get full context stack"""
        return self.dialogue_state['context_stack']
    
    def get_similar_contexts(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Get similar past contexts based on query similarity
        
        Args:
            query: Current query
            top_k: Number of similar contexts to return
            
        Returns:
            List of similar context entries
        """
        query_words = set(query.lower().split())
        
        scored_contexts = []
        for ctx in self.dialogue_state['context_stack']:
            ctx_words = set(ctx['query'].lower().split())
            
            # Calculate Jaccard similarity
            intersection = query_words & ctx_words
            union = query_words | ctx_words
            similarity = len(intersection) / len(union) if union else 0
            
            scored_contexts.append((similarity, ctx))
        
        # Sort by similarity and return top_k
        scored_contexts.sort(reverse=True, key=lambda x: x[0])
        return [ctx for score, ctx in scored_contexts[:top_k] if score > 0.3]
    
    def get_current_dialogue_state(self) -> Dict[str, Any]:
        """
        Get snapshot of current dialogue state
        
        Returns:
            Dictionary with current state summary
        """
        return {
            "entities": self.get_all_active_entities(),
            "context_stack_size": len(self.dialogue_state['context_stack']),
            "entity_stack": [
                {
                    'type': k,
                    'value': v['value'],
                    'timestamp': v['timestamp']
                }
                for k, v in self.dialogue_state['entities'].items()
            ]
        }
    
    def clear_dialogue_state(self):
        """Clear dialogue state (useful for new conversations)"""
        self.dialogue_state = {
            "entities": {},
            "context_stack": [],
            "query_results_cache": {}
        }
        logger.info("🧹 Dialogue state cleared")
    
    # ===========================
    # UTILITY METHODS
    # ===========================
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of loaded data"""
        if self.sales_df is None:
            return {"status": "No data loaded"}
        
        return {
            "total_records": len(self.sales_df),
            "date_range": {
                "start": self.sales_df['CreationDate'].min().isoformat() if pd.notna(self.sales_df['CreationDate'].min()) else None,
                "end": self.sales_df['CreationDate'].max().isoformat() if pd.notna(self.sales_df['CreationDate'].max()) else None
            },
            "total_revenue": float(self.sales_df['NetAmount'].sum()),
            "unique_customers": int(self.sales_df['SoldToParty'].nunique()),
            "unique_products": int(self.sales_df['Product'].nunique()),
            "loaded_at": self.data_timestamp.isoformat() if self.data_timestamp else None
        }
    
    def reset(self):
        """Reset entire store (data, contexts, history)"""
        self.sales_df = None
        self.agent_contexts = {}
        self.conversation_history = []
        self.data_timestamp = None
        self.clear_dialogue_state()
        logger.info("🔄 MCPBackedDataStore reset complete")


# ===========================
# SINGLETON INSTANCE
# ===========================

# Create singleton instance
mcp_store = MCPBackedDataStore()

# Load data once at module import (singleton pattern)
mcp_store.load_sales_data()

logger.info("✅ MCP Store singleton ready")
