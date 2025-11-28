import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import hashlib

import pandas as pd
from mcp_server.mcp_http_client import read_resource_sync, call_tool_sync

logger = logging.getLogger(__name__)


class MCPBackedDataStore:
    """
    MCP-backed data store.
    - Reads sales data from MCP resource `sales://data`
    - Accepts ALL columns from MCP (no schema enforcement)
    - Optionally triggers MCP tool `load_sales_data` if no data loaded
    - Manages dialogue state, agent contexts, conversation history
    - Stores enriched data (e.g., anomalies) in-process
    """

    def __init__(self):
        self.sales_df: Optional[pd.DataFrame] = None
        self.enriched_data: Dict[str, pd.DataFrame] = {}  # for anomalies, etc.
        self.agent_contexts: Dict[str, Any] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.data_timestamp: Optional[datetime] = None

        self.dialogue_state = {
            "entities": {},
            "context_stack": [],
            "query_results_cache": {}
        }

        logger.info("✅ MCPBackedDataStore initialized (MCP client mode)")

    # ===========================
    # DATA LOADING (FROM MCP)
    # ===========================

    def _fetch_sales_from_mcp(self) -> pd.DataFrame:
        """
        Fetch sales data from MCP resource `sales://data`.
        MCP server must have already run its `load_sales_data` tool at least once.
        
        Returns ALL columns from MCP without schema enforcement.
        """
        logger.info("📡 Fetching sales data from MCP resource 'sales://data'")
        res = read_resource_sync("sales://data")

        status = res.get("status")
        if status != "success":
            msg = res.get("message", "Unknown error")
            raise RuntimeError(f"MCP resource 'sales://data' not ready: {status} ({msg})")

        data = res.get("data", [])
        shape = res.get("shape", {})
        rows = shape.get("rows", len(data))
        cols = shape.get("columns", len(data[0]) if data else 0)
        logger.info(f"✅ MCP returned {rows} rows, {cols} columns")

        df = pd.DataFrame(data)

        if df.empty:
            logger.warning("⚠️ Empty DataFrame received from MCP")
            return df

        logger.info(f"📋 Columns received: {list(df.columns)}")

        # Auto-detect and convert datetime columns
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                logger.info(f"  🔄 Converting {col} to datetime...")
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Auto-detect and convert numeric columns
        numeric_candidates = ['revenue', 'amount', 'quantity', 'volume', 'price', 
                             'cost', 'tax', 'asp', 'cogs', 'margin']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in numeric_candidates):
                if df[col].dtype == 'object':
                    logger.info(f"  🔄 Converting {col} to numeric...")
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(f"✅ Final DataFrame shape: {df.shape}")
        logger.info(f"✅ Columns: {list(df.columns)}")
        
        return df  # Return ALL columns as-is

    def load_sales_data(
        self,
        force_reload: bool = False,
        auto_trigger_tool: bool = False,
        connection_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Load sales data into this process from MCP.

        Parameters:
        - force_reload: ignore local cache and re-fetch from MCP resource.
        - auto_trigger_tool: if True and MCP resource is empty, call `load_sales_data` tool once.
        - connection_params: passed to MCP tool `load_sales_data` if auto_trigger_tool=True.
        """
        if self.sales_df is not None and not force_reload:
            logger.info("📦 Data already loaded in MCPBackedDataStore, using cached DataFrame")
            return

        try:
            self.sales_df = self._fetch_sales_from_mcp()
            self.data_timestamp = datetime.now()
            logger.info(f"✅ Data loaded from MCP: {len(self.sales_df)} records, {len(self.sales_df.columns)} columns")
        except RuntimeError as e:
            logger.warning(f"⚠️ Could not fetch from MCP resource: {e}")
            if not auto_trigger_tool:
                raise

            if connection_params is None:
                raise RuntimeError("auto_trigger_tool=True but no connection_params provided")

            logger.info("🔄 Calling MCP tool 'load_sales_data' to populate server data")
            tool_res = call_tool_sync("load_sales_data", {"connection_params": connection_params})
            if tool_res.get("status") != "success":
                raise RuntimeError(f"load_sales_data tool failed: {tool_res}")

            # Try again after tool call
            self.sales_df = self._fetch_sales_from_mcp()
            self.data_timestamp = datetime.now()
            logger.info(f"✅ Data loaded from MCP after tool call: {len(self.sales_df)} records")

    def get_sales_data(self) -> pd.DataFrame:
        """
        Get loaded sales data with ALL columns.

        - If not yet loaded locally, lazy-fetch from MCP (no auto tool trigger here).
        """
        if self.sales_df is None:
            logger.info("ℹ️ sales_df is None, lazy-loading from MCP (no tool trigger)")
            self.load_sales_data(force_reload=False, auto_trigger_tool=False)
        return self.sales_df

    # ===========================
    # ENRICHED DATA (ANOMALIES, ETC.)
    # ===========================

    def set_enriched_data(self, key: str, data: pd.DataFrame):
        """
        Store enriched data in-process.

        Example keys:
        - 'anomalies'        -> full DF with is_anomaly, anomaly_score, anomaly_reason
        - 'anomaly_records'  -> only anomalous rows
        """
        self.enriched_data[key] = data
        logger.info(f"✅ Enriched data stored: '{key}' ({len(data)} rows)")

    def get_enriched_data(self, key: str) -> Optional[pd.DataFrame]:
        """
        Retrieve enriched data by key.
        """
        return self.enriched_data.get(key)

    # ===========================
    # AGENT CONTEXT MANAGEMENT
    # ===========================

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

    # ===========================
    # CONVERSATION HISTORY
    # ===========================

    def add_conversation_turn(self, role: str, content: str):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    # ===========================
    # DYNAMIC DIALOGUE STATE MANAGEMENT
    # ===========================

    def update_dialogue_state(self, entities: Dict[str, Any], query: str, response: str):
        if not entities:
            logger.debug("No entities to update in dialogue state")
            return

        timestamp = datetime.now().isoformat()

        for entity_type, entity_value in entities.items():
            self.dialogue_state['entities'][entity_type] = {
                'value': entity_value,
                'timestamp': timestamp,
                'query': query
            }
            logger.info(f"🔄 Updated entity: {entity_type} = {entity_value}")

        context_entry = {
            'query': query,
            'response': response[:500] if response else "",
            'entities': entities.copy(),
            'timestamp': timestamp,
            'query_type': self._infer_query_type(query)
        }

        self.dialogue_state['context_stack'].insert(0, context_entry)
        self.dialogue_state['context_stack'] = self.dialogue_state['context_stack'][:20]

        query_fingerprint = self._generate_query_fingerprint(query, entities)
        self.dialogue_state['query_results_cache'][query_fingerprint] = {
            'response': response,
            'entities': entities,
            'timestamp': timestamp
        }

        logger.info(
            f"💾 Dialogue state updated: {len(entities)} entities, "
            f"stack size: {len(self.dialogue_state['context_stack'])}"
        )

    def _generate_query_fingerprint(self, query: str, entities: Dict[str, Any]) -> str:
        query_normalized = query.lower().strip()
        entities_str = str(sorted(entities.items()))
        fingerprint_input = f"{query_normalized}|{entities_str}"
        return hashlib.md5(fingerprint_input.encode()).hexdigest()[:16]

    def _infer_query_type(self, query: str) -> str:
        q = query.lower()
        if any(kw in q for kw in ['predict', 'forecast', 'next', 'future', 'will be']):
            return 'prediction'
        elif any(kw in q for kw in ['anomaly', 'unusual', 'outlier', 'detect']):
            return 'anomaly'
        elif any(kw in q for kw in ['compare', 'difference', 'vs', 'versus']):
            return 'comparison'
        elif any(kw in q for kw in ['trend', 'pattern', 'over time']):
            return 'trend'
        elif any(kw in q for kw in ['dashboard', 'chart', 'graph', 'visualization']):
            return 'dashboard'
        else:
            return 'analysis'

    # ===========================
    # DIALOGUE STATE QUERIES
    # ===========================

    def get_entity(self, entity_type: str) -> Optional[Any]:
        info = self.dialogue_state['entities'].get(entity_type)
        return info['value'] if info else None

    def get_all_active_entities(self) -> Dict[str, Any]:
        return {k: v['value'] for k, v in self.dialogue_state['entities'].items()}

    def get_context_stack(self) -> List[Dict]:
        return self.dialogue_state['context_stack']

    def get_similar_contexts(self, query: str, top_k: int = 3) -> List[Dict]:
        query_words = set(query.lower().split())
        scored: List[Any] = []
        for ctx in self.dialogue_state['context_stack']:
            ctx_words = set(ctx['query'].lower().split())
            inter = query_words & ctx_words
            union = query_words | ctx_words
            sim = len(inter) / len(union) if union else 0
            scored.append((sim, ctx))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [ctx for sim, ctx in scored[:top_k] if sim > 0.3]

    def get_current_dialogue_state(self) -> Dict[str, Any]:
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
        """Generate summary with flexible column detection"""
        if self.sales_df is None:
            return {"status": "No data loaded"}
        
        df = self.sales_df
        summary = {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns)
        }
        
        # Try to find date column
        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break
        
        if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            summary["date_range"] = {
                "start": df[date_col].min().isoformat() if pd.notna(df[date_col].min()) else None,
                "end": df[date_col].max().isoformat() if pd.notna(df[date_col].max()) else None
            }
        
        # Try to find revenue column
        revenue_col = None
        for col in ['Revenue', 'NetAmount', 'Sales', 'Total']:
            if col in df.columns:
                revenue_col = col
                break
        
        if revenue_col:
            summary["total_revenue"] = float(df[revenue_col].sum())
        
        # Try to find customer column
        customer_col = None
        for col in ['Customer', 'SoldToParty', 'CustomerID']:
            if col in df.columns:
                customer_col = col
                break
        
        if customer_col:
            summary["unique_customers"] = int(df[customer_col].nunique())
        
        # Try to find product column
        product_col = None
        for col in ['Product', 'ProductID', 'Material']:
            if col in df.columns:
                product_col = col
                break
        
        if product_col:
            summary["unique_products"] = int(df[product_col].nunique())
        
        summary["loaded_at"] = self.data_timestamp.isoformat() if self.data_timestamp else None
        
        return summary

    def reset(self):
        self.sales_df = None
        self.enriched_data = {}
        self.agent_contexts = {}
        self.conversation_history = []
        self.data_timestamp = None
        self.clear_dialogue_state()
        logger.info("🔄 MCPBackedDataStore reset complete")


# Singleton
mcp_store = MCPBackedDataStore()
logger.info("✅ MCP Store singleton ready (no direct DB access, MCP-backed)")
