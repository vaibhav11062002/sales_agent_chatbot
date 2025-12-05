import pandas as pd
import logging
from typing import Dict, Any, List
from data_connector import mcp_store
import os
import json
import re
from datetime import datetime
from config import GEMINI_API_KEY
import base64
from io import BytesIO
import numpy as np

logger = logging.getLogger(__name__)

# Import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    sns.set_style("whitegrid")
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. Install with: pip install matplotlib seaborn")

# Import LLM
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LangChain Google GenAI not available")


class DashboardAgent:
    """âœ… 100% LLM-Driven Dynamic Dashboard Agent with LEFT JOIN anomaly integration"""
    
    def __init__(self):
        self.name = "DashboardAgent"
        self.llm = None
        self.analysis_agent_ref = None
        self.chart_data_cache = {}
        self.table_data_cache = None
        
        # Column mappings for flexibility
        self.date_column = None
        self.revenue_column = None
        self.customer_column = None
        self.product_column = None
        self.sales_org_column = None
        self.cost_column = None
        self.tax_column = None
        self.order_id_column = None
        
        if not MATPLOTLIB_AVAILABLE:
            logger.error("[DashboardAgent] Matplotlib not available!")
            return
        
        try:
            if LLM_AVAILABLE:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0.3,  # Slightly higher for creativity
                    api_key=GEMINI_API_KEY
                )
                logger.info("[DashboardAgent] âœ… Initialized with 100% LLM-DRIVEN mode")
            else:
                logger.error("[DashboardAgent] âŒ LLM not available - dashboard generation will fail")
        except Exception as e:
            logger.warning(f"[DashboardAgent] Could not initialize LLM: {e}")
    
    def _detect_columns(self, df: pd.DataFrame):
        """âœ… FULLY DYNAMIC: Detect ALL column names"""
        # Date column
        date_candidates = ['Date', 'CreationDate', 'SalesDocumentDate', 'TransactionDate', 'OrderDate']
        for col in date_candidates:
            if col in df.columns:
                self.date_column = col
                break
        
        # Revenue column
        revenue_candidates = ['Revenue', 'NetAmount', 'Sales', 'TotalSales', 'Amount', 'SalesAmount']
        for col in revenue_candidates:
            if col in df.columns:
                self.revenue_column = col
                break
        
        # Customer column
        customer_candidates = ['Customer', 'SoldToParty', 'CustomerID', 'CustomerId']
        for col in customer_candidates:
            if col in df.columns:
                self.customer_column = col
                break
        
        # Product column
        product_candidates = ['Product', 'ProductID', 'Material', 'ProductId', 'Item']
        for col in product_candidates:
            if col in df.columns:
                self.product_column = col
                break
        
        # Sales Org column
        sales_org_candidates = ['Sales Org', 'SalesOrganization', 'SalesOrg', 'Organization', 'Sales Org Name']
        for col in sales_org_candidates:
            if col in df.columns:
                self.sales_org_column = col
                break
        
        # Cost column
        cost_candidates = ['Cost', 'CostAmount', 'COGS', 'TotalCost']
        for col in cost_candidates:
            if col in df.columns:
                self.cost_column = col
                break
        
        # Tax column
        tax_candidates = ['Tax', 'TaxAmount', 'VAT', 'SalesTax']
        for col in tax_candidates:
            if col in df.columns:
                self.tax_column = col
                break
        
        # Order ID column
        order_id_candidates = ['OrderID', 'SalesDocument', 'InvoiceID', 'TransactionID', 'OrderNumber']
        for col in order_id_candidates:
            if col in df.columns:
                self.order_id_column = col
                break
        
        logger.info(f"[Column Detection] Date: {self.date_column}, Revenue: {self.revenue_column}, Customer: {self.customer_column}, Product: {self.product_column}, Cost: {self.cost_column}")
    
    def execute(self, query: str, entities: dict = None, analysis_agent=None) -> Dict[str, Any]:
        """Main execution method - LLM-driven with LEFT JOIN anomaly integration"""
        logger.info(f"[{self.name}] ðŸŽ Received query: '{query}' | entities={entities}")
        
        if not MATPLOTLIB_AVAILABLE:
            return {"status": "error", "message": "Matplotlib not installed"}
        
        if not self.llm:
            return {"status": "error", "message": "LLM not available - cannot generate dynamic dashboard"}
        
        try:
            self.analysis_agent_ref = analysis_agent
            self.chart_data_cache = {}
            self.table_data_cache = None
            
            # ===================================================================
            # âœ… NEW STRATEGY: Get original data, then LEFT JOIN anomaly data
            # ===================================================================
            
            # Step 1: Always start with original sales data
            df = mcp_store.get_sales_data()
            logger.info(f"[{self.name}] Loaded base data: {len(df)} rows")
            
            # Step 2: Check if anomaly detection has been run
            dialogue_state = mcp_store.get_current_dialogue_state()
            if dialogue_state.get('entities', {}).get('anomalies_detected'):
                logger.info(f"[{self.name}] Anomalies detected - attempting to enrich data...")
                
                # Step 3: Get anomaly enrichment data
                anomaly_df = mcp_store.get_enriched_data('anomalies')
                
                if anomaly_df is not None and not anomaly_df.empty:
                    # Step 4: Identify join key
                    join_key = self._detect_join_key(df, anomaly_df)
                    
                    if join_key:
                        logger.info(f"[{self.name}]Using join key: {join_key}")
                        
                        # Step 5: LEFT JOIN anomaly data
                        df = self._left_join_anomaly_data(df, anomaly_df, join_key)
                        
                        logger.info(f"[{self.name}] Enriched with anomaly data: {df['is_anomaly'].sum()} anomalies")
                    else:
                        logger.warning(f"[{self.name}] i Could not identify join key - using index-based merge")
                        # Fallback: join on index
                        df = df.merge(
                            anomaly_df[['is_anomaly', 'anomaly_score', 'anomaly_reason']], 
                            left_index=True, 
                            right_index=True, 
                            how='left'
                        )
                        df['is_anomaly'] = df['is_anomaly'].fillna(False)
                        df['anomaly_score'] = df['anomaly_score'].fillna(0.0)
                        df['anomaly_reason'] = df['anomaly_reason'].fillna('Normal')
                        logger.info(f"[{self.name}] Index-based merge complete: {df['is_anomaly'].sum()} anomalies")
                else:
                    logger.warning(f"[{self.name}] i Anomalies flagged but enriched data unavailable")
            
            # Step 6: Detect columns (after potential anomaly enrichment)
            self._detect_columns(df)
            
            # ===================================================================
            # Rest of the existing logic continues with unified dataframe
            # ===================================================================
            
            resolved_entities = self._resolve_dashboard_context(query, entities)
            
            logger.info(f"[{self.name}] Final entities: {resolved_entities}")
            
            # Build full context including anomaly info
            full_context = {
                **resolved_entities,
                'dialogue_state': dialogue_state.get('entities', {}),
                'anomalies_detected': dialogue_state.get('entities', {}).get('anomalies_detected', False),
                'anomaly_count': dialogue_state.get('entities', {}).get('anomaly_count', 0),
                'has_anomaly_columns': 'is_anomaly' in df.columns  # Tell LLM if columns exist
            }
            
            logger.info(f"[{self.name}] Full context: anomalies={full_context['anomalies_detected']}, has_columns={'is_anomaly' in df.columns}")
            
            # LLM decides everything
            dashboard_plan = self._llm_create_dashboard_plan(query, full_context, df)
            
            if not dashboard_plan or len(dashboard_plan.get('charts', [])) == 0:
                return {"status": "error", "message": "LLM could not generate a valid dashboard plan"}
            
            logger.info(f"[{self.name}] LLM-Generated Plan: {dashboard_plan.get('title')}")
            logger.info(f"[{self.name}] Charts to generate: {len(dashboard_plan.get('charts', []))}")
            
            # Pre-aggregate and generate
            self._preaggregate_chart_data(dashboard_plan, df, resolved_entities)
            charts = self._generate_charts(dashboard_plan, df, resolved_entities)

            # Generate aggregated data table
            data_table = self._generate_aggregated_table(dashboard_plan, df, resolved_entities)
            
            all_insights = self._generate_comprehensive_insights(query, df, resolved_entities, dashboard_plan, charts)
            dashboard_html = self._create_dashboard_html(charts, dashboard_plan, query, resolved_entities, all_insights, data_table)
            output_path = self._save_dashboard(dashboard_html, resolved_entities, query)
            
            result = {
                "status": "success",
                "dashboard_plan": dashboard_plan,
                "charts_generated": len(charts),
                "output_path": output_path,
                "dashboard_html": dashboard_html
            }
            
            mcp_store.update_agent_context(self.name, {
                "query": query,
                "entities": resolved_entities,
                "results": result,
                "dashboard_plan": dashboard_plan
            })
            
            dashboard_entities = {**resolved_entities, "dashboard_created": True}
            mcp_store.update_dialogue_state(
                dashboard_entities, 
                query, 
                f"Dashboard created with {len(charts)} visualizations at {output_path}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    # ===========================
    # NEW HELPER METHODS FOR ANOMALY JOIN
    # ===========================
    
    def _detect_join_key(self, df: pd.DataFrame, anomaly_df: pd.DataFrame) -> str:
        """Automatically detect the best join key between original and anomaly data"""
        
        # Priority order of potential join keys
        potential_keys = [
            'SalesDocument',      # SAP standard
            'OrderID',
            'TransactionID',
            'InvoiceID',
            'OrderNumber',
            'DocumentNumber',
            'Document'
        ]
        
        for key in potential_keys:
            if key in df.columns and key in anomaly_df.columns:
                logger.info(f"[Join Key Detection] Found: {key}")
                return key
        
        # Check if order_id_column was detected
        if self.order_id_column and self.order_id_column in df.columns and self.order_id_column in anomaly_df.columns:
            logger.info(f"[Join Key Detection] Using detected order_id_column: {self.order_id_column}")
            return self.order_id_column
        
        # âœ… NEW: Check for any common unique identifier columns
        common_cols = set(df.columns) & set(anomaly_df.columns)
        for col in common_cols:
            # Look for columns that look like IDs
            if any(keyword in col.lower() for keyword in ['id', 'document', 'number', 'key', 'order', 'invoice', 'transaction']):
                # Verify it's actually a unique identifier
                if df[col].nunique() == len(df) and anomaly_df[col].nunique() == len(anomaly_df):
                    logger.info(f"[Join Key Detection] Found unique ID column: {col}")
                    return col
        
        # No explicit key found
        logger.warning("[Join Key Detection] No explicit join key found")
        return None

    
    def _left_join_anomaly_data(self, df: pd.DataFrame, anomaly_df: pd.DataFrame, join_key: str) -> pd.DataFrame:
        """Perform LEFT JOIN to merge anomaly data with original data"""
        
        # Select only anomaly columns to merge
        anomaly_cols = ['is_anomaly', 'anomaly_score', 'anomaly_reason']
        merge_cols = [join_key] + [col for col in anomaly_cols if col in anomaly_df.columns]
        
        # Perform LEFT JOIN
        df_enriched = df.merge(
            anomaly_df[merge_cols],
            on=join_key,
            how='left'
        )
        
        # Fill NaN values for rows that weren't anomalies
        if 'is_anomaly' in df_enriched.columns:
            df_enriched['is_anomaly'] = df_enriched['is_anomaly'].fillna(False).astype(bool)
        
        if 'anomaly_score' in df_enriched.columns:
            df_enriched['anomaly_score'] = df_enriched['anomaly_score'].fillna(0.0).astype(float)
        
        if 'anomaly_reason' in df_enriched.columns:
            df_enriched['anomaly_reason'] = df_enriched['anomaly_reason'].fillna('Normal')
        
        logger.info(f"[Left Join] Original: {len(df)} rows â†’ Enriched: {len(df_enriched)} rows")
        
        return df_enriched
    
    # ===========================
    # CONTEXT RESOLUTION
    # ===========================
    
    def _resolve_dashboard_context(self, query: str, entities: dict = None) -> dict:
        """Dynamically resolve entities for dashboard based on query intent"""
        query_lower = query.lower()
        dialogue_state = mcp_store.get_current_dialogue_state()
        all_entities = dialogue_state.get("entities", {})
        entity_stack = dialogue_state.get("context_stack", [])
        
        logger.info("="*60)
        logger.info("DASHBOARD CONTEXT RESOLUTION")
        logger.info("="*60)
        logger.info(f"Query: '{query}'")
        logger.info(f"Incoming entities: {entities}")
        logger.info(f"Dialogue entities: {all_entities}")
        
        # âœ… FIX: Handle "for this/that" references - use incoming entities if provided
        reference_keywords = ['for this', 'for that', 'for it', 'same as', 'about this', 'about that']
        has_reference = any(phrase in query_lower for phrase in reference_keywords)
        
        if has_reference and entities:
            # User said "dashboard for this" - use the entities we just extracted (customer 1002)
            filtered = self._filter_dashboard_entities(entities)
            logger.info(f"Reference query with entities - using: {filtered}")
            return filtered
        
        # Detect if this is a NEW/GLOBAL/OVERALL dashboard request
        global_keywords = [
            'overall', '360', 'all', 'complete', 'comprehensive', 
            'full', 'entire', 'total', 'global', 'new dashboard',
            'create dashboard', 'build dashboard', 'generate dashboard'
        ]
        
        is_global_request = any(keyword in query_lower for keyword in global_keywords)
        
        # If it's a global request AND no specific entity mentioned in query
        if is_global_request:
            # Check if query explicitly mentions a customer/product/year
            has_specific_customer = bool(re.search(r'customer\s+(\d+)', query_lower))
            has_specific_product = bool(re.search(r'product\s+([A-Z0-9]+)', query_lower, re.IGNORECASE))
            has_specific_year = bool(re.search(r'\b(20\d{2})\b', query))
            
            if not (has_specific_customer or has_specific_product or has_specific_year):
                logger.info("Detected GLOBAL dashboard request without specific entities")
                logger.info("Clearing all entity filters for fresh analysis")
                return {}
        
        # Check if it's a comparison query
        if self._is_comparison_query(query_lower):
            comparison_context = self._extract_comparison_context(query, entities, all_entities)
            if comparison_context:
                logger.info(f"Comparison: {comparison_context}")
                return comparison_context
        
        # Use provided entities if they exist (with priority to specific ones)
        if entities:
            # Prioritize product over customer if both exist
            if 'product' in entities or 'product_id' in entities:
                product_filter = {
                    'product_id': entities.get('product_id') or entities.get('product')
                }
                logger.info(f"Using PRODUCT filter (ignoring other filters): {product_filter}")
                return product_filter
            
            if any(k in entities for k in ['customer_id', 'year', 'sales_org']):
                filtered = self._filter_dashboard_entities(entities)
                logger.info(f"Using provided entities: {filtered}")
                return filtered
        
        # Check if there are RELEVANT recent entities
        continuation_keywords = ['that', 'those', 'it', 'them', 'this', 'these']
        has_continuation = any(kw in query_lower for kw in continuation_keywords)
        
        if has_continuation:
            recent_context = self._filter_dashboard_entities(all_entities)
            logger.info(f"Continuation query - using recent context: {recent_context}")
            return recent_context
        
        # Default: Return empty for new standalone queries
        logger.info(f"New standalone query - no entity filters applied")
        return {}

    
    def _is_comparison_query(self, query_lower: str) -> bool:
        return any(kw in query_lower for kw in ['comparison', 'compare', 'vs', 'versus', 'difference between', 'contrast'])
    
    def _extract_comparison_context(self, query: str, entities: dict, all_entities: dict) -> dict:
        comparison_context = {'is_comparison': True, 'comparison_dimension': None, 'comparison_values': []}
        
        if entities and 'years' in entities and isinstance(entities['years'], list) and len(entities['years']) >= 2:
            comparison_context['comparison_dimension'] = 'year'
            comparison_context['comparison_values'] = entities['years']
            return comparison_context
        
        years = re.findall(r'\b(20\d{2})\b', query)
        if len(years) >= 2:
            comparison_context['comparison_dimension'] = 'year'
            comparison_context['comparison_values'] = [int(y) for y in years[:2]]
            return comparison_context
        
        return None
    
    def _get_most_recent_entities(self, entity_stack: list, all_entities: dict) -> dict:
        recent_entities = {}
        if entity_stack:
            for context in reversed(entity_stack):
                if 'year' in context and context['year'] and 'year' not in recent_entities:
                    recent_entities['year'] = context['year']
                    break
        if not recent_entities and all_entities.get('year'):
            recent_entities['year'] = all_entities['year']
        return recent_entities
    
    def _filter_dashboard_entities(self, entities: dict) -> dict:
        """Filter entities to only keep dashboard-relevant keys including aggregation context"""
        # ✅ UPDATED: Added aggregation_type, aggregation, and metric to preserve performance context
        DASHBOARD_KEYS = [
            'customer_id', 
            'product_id', 
            'year', 
            'sales_org', 
            'aggregation_type',  # ✅ NEW: Preserve "highest"/"least"/"most" etc.
            'aggregation',       # ✅ NEW: Alternative field name for aggregation
            'metric'             # ✅ NEW: Preserve metric type (revenue, orders, etc.)
        ]
        return {k: v for k, v in entities.items() if k in DASHBOARD_KEYS and v}

    
    def _filter_dataframe(self, df: pd.DataFrame, entities: dict) -> pd.DataFrame:
        """
        Filter dataframe - anomaly columns preserved through LEFT JOIN
        """
        filtered = df.copy()
        
        # Ensure date column is datetime
        if self.date_column and self.date_column in filtered.columns:
            filtered[self.date_column] = pd.to_datetime(filtered[self.date_column], errors='coerce')
        
        if entities.get('is_comparison'):
            comparison_dim = entities.get('comparison_dimension')
            comparison_values = entities.get('comparison_values', [])
            
            if comparison_dim == 'year' and comparison_values and self.date_column:
                years_int = [int(y) if isinstance(y, str) else y for y in comparison_values]
                filtered = filtered[filtered[self.date_column].dt.year.isin(years_int)]
        else:
            # Handle year filtering
            if 'year' in entities and self.date_column:
                year_int = int(entities['year']) if isinstance(entities['year'], str) else entities['year']
                before_count = len(filtered)
                filtered = filtered[filtered[self.date_column].dt.year == year_int]
                logger.info(f"[Filter] Year {year_int}: {before_count} â†’ {len(filtered)} rows")
            
            # Handle customer_id filtering with type coercion
            if 'customer_id' in entities and self.customer_column:
                customer_id = entities['customer_id']
                before_count = len(filtered)
                
                if isinstance(customer_id, list):
                    mask = filtered[self.customer_column].isin(customer_id)
                    
                    if mask.sum() == 0:
                        if filtered[self.customer_column].dtype in ['int64', 'float64']:
                            customer_id_numeric = [int(c) if isinstance(c, str) else c for c in customer_id]
                            mask = filtered[self.customer_column].isin(customer_id_numeric)
                            logger.info(f"[Filter] Converted customer_id list to numeric: {customer_id_numeric}")
                        else:
                            customer_id_str = [str(c) for c in customer_id]
                            mask = filtered[self.customer_column].isin(customer_id_str)
                            logger.info(f"[Filter] Converted customer_id list to string: {customer_id_str}")
                    
                    filtered = filtered[mask]
                else:
                    mask = filtered[self.customer_column] == customer_id
                    
                    if mask.sum() == 0:
                        if filtered[self.customer_column].dtype in ['int64', 'float64']:
                            try:
                                customer_id_numeric = int(customer_id) if isinstance(customer_id, str) else customer_id
                                mask = filtered[self.customer_column] == customer_id_numeric
                                logger.info(f"[Filter] Converted customer_id to numeric: {customer_id_numeric}")
                            except (ValueError, TypeError):
                                logger.warning(f"[Filter] Could not convert customer_id '{customer_id}' to numeric")
                        else:
                            customer_id_str = str(customer_id)
                            mask = filtered[self.customer_column] == customer_id_str
                            logger.info(f"[Filter] Converted customer_id to string: {customer_id_str}")
                    
                    filtered = filtered[mask]
                
                logger.info(f"[Filter] Customer filter: {before_count} â†’ {len(filtered)} rows")
            
            # Handle product_id filtering with type coercion
            if 'product_id' in entities and self.product_column:
                product_id = entities['product_id']
                before_count = len(filtered)
                
                if isinstance(product_id, list):
                    mask = filtered[self.product_column].isin(product_id)
                    
                    if mask.sum() == 0:
                        product_id_str = [str(p) for p in product_id]
                        mask = filtered[self.product_column].isin(product_id_str)
                        logger.info(f"[Filter] Converted product_id list to string: {product_id_str}")
                    
                    filtered = filtered[mask]
                else:
                    mask = filtered[self.product_column] == product_id
                    
                    if mask.sum() == 0:
                        product_id_str = str(product_id)
                        mask = filtered[self.product_column] == product_id_str
                        logger.info(f"[Filter] Converted product to string: {product_id_str}")
                    
                    filtered = filtered[mask]
                
                logger.info(f"[Filter] Product filter: {before_count} â†’ {len(filtered)} rows")
            
            # Handle sales_org filtering
            if 'sales_org' in entities and self.sales_org_column:
                before_count = len(filtered)
                filtered = filtered[filtered[self.sales_org_column] == str(entities['sales_org'])]
                logger.info(f"[Filter] Sales org filter: {before_count} â†’ {len(filtered)} rows")
        
        logger.info(f"[Filter] FINAL: {len(df)} â†’ {len(filtered)} rows")
        return filtered
    
    # ===========================
    # 100% LLM-DRIVEN DASHBOARD PLANNING
    # ===========================
    
    def _llm_create_dashboard_plan(self, query: str, full_context: dict, df: pd.DataFrame) -> Dict[str, Any]:
        """LLM DECIDES EVERYTHING with dynamic anomaly column awareness and SEMANTIC CONTEXT"""
        
        # âœ… NEW: Extract semantic context from previous analysis
        semantic_context = full_context.get('_semantic_context', '')
        previous_query = full_context.get('_previous_query', '')
        previous_results = full_context.get('_previous_results', {})
        
        # Extract filter entities
        entities = {k: v for k, v in full_context.items() if k in ['customer_id', 'product_id', 'year', 'sales_org', 'is_comparison', 'comparison_dimension', 'comparison_values']}
        
        # Prepare context for LLM
        filtered_df = self._filter_dataframe(df, entities)
        
        if len(filtered_df) == 0:
            logger.warning("[LLM] No data after filtering!")
            return {"title": "No Data", "description": "No data available", "charts": []}
        
        # Check if anomaly columns actually exist in filtered data
        has_anomaly_cols = 'is_anomaly' in filtered_df.columns
        
        # Get data statistics
        data_stats = self._get_comprehensive_data_stats(filtered_df, full_context)
        
        # Get available columns with examples
        column_info = self._get_column_information(filtered_df)
        
        # Build dynamic context description (now includes semantic context)
        context_desc = self._build_context_description(query, full_context, filtered_df)
        
        # âœ… NEW: Build semantic understanding section
        semantic_instruction = ""
        if semantic_context or previous_query:
            semantic_instruction = f"""
    **CRITICAL: SEMANTIC CONTEXT FROM PREVIOUS ANALYSIS**

    Previous User Question: "{previous_query}"

    Analysis Result: {semantic_context[:400]}

    **DASHBOARD INTERPRETATION RULES:**
    """
            
            # Detect performance level from semantic context
            semantic_lower = semantic_context.lower() if semantic_context else ""
            previous_lower = previous_query.lower() if previous_query else ""
            
            # Check both semantic context and previous query for performance indicators
            combined_text = (semantic_lower + " " + previous_lower).lower()
            
            if any(word in combined_text for word in ['lowest', 'worst', 'bottom', 'minimum', 'poorest', 'weakest']):
                semantic_instruction += """
    - This entity has the **LOWEST/WORST performance** (underperformer)
    - Dashboard title MUST reflect LOW performance (e.g., "Low Revenue Customer Investigation", "Underperforming Product Analysis")
    - Focus on: Why is performance poor? What's causing issues? Where are improvement opportunities?
    - Tone: Diagnostic, problem-solving, identifying challenges
    - **FORBIDDEN TERMS**: Do NOT use "high-revenue", "top performer", "critical client", "key account", "success story"
    - **REQUIRED TERMS**: Use "low performance", "underperforming", "challenges", "improvement needed"
    - Description should explain WHY this entity needs attention and improvement
    """
            
            elif any(word in combined_text for word in ['highest', 'best', 'top', 'maximum', 'leading', 'strongest', 'first']):
                semantic_instruction += """
    - This entity has the **HIGHEST/BEST performance** (top performer)
    - Dashboard title MUST reflect HIGH performance (e.g., "Top Revenue Generator", "Leading Product Performance")
    - Focus on: What drives success? How to maintain growth? What are expansion opportunities?
    - Tone: Success-oriented, celebratory, strategic growth
    - **REQUIRED TERMS**: Use "high-revenue", "top performer", "key account", "leading", "success driver"
    - **FORBIDDEN TERMS**: Do NOT use "low performance", "underperforming", "struggling"
    - Description should highlight strengths and success factors
    """
            
            elif any(word in combined_text for word in ['average', 'median', 'middle', 'moderate']):
                semantic_instruction += """
    - This entity has **AVERAGE/MEDIAN performance**
    - Dashboard should show balanced view of strengths and weaknesses
    - Focus on: Optimization opportunities, benchmarking against top performers, growth potential
    - Tone: Balanced, analytical, opportunity-focused
    """
            
            else:
                semantic_instruction += """
    - Standard analysis requested
    - Provide comprehensive, balanced view of this entity
    - Focus on: Key metrics, trends, patterns, and actionable insights
    """
            
            semantic_instruction += "\n**REMEMBER**: The dashboard narrative MUST match the previous analysis findings. Do not contradict what was already discovered.\n"
        
        # Build dynamic column mapping for LLM
        column_mapping = f"""
    **DETECTED COLUMN NAMES (USE THESE EXACT NAMES):**
    - Date Column: `{self.date_column}`
    - Revenue Column: `{self.revenue_column}`
    - Customer Column: `{self.customer_column}`
    - Product Column: `{self.product_column}`
    - Sales Org Column: `{self.sales_org_column}`
    - Cost Column: `{self.cost_column}`
    - Tax Column: `{self.tax_column}`
    - Order ID Column: `{self.order_id_column}`
    """
        
        # âœ… DYNAMIC ANOMALY INSTRUCTION based on actual column availability
        # DYNAMIC ANOMALY INSTRUCTION - Only mention availability, don't force it
        anomaly_instruction = ""
        if full_context.get('anomalies_detected') and has_anomaly_cols:
            anomaly_instruction = """
        **NOTE: Anomaly Data Available (Optional)**
        The dataset includes anomaly detection results with columns: `is_anomaly`, `anomaly_score`, `anomaly_reason`.

        **IMPORTANT RULE:**
        - ONLY create anomaly-focused charts if the user's query explicitly asks about anomalies
        - If the query is about product/customer performance WITHOUT mentioning anomalies, create standard business dashboards (revenue trends, top customers, sales patterns, etc.)
        - The presence of anomaly columns does NOT mean you should focus on anomalies
        - Let the USER QUERY guide the dashboard focus, not the data availability

        **Examples:**
        - "dashboard for this product" → Product performance dashboard (revenue, customers, trends)
        - "show anomalies for this product" → Anomaly analysis dashboard
        """
        elif full_context.get('anomalies_detected') and not has_anomaly_cols:
            anomaly_instruction = ""  # Remove the warning, just ignore it

        
        # 100% OPEN-ENDED PROMPT - LLM IS TOTALLY FREE
        prompt = f"""You are a creative data visualization and analytics expert. Design a PERFECT, UNIQUE dashboard for this user query.

**USER QUERY:** "{query}"

{semantic_instruction}

**CONTEXT:**
{context_desc}

{anomaly_instruction}

**DATA SUMMARY:**
{data_stats}

{column_mapping}

**ALL AVAILABLE COLUMNS:**
{column_info}

**YOUR MISSION:**
Create a dashboard with:
1. **6-10 VISUALIZATIONS** that PERFECTLY answer the user's question
2. **1 DATA TABLE (MANDATORY)** with top 10 aggregated records relevant to the query

You have COMPLETE CREATIVE FREEDOM:
- **Choose ANY chart types:** line, bar, pie, area, scatter
- **Design ANY metrics:** growth rates, percentages, ratios, rankings, distributions
- **Create ANY transformations:** groupby, pivot, window functions, custom calculations
- **Focus on INSIGHTS:** What would be most valuable to see?

**CRITICAL ACCURACY REQUIREMENTS:**
- Dashboard title and description MUST match the semantic context above
- If entity is LOW/WORST performer, title must reflect that (e.g., "Low Performance Analysis")
- If entity is HIGH/BEST performer, title must reflect that (e.g., "Top Performer Analysis")
- DO NOT contradict previous analysis findings
- Use appropriate terminology based on performance level
- Maintain factual accuracy throughout
- **ALL content (title, description, strategies, recommendations) MUST be LLM-generated and dynamic**

**IMPORTANT CHART RULES:**
- Each chart should show ONE metric on Y-axis (NOT multiple columns)
- For multi-metric visualizations, create SEPARATE charts
- Use transformations to create calculated fields BEFORE plotting
- Be creative - avoid generic dashboards!
- ONLY use columns that exist in the dataframe
- Y-axis must be a SINGLE column name (string), never a list

**DATA TABLE REQUIREMENTS (MANDATORY):**
- The data_table field is **MANDATORY** - you MUST include it in your JSON response
- Table should show TOP 10 aggregated entries specifically relevant to the user query
- YOU decide which dimensions to group by (e.g., customer, product, date, region)
- YOU decide which metrics to aggregate (e.g., sum revenue, count orders, avg price)
- Maximum 3 columns in the table (1 dimension + 2 metrics OR 1 dimension + 1 metric)
- The table should be highly specific to the user's question
- Table must be sortable by the primary metric (descending)

**DEFAULT TABLE FALLBACKS (if you cannot think of a relevant table):**
- For product analysis: Top 10 customers by revenue for this product
- For customer analysis: Top 10 products by revenue for this customer  
- For overall analysis: Top 10 products or customers by revenue
- **NEVER skip the data_table field - it is REQUIRED**

**JSON FORMAT (REQUIRED):**
{{
  "title": "Dashboard Title (MUST MATCH SEMANTIC CONTEXT and USER QUERY FOCUS)",
  "description": "What this dashboard reveals (MUST BE FACTUALLY ACCURATE and DYNAMICALLY GENERATED)",
  "strategies": "Strategic insights paragraph - what approach should be taken based on this analysis (LLM-GENERATED, 2-3 sentences)",
  "recommendations": "Actionable recommendations paragraph - specific actions to take (LLM-GENERATED, 2-3 sentences)",
  "charts": [
    {{
      "type": "line|bar|pie|area|scatter",
      "title": "Chart Title (LLM-GENERATED)",
      "description": "What this shows and why it matters (LLM-GENERATED)",
      "data_transformation": "PYTHON CODE that creates df_chart",
      "x_column": "column_name",
      "y_column": "single_column_name"
    }}
  ],
  "data_table": {{
    "title": "Table Title (LLM-GENERATED, specific to user query)",
    "description": "What this table shows (LLM-GENERATED)",
    "transformation_code": "PYTHON CODE that creates df_table (top 10 aggregated records)",
    "columns": ["col1", "col2", "col3"]
  }}
}}

**DATA TRANSFORMATION GUIDELINES:**
Your code receives df (filtered dataframe) and must create df_chart or df_table

**Chart Examples:**
Example 1: Monthly revenue trend
df_chart = df.groupby(pd.Grouper(key='{self.date_column}', freq='ME'))['{self.revenue_column}'].sum().reset_index()

Example 2: Top 10 customers by revenue
df_chart = df.groupby('{self.customer_column}')['{self.revenue_column}'].sum().nlargest(10).reset_index()

Example 3: Product revenue mix (Top 5)
df_chart = df.groupby('{self.product_column}')['{self.revenue_column}'].sum().nlargest(5).reset_index()


**Data Table Examples (MANDATORY - Choose one relevant to query):**
Example 1: Top 10 customers with revenue and order count
df_table = df.groupby('{self.customer_column}').agg({{
'{self.revenue_column}': 'sum',
'{self.order_id_column}': 'count'
}}).nlargest(10, '{self.revenue_column}').reset_index()
df_table.columns = ['Customer', 'Total_Revenue', 'Order_Count']

Example 2: Top 10 products by revenue
df_table = df.groupby('{self.product_column}')['{self.revenue_column}'].sum().nlargest(10).reset_index()
df_table.columns = ['Product', 'Total_Revenue']

Example 3: Monthly revenue summary (last 10 months)
df_table = df.groupby(pd.Grouper(key='{self.date_column}', freq='ME'))['{self.revenue_column}'].sum().tail(10).reset_index()
df_table.columns = ['Month', 'Revenue']


{'''**OPTIONAL: Anomaly Data Available**
If relevant to the user query, you MAY use these anomaly columns:

Anomaly count over time
df_chart = df.groupby(pd.Grouper(key='Date', freq='ME'))['is_anomaly'].sum().reset_index()
df_chart.columns = ['Date', 'Anomaly_Count']

Top affected products
df_anomalies = df[df['is_anomaly'] == True]
df_chart = df_anomalies.groupby('Product').size().nlargest(10).reset_index(name='Anomaly_Count')

Normal vs Anomalous revenue
df_chart = df.groupby('is_anomaly')['Revenue'].sum().reset_index()
df_chart['Category'] = df_chart['is_anomaly'].map({{True: 'Anomalous', False: 'Normal'}})
df_chart = df_chart[['Category', 'Revenue']]
 
**IMPORTANT:** Only create anomaly-focused dashboards if the user query explicitly mentions anomalies, outliers, or unusual patterns.
''' if has_anomaly_cols else ''}

**CRITICAL VALIDATION BEFORE RESPONDING:**
1. ✓ Have I included the "data_table" field in my JSON? (MANDATORY)
2. ✓ Does my dashboard focus match the USER QUERY (not just data availability)?
3. ✓ Have I created 6-10 charts?
4. ✓ Is the table relevant to the user's question?
5. ✓ Are all field names correct (title, description, strategies, recommendations, charts, data_table)?

**NOW CREATE THE DASHBOARD:**
- Focus on what the USER QUERY explicitly asks for
- If query asks about product/customer performance WITHOUT mentioning anomalies → create standard business dashboard
- If query asks about anomalies/outliers → create anomaly-focused dashboard  
- Match the narrative to semantic context (LOW vs HIGH performer) if applicable
- Be creative and insightful
- ENSURE FACTUAL ACCURACY - do not contradict previous analysis
- **MANDATORY: Include the data_table field**
- Return ONLY valid JSON

Query Focus: "{query}"
Dashboard Type: {'Anomaly Analysis' if any(word in query.lower() for word in ['anomaly', 'anomalies', 'outlier', 'unusual', 'abnormal']) else 'Business Performance Analysis'}

Generate your complete dashboard plan with MANDATORY data_table:"""


        try:
            logger.info("="*60)
            logger.info("CALLING LLM FOR 100% DYNAMIC DASHBOARD...")
            logger.info(f"Query: {query}")
            logger.info(f"Context: {context_desc[:150]}...")
            logger.info(f"Semantic Context: {semantic_context[:100]}..." if semantic_context else "No semantic context")
            logger.info(f"Previous Query: {previous_query}")
            logger.info(f"Anomalies: {full_context.get('anomaly_count', 0)}")
            logger.info(f"Has anomaly columns: {has_anomaly_cols}")
            logger.info("="*60)
            
            response = self.llm.invoke(prompt)
            content = response.content
            
            logger.info("="*60)
            logger.info("LLM RESPONSE (first 1000 chars):")
            logger.info(content[:1000])
            logger.info("="*60)
            
            # Extract JSON from response
            if '```json' in content:
                # Split by '```json', take the part after it (index 1), then split that
                # by the next '```' and take the content before it (index 0).
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                # If the above isn't found, but a general '```' is,
                # assume it's at the start and take everything after the first '```'.
                content = content.split('```', 1)[1].split('```')[0].strip()
            
            content = content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                plan = json.loads(json_match.group())
                logger.info(f"[LLM] Successfully parsed plan with {len(plan.get('charts', []))} charts")
                
                # Minimal validation
                charts = plan.get('charts', [])
                valid_charts = []

                for i, chart in enumerate(charts):
                    x_col = chart.get('x_column')
                    y_col = chart.get('y_column')
                    chart_type = chart.get('type')
                    transformation_code = chart.get('data_transformation')
                    
                    # Reject if y_col is a list
                    if isinstance(y_col, list):
                        logger.warning(f"[LLM Chart {i+1}] y_column is a LIST {y_col} - rejecting (must be single column)")
                        continue
                    
                    if isinstance(x_col, list):
                        logger.warning(f"[LLM Chart {i+1}] x_column is a LIST {x_col} - rejecting (must be single column)")
                        continue
                    
                    if not x_col or not y_col:
                        logger.warning(f"[LLM Chart {i+1}] Missing x_column or y_column - skipping")
                        continue
                    
                    if not chart_type:
                        logger.warning(f"[LLM Chart {i+1}] Missing chart type - skipping")
                        continue
                    
                    if not transformation_code:
                        logger.warning(f"[LLM Chart {i+1}] Missing data_transformation code - skipping")
                        continue
                    
                    valid_charts.append(chart)
                    logger.info(f"[LLM Chart {i+1}] VALID: {chart.get('title')}")
                
                plan['charts'] = valid_charts
                logger.info(f"[LLM] Final plan: {len(valid_charts)}/{len(charts)} valid charts")
                
                if len(valid_charts) == 0:
                    logger.error("[LLM] No valid charts generated!")
                    return None
                
                return plan
            
            logger.error("[LLM] Could not extract JSON from response")
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"[LLM] JSON parsing error: {e}")
            logger.error(f"Content: {content[:500]}")
            return None
        except Exception as e:
            logger.error(f"[LLM] Failed to create dashboard plan: {e}", exc_info=True)
            return None

    
    def _get_comprehensive_data_stats(self, df: pd.DataFrame, full_context: dict) -> str:
        """ FULLY DYNAMIC: Get comprehensive statistics including ANOMALY context"""
        try:
            stats = []
            stats.append(f"Total Records: {len(df):,}")
            
            if self.date_column and self.date_column in df.columns:
                stats.append(f"Date Range: {df[self.date_column].min().strftime('%Y-%m-%d')} to {df[self.date_column].max().strftime('%Y-%m-%d')}")
            
            if self.revenue_column and self.revenue_column in df.columns:
                stats.append(f"Total Revenue: ${df[self.revenue_column].sum():,.2f}")
                stats.append(f"Avg Order Value: ${df[self.revenue_column].mean():,.2f}")
            
            if self.customer_column and self.customer_column in df.columns:
                stats.append(f"Unique Customers: {df[self.customer_column].nunique()}")
            
            if self.product_column and self.product_column in df.columns:
                stats.append(f"Unique Products: {df[self.product_column].nunique()}")
            
            # Add anomaly statistics if available
            if 'is_anomaly' in df.columns:
                anomaly_count = df['is_anomaly'].sum()
                anomaly_pct = (anomaly_count / len(df)) * 100 if len(df) > 0 else 0
                stats.append(f"")
                stats.append(f" ANOMALIES DETECTED: {anomaly_count:,} ({anomaly_pct:.1f}% of data)")
                
                if self.revenue_column and self.revenue_column in df.columns:
                    anomalous_revenue = df[df['is_anomaly'] == True][self.revenue_column].sum()
                    normal_revenue = df[df['is_anomaly'] == False][self.revenue_column].sum()
                    stats.append(f"   - Anomalous Revenue: ${anomalous_revenue:,.2f}")
                    stats.append(f"   - Normal Revenue: ${normal_revenue:,.2f}")
                
                if self.customer_column and self.customer_column in df.columns:
                    affected_customers = df[df['is_anomaly'] == True][self.customer_column].nunique()
                    stats.append(f"   - Affected Customers: {affected_customers}")
                
                if self.product_column and self.product_column in df.columns:
                    affected_products = df[df['is_anomaly'] == True][self.product_column].nunique()
                    stats.append(f"   - Affected Products: {affected_products}")
                
                stats.append(f"   - IMPORTANT: Use 'is_anomaly' column to analyze anomalies!")
            
            # Add context-specific stats
            entities = {k: v for k, v in full_context.items() if k in ['customer_id', 'product_id', 'year', 'sales_org']}
            
            if entities.get('customer_id'):
                stats.append(f"")
                stats.append(f" FILTERING: Customer {entities['customer_id']} ONLY")
            elif entities.get('product_id'):
                stats.append(f"")
                stats.append(f" FILTERING: Product {entities['product_id']} ONLY")
            elif entities.get('year'):
                stats.append(f"")
                stats.append(f" FILTERING: Year {entities['year']} ONLY")
            else:
                stats.append(f"")
                stats.append(f" SCOPE: ALL data (no filters)")
            
            return "\n".join(stats)
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return f"Total Records: {len(df):,}"
    
    def _get_column_information(self, df: pd.DataFrame) -> str:
        """Get detailed column information with examples"""
        column_details = []
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            
            # Get sample values
            if df[col].dtype == 'object' or df[col].dtype == 'category':
                unique_count = df[col].nunique()
                samples = df[col].dropna().unique()[:3].tolist()
                column_details.append(f"- {col} ({dtype}): {unique_count} unique values. Examples: {samples}")
            elif 'datetime' in str(df[col].dtype):
                min_date = df[col].min()
                max_date = df[col].max()
                column_details.append(f"- {col} (datetime): Range from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
            elif df[col].dtype == 'bool':
                true_count = df[col].sum()
                false_count = len(df) - true_count
                column_details.append(f"- {col} (boolean): True={true_count:,}, False={false_count:,}")
            else:
                min_val = df[col].min()
                max_val = df[col].max()
                avg_val = df[col].mean()
                column_details.append(f"- {col} ({dtype}): Range [{min_val:,.0f} to {max_val:,.0f}], Avg: {avg_val:,.0f}")
        
        return "\n".join(column_details)
    
    def _build_context_description(self, query: str, full_context: dict, df: pd.DataFrame) -> str:
        """ FULLY DYNAMIC: Build a natural language description including ANOMALIES and SEMANTIC CONTEXT"""
        descriptions = []
        
        # ✅ NEW: Extract semantic context from previous analysis
        semantic_context = full_context.get('_semantic_context', '')
        previous_query = full_context.get('_previous_query', '')
        previous_results = full_context.get('_previous_results', {})
        
        # ✅ NEW: Add semantic understanding FIRST (most important)
        if semantic_context or previous_query:
            descriptions.append("🎯 CRITICAL CONTEXT FROM PREVIOUS ANALYSIS:")
            
            if previous_query:
                descriptions.append(f"Previous question: '{previous_query}'")
            
            if semantic_context:
                # Extract key insights from semantic context
                semantic_lower = semantic_context.lower()
                
                if 'lowest' in semantic_lower or 'least' in semantic_lower:
                    descriptions.append("⚠️ IMPORTANT: This is about the LOWEST performing entity (worst performer).")
                    descriptions.append("Dashboard should focus on: why performance is poor, opportunities for improvement, risk factors.")
                elif 'highest' in semantic_lower or 'most' in semantic_lower:
                    descriptions.append("✅ IMPORTANT: This is about the HIGHEST performing entity (top performer).")
                    descriptions.append("Dashboard should focus on: success factors, what's driving growth, retention strategies.")
                elif 'top' in semantic_lower and 'bottom' not in semantic_lower:
                    descriptions.append("🔝 FOCUS: Highlighting top performers.")
                elif 'bottom' in semantic_lower or 'worst' in semantic_lower:
                    descriptions.append("📉 FOCUS: Identifying underperformers.")
            
            descriptions.append("")  # Blank line for separation
        
        # Extract filter entities
        entities = {k: v for k, v in full_context.items() if k in ['customer_id', 'product_id', 'year', 'sales_org', 'is_comparison', 'comparison_dimension', 'comparison_values', 'aggregation_type', 'aggregation', 'metric']}
        
        # ✅ NEW: Extract aggregation_type for performance labeling
        aggregation_type = entities.get('aggregation_type') or entities.get('aggregation')
        
        # Add anomaly context if available
        if full_context.get('anomalies_detected') and full_context.get('anomaly_count', 0) > 0:
            anomaly_count = full_context.get('anomaly_count')
            # Only mention anomalies are available, don't mandate their use
            descriptions.append(f"Note: Dataset includes {anomaly_count} detected anomalies (available for analysis if relevant to query).")
        
        # Entity-specific context WITH PERFORMANCE LABELING
        if entities.get('customer_id'):
            customer_id = entities['customer_id']
            customer_orders = len(df)
            customer_revenue = df[self.revenue_column].sum() if self.revenue_column and self.revenue_column in df.columns else 0
            
            # ✅ USE aggregation_type to determine performance level
            if aggregation_type:
                agg_lower = str(aggregation_type).lower()
                if agg_lower in ['least', 'lowest', 'minimum', 'min', 'bottom', 'worst']:
                    descriptions.append(f"📉 Customer {customer_id}: **LOWEST REVENUE** customer with ${customer_revenue:,.2f} from {customer_orders} orders.")
                    descriptions.append("Analysis focus: Why is revenue low? What products do they buy? Growth opportunities?")
                elif agg_lower in ['highest', 'most', 'maximum', 'max', 'top', 'best']:
                    descriptions.append(f"🏆 Customer {customer_id}: **HIGHEST REVENUE** customer with ${customer_revenue:,.2f} from {customer_orders} orders.")
                    descriptions.append("Analysis focus: What drives their success? How to retain them? Expansion opportunities?")
                else:
                    descriptions.append(f"Customer {customer_id}: {customer_orders} orders, ${customer_revenue:,.2f} revenue.")
            else:
                # Fallback to semantic context if no aggregation_type
                if semantic_context:
                    semantic_lower = semantic_context.lower()
                    if 'lowest' in semantic_lower or 'least' in semantic_lower:
                        descriptions.append(f"📉 Customer {customer_id}: **LOWEST REVENUE** customer with ${customer_revenue:,.2f} from {customer_orders} orders.")
                        descriptions.append("Analysis focus: Why is revenue low? What products do they buy? Growth opportunities?")
                    elif 'highest' in semantic_lower or 'most' in semantic_lower:
                        descriptions.append(f"🏆 Customer {customer_id}: **HIGHEST REVENUE** customer with ${customer_revenue:,.2f} from {customer_orders} orders.")
                        descriptions.append("Analysis focus: What drives their success? How to retain them? Expansion opportunities?")
                    else:
                        descriptions.append(f"Customer {customer_id}: {customer_orders} orders, ${customer_revenue:,.2f} revenue.")
                else:
                    descriptions.append(f"Customer {customer_id}: {customer_orders} orders, ${customer_revenue:,.2f} revenue.")
            
            if 'is_anomaly' in df.columns:
                customer_anomalies = df['is_anomaly'].sum()
                if customer_anomalies > 0:
                    descriptions.append(f"Anomalies: {customer_anomalies} anomalous transactions detected.")
        
        elif entities.get('product_id'):
            product_id = entities['product_id']
            product_orders = len(df)
            product_revenue = df[self.revenue_column].sum() if self.revenue_column and self.revenue_column in df.columns else 0
            
            # ✅ USE aggregation_type for products too
            if aggregation_type:
                agg_lower = str(aggregation_type).lower()
                if agg_lower in ['least', 'lowest', 'minimum', 'min', 'bottom', 'worst']:
                    descriptions.append(f"Product {product_id}: **LOWEST REVENUE** product with ${product_revenue:,.2f} from {product_orders} orders.")
                    descriptions.append("Analysis focus: Why is demand low? Pricing issues? Marketing needed?")
                elif agg_lower in ['highest', 'most', 'maximum', 'max', 'top', 'best']:
                    descriptions.append(f"Product {product_id}: **HIGHEST REVENUE** product with ${product_revenue:,.2f} from {product_orders} orders.")
                    descriptions.append("Analysis focus: What makes it popular? How to sustain growth? Cross-sell opportunities?")
                else:
                    descriptions.append(f"Product {product_id}: {product_orders} orders, ${product_revenue:,.2f} revenue.")
            else:
                # Fallback to semantic context
                if semantic_context:
                    semantic_lower = semantic_context.lower()
                    if 'lowest' in semantic_lower or 'least' in semantic_lower:
                        descriptions.append(f"Product {product_id}: **LOWEST REVENUE** product with ${product_revenue:,.2f} from {product_orders} orders.")
                        descriptions.append("Analysis focus: Why is demand low? Pricing issues? Marketing needed?")
                    elif 'highest' in semantic_lower or 'most' in semantic_lower:
                        descriptions.append(f"Product {product_id}: **HIGHEST REVENUE** product with ${product_revenue:,.2f} from {product_orders} orders.")
                        descriptions.append("Analysis focus: What makes it popular? How to sustain growth? Cross-sell opportunities?")
                    else:
                        descriptions.append(f"Product {product_id}: {product_orders} orders, ${product_revenue:,.2f} revenue.")
                else:
                    descriptions.append(f"Product {product_id}: {product_orders} orders, ${product_revenue:,.2f} revenue.")
            
            if 'is_anomaly' in df.columns:
                product_anomalies = df['is_anomaly'].sum()
                if product_anomalies > 0:
                    descriptions.append(f"Anomalies: {product_anomalies} anomalous transactions detected.")
        
        elif entities.get('year'):
            year = entities['year']
            descriptions.append(f"Year Focus: Analyzing year {year} specifically.")
        
        elif entities.get('is_comparison'):
            dim = entities.get('comparison_dimension', 'unknown')
            values = entities.get('comparison_values', [])
            descriptions.append(f"Comparison Analysis: Comparing {dim} between {' vs '.join(map(str, values))}.")
        
        else:
            descriptions.append("Overall Business Analysis: No specific filters applied.")
        
        # ✅ NEW: Add data quality info
        total_records = len(df)
        descriptions.append(f"\n📊 Data: {total_records:,} records loaded for analysis.")
        
        return " ".join(descriptions)


    
    # ===========================
    # PRE-AGGREGATION
    # ===========================
    
    def _preaggregate_chart_data(self, dashboard_plan: Dict[str, Any], df: pd.DataFrame, entities: dict):
        """Execute LLM-generated data transformation code"""
        logger.info("[Dashboard] Executing LLM data transformations...")
        
        filtered_df = self._filter_dataframe(df, entities)
        
        for i, chart_config in enumerate(dashboard_plan.get('charts', [])):
            chart_key = f"chart_{i}"
            transformation_code = chart_config.get('data_transformation', '')
            x_col = chart_config.get('x_column')
            y_col = chart_config.get('y_column')
            
            logger.info(f"[Chart {i+1}] Executing transformation...")
            
            if not transformation_code:
                logger.error(f"[Chart {i+1}] âŒ No transformation code provided")
                self.chart_data_cache[chart_key] = None
                continue
            
            try:
                # Create safe execution environment
                local_namespace = {
                    'df': filtered_df.copy(),
                    'pd': pd,
                    'np': np,
                    'datetime': datetime
                }
                
                # Execute LLM-generated code
                exec(transformation_code, {}, local_namespace)
                
                # Get the result
                df_chart = local_namespace.get('df_chart')
                
                if df_chart is None or not isinstance(df_chart, pd.DataFrame):
                    logger.error(f"[Chart {i+1}] Code did not create 'df_chart'")
                    self.chart_data_cache[chart_key] = None
                    continue
                
                # Validate that x_col and y_col are strings, not lists
                if isinstance(x_col, list) or isinstance(y_col, list):
                    logger.error(f"[Chart {i+1}] x_col or y_col is a LIST - must be single column name")
                    self.chart_data_cache[chart_key] = None
                    continue
                
                if x_col not in df_chart.columns or y_col not in df_chart.columns:
                    logger.error(f"[Chart {i+1}] Required columns missing: {x_col}, {y_col}")
                    logger.error(f"Available columns: {list(df_chart.columns)}")
                    self.chart_data_cache[chart_key] = None
                    continue
                
                if len(df_chart) == 0:
                    logger.warning(f"[Chart {i+1}] Transformation produced empty dataframe")
                    self.chart_data_cache[chart_key] = None
                    continue
                
                # Success!
                self.chart_data_cache[chart_key] = {
                    'data': df_chart,
                    'config': chart_config
                }
                logger.info(f"[Chart {i+1}] Transformation successful: {len(df_chart)} rows")
                
            except Exception as e:
                logger.error(f"[Chart {i+1}] Transformation failed: {e}")
                logger.error(f"Code was: {transformation_code}")
                self.chart_data_cache[chart_key] = None
    
    # ===========================
    # CHART GENERATION
    # ===========================
    
    def _generate_charts(self, dashboard_plan: Dict[str, Any], df: pd.DataFrame, entities: dict) -> List[Dict[str, Any]]:
        """Generate charts using Matplotlib"""
        charts = []
        
        logger.info(f"[Charts] Generating {len(dashboard_plan.get('charts', []))} charts...")
        
        for i, chart_config in enumerate(dashboard_plan.get('charts', [])):
            chart_key = f"chart_{i}"
            
            try:
                logger.info(f"[Chart {i+1}] Creating: {chart_config.get('title')}")
                
                cached_data = self.chart_data_cache.get(chart_key)
                
                if not cached_data or cached_data.get('data') is None or len(cached_data.get('data', [])) == 0:
                    logger.warning(f"[Chart {i+1}] No cached data - skipping")
                    continue
                
                img_base64 = self._create_matplotlib_chart(cached_data, entities)
                
                if img_base64:
                    charts.append({
                        'image_base64': img_base64,
                        'title': chart_config.get('title', f'Chart {i+1}'),
                        'description': chart_config.get('description', ''),
                        'config': chart_config,
                        'chart_key': chart_key
                    })
                    logger.info(f"[Chart {i+1}] SUCCESS")
            except Exception as e:
                logger.error(f"[Chart {i+1}] FAILED: {str(e)}", exc_info=True)
        
        logger.info(f"[Charts] Generated {len(charts)} charts")
        return charts
    

    def _generate_aggregated_table(self, dashboard_plan: Dict[str, Any], df: pd.DataFrame, entities: dict) -> Dict[str, Any]:
        """Generate aggregated data table based on LLM's specification"""
        logger.info("[Table] Generating aggregated data table...")

        table_config = dashboard_plan.get('data_table')

        if not table_config:
            logger.info("[Table] No table configuration in dashboard plan")
            return None

        filtered_df = self._filter_dataframe(df, entities)

        if len(filtered_df) == 0:
            logger.warning("[Table] No data after filtering")
            return None

        transformation_code = table_config.get('transformation_code', '')
        columns = table_config.get('columns', [])
        title = table_config.get('title', 'Data Table')
        description = table_config.get('description', '')

        if not transformation_code:
            logger.error("[Table] No transformation code provided")
            return None

        try:
            # Create safe execution environment
            local_namespace = {
                'df': filtered_df.copy(),
                'pd': pd,
                'np': np,
                'datetime': datetime
            }

            # Execute LLM-generated code
            exec(transformation_code, {}, local_namespace)

            # Get the result
            df_table = local_namespace.get('df_table')

            if df_table is None or not isinstance(df_table, pd.DataFrame):
                logger.error("[Table] Code did not create 'df_table'")
                return None

            if len(df_table) == 0:
                logger.warning("[Table] Transformation produced empty dataframe")
                return None

            # Validate columns
            for col in columns:
                if col not in df_table.columns:
                    logger.error(f"[Table] Required column missing: {col}")
                    return None

            # Ensure top 10 or less
            if len(df_table) > 10:
                df_table = df_table.head(10)

            logger.info(f"[Table] Successfully generated table with {len(df_table)} rows and {len(columns)} columns")

            return {
                'data': df_table[columns],
                'title': title,
                'description': description,
                'columns': columns
            }

        except Exception as e:
            logger.error(f"[Table] Transformation failed: {e}")
            logger.error(f"Code was: {transformation_code}")
            return None

    def _create_matplotlib_chart(self, cached_data: dict, entities: dict) -> str:
        """Create chart - data is already transformed by LLM"""
        config = cached_data['config']
        df_chart = cached_data['data'].copy()
        
        chart_type = config.get('type', 'bar')
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        title = config.get('title', 'Chart')
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
            
            # Data is already aggregated/transformed by LLM code
            x_data = df_chart[x_col]
            y_data = df_chart[y_col]
            
            if chart_type == 'bar':
                ax.bar(range(len(df_chart)), y_data, color=colors[0], alpha=0.8)
                ax.set_xticks(range(len(df_chart)))
                ax.set_xticklabels(x_data.astype(str), rotation=45, ha='right')
                ax.set_ylabel(y_col)
                ax.grid(axis='y', alpha=0.3)
            
            elif chart_type == 'line':
                ax.plot(x_data, y_data, marker='o', linewidth=2, 
                        markersize=6, color=colors[0])
                ax.set_ylabel(y_col)
                ax.grid(True, alpha=0.3)
                
                # Handle datetime x-axis
                if pd.api.types.is_datetime64_any_dtype(x_data):
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            elif chart_type == 'pie':
                colors_pie = plt.cm.Purples(np.linspace(0.3, 0.9, len(df_chart)))
                wedges, texts, autotexts = ax.pie(y_data, labels=x_data.astype(str), 
                                                autopct='%1.1f%%', colors=colors_pie,
                                                startangle=90)
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            
            elif chart_type == 'area':
                ax.fill_between(range(len(df_chart)), y_data, alpha=0.6, color=colors[0])
                ax.plot(range(len(df_chart)), y_data, linewidth=2, color=colors[1])
                ax.set_ylabel(y_col)
                ax.grid(True, alpha=0.3)
                
                if pd.api.types.is_datetime64_any_dtype(x_data):
                    ax.set_xticks(range(len(df_chart)))
                    ax.set_xticklabels(x_data.dt.strftime('%b %Y'), rotation=45, ha='right')
            
            elif chart_type == 'scatter':
                ax.scatter(x_data, y_data, alpha=0.6, s=50, color=colors[0])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.grid(True, alpha=0.3)
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close(fig)
            
            return img_base64
            
        except Exception as e:
            logger.error(f"[Chart] Matplotlib failed: {e}", exc_info=True)
            plt.close('all')
            return None
    
    # ===========================
    # INSIGHTS GENERATION
    # ===========================
    
    def _generate_comprehensive_insights(self, query: str, df: pd.DataFrame, entities: dict, 
                                        dashboard_plan: Dict, charts: List[Dict]) -> Dict[str, Any]:
        """Generate insights"""
        filtered_df = self._filter_dataframe(df, entities)
        
        if len(filtered_df) == 0:
            return {
                'executive_summary': 'No data available for the selected filters.',
                'key_findings': ['No data to analyze'],
                'recommendations': ['Adjust filters or select different criteria'],
                'chart_insights': []
            }
        
        metrics = self._calculate_metrics(filtered_df, entities)
        
        if self.llm:
            insights = self._llm_generate_insights(query, metrics, entities, filtered_df)
        else:
            insights = self._fallback_insights(metrics)
        
        chart_insights = self._generate_chart_insights(charts)
        insights['chart_insights'] = chart_insights
        
        return insights
    
    def _calculate_metrics(self, df: pd.DataFrame, entities: dict) -> Dict[str, Any]:
        """âœ… FULLY DYNAMIC: Calculate key metrics"""
        try:
            metrics = {}
            
            if self.revenue_column and self.revenue_column in df.columns:
                metrics['total_revenue'] = float(df[self.revenue_column].sum())
                metrics['avg_order_value'] = float(df[self.revenue_column].mean())
            else:
                metrics['total_revenue'] = 0.0
                metrics['avg_order_value'] = 0.0
            
            metrics['total_orders'] = len(df)
            
            if self.customer_column and self.customer_column in df.columns:
                metrics['unique_customers'] = int(df[self.customer_column].nunique())
            else:
                metrics['unique_customers'] = 0
            
            if self.product_column and self.product_column in df.columns:
                metrics['unique_products'] = int(df[self.product_column].nunique())
            else:
                metrics['unique_products'] = 0
            
            if self.date_column and self.date_column in df.columns:
                metrics['date_range'] = f"{df[self.date_column].min().strftime('%Y-%m-%d')} to {df[self.date_column].max().strftime('%Y-%m-%d')}"
            else:
                metrics['date_range'] = "N/A"
            
            if self.cost_column and self.cost_column in df.columns:
                metrics['total_cost'] = float(df[self.cost_column].sum())
            else:
                metrics['total_cost'] = 0.0
            
            if self.tax_column and self.tax_column in df.columns:
                metrics['total_tax'] = float(df[self.tax_column].sum())
            else:
                metrics['total_tax'] = 0.0
            
            # Add anomaly metrics
            if 'is_anomaly' in df.columns:
                metrics['anomaly_count'] = int(df['is_anomaly'].sum())
                metrics['anomaly_percentage'] = (metrics['anomaly_count'] / len(df)) * 100 if len(df) > 0 else 0
                
                if self.revenue_column and self.revenue_column in df.columns:
                    metrics['anomalous_revenue'] = float(df[df['is_anomaly'] == True][self.revenue_column].sum())
            else:
                metrics['anomaly_count'] = 0
                metrics['anomaly_percentage'] = 0
                metrics['anomalous_revenue'] = 0
            
            # Top customer
            if self.customer_column and self.customer_column in df.columns and self.revenue_column and self.revenue_column in df.columns:
                try:
                    top_customer_series = df.groupby(self.customer_column)[self.revenue_column].sum()
                    metrics['top_customer'] = str(top_customer_series.idxmax())
                    metrics['top_customer_revenue'] = float(top_customer_series.max())
                except:
                    metrics['top_customer'] = "N/A"
                    metrics['top_customer_revenue'] = 0.0
            else:
                metrics['top_customer'] = "N/A"
                metrics['top_customer_revenue'] = 0.0
            
            # Top product
            if self.product_column and self.product_column in df.columns and self.revenue_column and self.revenue_column in df.columns:
                try:
                    top_product_series = df.groupby(self.product_column)[self.revenue_column].sum()
                    metrics['top_product'] = str(top_product_series.idxmax())
                    metrics['top_product_revenue'] = float(top_product_series.max())
                except:
                    metrics['top_product'] = "N/A"
                    metrics['top_product_revenue'] = 0.0
            else:
                metrics['top_product'] = "N/A"
                metrics['top_product_revenue'] = 0.0
            
            # Profit margin
            if metrics['total_revenue'] > 0 and metrics['total_cost'] > 0:
                metrics['profit_margin'] = ((metrics['total_revenue'] - metrics['total_cost']) / metrics['total_revenue']) * 100
            else:
                metrics['profit_margin'] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"[Metrics] Failed: {e}")
            return {}
    
    def _llm_generate_insights(self, query: str, metrics: Dict, entities: dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate insights using LLM"""
        
        context_desc = self._build_context_description(query, {'dialogue_state': entities, **entities}, df)
        
        anomaly_context = ""
        if metrics.get('anomaly_count', 0) > 0:
            anomaly_context = f"""
 ANOMALY ANALYSIS:
- Anomalies Detected: {metrics['anomaly_count']} ({metrics['anomaly_percentage']:.1f}%)
- Anomalous Revenue: ${metrics['anomalous_revenue']:,.2f}
- This is a KEY focus area!
"""
        
        prompt = f"""You are a business analyst. Provide insights for this dashboard.

**Dashboard Query:** "{query}"
**Context:** {context_desc}

**Key Metrics:**
- Period: {metrics.get('date_range', 'N/A')}
- Total Revenue: ${metrics.get('total_revenue', 0):,.2f}
- Total Orders: {metrics.get('total_orders', 0):,}
- Avg Order Value: ${metrics.get('avg_order_value', 0):,.2f}
- Unique Customers: {metrics.get('unique_customers', 0)}
- Unique Products: {metrics.get('unique_products', 0)}
- Top Customer: {metrics.get('top_customer', 'N/A')} (${metrics.get('top_customer_revenue', 0):,.2f})
- Top Product: {metrics.get('top_product', 'N/A')} (${metrics.get('top_product_revenue', 0):,.2f})
- Profit Margin: {metrics.get('profit_margin', 0):.1f}%
{anomaly_context}

Provide:

EXECUTIVE_SUMMARY:
[2-3 sentences summarizing key insights, focusing on anomalies if present]

KEY_FINDINGS:
- [Finding 1]
- [Finding 2]
- [Finding 3]
- [Finding 4]

RECOMMENDATIONS:
- [Action 1]
- [Action 2]
- [Action 3]"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content
            
            exec_summary = self._extract_section(content, "EXECUTIVE_SUMMARY:", "KEY_FINDINGS:")
            findings = self._extract_bullets(content, "KEY_FINDINGS:", "RECOMMENDATIONS:")
            recommendations = self._extract_bullets(content, "RECOMMENDATIONS:", None)
            
            return {
                'executive_summary': exec_summary or f"Analysis of ${metrics.get('total_revenue', 0):,.2f} in revenue from {metrics.get('total_orders', 0):,} orders.",
                'key_findings': findings or ["Analysis completed"],
                'recommendations': recommendations or ["Monitor performance"]
            }
        except Exception as e:
            logger.error(f"[Insights] LLM failed: {e}")
            return self._fallback_insights(metrics)
    
    def _fallback_insights(self, metrics: Dict) -> Dict[str, Any]:
        findings = [
            f"Total Revenue: ${metrics.get('total_revenue', 0):,.2f}",
            f"Average Order: ${metrics.get('avg_order_value', 0):,.2f}",
            f"Top Customer: {metrics.get('top_customer', 'N/A')}",
            f"Profit Margin: {metrics.get('profit_margin', 0):.1f}%"
        ]
        
        if metrics.get('anomaly_count', 0) > 0:
            findings.insert(0, f" {metrics['anomaly_count']} anomalies detected ({metrics['anomaly_percentage']:.1f}%)")
        
        return {
            'executive_summary': f"Analysis period: {metrics.get('date_range', 'N/A')}. Revenue: ${metrics.get('total_revenue', 0):,.2f} from {metrics.get('total_orders', 0):,} orders.",
            'key_findings': findings,
            'recommendations': ["Focus on top accounts", "Investigate anomalies", "Monitor trends"]
        }
    
    def _generate_chart_insights(self, charts: List[Dict]) -> List[str]:
        """Generate dynamic insights for each chart using LLM"""
        insights = []
        
        if not self.llm:
            return [chart.get('config', {}).get('description', 'Analysis') for chart in charts]
        
        for chart_data in charts:
            chart_key = chart_data.get('chart_key')
            config = chart_data.get('config', {})
            title = config.get('title', 'Chart')
            
            try:
                cached_data = self.chart_data_cache.get(chart_key)
                
                if not cached_data or cached_data.get('data') is None:
                    insights.append("Data unavailable")
                    continue
                
                df_chart = cached_data['data']
                y_col = config.get('y_column')
                
                if len(df_chart) == 0:
                    insights.append("No data available for this visualization")
                    continue
                
                # Prepare data summary for LLM
                data_summary = f"""
Chart: {title}
Data points: {len(df_chart)}
Metric: {y_col}
Min: {df_chart[y_col].min():,.2f}
Max: {df_chart[y_col].max():,.2f}
Average: {df_chart[y_col].mean():,.2f}
"""
                
                # Get top/bottom values if applicable
                if len(df_chart) > 3:
                    x_col = config.get('x_column')
                    top_3 = df_chart.nlargest(3, y_col)
                    data_summary += f"\nTop 3 values:\n"
                    for idx, row in top_3.iterrows():
                        data_summary += f"  - {row[x_col]}: {row[y_col]:,.2f}\n"
                
                # Ask LLM for insight
                prompt = f"""Provide a 1-sentence business insight (max 25 words) for this chart data:

{data_summary}

Focus on: trends, patterns, outliers, or actionable observations.
Be specific and concise. No generic statements."""
                
                response = self.llm.invoke(prompt)
                insight = response.content.strip()
                
                # Clean up response
                insight = insight.replace('\n', ' ').strip()
                if len(insight) > 200:
                    insight = insight[:197] + "..."
                
                insights.append(insight)
                
            except Exception as e:
                logger.error(f"[Chart Insight] Error: {e}")
                insights.append(config.get('description', f"Analysis of {title}"))
        
        return insights
    
    def _extract_section(self, text: str, start: str, end: str = None) -> str:
        try:
            start_idx = text.find(start)
            if start_idx == -1:
                return ""
            start_idx += len(start)
            
            if end:
                end_idx = text.find(end, start_idx)
                section = text[start_idx:end_idx] if end_idx != -1 else text[start_idx:]
            else:
                section = text[start_idx:]
            
            return section.strip()
        except:
            return ""
    
    def _extract_bullets(self, text: str, start: str, end: str = None) -> List[str]:
        """Extract bullet points from text"""
        section = self._extract_section(text, start, end)
        bullets = []
        
        for line in section.split('\n'):
            line = line.strip()
            if line.startswith(('-', '- ', '*')) or re.match(r'^\d+[\.\)]\s', line):
                # Remove bullet characters and numbers
                bullet = re.sub(r'^[-* ]|\d+[\.\)]\s*', '', line).strip()
                if bullet:
                    bullets.append(bullet)
        
        return bullets[:6]
    
    # ===========================
    # HTML GENERATION
    # ===========================

    def _render_data_table(self, data_table: Dict[str, Any]) -> str:
        """Render the data table HTML"""
        rows_html = ""
        for _, row in data_table['data'].iterrows():
            cells = "".join(f"<td style='padding: 10px;'>{row[col] if pd.notna(row[col]) else 'N/A'}</td>" 
                        for col in data_table['columns'])
            rows_html += f"<tr style='border-bottom: 1px solid #e0e0e0;'>{cells}</tr>"
        
        headers = "".join(f"<th style='padding: 12px; text-align: left; font-weight: 600;'>{col}</th>" 
                        for col in data_table['columns'])
        
        return f"""
            <div class="insights-box">
                <h2>{data_table['title']}</h2>
                <p>{data_table['description']}</p>
                <div style="overflow-x: auto; margin-top: 20px;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.95rem;">
                        <thead>
                            <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                                {headers}
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html}
                        </tbody>
                    </table>
                </div>
            </div>
            """

    
    def _create_dashboard_html(self, charts: List[Dict[str, Any]], dashboard_plan: Dict[str, Any],
                               query: str, entities: dict, insights: Dict[str, Any], data_table: Dict[str, Any] = None) -> str:
        """Create dashboard HTML"""
        title = dashboard_plan.get('title', 'Dashboard')
        description = dashboard_plan.get('description', '')
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; line-height: 1.6; color: #333; }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        .header {{ background: white; padding: 40px; border-radius: 16px; margin-bottom: 24px; box-shadow: 0 8px 32px rgba(0,0,0,0.12); }}
        .header h1 {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; font-weight: 700; margin-bottom: 12px; }}
        .header .description {{ color: #666; font-size: 1.1rem; margin-bottom: 20px; }}
        .metadata {{ display: flex; flex-wrap: wrap; gap: 16px; padding-top: 20px; border-top: 2px solid #f0f0f0; }}
        .metadata-item {{ padding: 8px 16px; background: #f8f9fa; border-radius: 8px; font-size: 0.9rem; }}
        .metadata-item strong {{ color: #667eea; margin-right: 6px; }}
        .insights-box {{ background: white; padding: 32px; border-radius: 16px; margin-bottom: 24px; box-shadow: 0 8px 32px rgba(0,0,0,0.12); }}
        .insights-box h2 {{ color: #2c3e50; font-size: 1.8rem; margin-bottom: 20px; font-weight: 700; }}
        .insights-box h3 {{ color: #34495e; font-size: 1.3rem; margin-top: 24px; margin-bottom: 12px; font-weight: 600; }}
        .insights-box p {{ color: #555; font-size: 1.05rem; line-height: 1.7; margin-bottom: 16px; }}
        .insights-box ul {{ list-style: none; padding-left: 0; }}
        .insights-box li {{ color: #555; font-size: 1.05rem; padding: 10px 0; padding-left: 28px; position: relative; line-height: 1.6; }}
        .insights-box li::before {{ content: "â–"; color: #667eea; font-weight: bold; position: absolute; left: 8px; top: 10px; }}
        .charts-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(700px, 1fr)); gap: 24px; margin-bottom: 24px; }}
        .chart-card {{ background: white; border-radius: 16px; overflow: hidden; box-shadow: 0 8px 32px rgba(0,0,0,0.12); transition: transform 0.3s; }}
        .chart-card:hover {{ transform: translateY(-4px); box-shadow: 0 12px 48px rgba(0,0,0,0.18); }}
        .chart-content {{ padding: 20px; text-align: center; }}
        .chart-content img {{ max-width: 100%; height: auto; }}
        .chart-insight-box {{ padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-top: 1px solid #dee2e6; }}
        .chart-insight-box p {{ color: #495057; font-size: 0.95rem; line-height: 1.6; margin: 0; }}
        .footer {{ background: white; padding: 32px; border-radius: 16px; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.12); }}
        .footer strong {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; }}
        @media (max-width: 1400px) {{ .charts-container {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ðŸ“Š {title}</h1>
            <p class="description">{description}</p>
            <div class="metadata">
                <div class="metadata-item"><strong>Generated:</strong> {datetime.now().strftime('%B %d, %Y at %H:%M')}</div>
                <div class="metadata-item"><strong>Query:</strong> "{query}"</div>
                <div class="metadata-item"><strong>Charts:</strong> {len(charts)}</div>
                <div class="metadata-item"><strong>ðŸ¤– 100% LLM-Powered:</strong> Fully Dynamic</div>
            </div>
        </div>
        
        <div class="insights-box">
            <h2>ðŸ“ˆ Executive Summary</h2>
            <p>{insights.get('executive_summary', 'No summary.')}</p>
            
            <h3>ðŸ”‘ Key Findings</h3>
            <ul>
                {"".join(f"<li>{f}</li>" for f in insights.get('key_findings', ['No findings']))}
            </ul>
            
            <h3>ðŸ’¡ Recommendations</h3>
            <ul>
                {"".join(f"<li>{r}</li>" for r in insights.get('recommendations', ['No recommendations']))}
            </ul>
        </div>
        
        
        {self._render_data_table(data_table) if data_table else ""}

        <div class="charts-container">"""
        
        chart_insights = insights.get('chart_insights', [])
        for i, chart_data in enumerate(charts):
            img_base64 = chart_data['image_base64']
            insight = chart_insights[i] if i < len(chart_insights) else chart_data.get('description', '')
            
            html += f"""
            <div class="chart-card">
                <div class="chart-content">
                    <img src="data:image/png;base64,{img_base64}" alt="{chart_data['title']}">
                </div>
                <div class="chart-insight-box"><p>{insight}</p></div>
            </div>"""
        
        html += f"""
        </div>
        
        <div class="footer">
            <p><strong>100% AI-Generated Dashboard</strong></p>
            <p style="margin-top: 8px; color: #666; font-size: 0.9rem;">Powered by Gemini LLM - Zero Templates, Full Creativity</p>
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def _save_dashboard(self, html_content: str, entities: dict, query: str) -> str:
        """Save dashboard to file"""
        os.makedirs('dashboards', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create meaningful filename from query
        query_slug = re.sub(r'[^\w\s-]', '', query.lower())[:30]
        query_slug = re.sub(r'[-\s]+', '_', query_slug)
        
        filename = f"dashboard_{query_slug}_{timestamp}.html"
        filepath = os.path.join('dashboards', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"âœ… [Dashboard] Saved: {filepath}")
        return filepath