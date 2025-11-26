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

class DashboardAgent:
    """✅ Fully LLM-Driven Dynamic Dashboard Agent - LLM decides EVERYTHING"""
    
    def __init__(self):
        self.name = "DashboardAgent"
        self.llm = None
        self.analysis_agent_ref = None
        self.chart_data_cache = {}
        
        if not MATPLOTLIB_AVAILABLE:
            logger.error("[DashboardAgent] Matplotlib not available!")
            return
        
        try:
            if LLM_AVAILABLE:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0,
                    api_key=GEMINI_API_KEY
                )
                logger.info("[DashboardAgent] ✅ Initialized with FULLY LLM-DRIVEN mode")
            else:
                logger.error("[DashboardAgent] ❌ LLM not available - dashboard generation will fail")
        except Exception as e:
            logger.warning(f"[DashboardAgent] Could not initialize LLM: {e}")
    
    def execute(self, query: str, entities: dict = None, analysis_agent=None) -> Dict[str, Any]:
        """Main execution method - LLM-driven"""
        logger.info(f"[{self.name}] 🎯 Received query: '{query}' | entities={entities}")
        
        if not MATPLOTLIB_AVAILABLE:
            return {"status": "error", "message": "Matplotlib not installed"}
        
        if not self.llm:
            return {"status": "error", "message": "LLM not available - cannot generate dynamic dashboard"}
        
        try:
            self.analysis_agent_ref = analysis_agent
            self.chart_data_cache = {}
            
            # Get data
            df = mcp_store.get_sales_data()
            resolved_entities = self._resolve_dashboard_context(query, entities)
            
            logger.info(f"[{self.name}] Final entities: {resolved_entities}")
            
            # ✅ LLM DECIDES EVERYTHING
            dashboard_plan = self._llm_create_dashboard_plan(query, resolved_entities, df)
            
            if not dashboard_plan or len(dashboard_plan.get('charts', [])) == 0:
                return {"status": "error", "message": "LLM could not generate a valid dashboard plan"}
            
            logger.info(f"[{self.name}] 📊 LLM-Generated Plan: {dashboard_plan.get('title')}")
            logger.info(f"[{self.name}] 📈 Charts to generate: {len(dashboard_plan.get('charts', []))}")
            
            # Pre-aggregate and generate
            self._preaggregate_chart_data(dashboard_plan, df, resolved_entities)
            charts = self._generate_charts(dashboard_plan, df, resolved_entities)
            all_insights = self._generate_comprehensive_insights(query, df, resolved_entities, dashboard_plan, charts)
            dashboard_html = self._create_dashboard_html(charts, dashboard_plan, query, resolved_entities, all_insights)
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
    # CONTEXT RESOLUTION
    # ===========================
    
    def _resolve_dashboard_context(self, query: str, entities: dict = None) -> dict:
        """Dynamically resolve entities for dashboard based on query intent"""
        query_lower = query.lower()
        dialogue_state = mcp_store.get_current_dialogue_state()
        all_entities = dialogue_state.get("entities", {})
        entity_stack = dialogue_state.get("context_stack", [])
        
        logger.info("="*60)
        logger.info("🔍 DASHBOARD CONTEXT RESOLUTION")
        logger.info("="*60)
        logger.info(f"Query: '{query}'")
        logger.info(f"Incoming entities: {entities}")
        logger.info(f"Dialogue entities: {all_entities}")
        
        # ✅ CRITICAL: Detect if this is a NEW/GLOBAL/OVERALL dashboard request
        global_keywords = [
            'overall', '360', 'all', 'complete', 'comprehensive', 
            'full', 'entire', 'total', 'global', 'new dashboard',
            'create dashboard', 'build dashboard', 'generate dashboard'
        ]
        
        is_global_request = any(keyword in query_lower for keyword in global_keywords)
        
        # ✅ If it's a global request AND no specific entity mentioned in query
        if is_global_request:
            # Check if query explicitly mentions a customer/product/year
            has_specific_customer = bool(re.search(r'customer\s+(\d+)', query_lower))
            has_specific_product = bool(re.search(r'product\s+([A-Z0-9]+)', query_lower, re.IGNORECASE))
            has_specific_year = bool(re.search(r'\b(20\d{2})\b', query))
            
            if not (has_specific_customer or has_specific_product or has_specific_year):
                logger.info("🌎 Detected GLOBAL dashboard request without specific entities")
                logger.info("✅ Clearing all entity filters for fresh analysis")
                return {}
        
        # ✅ Check if it's a comparison query
        if self._is_comparison_query(query_lower):
            comparison_context = self._extract_comparison_context(query, entities, all_entities)
            if comparison_context:
                logger.info(f"✅ Comparison: {comparison_context}")
                return comparison_context
        
        # ✅ Handle reference queries ("for the same", "for that")
        if any(phrase in query_lower for phrase in ['for the same', 'for that', 'same as']):
            recent_entities = self._get_most_recent_entities(entity_stack, all_entities)
            logger.info(f"✅ Same reference: {recent_entities}")
            return recent_entities
        
        # ✅ Use provided entities if they exist
        if entities and any(k in entities for k in ['customer_id', 'product_id', 'year', 'sales_org']):
            filtered = self._filter_dashboard_entities(entities)
            logger.info(f"✅ Using provided entities: {filtered}")
            return filtered
        
        # ✅ Check if there are RELEVANT recent entities (not just any old filters)
        # Only use recent context if query implies continuation
        continuation_keywords = ['that', 'those', 'it', 'them', 'this', 'these']
        has_continuation = any(kw in query_lower for kw in continuation_keywords)
        
        if has_continuation:
            recent_context = self._filter_dashboard_entities(all_entities)
            logger.info(f"✅ Continuation query - using recent context: {recent_context}")
            return recent_context
        
        # ✅ Default: Return empty for new standalone queries
        logger.info(f"✅ New standalone query - no entity filters applied")
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
        DASHBOARD_KEYS = ['customer_id', 'product_id', 'year', 'sales_org']
        return {k: v for k, v in entities.items() if k in DASHBOARD_KEYS and v}
    
    def _filter_dataframe(self, df: pd.DataFrame, entities: dict) -> pd.DataFrame:
        """Filter dataframe based on entities - handles both single values and lists"""
        filtered = df.copy()
        
        if entities.get('is_comparison'):
            comparison_dim = entities.get('comparison_dimension')
            comparison_values = entities.get('comparison_values', [])
            
            if comparison_dim == 'year' and comparison_values:
                years_int = [int(y) if isinstance(y, str) else y for y in comparison_values]
                filtered = filtered[filtered['CreationDate'].dt.year.isin(years_int)]
        else:
            # ✅ Handle year filtering
            if 'year' in entities:
                year_int = int(entities['year']) if isinstance(entities['year'], str) else entities['year']
                filtered = filtered[filtered['CreationDate'].dt.year == year_int]
            
            # ✅ Handle customer_id filtering (single or list)
            if 'customer_id' in entities:
                customer_id = entities['customer_id']
                if isinstance(customer_id, list):
                    # Convert all to strings for comparison
                    customer_id_str = [str(c) for c in customer_id]
                    filtered = filtered[filtered['SoldToParty'].isin(customer_id_str)]
                    logger.info(f"[Filter] Filtering by customer_id list: {customer_id_str}")
                else:
                    filtered = filtered[filtered['SoldToParty'] == str(customer_id)]
                    logger.info(f"[Filter] Filtering by customer_id: {customer_id}")
            
            # ✅ Handle product_id filtering (single or list)
            if 'product_id' in entities:
                product_id = entities['product_id']
                if isinstance(product_id, list):
                    # Convert all to strings for comparison
                    product_id_str = [str(p) for p in product_id]
                    filtered = filtered[filtered['Product'].isin(product_id_str)]
                    logger.info(f"[Filter] Filtering by product_id list: {product_id_str}")
                else:
                    filtered = filtered[filtered['Product'] == str(product_id)]
                    logger.info(f"[Filter] Filtering by product_id: {product_id}")
            
            # ✅ Handle sales_org filtering
            if 'sales_org' in entities:
                filtered = filtered[filtered['SalesOrganization'] == str(entities['sales_org'])]
                logger.info(f"[Filter] Filtering by sales_org: {entities['sales_org']}")
        
        logger.info(f"[Filter] {len(df)} → {len(filtered)} rows")
        return filtered

    
    # ===========================
    # ✅ FULLY LLM-DRIVEN DASHBOARD PLANNING
    # ===========================
    
    def _llm_create_dashboard_plan(self, query: str, entities: dict, df: pd.DataFrame) -> Dict[str, Any]:
        """✅ LLM DECIDES EVERYTHING - No hardcoded fallbacks"""
        
        # Prepare context for LLM
        filtered_df = self._filter_dataframe(df, entities)
        
        if len(filtered_df) == 0:
            logger.warning("[LLM] No data after filtering!")
            return {"title": "No Data", "description": "No data available", "charts": []}
        
        # Get data statistics
        data_stats = self._get_comprehensive_data_stats(filtered_df, entities)
        
        # Get available columns with examples
        column_info = self._get_column_information(filtered_df)
        
        # Build dynamic context description
        context_desc = self._build_context_description(query, entities, filtered_df)
        
        # ✅ POWERFUL LLM PROMPT - LLM has full creative freedom
        prompt = f"""You are an expert data analyst and dashboard designer. Create a PERFECT dashboard plan for this specific query.

**USER QUERY:** "{query}"

**CONTEXT:**
{context_desc}

**DATA AVAILABLE:**
{data_stats}

**COLUMNS YOU CAN USE:**
{column_info}

**YOUR TASK:**
Design a dashboard with 7-10 visualizations that PERFECTLY answer the user's query. You have COMPLETE FREEDOM to:
- Write ANY Python/Pandas code to transform the data
- Calculate ANY metric (growth rates, ratios, percentages, rankings, etc.)
- Create derived columns (profit = revenue - cost, conversion rates, etc.)
- Aggregate however you want (custom groupby, rolling windows, cumsum, etc.)
- Use ANY chart type that makes sense

**AVAILABLE DATAFRAME:**
The dataframe is called `df` and has these columns:
{column_info}

**REQUIRED JSON STRUCTURE:**
{{
  "title": "Dashboard Title",
  "description": "Brief description",
  "charts": [
    {{
      "type": "line|bar|pie|area|scatter",
      "title": "Chart Title",
      "description": "What this shows",
      "data_transformation": "PYTHON CODE to create df_chart with columns for plotting",
      "x_column": "column_name_after_transformation",
      "y_column": "column_name_after_transformation"
    }}
  ]
}}

**DATA TRANSFORMATION RULES:**
1. Your code receives a filtered dataframe called `df`
2. You MUST create a dataframe called `df_chart` with the final data to plot
3. `df_chart` must have the columns specified in `x_column` and `y_column`
4. You can use any Pandas operations: groupby, merge, pivot, apply, etc.
5. You can create calculated fields: df_chart['profit'] = df_chart['revenue'] - df_chart['cost']
6. For time series: use pd.Grouper(key='CreationDate', freq='ME') for monthly
7. Handle edge cases (division by zero, missing values)

**EXAMPLES:**

**Example 1: Growth Rate**
{{
  "type": "line",
  "title": "Monthly Revenue Growth Rate",
  "data_transformation": "df_chart = df.groupby(pd.Grouper(key='CreationDate', freq='ME'))['NetAmount'].sum().reset_index(); df_chart['GrowthRate'] = df_chart['NetAmount'].pct_change() * 100",
  "x_column": "CreationDate",
  "y_column": "GrowthRate"
}}

**Example 2: Profit Margin by Product**
{{
  "type": "bar",
  "title": "Product Profitability",
  "data_transformation": "df_chart = df.groupby('Product').agg({{'NetAmount': 'sum', 'CostAmount': 'sum'}}).reset_index(); df_chart['ProfitMargin'] = ((df_chart['NetAmount'] - df_chart['CostAmount']) / df_chart['NetAmount'] * 100); df_chart = df_chart.nlargest(10, 'ProfitMargin')",
  "x_column": "Product",
  "y_column": "ProfitMargin"
}}

**Example 3: Customer Concentration**
{{
  "type": "pie",
  "title": "Revenue Concentration (Top 10 vs Rest)",
  "data_transformation": "top10_revenue = df.groupby('SoldToParty')['NetAmount'].sum().nlargest(10).sum(); other_revenue = df['NetAmount'].sum() - top10_revenue; df_chart = pd.DataFrame({{'Category': ['Top 10 Customers', 'Other Customers'], 'Revenue': [top10_revenue, other_revenue]}})",
  "x_column": "Category",
  "y_column": "Revenue"
}}

**Example 4: Customer Lifetime Value Ranking**
{{
  "type": "bar",
  "title": "Top 15 Customers by Lifetime Value",
  "data_transformation": "df_chart = df.groupby('SoldToParty').agg({{'NetAmount': 'sum', 'SalesDocument': 'count'}}).reset_index(); df_chart.columns = ['Customer', 'TotalRevenue', 'OrderCount']; df_chart['AvgOrderValue'] = df_chart['TotalRevenue'] / df_chart['OrderCount']; df_chart = df_chart.nlargest(15, 'TotalRevenue')",
  "x_column": "Customer",
  "y_column": "TotalRevenue"
}}

Now create the PERFECT dashboard for: "{query}"
Return ONLY valid JSON."""

        try:
            logger.info("="*60)
            logger.info("🤖 CALLING LLM FOR DASHBOARD PLANNING...")
            logger.info(f"Query: {query}")
            logger.info(f"Context: {context_desc}")
            logger.info("="*60)
            
            response = self.llm.invoke(prompt)
            content = response.content
            
            logger.info("="*60)
            logger.info("📥 LLM RESPONSE (first 800 chars):")
            logger.info(content[:800])
            logger.info("="*60)
            
            # Extract JSON from response
            if '```json' in content:
                    # Split by '```json' and then take the second part,
                    # then split by '```' to get the content within the JSON block
                    parts = content.split('```json', 1)
                    if len(parts) > 1:
                        content = parts[1].split('```', 1)[0]
                    else:
                        # Handle the case where '```json' exists but there's no content after it
                        content = ""
            elif '```' in content:
                    # If only '```' is present, take the content before the first '```'
                    content = content.split('```', 1)[0]
            
            content = content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                plan = json.loads(json_match.group())
                logger.info(f"✅ [LLM] Successfully parsed plan with {len(plan.get('charts', []))} charts")
                
                # Validate charts
                charts = plan.get('charts', [])
                # ✅ NEW CODE - Minimal validation only
                valid_charts = []

                for i, chart in enumerate(charts):
                    x_col = chart.get('x_column')
                    y_col = chart.get('y_column')
                    chart_type = chart.get('type')
                    transformation_code = chart.get('data_transformation')
                    
                    # Only validate that required fields exist
                    if not x_col or not y_col:
                        logger.warning(f"[LLM Chart {i+1}] ❌ Missing x_column or y_column - skipping")
                        continue
                    
                    if not chart_type:
                        logger.warning(f"[LLM Chart {i+1}] ❌ Missing chart type - skipping")
                        continue
                    
                    if not transformation_code:
                        logger.warning(f"[LLM Chart {i+1}] ❌ Missing data_transformation code - skipping")
                        continue
                    
                    # Fix aggregation naming (keep this)
                    if chart.get('aggregation') == 'avg':
                        chart['aggregation'] = 'mean'
                    
                    if chart_type == 'scatter' and chart.get('aggregation') != 'none':
                        chart['aggregation'] = 'none'
                        logger.info(f"[LLM Chart {i+1}] Fixed scatter plot aggregation to 'none'")
                    
                    valid_charts.append(chart)
                    logger.info(f"[LLM Chart {i+1}] ✅ VALID: {chart.get('title')}")

                plan['charts'] = valid_charts
                logger.info(f"✅ [LLM] Final plan: {len(valid_charts)}/{len(charts)} valid charts")
                
                if len(valid_charts) == 0:
                    logger.error("[LLM] ❌ No valid charts generated!")
                    return None
                
                return plan
            
            logger.error("[LLM] ❌ Could not extract JSON from response")
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"[LLM] ❌ JSON parsing error: {e}")
            logger.error(f"Content: {content[:500]}")
            return None
        except Exception as e:
            logger.error(f"[LLM] ❌ Failed to create dashboard plan: {e}", exc_info=True)
            return None
    
    def _get_comprehensive_data_stats(self, df: pd.DataFrame, entities: dict) -> str:
        """Get comprehensive statistics about the data"""
        try:
            stats = []
            stats.append(f"Total Records: {len(df):,}")
            stats.append(f"Date Range: {df['CreationDate'].min().strftime('%Y-%m-%d')} to {df['CreationDate'].max().strftime('%Y-%m-%d')}")
            stats.append(f"Total Revenue: ${df['NetAmount'].sum():,.2f}")
            stats.append(f"Total Orders: {len(df):,}")
            stats.append(f"Avg Order Value: ${df['NetAmount'].mean():,.2f}")
            stats.append(f"Unique Customers: {df['SoldToParty'].nunique()}")
            stats.append(f"Unique Products: {df['Product'].nunique()}")
            stats.append(f"Sales Organizations: {df['SalesOrganization'].nunique()}")
            
            # Add context-specific stats
            if entities.get('customer_id'):
                stats.append(f"📍 Analyzing ONLY Customer {entities['customer_id']}")
            elif entities.get('product_id'):
                stats.append(f"📍 Analyzing ONLY Product {entities['product_id']}")
            elif entities.get('year'):
                stats.append(f"📍 Analyzing ONLY Year {entities['year']}")
            elif entities.get('is_comparison'):
                values = entities.get('comparison_values', [])
                stats.append(f"📍 COMPARISON: {' vs '.join(map(str, values))}")
            else:
                stats.append(f"📍 Analyzing ALL data (overall view)")
            
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
            else:
                min_val = df[col].min()
                max_val = df[col].max()
                avg_val = df[col].mean()
                column_details.append(f"- {col} ({dtype}): Range [{min_val:,.0f} to {max_val:,.0f}], Avg: {avg_val:,.0f}")
        
        return "\n".join(column_details)
    
    def _build_context_description(self, query: str, entities: dict, df: pd.DataFrame) -> str:
        """Build a natural language description of the context"""
        descriptions = []
        
        if entities.get('customer_id'):
            customer_id = entities['customer_id']
            customer_orders = len(df)
            customer_revenue = df['NetAmount'].sum()
            descriptions.append(f"Customer Focus: Analyzing customer {customer_id} with {customer_orders} orders and ${customer_revenue:,.2f} in revenue.")
            descriptions.append("The dashboard should focus on THIS customer's behavior, not compare with other customers.")
        
        elif entities.get('product_id'):
            product_id = entities['product_id']
            product_orders = len(df)
            product_revenue = df['NetAmount'].sum()
            descriptions.append(f"Product Focus: Analyzing product {product_id} with {product_orders} orders and ${product_revenue:,.2f} in revenue.")
            descriptions.append("The dashboard should focus on THIS product's performance and who buys it.")
        
        elif entities.get('year'):
            year = entities['year']
            descriptions.append(f"Year Focus: Analyzing year {year} specifically.")
            descriptions.append("Show monthly trends and patterns within this year.")
        
        elif entities.get('is_comparison'):
            dim = entities.get('comparison_dimension', 'unknown')
            values = entities.get('comparison_values', [])
            descriptions.append(f"Comparison Analysis: Comparing {dim} between {' vs '.join(map(str, values))}.")
            descriptions.append("Show side-by-side comparisons and highlight differences.")
        
        else:
            descriptions.append("Overall Business Analysis: No specific filters applied.")
            descriptions.append("Show comprehensive view with top performers, trends, and distributions.")
        
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
                logger.error(f"[Chart {i+1}] ❌ No transformation code provided")
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
                    logger.error(f"[Chart {i+1}] ❌ Code did not create 'df_chart'")
                    self.chart_data_cache[chart_key] = None
                    continue
                
                if x_col not in df_chart.columns or y_col not in df_chart.columns:
                    logger.error(f"[Chart {i+1}] ❌ Required columns missing: {x_col}, {y_col}")
                    logger.error(f"Available columns: {list(df_chart.columns)}")
                    self.chart_data_cache[chart_key] = None
                    continue
                
                if len(df_chart) == 0:
                    logger.warning(f"[Chart {i+1}] ⚠️ Transformation produced empty dataframe")
                    self.chart_data_cache[chart_key] = None
                    continue
                
                # Success!
                self.chart_data_cache[chart_key] = {
                    'data': df_chart,
                    'config': chart_config
                }
                logger.info(f"[Chart {i+1}] ✅ Transformation successful: {len(df_chart)} rows")
                
            except Exception as e:
                logger.error(f"[Chart {i+1}] ❌ Transformation failed: {e}", exc_info=True)
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
                    logger.warning(f"[Chart {i+1}] ⚠️ No cached data - skipping")
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
                    logger.info(f"[Chart {i+1}] ✅ SUCCESS")
            except Exception as e:
                logger.error(f"[Chart {i+1}] ❌ FAILED: {str(e)}", exc_info=True)
        
        logger.info(f"[Charts] ✅ Generated {len(charts)} charts")
        return charts
    
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
        """Calculate key metrics"""
        try:
            metrics = {
                'total_revenue': float(df['NetAmount'].sum()),
                'total_orders': len(df),
                'avg_order_value': float(df['NetAmount'].mean()),
                'unique_customers': int(df['SoldToParty'].nunique()),
                'unique_products': int(df['Product'].nunique()),
                'date_range': f"{df['CreationDate'].min().strftime('%Y-%m-%d')} to {df['CreationDate'].max().strftime('%Y-%m-%d')}",
                'total_cost': float(df['CostAmount'].sum()),
                'total_tax': float(df['TaxAmount'].sum()),
            }
            
            # Top customer
            try:
                top_customer_series = df.groupby('SoldToParty')['NetAmount'].sum()
                metrics['top_customer'] = str(top_customer_series.idxmax())
                metrics['top_customer_revenue'] = float(top_customer_series.max())
            except:
                metrics['top_customer'] = "N/A"
                metrics['top_customer_revenue'] = 0.0
            
            # Top product
            try:
                top_product_series = df.groupby('Product')['NetAmount'].sum()
                metrics['top_product'] = str(top_product_series.idxmax())
                metrics['top_product_revenue'] = float(top_product_series.max())
            except:
                metrics['top_product'] = "N/A"
                metrics['top_product_revenue'] = 0.0
            
            # Profit margin
            if metrics['total_revenue'] > 0:
                metrics['profit_margin'] = ((metrics['total_revenue'] - metrics['total_cost']) / metrics['total_revenue']) * 100
            else:
                metrics['profit_margin'] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"[Metrics] Failed: {e}")
            return {}
    
    def _llm_generate_insights(self, query: str, metrics: Dict, entities: dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate insights using LLM"""
        
        context_desc = self._build_context_description(query, entities, df)
        
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

Provide:

EXECUTIVE_SUMMARY:
[2-3 sentences summarizing key insights]

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
        return {
            'executive_summary': f"Analysis period: {metrics.get('date_range', 'N/A')}. Revenue: ${metrics.get('total_revenue', 0):,.2f} from {metrics.get('total_orders', 0):,} orders.",
            'key_findings': [
                f"Total Revenue: ${metrics.get('total_revenue', 0):,.2f}",
                f"Average Order: ${metrics.get('avg_order_value', 0):,.2f}",
                f"Top Customer: {metrics.get('top_customer', 'N/A')}",
                f"Profit Margin: {metrics.get('profit_margin', 0):.1f}%"
            ],
            'recommendations': ["Focus on top accounts", "Diversify customer base", "Monitor trends"]
        }
    
    def _generate_chart_insights(self, charts: List[Dict]) -> List[str]:
        """Generate insights for each chart"""
        insights = []
        
        for chart_data in charts:
            chart_key = chart_data.get('chart_key')
            config = chart_data.get('config', {})
            title = chart_data.get('title', 'Chart')
            
            try:
                cached_data = self.chart_data_cache.get(chart_key)
                
                if not cached_data or cached_data.get('data') is None:
                    insights.append("Data unavailable")
                    continue
                
                df_agg = cached_data['data']
                y_col = config.get('y_column')
                
                # Simple insight generation
                if len(df_agg) > 0:
                    max_val = df_agg[y_col].max()
                    min_val = df_agg[y_col].min()
                    avg_val = df_agg[y_col].mean()
                    
                    insights.append(f"{title}: Range from {min_val:,.0f} to {max_val:,.0f}, Average: {avg_val:,.0f}")
                else:
                    insights.append("No data")
                
            except Exception as e:
                logger.error(f"[Insight] Error: {e}")
                insights.append(f"Analysis of {title}")
        
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
            # ✅ FIX: Corrected regex pattern
            if line.startswith(('-', '•', '*')) or re.match(r'^\d+[\.\)]\s', line):
                # Remove bullet characters and numbers
                bullet = re.sub(r'^[-•*]|\d+[\.\)]\s*', '', line).strip()  # ← FIXED REGEX
                if bullet:
                    bullets.append(bullet)
        
        return bullets[:6]
    
    # ===========================
    # HTML GENERATION
    # ===========================
    
    def _create_dashboard_html(self, charts: List[Dict[str, Any]], dashboard_plan: Dict[str, Any],
                               query: str, entities: dict, insights: Dict[str, Any]) -> str:
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
        .insights-box li::before {{ content: "▸"; color: #667eea; font-weight: bold; position: absolute; left: 8px; top: 10px; }}
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
            <h1>📊 {title}</h1>
            <p class="description">{description}</p>
            <div class="metadata">
                <div class="metadata-item"><strong>Generated:</strong> {datetime.now().strftime('%B %d, %Y at %H:%M')}</div>
                <div class="metadata-item"><strong>Query:</strong> "{query}"</div>
                <div class="metadata-item"><strong>Charts:</strong> {len(charts)}</div>
                <div class="metadata-item"><strong>🤖 LLM-Powered:</strong> Fully Dynamic</div>
            </div>
        </div>
        
        <div class="insights-box">
            <h2>📈 Executive Summary</h2>
            <p>{insights.get('executive_summary', 'No summary.')}</p>
            
            <h3>🔑 Key Findings</h3>
            <ul>
                {"".join(f"<li>{f}</li>" for f in insights.get('key_findings', ['No findings']))}
            </ul>
            
            <h3>💡 Recommendations</h3>
            <ul>
                {"".join(f"<li>{r}</li>" for r in insights.get('recommendations', ['No recommendations']))}
            </ul>
        </div>
        
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
            <p><strong>🤖 AI-Generated Dashboard</strong></p>
            <p style="margin-top: 8px; color: #666; font-size: 0.9rem;">Powered by Gemini LLM -  Fully Dynamic Analysis</p>
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
        
        logger.info(f"✅ [Dashboard] Saved: {filepath}")
        return filepath
