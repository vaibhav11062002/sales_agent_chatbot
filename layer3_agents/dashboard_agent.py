import pandas as pd
import logging
from typing import Dict, Any, List
from data_connector import mcp_store
import os
import json
import re
from datetime import datetime

logger = logging.getLogger(__name__)

# Import visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Install with: pip install plotly")

# Import LLM
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class DashboardAgent:
    """Dynamic Dashboard Agent - Creates visualizations based on context and user requests"""
    
    def __init__(self):
        self.name = "DashboardAgent"
        self.llm = None
        
        if not PLOTLY_AVAILABLE:
            logger.error("[DashboardAgent] Plotly not available! Dashboard generation will fail.")
            return
        
        # Initialize LLM for dashboard planning
        try:
            if LLM_AVAILABLE:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0.3,
                    api_key="AIzaSyBvGk-pDi2hqdq0CLSoKV2Sa8TH5IWShtE"
                )
                logger.info("[DashboardAgent] Initialized with LLM-powered dashboard generation")
        except Exception as e:
            logger.warning(f"[DashboardAgent] Could not initialize LLM: {e}")
    
    def execute(self, query: str, entities: dict = None) -> Dict[str, Any]:
        """Main execution method - generates dashboard based on query and context"""
        logger.info(f"[{self.name}] Received query: '{query}' | entities={entities}")
        
        if not PLOTLY_AVAILABLE:
            return {
                "status": "error",
                "message": "Plotly not installed. Run: pip install plotly"
            }
        
        try:
            # Get data
            df = mcp_store.get_sales_data()
            
            # ✅ USE INTELLIGENT CONTEXT RESOLUTION
            resolved_entities = self._resolve_dashboard_context(query, entities)
            
            logger.info(f"[{self.name}] Final resolved entities for dashboard: {resolved_entities}")
            
            # Get previous analysis context
            context_stack = mcp_store.get_context_stack()
            previous_analysis = self._get_previous_analysis_context(context_stack)
            
            # Use LLM to plan dashboard
            if self.llm:
                dashboard_plan = self._llm_plan_dashboard(
                    query, 
                    resolved_entities, 
                    previous_analysis,
                    df
                )
            else:
                dashboard_plan = self._fallback_dashboard_plan(resolved_entities, df)
            
            logger.info(f"[{self.name}] Dashboard plan: {dashboard_plan.get('title')}")
            
            # Generate visualizations
            charts = self._generate_charts(dashboard_plan, df, resolved_entities)
            
            # Create dashboard HTML
            dashboard_html = self._create_dashboard_html(charts, dashboard_plan, query, resolved_entities)
            
            # Save dashboard
            output_path = self._save_dashboard(dashboard_html, resolved_entities)
            
            # Prepare result
            result = {
                "status": "success",
                "dashboard_plan": dashboard_plan,
                "charts_generated": len(charts),
                "output_path": output_path,
                "dashboard_html": dashboard_html
            }
            
            # Update context
            mcp_store.update_agent_context(self.name, {
                "query": query,
                "entities": resolved_entities,
                "results": result,
                "dashboard_plan": dashboard_plan
            })
            
            # Update dialogue state
            dashboard_entities = {**resolved_entities, "dashboard_created": True}
            mcp_store.update_dialogue_state(
                dashboard_entities, 
                query, 
                f"Dashboard created with {len(charts)} visualizations at {output_path}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }
    
    # ===========================
    # INTELLIGENT CONTEXT RESOLUTION
    # ===========================
    
    def _resolve_dashboard_context(self, query: str, entities: dict = None) -> dict:
        """
        Intelligently resolve which entities to use for dashboard based on query.
        
        Handles:
        - "for the same" -> use most recent entity
        - "above customer and product" -> combine entities
        - "product analysis" -> clear filters (general view)
        - Specific mentions -> use only those
        
        Returns:
            dict: Resolved entities to use for filtering
        """
        query_lower = query.lower()
        
        # Get dialogue state
        dialogue_state = mcp_store.get_current_dialogue_state()
        all_entities = dialogue_state.get("entities", {})
        entity_stack = dialogue_state.get("context_stack", [])
        
        logger.info("="*60)
        logger.info("🔍 DASHBOARD CONTEXT RESOLUTION")
        logger.info("="*60)
        logger.info(f"Query: {query}")
        logger.info(f"Passed entities: {entities}")
        logger.info(f"All entities in state: {list(all_entities.keys())}")
        logger.info(f"Context stack size: {len(entity_stack)}")
        
        # ✅ CASE 1: "for the same" or "for that" - use most recent relevant entity
        if any(phrase in query_lower for phrase in ['for the same', 'for that', 'for this', 'same thing', 'that one', 'that particular']):
            logger.info("📌 Detected: 'for the same' reference")
            
            recent_entities = self._get_most_recent_entities(entity_stack, all_entities)
            
            logger.info(f"✅ Using 'same' reference: {recent_entities}")
            logger.info("="*60)
            return recent_entities
        
        # ✅ CASE 2: "above customer and product" or "both" - combine specific entities
        if any(phrase in query_lower for phrase in ['above', 'both', 'customer and product', 'product and customer', 'that customer and product']):
            logger.info("📌 Detected: Combined reference (above/both)")
            
            combined = {}
            
            # Find most recent customer and product
            if entity_stack:
                for context in reversed(entity_stack):
                    if 'customer_id' in context and context['customer_id'] and 'customer_id' not in combined:
                        combined['customer_id'] = context['customer_id']
                        logger.info(f"  └─ Found customer from context: {combined['customer_id']}")
                    
                    if 'product_id' in context and context['product_id'] and 'product_id' not in combined:
                        combined['product_id'] = context['product_id']
                        logger.info(f"  └─ Found product from context: {combined['product_id']}")
            
            # Fallback to all_entities
            if not combined.get('customer_id') and all_entities.get('customer_id'):
                combined['customer_id'] = all_entities['customer_id']
            
            if not combined.get('product_id') and all_entities.get('product_id'):
                combined['product_id'] = all_entities['product_id']
            
            logger.info(f"✅ Using combined entities: {combined}")
            logger.info("="*60)
            return combined
        
        # ✅ CASE 3: General analysis (no specific entity) - clear filters
        if any(phrase in query_lower for phrase in [
            'product analysis',
            'all products',
            'customer analysis',
            'all customers',
            'overall',
            'general',
            'summary',
            'overview'
        ]):
            logger.info("📌 Detected: General analysis - clearing entity filters")
            
            # Keep only time filters (year) if present
            general_context = {}
            
            # Check for year in passed entities
            if entities and 'year' in entities:
                general_context['year'] = entities['year']
                logger.info(f"  └─ Keeping year from entities: {general_context['year']}")
            # Check for year in query
            elif any(str(year) in query for year in [2023, 2024, 2025]):
                for year in [2023, 2024, 2025]:
                    if str(year) in query:
                        general_context['year'] = year
                        logger.info(f"  └─ Extracted year from query: {year}")
                        break
            
            logger.info(f"✅ Using general context: {general_context}")
            logger.info("="*60)
            return general_context
        
        # ✅ CASE 4: Specific entity mentioned in query - use only that
        specific_entities = self._extract_specific_entities_from_query(query)
        
        if specific_entities:
            logger.info(f"✅ Using specific entities from query: {specific_entities}")
            logger.info("="*60)
            return specific_entities
        
        # ✅ CASE 5: Passed entities from router
        if entities and any(k in entities for k in ['customer_id', 'product_id', 'year', 'sales_org']):
            logger.info(f"📌 Using entities from router")
            
            # Filter out non-dashboard entities
            dashboard_entities = self._filter_dashboard_entities(entities)
            
            logger.info(f"✅ Filtered dashboard entities: {dashboard_entities}")
            logger.info("="*60)
            return dashboard_entities
        
        # ✅ CASE 6: Default - use most recent context but only dashboard-relevant entities
        logger.info("📌 Using recent context (default)")
        
        recent_context = self._filter_dashboard_entities(all_entities)
        
        logger.info(f"✅ Using recent context: {recent_context}")
        logger.info("="*60)
        return recent_context
    
    def _get_most_recent_entities(self, entity_stack: list, all_entities: dict) -> dict:
        """Get most recent entity of each type from context stack"""
        recent_entities = {}
        
        # Check what was most recently discussed
        if entity_stack:
            # Get the most recent context that has actual entity values
            for context in reversed(entity_stack):
                if 'customer_id' in context and context['customer_id'] and 'customer_id' not in recent_entities:
                    recent_entities['customer_id'] = context['customer_id']
                    logger.info(f"  └─ Found recent customer: {context['customer_id']}")
                
                if 'product_id' in context and context['product_id'] and 'product_id' not in recent_entities:
                    recent_entities['product_id'] = context['product_id']
                    logger.info(f"  └─ Found recent product: {context['product_id']}")
                
                if 'year' in context and context['year'] and 'year' not in recent_entities:
                    recent_entities['year'] = context['year']
                    logger.info(f"  └─ Found recent year: {context['year']}")
                
                # If we found any entity, use just the most recent one (not all)
                if recent_entities:
                    break
        
        # Fallback to all_entities if nothing in stack
        if not recent_entities:
            if all_entities.get('customer_id'):
                recent_entities['customer_id'] = all_entities['customer_id']
            
            if all_entities.get('product_id'):
                recent_entities['product_id'] = all_entities['product_id']
            
            if all_entities.get('year'):
                recent_entities['year'] = all_entities['year']
        
        return recent_entities
    
    def _extract_specific_entities_from_query(self, query: str) -> dict:
        """Extract specific entities mentioned in the query"""
        query_lower = query.lower()
        specific_entities = {}
        
        # Check for specific customer mention
        customer_match = re.search(r'customer\s+(\d+)', query_lower)
        if customer_match:
            specific_entities['customer_id'] = customer_match.group(1)
            logger.info(f"📌 Detected specific customer: {specific_entities['customer_id']}")
        
        # Check for specific product mention
        product_match = re.search(r'product\s+([A-Z0-9]+)', query, re.IGNORECASE)
        if product_match:
            specific_entities['product_id'] = product_match.group(1)
            logger.info(f"📌 Detected specific product: {specific_entities['product_id']}")
        
        # Check for year mention
        year_match = re.search(r'(20\d{2})', query)
        if year_match:
            specific_entities['year'] = int(year_match.group(1))
            logger.info(f"📌 Detected specific year: {specific_entities['year']}")
        
        return specific_entities
    
    def _filter_dashboard_entities(self, entities: dict) -> dict:
        """Filter to keep only dashboard-relevant entities"""
        DASHBOARD_KEYS = ['customer_id', 'product_id', 'year', 'sales_org', 'category']
        
        dashboard_entities = {}
        for key in DASHBOARD_KEYS:
            if key in entities and entities[key]:
                dashboard_entities[key] = entities[key]
        
        return dashboard_entities
    
    # ===========================
    # DASHBOARD PLANNING & GENERATION
    # ===========================
    
    def _get_previous_analysis_context(self, context_stack: List[Dict]) -> Dict[str, Any]:
        """Extract previous analysis results from context stack"""
        if not context_stack:
            return {}
        
        for ctx in context_stack[:5]:
            if ctx.get('query_type') in ['analysis', 'comparison', 'trend']:
                return {
                    'query': ctx.get('query'),
                    'entities': ctx.get('entities', {}),
                    'response': ctx.get('response', '')
                }
        
        return {}
    
    def _llm_plan_dashboard(
        self, 
        query: str, 
        entities: dict, 
        previous_analysis: dict,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Use LLM to plan dashboard layout"""
        
        data_summary = self._get_data_summary(df, entities)
        
        prompt = f"""You are a business intelligence dashboard designer. Create a comprehensive, insightful dashboard plan.

**User Query:** "{query}"

**Available Entities:** {json.dumps(entities, indent=2)}

**Previous Analysis:**
{previous_analysis.get('response', 'No previous analysis')[:500] if previous_analysis else 'No previous analysis'}

**Data Summary:**
{data_summary}

**Available Columns:** CreationDate, NetAmount, OrderQuantity, TaxAmount, CostAmount, SoldToParty (customer), Product, SalesDocument, SalesOrganization

**Task:** Design a dashboard with 4-6 powerful visualizations that tell a complete story.

**Chart Types Available:**
- bar: Compare categories
- line: Show trends over time
- pie: Show proportions
- scatter: Show correlations
- area: Show cumulative trends

**Response Format (JSON only):**
{{
  "title": "Clear Dashboard Title",
  "description": "What insights this dashboard provides",
  "charts": [
    {{
      "type": "bar|line|pie|scatter|area",
      "title": "Specific Chart Title",
      "description": "What this shows",
      "x_column": "column_name",
      "y_column": "column_name",
      "aggregation": "sum|mean|count|none",
      "top_n": 10,
      "sort_by": "value|name"
    }},
    ...
  ]
}}

Be creative and create visualizations that provide deep business insights."""

        try:
            response = self.llm.invoke(prompt)
            content = response.content
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                logger.info(f"[LLM Dashboard] Planned: {plan.get('title')}")
                return plan
            else:
                logger.warning("[LLM Dashboard] No JSON found")
                return self._fallback_dashboard_plan(entities, df)
                
        except Exception as e:
            logger.error(f"[LLM Dashboard] Failed: {e}")
            return self._fallback_dashboard_plan(entities, df)
    
    def _fallback_dashboard_plan(self, entities: dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback dashboard plan"""
        year = entities.get('year', entities.get('target_year'))
        customer = entities.get('customer_id')
        product = entities.get('product_id')
        
        # Determine title based on context
        title_parts = ["Sales Analytics Dashboard"]
        if customer:
            title_parts.append(f"Customer {customer}")
        if product:
            title_parts.append(f"Product {product}")
        if year:
            title_parts.append(str(year))
        
        title = " - ".join(title_parts)
        
        return {
            "title": title,
            "description": "Comprehensive sales performance analysis",
            "charts": [
                {
                    "type": "line",
                    "title": "Revenue Trend Over Time",
                    "x_column": "CreationDate",
                    "y_column": "NetAmount",
                    "aggregation": "sum"
                },
                {
                    "type": "bar",
                    "title": "Top 10 Customers by Revenue",
                    "x_column": "SoldToParty",
                    "y_column": "NetAmount",
                    "aggregation": "sum",
                    "top_n": 10
                },
                {
                    "type": "pie",
                    "title": "Revenue Distribution by Product",
                    "x_column": "Product",
                    "y_column": "NetAmount",
                    "aggregation": "sum",
                    "top_n": 8
                },
                {
                    "type": "bar",
                    "title": "Order Volume by Sales Organization",
                    "x_column": "SalesOrganization",
                    "y_column": "OrderQuantity",
                    "aggregation": "sum"
                }
            ]
        }
    
    def _get_data_summary(self, df: pd.DataFrame, entities: dict) -> str:
        """Generate data summary"""
        filtered_df = self._filter_dataframe(df, entities)
        
        if len(filtered_df) == 0:
            return "No data available after filtering"
        
        return f"""
- Records: {len(filtered_df):,}
- Date Range: {filtered_df['CreationDate'].min()} to {filtered_df['CreationDate'].max()}
- Total Revenue: ${filtered_df['NetAmount'].sum():,.2f}
- Customers: {filtered_df['SoldToParty'].nunique()}
- Products: {filtered_df['Product'].nunique()}
- Avg Order: ${filtered_df['NetAmount'].mean():,.2f}
"""
    
    def _filter_dataframe(self, df: pd.DataFrame, entities: dict) -> pd.DataFrame:
        """Filter dataframe by entities"""
        filtered = df.copy()
        initial_count = len(filtered)
        
        if 'year' in entities or 'target_year' in entities:
            year = entities.get('year') or entities.get('target_year')
            filtered = filtered[filtered['CreationDate'].dt.year == year]
            logger.info(f"  └─ Year filter {year}: {initial_count} → {len(filtered)} rows")
        
        if 'customer_id' in entities or 'customer' in entities:
            customer = entities.get('customer_id') or entities.get('customer')
            filtered = filtered[filtered['SoldToParty'] == str(customer)]
            logger.info(f"  └─ Customer filter {customer}: {initial_count} → {len(filtered)} rows")
        
        if 'product_id' in entities or 'product' in entities:
            product = entities.get('product_id') or entities.get('product')
            filtered = filtered[filtered['Product'] == str(product)]
            logger.info(f"  └─ Product filter {product}: {initial_count} → {len(filtered)} rows")
        
        logger.info(f"[Filter] Final: {len(df)} → {len(filtered)} rows")
        return filtered
    
    def _generate_charts(
        self, 
        dashboard_plan: Dict[str, Any], 
        df: pd.DataFrame, 
        entities: dict
    ) -> List[go.Figure]:
        """Generate Plotly charts"""
        charts = []
        filtered_df = self._filter_dataframe(df, entities)
        
        if len(filtered_df) == 0:
            logger.warning(f"[Charts] No data after filtering! Creating empty state.")
            return []
        
        for chart_config in dashboard_plan.get('charts', []):
            try:
                chart = self._create_chart(chart_config, filtered_df)
                if chart:
                    charts.append(chart)
            except Exception as e:
                logger.error(f"[Chart] Failed {chart_config.get('title')}: {e}")
        
        return charts
    
    def _create_chart(self, config: dict, df: pd.DataFrame) -> go.Figure:
        """Create individual chart"""
        chart_type = config.get('type', 'bar')
        title = config.get('title', 'Chart')
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        agg = config.get('aggregation', 'sum')
        top_n = config.get('top_n')
        
        logger.info(f"[Chart] Creating {chart_type}: {title}")
        
        # Aggregate data
        if agg and agg != 'none':
            if chart_type == 'line' and x_col == 'CreationDate':
                df_agg = df.groupby(pd.Grouper(key='CreationDate', freq='ME')).agg({
                    y_col: agg
                }).reset_index()
            else:
                df_agg = df.groupby(x_col).agg({y_col: agg}).reset_index()
                df_agg = df_agg.sort_values(y_col, ascending=False)
                if top_n:
                    df_agg = df_agg.head(top_n)
        else:
            df_agg = df[[x_col, y_col]].copy()
        
        # Create chart
        if chart_type == 'bar':
            fig = px.bar(df_agg, x=x_col, y=y_col, title=title)
        elif chart_type == 'line':
            fig = px.line(df_agg, x=x_col, y=y_col, title=title, markers=True)
        elif chart_type == 'pie':
            fig = px.pie(df_agg, names=x_col, values=y_col, title=title)
        elif chart_type == 'area':
            fig = px.area(df_agg, x=x_col, y=y_col, title=title)
        elif chart_type == 'scatter':
            fig = px.scatter(df_agg, x=x_col, y=y_col, title=title)
        else:
            fig = px.bar(df_agg, x=x_col, y=y_col, title=title)
        
        # Styling
        fig.update_layout(
            template='plotly_white',
            height=450,
            margin=dict(l=50, r=50, t=60, b=50),
            font=dict(size=12)
        )
        
        return fig
    
    def _create_dashboard_html(
        self, 
        charts: List[go.Figure], 
        dashboard_plan: Dict[str, Any],
        query: str,
        entities: dict
    ) -> str:
        """Create complete dashboard HTML"""
        
        title = dashboard_plan.get('title', 'Sales Dashboard')
        description = dashboard_plan.get('description', '')
        
        # Filter entities for display
        display_entities = {k: v for k, v in entities.items() if k in ['customer_id', 'product_id', 'year', 'sales_org']}
        
        html_parts = [f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .header h1 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .header .description {{
            color: #666;
            font-size: 1.1em;
            margin-bottom: 15px;
        }}
        .header .metadata {{
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 2px solid #f0f0f0;
        }}
        .metadata-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: #555;
        }}
        .metadata-item strong {{
            color: #667eea;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .chart-container:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        }}
        .footer {{
            text-align: center;
            padding: 30px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
            color: #666;
        }}
        .footer strong {{
            color: #667eea;
        }}
        @media print {{
            body {{ background: white; }}
            .header, .chart-container, .footer {{ box-shadow: none; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {title}</h1>
            <p class="description">{description}</p>
            <div class="metadata">
                <div class="metadata-item">
                    <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
                <div class="metadata-item">
                    <strong>Query:</strong> "{query}"
                </div>
                <div class="metadata-item">
                    <strong>Charts:</strong> {len(charts)}
                </div>
                {"".join(f'<div class="metadata-item"><strong>{k.replace("_", " ").title()}:</strong> {v}</div>' for k, v in display_entities.items())}
            </div>
        </div>
        
        <div class="charts-grid">
"""]
        
        # Add charts
        for i, fig in enumerate(charts):
            chart_div = fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"chart-{i}")
            html_parts.append(f'<div class="chart-container">{chart_div}</div>')
        
        html_parts.append(f"""
        </div>
        
        <div class="footer">
            <p><strong>Sales Agent AI</strong> • Powered by Gemini & Plotly</p>
            <p style="margin-top: 10px; font-size: 0.9em;">Generated {len(charts)} visualizations from {len(mcp_store.get_sales_data())} sales records</p>
        </div>
    </div>
</body>
</html>
""")
        
        return '\n'.join(html_parts)
    
    def _save_dashboard(self, html_content: str, entities: dict) -> str:
        """Save dashboard to file"""
        os.makedirs('dashboards', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Build filename based on entities
        filename_parts = ["dashboard"]
        
        if entities.get('customer_id'):
            filename_parts.append(f"cust{entities['customer_id']}")
        
        if entities.get('product_id'):
            filename_parts.append(f"prod{entities['product_id']}")
        
        year = entities.get('year', entities.get('target_year'))
        if year:
            filename_parts.append(str(year))
        else:
            filename_parts.append("all")
        
        filename_parts.append(timestamp)
        
        filename = "_".join(filename_parts) + ".html"
        filepath = os.path.join('dashboards', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"[Dashboard] Saved: {filepath}")
        return filepath
