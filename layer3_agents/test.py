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

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class DashboardAgent:
    """✅ CONTEXT-AWARE & ROBUST Dashboard Generation"""
    
    def __init__(self):
        self.name = "DashboardAgent"
        self.llm = None
        self.analysis_agent_ref = None
        self.chart_data_cache = {}
        self.table_data = None
        
        # Column mappings
        self.date_column = None
        self.revenue_column = None
        self.customer_column = None
        self.product_column = None
        self.sales_org_column = None
        self.cost_column = None
        self.order_id_column = None
        
        if not MATPLOTLIB_AVAILABLE:
            logger.error("[DashboardAgent] Matplotlib not available!")
            return
        
        try:
            if LLM_AVAILABLE:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0.2,  # ✅ Lower temp for more consistent output
                    api_key=GEMINI_API_KEY,
                    max_output_tokens=8192
                )
                logger.info("[DashboardAgent] ✅ Context-aware mode initialized")
            else:
                logger.error("[DashboardAgent] ❌ LLM not available")
        except Exception as e:
            logger.warning(f"[DashboardAgent] Could not initialize LLM: {e}")
    
    def _detect_columns(self, df: pd.DataFrame):
        """Detect column names"""
        date_candidates = ['Date', 'CreationDate', 'SalesDocumentDate', 'TransactionDate', 'OrderDate']
        for col in date_candidates:
            if col in df.columns:
                self.date_column = col
                break
        
        revenue_candidates = ['Revenue', 'NetAmount', 'Sales', 'TotalSales', 'Amount']
        for col in revenue_candidates:
            if col in df.columns:
                self.revenue_column = col
                break
        
        customer_candidates = ['Customer', 'SoldToParty', 'CustomerID', 'CustomerId']
        for col in customer_candidates:
            if col in df.columns:
                self.customer_column = col
                break
        
        product_candidates = ['Product', 'ProductID', 'Material', 'ProductId']
        for col in product_candidates:
            if col in df.columns:
                self.product_column = col
                break
        
        order_id_candidates = ['OrderID', 'SalesDocument', 'InvoiceID', 'TransactionID']
        for col in order_id_candidates:
            if col in df.columns:
                self.order_id_column = col
                break
    
    def _determine_dashboard_type(self, entities: dict, performance: str, entitytype: str) -> dict:
        """
        🎯 FULLY DYNAMIC: Determines dashboard context based on ANY entity type
        Works for customers, products, sales orgs, regions, etc.
        """
        
        # Detect the entity being analyzed
        entity_key = None
        entity_value = None
        entity_label = "Entity"
        
        # Map entity keys to human-readable labels
        entity_map = {
            'customer_id': ('Customer', self.customer_column),
            'product_id': ('Product', self.product_column),
            'sales_org': ('Sales Organization', self.sales_org_column),
            'year': ('Year', None),
            'region': ('Region', 'Region'),
            'material': ('Material', 'Material'),
            'sold_to_party': ('Sold To Party', 'SoldToParty')
        }
        
        # Find which entity is being analyzed
        for key, (label, column) in entity_map.items():
            if key in entities and entities[key]:
                entity_key = key
                entity_value = entities[key]
                entity_label = label
                break
        
        # Build dynamic scope description
        if entity_value:
            scope_base = f"{entity_label} {entity_value}"
            
            # Add performance indicator
            if performance == 'HIGH':
                scope = f"{scope_base} (🏆 TOP PERFORMER)"
                performance_emoji = "🏆"
                performance_word = "HIGHEST"
            elif performance == 'LOW':
                scope = f"{scope_base} (🚨 UNDERPERFORMER - CRITICAL)"
                performance_emoji = "🚨"
                performance_word = "LOWEST"
            else:
                scope = f"{scope_base} Analysis"
                performance_emoji = "📊"
                performance_word = "STANDARD"
        else:
            scope = "Overall Business Performance"
            performance_emoji = "📊"
            performance_word = "OVERALL"
        
        # Return dynamic configuration that LLM will interpret
        return {
            'scope': scope,
            'entity_type': entity_label,
            'entity_value': entity_value,
            'entity_key': entity_key,
            'performance': performance,
            'performance_emoji': performance_emoji,
            'performance_word': performance_word,
            
            # Dynamic instructions for LLM
            'instructions': f'''
    You are creating a dashboard for: **{scope}**

    **PERFORMANCE LEVEL: {performance}**

    **CRITICAL CONTEXT RULES:**

    {'**🏆 HIGH PERFORMER DASHBOARD:**' if performance == 'HIGH' else ''}
    {'- This is a TOP-PERFORMING ' + entity_label.lower() if performance == 'HIGH' else ''}
    {'- Tone: POSITIVE, CELEBRATORY, SUCCESS-FOCUSED' if performance == 'HIGH' else ''}
    {'- Focus: achievements, growth, scaling opportunities, market leadership' if performance == 'HIGH' else ''}
    {'- Language: "top performer", "market leader", "strong growth", "scaling opportunity"' if performance == 'HIGH' else ''}
    {'- Recommendations: expansion, scaling, premium strategies, capacity planning' if performance == 'HIGH' else ''}
    {'- FORBIDDEN: "at-risk", "declining", "needs improvement", "retention"' if performance == 'HIGH' else ''}

    {'**🚨 LOW PERFORMER DASHBOARD:**' if performance == 'LOW' else ''}
    {'- This is an UNDERPERFORMING ' + entity_label.lower() if performance == 'LOW' else ''}
    {'- Tone: URGENT, CONCERNED, ACTION-ORIENTED' if performance == 'LOW' else ''}
    {'- Focus: decline analysis, churn risk, retention, root cause investigation' if performance == 'LOW' else ''}
    {'- Language: "critical", "alarming decline", "at-risk", "urgent intervention needed"' if performance == 'LOW' else ''}
    {'- Recommendations: retention, recovery, investigation, intervention, possibly discontinuation' if performance == 'LOW' else ''}
    {'- FORBIDDEN: "growth opportunity", "strong performance", "cross-sell", "upsell"' if performance == 'LOW' else ''}

    {'**📊 STANDARD DASHBOARD:**' if performance == 'STANDARD' else ''}
    {'- Balanced analysis of ' + entity_label.lower() + ' performance' if performance == 'STANDARD' else ''}
    {'- Tone: ANALYTICAL, INFORMATIVE, BALANCED' if performance == 'STANDARD' else ''}
    {'- Focus: patterns, trends, insights, opportunities' if performance == 'STANDARD' else ''}

    **ENTITY-SPECIFIC GUIDANCE:**
    - For {entity_label}: Analyze from the perspective of THIS SPECIFIC {entity_label.lower()} only
    - Compare to company average/peers when relevant
    - Focus on metrics relevant to {entity_label.lower()} analysis
    - Table should show breakdown relevant to this {entity_label.lower()}
    - Charts should reveal patterns specific to this {entity_label.lower()}

    **MANDATORY:**
    1. Title MUST include "{entity_label} {entity_value}" and reflect {performance} level
    2. Executive summary MUST acknowledge this is a {performance} performer
    3. All insights must be specific to THIS {entity_label.lower()}, not general company
    4. Use actual entity ID "{entity_value}" (not placeholders like "Entity X")
    5. Recommendations must match performance level (retention for LOW, scaling for HIGH)
    ''',
            
            # Dynamic title guidance
            'title_guidance': f'Include "{entity_label} {entity_value}" and {"success/achievement" if performance == "HIGH" else "critical/urgent" if performance == "LOW" else "analysis/performance"} language',
            
            'title_example': f'{performance_emoji} {entity_label} {entity_value}: ' + (
                'Market Leader & Top Revenue Driver' if performance == 'HIGH'
                else 'Critical Alert - Underperforming & At-Risk' if performance == 'LOW'
                else 'Performance Analysis & Trends'
            ),
            
            'focus_areas': (
                'revenue excellence, market leadership, growth trends, scaling strategies' if performance == 'HIGH'
                else 'decline analysis, churn risk, retention urgency, root cause investigation' if performance == 'LOW'
                else 'trends, patterns, opportunities, performance metrics'
            ),
            
            # Dynamic table configuration
            'table_type': f'{entity_label} Performance Breakdown',
            'table_title': f'{performance_emoji} {entity_label} {entity_value} - Key Metrics',
            'table_code': self._generate_table_code(entity_key, entity_value),
            'table_columns': self._get_table_columns(entity_key),
            
            # Dynamic chart guidance
            'second_chart_title': f'Trends for {entity_label} {entity_value}',
            'second_chart_desc': f'Temporal patterns for this {entity_label.lower()}',
            'second_chart_code': self._generate_chart_code(entity_key, entity_value),
            'second_chart_x': self.date_column,
            
            'performance_phrase': f'{performance_word} performing {entity_label.lower()}',
            'tone': (
                'POSITIVE, ACHIEVEMENT-FOCUSED, and STRATEGIC' if performance == 'HIGH'
                else 'URGENT, ALARMED, and ACTION-ORIENTED' if performance == 'LOW'
                else 'analytical, balanced, and informative'
            ),
            
            'finding_example': f'{performance_emoji} {entity_label} {entity_value} ' + (
                f'leads with exceptional performance metrics' if performance == 'HIGH'
                else f'shows critical decline requiring immediate intervention' if performance == 'LOW'
                else f'demonstrates typical performance patterns'
            ),
            
            'recommendation_example': (
                f'Scale operations for {entity_label} {entity_value} to capitalize on strong market position' if performance == 'HIGH'
                else f'🚨 URGENT: Launch retention strategy for {entity_label} {entity_value} within 48 hours' if performance == 'LOW'
                else f'Continue monitoring {entity_label} {entity_value} and adjust strategies as needed'
            )
        }

    def _generate_table_code(self, entity_key: str, entity_value: str) -> str:
        """Generate dynamic table aggregation code based on entity type"""
        if entity_key == 'customer_id':
            # For customers: show quarterly revenue
            return f"df_table = df.copy(); df_table['Quarter'] = df_table['{self.date_column}'].dt.to_period('Q').astype(str); df_table = df_table.groupby('Quarter')['{self.revenue_column}'].sum().reset_index().sort_values('Quarter'); df_table['Quarter'] = df_table['Quarter'].apply(lambda x: f'Q{{x[-1]}} {{x[:4]}}'); df_table['Revenue'] = df_table['{self.revenue_column}']"
        
        elif entity_key == 'product_id':
            # For products: show top customers buying it
            return f"df_table = df.groupby('{self.customer_column}')['{self.revenue_column}'].sum().nlargest(10).reset_index(); df_table.columns = ['Customer', 'Revenue']"
        
        elif entity_key == 'sales_org':
            # For sales org: show product breakdown
            return f"df_table = df.groupby('{self.product_column}')['{self.revenue_column}'].sum().nlargest(10).reset_index(); df_table.columns = ['Product', 'Revenue']"
        
        else:
            # Default: quarterly breakdown
            return f"df_table = df.copy(); df_table['Quarter'] = df_table['{self.date_column}'].dt.to_period('Q').astype(str); df_table = df_table.groupby('Quarter')['{self.revenue_column}'].sum().reset_index().sort_values('Quarter'); df_table['Revenue'] = df_table['{self.revenue_column}']"

    def _get_table_columns(self, entity_key: str) -> str:
        """Get table column names based on entity type"""
        if entity_key == 'customer_id':
            return '["Quarter", "Revenue"]'
        elif entity_key == 'product_id':
            return '["Customer", "Revenue"]'
        elif entity_key == 'sales_org':
            return '["Product", "Revenue"]'
        else:
            return '["Period", "Revenue"]'

    def _generate_chart_code(self, entity_key: str, entity_value: str) -> str:
        """Generate dynamic chart code based on entity type"""
        if entity_key == 'product_id':
            # For products: show customer adoption
            return f"df_chart = df.groupby(pd.Grouper(key='{self.date_column}', freq='ME'))['{self.customer_column}'].nunique().reset_index(); df_chart.columns = ['{self.date_column}', 'Customer_Count']"
        
        elif entity_key == 'customer_id':
            # For customers: show product diversity
            return f"df_chart = df.groupby('{self.product_column}')['{self.revenue_column}'].sum().nlargest(10).reset_index()"
        
        else:
            # Default: revenue over time
            return f"df_chart = df.groupby(pd.Grouper(key='{self.date_column}', freq='ME'))['{self.revenue_column}'].sum().reset_index()"

    def execute(self, query: str, entities: dict = None, analysis_agent=None) -> Dict[str, Any]:
        """
        ✅ COMPLETE UPDATED: Main execution with all fixes
        - Uses anomaly-enriched data when available
        - Passes aggregation_type for performance detection
        - Handles entity context properly
        """
        logger.info(f"[{self.name}] 🎯 Query: {query}")
        
        if not MATPLOTLIB_AVAILABLE or not self.llm:
            return {'status': 'error', 'message': 'Required dependencies not available'}
        
        try:
            self.analysis_agent_ref = analysis_agent
            self.chart_data_cache = {}
            self.table_data = None
            
            # ✅ FIX 1: Check if anomaly-enriched data is available
            has_anomaly_data = False
            if mcp_store.has_enriched_data('anomalies'):
                logger.info(f"[{self.name}] 📊 Using ANOMALY-ENRICHED dataframe")
                df = mcp_store.get_enriched_data('anomalies')
                has_anomaly_data = 'is_anomaly' in df.columns
                
                if has_anomaly_data:
                    anomaly_count = df['is_anomaly'].sum()
                    logger.info(f"[{self.name}] ✅ Anomaly column detected: {anomaly_count} anomalies in {len(df)} records")
            else:
                logger.info(f"[{self.name}] 📊 Using STANDARD sales dataframe")
                df = mcp_store.get_sales_data()
            
            logger.info(f"[{self.name}] 📊 Data loaded: {len(df)} rows")
            
            # Detect columns FIRST before any stats calculation
            self.detect_columns(df)
            logger.info(f"[{self.name}] 🔍 Columns detected: Date={self.date_column}, Revenue={self.revenue_column}, Customer={self.customer_column}")
            
            # NOW get unfiltered stats (columns are already detected)
            unfiltered_stats = self._get_unfiltered_stats(df)
            logger.info(f"[{self.name}] 📈 Unfiltered stats: Total revenue=${unfiltered_stats.get('total_company_revenue', 0):,.0f}, Customers={unfiltered_stats.get('total_customers', 0)}")
            
            # ✅ FIX 2: Handle None entities gracefully
            if entities is None:
                entities = {}
            
            # Get dialogue state
            dialogue_state = mcp_store.get_current_dialogue_state()
            
            # Resolve entities from context
            resolved_entities = self._resolve_dashboard_context(query, entities)
            logger.info(f"[{self.name}] 📋 Resolved entities: {resolved_entities}")
            
            # ✅ FIX 3: Pass entities to semantic cache extraction (including aggregation_type)
            semantic_cache = self._extract_minimal_semantic_cache(dialogue_state, entities)
            logger.info(f"[{self.name}] 🧠 Semantic cache: Performance={semantic_cache.get('_performance')}, Entity={semantic_cache.get('_entity')}, ID={semantic_cache.get('_specific_entity_id')}")
            
            # Build full context for LLM
            full_context = {
                **resolved_entities,
                **semantic_cache,
                'unfiltered_stats': unfiltered_stats,
                'has_anomaly_data': has_anomaly_data  # ✅ Pass anomaly flag to LLM
            }
            
            # Step 1: LLM creates complete dashboard plan
            logger.info(f"[{self.name}] 🔄 Step 1/6: Calling LLM for dashboard plan...")
            dashboard_result = self._llm_create_complete_dashboard(query, full_context, df)
            
            if not dashboard_result:
                logger.error(f"[{self.name}] ❌ Step 1 FAILED: LLM returned None")
                return {'status': 'error', 'message': 'LLM could not generate dashboard plan'}
            
            if len(dashboard_result.get('plan', {}).get('charts', [])) == 0:
                logger.error(f"[{self.name}] ❌ Step 1 FAILED: No charts in plan")
                return {'status': 'error', 'message': 'LLM generated plan with no charts'}
            
            dashboard_plan = dashboard_result['plan']
            insights = dashboard_result['insights']
            chart_count = len(dashboard_plan.get('charts', []))
            
            logger.info(f"[{self.name}] ✅ Step 1 SUCCESS: Plan created with {chart_count} charts")
            logger.info(f"[{self.name}]   └─ Performance level detected: {semantic_cache.get('_performance', 'STANDARD')}")
            
            # Step 2: Generate aggregated table
            logger.info(f"[{self.name}] 🔄 Step 2/6: Generating aggregated table...")
            try:
                if dashboard_plan.get('table_spec'):
                    self._generate_aggregated_table(dashboard_plan, df, resolved_entities)
                    if self.table_data:
                        logger.info(f"[{self.name}] ✅ Step 2 SUCCESS: Table with {len(self.table_data['df'])} rows")
                    else:
                        logger.warning(f"[{self.name}] ⚠️ Step 2 WARNING: Table generation returned empty")
                else:
                    logger.info(f"[{self.name}] ⏭️ Step 2 SKIPPED: No table spec in plan")
            except Exception as e:
                logger.error(f"[{self.name}] ❌ Step 2 ERROR: {e}", exc_info=True)
            
            # Step 3: Pre-aggregate chart data
            logger.info(f"[{self.name}] 🔄 Step 3/6: Pre-aggregating chart data...")
            try:
                self._pre_aggregate_chart_data(dashboard_plan, df, resolved_entities)
                
                successful_aggs = sum(1 for v in self.chart_data_cache.values() if v is not None)
                total_expected = len(dashboard_plan.get('charts', []))
                logger.info(f"[{self.name}] ✅ Step 3 SUCCESS: {successful_aggs}/{total_expected} charts aggregated")
                
                if successful_aggs == 0:
                    logger.error(f"[{self.name}] ❌ Step 3 FAILED: No charts could be aggregated")
                    return {'status': 'error', 'message': 'Chart data aggregation failed - no valid transformations'}
                
                if successful_aggs < total_expected:
                    logger.warning(f"[{self.name}] ⚠️ Step 3 WARNING: Only {successful_aggs}/{total_expected} charts aggregated successfully")
            except Exception as e:
                logger.error(f"[{self.name}] ❌ Step 3 FAILED: {e}", exc_info=True)
                return {'status': 'error', 'message': f'Chart aggregation error: {str(e)}'}
            
            # Step 4: Generate chart images
            logger.info(f"[{self.name}] 🔄 Step 4/6: Generating chart images with matplotlib...")
            try:
                charts = self._generate_charts(dashboard_plan, df, resolved_entities)
                
                if len(charts) == 0:
                    logger.error(f"[{self.name}] ❌ Step 4 FAILED: No chart images generated")
                    return {'status': 'error', 'message': 'Chart image generation failed - all charts failed to render'}
                
                logger.info(f"[{self.name}] ✅ Step 4 SUCCESS: {len(charts)} chart images created")
                
                if len(charts) < successful_aggs:
                    logger.warning(f"[{self.name}] ⚠️ Step 4 WARNING: Only {len(charts)}/{successful_aggs} charts rendered (some failed)")
            except Exception as e:
                logger.error(f"[{self.name}] ❌ Step 4 FAILED: {e}", exc_info=True)
                return {'status': 'error', 'message': f'Chart rendering error: {str(e)}'}
            
            # Step 5: Create HTML dashboard
            logger.info(f"[{self.name}] 🔄 Step 5/6: Creating HTML dashboard...")
            try:
                dashboard_html = self._create_dashboard_html(charts, dashboard_plan, query, resolved_entities, insights)
                
                if not dashboard_html or len(dashboard_html) < 1000:
                    logger.error(f"[{self.name}] ❌ Step 5 FAILED: HTML too short ({len(dashboard_html)} chars)")
                    return {'status': 'error', 'message': 'HTML generation failed - output too short'}
                
                logger.info(f"[{self.name}] ✅ Step 5 SUCCESS: HTML created ({len(dashboard_html):,} chars)")
            except Exception as e:
                logger.error(f"[{self.name}] ❌ Step 5 FAILED: {e}", exc_info=True)
                return {'status': 'error', 'message': f'HTML creation error: {str(e)}'}
            
            # Step 6: Save dashboard
            logger.info(f"[{self.name}] 🔄 Step 6/6: Saving dashboard to file...")
            try:
                output_path = self._save_dashboard(dashboard_html, resolved_entities, query)
                logger.info(f"[{self.name}] ✅ Step 6 SUCCESS: Saved to {output_path}")
            except Exception as e:
                logger.error(f"[{self.name}] ❌ Step 6 FAILED: {e}", exc_info=True)
                return {'status': 'error', 'message': f'Save error: {str(e)}'}
            
            # Build result
            result = {
                'status': 'success',
                'dashboard_plan': dashboard_plan,
                'charts_generated': len(charts),
                'charts_planned': chart_count,
                'table_included': self.table_data is not None,
                'output_path': output_path,
                'dashboard_html': dashboard_html,
                'title': dashboard_plan.get('title', 'Dashboard'),
                'performance_level': semantic_cache.get('_performance', 'STANDARD')
            }
            
            logger.info(f"[{self.name}] 🎉 COMPLETE SUCCESS!")
            logger.info(f"[{self.name}]   └─ Title: {result['title']}")
            logger.info(f"[{self.name}]   └─ Performance: {result['performance_level']}")
            logger.info(f"[{self.name}]   └─ Charts: {len(charts)}/{chart_count}")
            logger.info(f"[{self.name}]   └─ Table: {'Yes' if self.table_data else 'No'}")
            logger.info(f"[{self.name}]   └─ File: {output_path}")
            
            # Update MCP store with agent context
            mcp_store.update_agent_context(
                self.name,
                query=query,
                entities=resolved_entities,
                results=result,
                timestamp=datetime.now().isoformat()
            )
            
            # Update dialogue state
            dashboard_entities = {
                **resolved_entities,
                'dashboard_created': True,
                'dashboard_path': output_path,
                'performance_level': semantic_cache.get('_performance', 'STANDARD')
            }
            
            mcp_store.update_dialogue_state(
                dashboard_entities,
                query,
                f"Dashboard created: {len(charts)} charts at {output_path}"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"[{self.name}] ❌ CRITICAL UNEXPECTED ERROR: {e}", exc_info=True)
            logger.error(f"[{self.name}] Error type: {type(e).__name__}")
            logger.error(f"[{self.name}] Error args: {e.args}")
            return {
                'status': 'error',
                'message': f'Unexpected error: {str(e)}',
                'error_type': type(e).__name__
            }


    
    # ===========================
    # ✅ NEW: GET UNFILTERED STATS FOR COMPARISON
    # ===========================
    
    def _get_unfiltered_stats(self, df: pd.DataFrame) -> dict:
        """Get company-wide stats for comparison"""
        stats = {}
        
        if self.revenue_column and self.revenue_column in df.columns:
            stats['total_company_revenue'] = float(df[self.revenue_column].sum())
            stats['avg_customer_revenue'] = float(df.groupby(self.customer_column)[self.revenue_column].sum().mean()) if self.customer_column in df.columns else 0
            stats['median_customer_revenue'] = float(df.groupby(self.customer_column)[self.revenue_column].sum().median()) if self.customer_column in df.columns else 0
        
        if self.customer_column and self.customer_column in df.columns:
            stats['total_customers'] = int(df[self.customer_column].nunique())
        
        if self.date_column and self.date_column in df.columns:
            stats['date_range_start'] = df[self.date_column].min().strftime('%Y-%m-%d')
            stats['date_range_end'] = df[self.date_column].max().strftime('%Y-%m-%d')
        
        logger.info(f"[Unfiltered Stats] Total revenue: ${stats.get('total_company_revenue', 0):,.0f}, Customers: {stats.get('total_customers', 0)}")
        
        return stats
    
    # ===========================
    # CONTEXT RESOLUTION
    # ===========================
    
    def _resolve_dashboard_context(self, query: str, entities: dict = None) -> dict:
        query_lower = query.lower()
        dialogue_state = mcp_store.get_current_dialogue_state()
        all_entities = dialogue_state.get("entities", {})
        
        if any(phrase in query_lower for phrase in ['for this', 'for that', 'for it', 'for the same']) and entities:
            return self._filter_dashboard_entities(entities)
        
        if any(kw in query_lower for kw in ['overall', 'all', 'complete', 'full']):
            return {}
        
        if entities and any(k in entities for k in ['customer_id', 'product_id', 'year']):
            return self._filter_dashboard_entities(entities)
        
        return {}
    
    def _filter_dashboard_entities(self, entities: dict) -> dict:
        KEYS = ['customer_id', 'product_id', 'year', 'sales_org']
        return {k: v for k, v in entities.items() if k in KEYS and v}
    
    def _filter_dataframe(self, df: pd.DataFrame, entities: dict) -> pd.DataFrame:
        """Filter dataframe with validation"""
        filtered = df.copy()
        
        if self.date_column and self.date_column in filtered.columns:
            filtered[self.date_column] = pd.to_datetime(filtered[self.date_column], errors='coerce')
            
            # ✅ CRITICAL: Filter out future dates
            today = pd.Timestamp.now()
            filtered = filtered[filtered[self.date_column] <= today]
            logger.info(f"[Filter] Removed future dates, keeping data up to {today.strftime('%Y-%m-%d')}")
        
        if 'year' in entities and self.date_column:
            year = int(entities['year']) if isinstance(entities['year'], str) else entities['year']
            filtered = filtered[filtered[self.date_column].dt.year == year]
            logger.info(f"[Filter] Year {year}: {len(filtered)} rows")
        
        if 'customer_id' in entities and self.customer_column:
            cid = entities['customer_id']
            mask = filtered[self.customer_column].astype(str) == str(cid)
            filtered = filtered[mask]
            logger.info(f"[Filter] Customer {cid}: {len(filtered)} rows")
        
        if 'product_id' in entities and self.product_column:
            pid = entities['product_id']
            mask = filtered[self.product_column].astype(str) == str(pid)
            filtered = filtered[mask]
            logger.info(f"[Filter] Product {pid}: {len(filtered)} rows")
        
        return filtered
    
    def _extract_minimal_semantic_cache(self, dialogue_state: dict, passed_entities: dict = None) -> dict:
        """Extract semantic context with COMPREHENSIVE aggregation key detection"""
        semantic_cache = {
            '_performance': 'STANDARD',
            '_entity': 'GENERAL',
            '_context': '',
            '_specific_entity_id': None
        }
        
        try:
            # ✅ PRIORITY 1: Check entities passed directly to agent
            if passed_entities:
                logger.info(f"[Semantic] Checking passed entities: {passed_entities}")
                
                # ✅ COMPREHENSIVE: Check ALL possible aggregation key names
                aggregation_value = None
                found_key = None
                
                # Check all known variations of aggregation keys
                possible_keys = [
                    'aggregation_type', 
                    'metric_qualifier',        # ← NEW: This is what routing agent uses!
                    'aggregation_type',
                    'aggregation',
                    'query_metric_aggregation',
                    'metric_aggregation',
                    'qualifier',
                    'metric_type',
                    'analysis_type'
                ]
                
                for key in possible_keys:
                    if key in passed_entities and passed_entities[key]:
                        aggregation_value = str(passed_entities[key]).lower()
                        found_key = key
                        logger.info(f"[Semantic] ✅ Found aggregation in key '{key}': '{aggregation_value}'")
                        break
                
                # Detect performance level from aggregation value
                if aggregation_value:
                    low_indicators = ['least', 'min', 'minimum', 'lowest', 'bottom', 'worst', 'smallest', 'underperform', 'poor', 'weak']
                    high_indicators = ['most', 'max', 'maximum', 'highest', 'top', 'best', 'largest', 'greatest', 'strong']
                    
                    if aggregation_value in low_indicators:
                        semantic_cache['_performance'] = 'LOW'
                        logger.info(f"[Semantic] 🚨 Detected LOW performance from '{found_key}': '{aggregation_value}'")
                    elif aggregation_value in high_indicators:
                        semantic_cache['_performance'] = 'HIGH'
                        logger.info(f"[Semantic] ✅ Detected HIGH performance from '{found_key}': '{aggregation_value}'")
                else:
                    logger.warning(f"[Semantic] ⚠️ No aggregation value found in entities. Keys checked: {possible_keys}")
                
                # Extract entity ID and type
                if 'customer_id' in passed_entities and passed_entities['customer_id']:
                    semantic_cache['_specific_entity_id'] = str(passed_entities['customer_id'])
                    semantic_cache['_entity'] = 'CUSTOMER'
                    logger.info(f"[Semantic] ✅ Customer ID from entities: {semantic_cache['_specific_entity_id']}")
                
                if 'product_id' in passed_entities and passed_entities['product_id']:
                    semantic_cache['_specific_entity_id'] = str(passed_entities['product_id'])
                    semantic_cache['_entity'] = 'PRODUCT'
                    logger.info(f"[Semantic] ✅ Product ID from entities: {semantic_cache['_specific_entity_id']}")
            
            # ✅ PRIORITY 2: Check conversation history (fallback)
            if semantic_cache['_performance'] == 'STANDARD':  # Only if not already detected
                try:
                    conversation = mcp_store.conversation_history[-5:] if hasattr(mcp_store, 'conversation_history') else []
                except:
                    conversation = []
                
                if len(conversation) == 0:
                    logger.warning(f"[Semantic] ⚠️ Conversation history is empty, using entities only")
                
                previous_user_query = ""
                previous_assistant_response = ""
                
                # Find most recent NON-dashboard query
                for i in range(len(conversation) - 1, -1, -1):
                    turn = conversation[i]
                    if turn.get('role') == 'user':
                        msg = turn.get('message', '').lower()
                        if 'dashboard' not in msg and 'visualiz' not in msg and 'chart' not in msg and 'generate' not in msg:
                            previous_user_query = turn.get('message', '')
                            if i + 1 < len(conversation):
                                previous_assistant_response = conversation[i + 1].get('message', '')
                            break
                
                semantic_cache['_context'] = previous_user_query[:200]
                
                logger.info(f"[Semantic Debug] Previous query: '{previous_user_query[:100] if previous_user_query else '(empty)'}'")
                logger.info(f"[Semantic Debug] Previous response preview: '{previous_assistant_response[:100] if previous_assistant_response else '(empty)'}'")
                
                if previous_user_query or previous_assistant_response:
                    combined_text = previous_user_query + " " + previous_assistant_response
                    combined_lower = combined_text.lower()
                    
                    # Extract customer/product ID from conversation
                    if not semantic_cache['_specific_entity_id']:
                        customer_match = re.search(r'customer\s+(?:id[:\s]*)?(\d+)', combined_lower)
                        if not customer_match:
                            customer_match = re.search(r'\(customer\s+id:\s*(\d+)\)', combined_lower)
                        if not customer_match:
                            customer_match = re.search(r'customer.*?revenue.*?[:\*\s]+(\d+)', combined_lower)
                        
                        if customer_match:
                            semantic_cache['_specific_entity_id'] = customer_match.group(1)
                            semantic_cache['_entity'] = 'CUSTOMER'
                            logger.info(f"[Semantic] ✅ Customer ID from conversation: {customer_match.group(1)}")
                    
                    # Detect performance from conversation
                    low_keywords = ['min ', 'minimum', 'lowest', 'least', 'worst', 'bottom', 'smallest', 'underperform', 'poor', 'weak', 'declining']
                    high_keywords = ['max ', 'maximum', 'highest', 'most', 'best', 'top', 'largest', 'greatest', 'strong', 'leading']
                    
                    for keyword in low_keywords:
                        if keyword in combined_lower:
                            semantic_cache['_performance'] = 'LOW'
                            logger.info(f"[Semantic] 🚨 Detected LOW from conversation keyword: '{keyword}'")
                            break
                    
                    if semantic_cache['_performance'] == 'STANDARD':
                        for keyword in high_keywords:
                            if keyword in combined_lower:
                                semantic_cache['_performance'] = 'HIGH'
                                logger.info(f"[Semantic] ✅ Detected HIGH from conversation keyword: '{keyword}'")
                                break
            
            logger.info(f"[Semantic] 🎯 FINAL RESULT: Performance={semantic_cache['_performance']}, Entity={semantic_cache['_entity']}, ID={semantic_cache.get('_specific_entity_id')}")
            
        except Exception as e:
            logger.error(f"[Semantic Cache] ❌ FAILED: {e}", exc_info=True)
        
        return semantic_cache


    
    # ===========================
    # ✅ IMPROVED: LLM DASHBOARD CREATION
    # ===========================
    
    def _llm_create_complete_dashboard(self, query: str, full_context: dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Context-aware dashboard with proper comparison"""
        
        entities = {k: v for k, v in full_context.items() if k in ['customer_id', 'product_id', 'year']}
        filtered_df = self._filter_dataframe(df, entities)
        
        if len(filtered_df) == 0:
            return None
        
        # Get stats
        unfiltered_stats = full_context.get('unfiltered_stats', {})
        filtered_stats = self._get_filtered_stats(filtered_df, unfiltered_stats)
        
        # Extract context
        performance = full_context.get('_performance', 'STANDARD')
        entity_type = full_context.get('_entity', 'GENERAL')
        entity_id = full_context.get('_specific_entity_id')
        prev_context = full_context.get('_context', '')[:150]
        
        # ✅ Determine dashboard type
        dashboard_type = self._determine_dashboard_type(entities, performance, entity_type)
        
        # ✅ Build context-specific prompt
        prompt = f"""Create a dashboard for: "{query}"

**CRITICAL CONTEXT:**
{dashboard_type['instructions']}

**DATA SUMMARY:**
{filtered_stats}

**COMPARISON TO COMPANY:**
{self._format_comparison_stats(filtered_stats, unfiltered_stats, entities)}

**COLUMNS:** Date={self.date_column}, Revenue={self.revenue_column}, Customer={self.customer_column}, Product={self.product_column}

**⚠️ ANOMALY DATA:** {{'Yes - is_anomaly column available' if 'is_anomaly' in df.columns else 'No anomaly data'}}

**STRICT RULES:**
1. Title MUST reflect performance level: {dashboard_type['title_guidance']}
2. Use ACTUAL entity IDs from data (no "Customer X" or "Product A")
3. Insights must acknowledge this is {dashboard_type['scope']} (not entire company)
4. Focus on: {dashboard_type['focus_areas']}
5. Table: {dashboard_type['table_type']}
6. NO future dates beyond {datetime.now().strftime('%Y-%m-%d')}
7. Format quarters as "Q1 2024" not "2024-03-31"
8. generate MIN 6 charts and MAX 10 charts
9. If is_anomaly column exists, include anomaly visualization chart' if 'is_anomaly' in df.columns else ''

**JSON (be specific and accurate):**
{{
  "plan": {{
    "title": "{dashboard_type['title_example']}",
    "description": "Brief context-aware description max 100 chars",
    "table_spec": {{
      "title": "{dashboard_type['table_title']}",
      "transformation_code": "{dashboard_type['table_code']}",
      "display_columns": {dashboard_type['table_columns']},
      "format_columns": {{"Revenue": "$,.2f"}},
      "limit": 10
    }},
    "charts": [
      {{
        "type": "line",
        "title": "Monthly Revenue Trend",
        "description": "Revenue pattern over time",
        "data_transformation": "df_chart = df.groupby(pd.Grouper(key='{self.date_column}', freq='ME'))['{self.revenue_column}'].sum().reset_index()",
        "x_column": "{self.date_column}",
        "y_column": "{self.revenue_column}"
      }},
      {{
        "type": "bar",
        "title": "{dashboard_type['second_chart_title']}",
        "description": "{dashboard_type['second_chart_desc']}",
        "data_transformation": "{dashboard_type['second_chart_code']}",
        "x_column": "{dashboard_type['second_chart_x']}",
        "y_column": "{self.revenue_column}"
      }}
    ]
  }},
  "insights": {{
    "executive_summary": "2 sentences acknowledging {dashboard_type['performance_phrase']} with specific numbers and context",
    "key_findings": ["{dashboard_type['finding_example']}", "Temporal pattern observed", "Product preference identified", "Growth/decline trend noted"],
    "recommendations": ["{dashboard_type['recommendation_example']}", "Monitor specific metrics", "Take corrective action"],
    "chart_insights": ["Monthly trend shows...", "Top products reveal...", "Order patterns indicate..."]
  }}
}}

**Remember:** This is a {dashboard_type['performance_phrase']} analysis. Be {dashboard_type['tone']}.

Return ONLY JSON."""

        try:
            logger.info(f"🤖 LLM: Creating {dashboard_type['scope']} dashboard...")
            
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            extracted_json = self._extract_json_robust(content)
            
            if not extracted_json:
                logger.error("❌ Could not extract JSON")
                if len(content) > 6000:
                    extracted_json = self._fix_truncated_json(content)
                if not extracted_json:
                    return None
            
            result = json.loads(extracted_json)
            
            # Validate structure
            if 'plan' not in result or 'insights' not in result:
                logger.error("❌ Invalid JSON structure")
                return None
            
            plan = result.get('plan', {})
            valid_charts = []
            
            for chart in plan.get('charts', []):
                if (chart.get('x_column') and chart.get('y_column') and 
                    chart.get('type') and chart.get('data_transformation')):
                    if not isinstance(chart.get('y_column'), list):
                        valid_charts.append(chart)
            
            if len(valid_charts) == 0:
                logger.error("❌ No valid charts")
                return None
            
            # ✅ Limit to 6 charts max for consistency
            plan['charts'] = valid_charts[:6]
            result['plan'] = plan
            
            logger.info(f"✅ Generated: {len(valid_charts[:6])} charts")
            return result
            
        except Exception as e:
            logger.error(f"❌ LLM failed: {e}", exc_info=True)
            return None
    
    def _get_filtered_stats(self, filtered_df: pd.DataFrame, unfiltered_stats: dict) -> str:
        """Get filtered stats with comparison"""
        parts = [f"Filtered Rows: {len(filtered_df):,}"]
        
        if self.revenue_column and self.revenue_column in filtered_df.columns:
            total_rev = filtered_df[self.revenue_column].sum()
            company_rev = unfiltered_stats.get('total_company_revenue', total_rev)
            pct_of_total = (total_rev / company_rev * 100) if company_rev > 0 else 0
            
            parts.append(f"Filtered Revenue: ${total_rev:,.0f} ({pct_of_total:.2f}% of company total)")
        
        if self.date_column and self.date_column in filtered_df.columns:
            min_date = filtered_df[self.date_column].min()
            max_date = filtered_df[self.date_column].max()
            parts.append(f"Period: {min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}")
        
        if self.product_column and self.product_column in filtered_df.columns and self.revenue_column:
            top_products = filtered_df.groupby(self.product_column)[self.revenue_column].sum().nlargest(3)
            product_list = [f"{prod}(${rev:,.0f})" for prod, rev in top_products.items()]
            parts.append(f"Top Products: {', '.join(product_list)}")
        
        return " | ".join(parts)
    
    def _format_comparison_stats(self, filtered_stats: str, unfiltered_stats: dict, entities: dict) -> str:
        """Format comparison context"""
        if 'customer_id' not in entities:
            return "N/A (overall analysis)"
        
        avg_cust_rev = unfiltered_stats.get('avg_customer_revenue', 0)
        median_cust_rev = unfiltered_stats.get('median_customer_revenue', 0)
        
        return f"Avg Customer Revenue: ${avg_cust_rev:,.0f} | Median: ${median_cust_rev:,.0f}"
    
    def _determine_dashboard_type(self, entities: dict, performance: str, entitytype: str) -> dict:
        """
        🎯 FULLY DYNAMIC: LLM determines dashboard context based on ANY entity type
        No hardcoded customer/product logic - works for ANY dimension!
        """
        
        # Detect the entity being analyzed
        entity_key = None
        entity_value = None
        entity_label = "Entity"
        
        # Map entity keys to human-readable labels
        entity_map = {
            'customer_id': ('Customer', self.customer_column),
            'product_id': ('Product', self.product_column),
            'sales_org': ('Sales Organization', self.sales_org_column),
            'year': ('Year', None),
            'region': ('Region', 'Region'),
            'material': ('Material', 'Material'),
            'sold_to_party': ('Sold To Party', 'SoldToParty')
        }
        
        # Find which entity is being analyzed
        for key, (label, column) in entity_map.items():
            if key in entities and entities[key]:
                entity_key = key
                entity_value = entities[key]
                entity_label = label
                break
        
        # Build dynamic scope description
        if entity_value:
            scope_base = f"{entity_label} {entity_value}"
            
            # Add performance indicator
            if performance == 'HIGH':
                scope = f"{scope_base} (🏆 TOP PERFORMER)"
                performance_emoji = "🏆"
                performance_word = "HIGHEST"
            elif performance == 'LOW':
                scope = f"{scope_base} (🚨 UNDERPERFORMER - CRITICAL)"
                performance_emoji = "🚨"
                performance_word = "LOWEST"
            else:
                scope = f"{scope_base} Analysis"
                performance_emoji = "📊"
                performance_word = "STANDARD"
        else:
            scope = "Overall Business Performance"
            performance_emoji = "📊"
            performance_word = "OVERALL"
        
        # Return dynamic configuration that LLM will interpret
        return {
            'scope': scope,
            'entity_type': entity_label,
            'entity_value': entity_value,
            'entity_key': entity_key,
            'performance': performance,
            'performance_emoji': performance_emoji,
            'performance_word': performance_word,
            
            # Dynamic instructions for LLM
            'instructions': f'''
    You are creating a dashboard for: **{scope}**

    **PERFORMANCE LEVEL: {performance}**

    **CRITICAL CONTEXT RULES:**

    {'**🏆 HIGH PERFORMER DASHBOARD:**' if performance == 'HIGH' else ''}
    {'- This is a TOP-PERFORMING ' + entity_label.lower() if performance == 'HIGH' else ''}
    {'- Tone: POSITIVE, CELEBRATORY, SUCCESS-FOCUSED' if performance == 'HIGH' else ''}
    {'- Focus: achievements, growth, scaling opportunities, market leadership' if performance == 'HIGH' else ''}
    {'- Language: "top performer", "market leader", "strong growth", "scaling opportunity"' if performance == 'HIGH' else ''}
    {'- Recommendations: expansion, scaling, premium strategies, capacity planning' if performance == 'HIGH' else ''}
    {'- FORBIDDEN: "at-risk", "declining", "needs improvement", "retention"' if performance == 'HIGH' else ''}

    {'**🚨 LOW PERFORMER DASHBOARD:**' if performance == 'LOW' else ''}
    {'- This is an UNDERPERFORMING ' + entity_label.lower() if performance == 'LOW' else ''}
    {'- Tone: URGENT, CONCERNED, ACTION-ORIENTED' if performance == 'LOW' else ''}
    {'- Focus: decline analysis, churn risk, retention, root cause investigation' if performance == 'LOW' else ''}
    {'- Language: "critical", "alarming decline", "at-risk", "urgent intervention needed"' if performance == 'LOW' else ''}
    {'- Recommendations: retention, recovery, investigation, intervention, possibly discontinuation' if performance == 'LOW' else ''}
    {'- FORBIDDEN: "growth opportunity", "strong performance", "cross-sell", "upsell"' if performance == 'LOW' else ''}

    {'**📊 STANDARD DASHBOARD:**' if performance == 'STANDARD' else ''}
    {'- Balanced analysis of ' + entity_label.lower() + ' performance' if performance == 'STANDARD' else ''}
    {'- Tone: ANALYTICAL, INFORMATIVE, BALANCED' if performance == 'STANDARD' else ''}
    {'- Focus: patterns, trends, insights, opportunities' if performance == 'STANDARD' else ''}

    **ENTITY-SPECIFIC GUIDANCE:**
    - For {entity_label}: Analyze from the perspective of THIS SPECIFIC {entity_label.lower()} only
    - Compare to company average/peers when relevant
    - Focus on metrics relevant to {entity_label.lower()} analysis
    - Table should show breakdown relevant to this {entity_label.lower()}
    - Charts should reveal patterns specific to this {entity_label.lower()}

    **MANDATORY:**
    1. Title MUST include "{entity_label} {entity_value}" and reflect {performance} level
    2. Executive summary MUST acknowledge this is a {performance} performer
    3. All insights must be specific to THIS {entity_label.lower()}, not general company
    4. Use actual entity ID "{entity_value}" (not placeholders like "Entity X")
    5. Recommendations must match performance level (retention for LOW, scaling for HIGH)
    ''',
            
            # Dynamic title guidance
            'title_guidance': f'Include "{entity_label} {entity_value}" and {"success/achievement" if performance == "HIGH" else "critical/urgent" if performance == "LOW" else "analysis/performance"} language',
            
            'title_example': f'{performance_emoji} {entity_label} {entity_value}: ' + (
                'Market Leader & Top Revenue Driver' if performance == 'HIGH'
                else 'Critical Alert - Underperforming & At-Risk' if performance == 'LOW'
                else 'Performance Analysis & Trends'
            ),
            
            'focus_areas': (
                'revenue excellence, market leadership, growth trends, scaling strategies' if performance == 'HIGH'
                else 'decline analysis, churn risk, retention urgency, root cause investigation' if performance == 'LOW'
                else 'trends, patterns, opportunities, performance metrics'
            ),
            
            # Dynamic table configuration
            'table_type': f'{entity_label} Performance Breakdown',
            'table_title': f'{performance_emoji} {entity_label} {entity_value} - Key Metrics',
            'table_code': self._generate_table_code(entity_key, entity_value),
            'table_columns': self._get_table_columns(entity_key),
            
            # Dynamic chart guidance
            'second_chart_title': f'Trends for {entity_label} {entity_value}',
            'second_chart_desc': f'Temporal patterns for this {entity_label.lower()}',
            'second_chart_code': self._generate_chart_code(entity_key, entity_value),
            'second_chart_x': self.date_column,
            
            'performance_phrase': f'{performance_word} performing {entity_label.lower()}',
            'tone': (
                'POSITIVE, ACHIEVEMENT-FOCUSED, and STRATEGIC' if performance == 'HIGH'
                else 'URGENT, ALARMED, and ACTION-ORIENTED' if performance == 'LOW'
                else 'analytical, balanced, and informative'
            ),
            
            'finding_example': f'{performance_emoji} {entity_label} {entity_value} ' + (
                f'leads with exceptional performance metrics' if performance == 'HIGH'
                else f'shows critical decline requiring immediate intervention' if performance == 'LOW'
                else f'demonstrates typical performance patterns'
            ),
            
            'recommendation_example': (
                f'Scale operations for {entity_label} {entity_value} to capitalize on strong market position' if performance == 'HIGH'
                else f'🚨 URGENT: Launch retention strategy for {entity_label} {entity_value} within 48 hours' if performance == 'LOW'
                else f'Continue monitoring {entity_label} {entity_value} and adjust strategies as needed'
            )
        }

    def _generate_table_code(self, entity_key: str, entity_value: str) -> str:
        """Generate dynamic table aggregation code based on entity type"""
        if entity_key == 'customer_id':
            # For customers: show quarterly revenue
            return f"df_table = df.copy(); df_table['Quarter'] = df_table['{self.date_column}'].dt.to_period('Q').astype(str); df_table = df_table.groupby('Quarter')['{self.revenue_column}'].sum().reset_index().sort_values('Quarter'); df_table['Quarter'] = df_table['Quarter'].apply(lambda x: f'Q{{x[-1]}} {{x[:4]}}'); df_table['Revenue'] = df_table['{self.revenue_column}']"
        
        elif entity_key == 'product_id':
            # For products: show top customers buying it
            return f"df_table = df.groupby('{self.customer_column}')['{self.revenue_column}'].sum().nlargest(10).reset_index(); df_table.columns = ['Customer', 'Revenue']"
        
        elif entity_key == 'sales_org':
            # For sales org: show product breakdown
            return f"df_table = df.groupby('{self.product_column}')['{self.revenue_column}'].sum().nlargest(10).reset_index(); df_table.columns = ['Product', 'Revenue']"
        
        else:
            # Default: quarterly breakdown
            return f"df_table = df.copy(); df_table['Quarter'] = df_table['{self.date_column}'].dt.to_period('Q').astype(str); df_table = df_table.groupby('Quarter')['{self.revenue_column}'].sum().reset_index().sort_values('Quarter'); df_table['Revenue'] = df_table['{self.revenue_column}']"

    def _get_table_columns(self, entity_key: str) -> str:
        """Get table column names based on entity type"""
        if entity_key == 'customer_id':
            return '["Quarter", "Revenue"]'
        elif entity_key == 'product_id':
            return '["Customer", "Revenue"]'
        elif entity_key == 'sales_org':
            return '["Product", "Revenue"]'
        else:
            return '["Period", "Revenue"]'

    def _generate_chart_code(self, entity_key: str, entity_value: str) -> str:
        """Generate dynamic chart code based on entity type"""
        if entity_key == 'product_id':
            # For products: show customer adoption
            return f"df_chart = df.groupby(pd.Grouper(key='{self.date_column}', freq='ME'))['{self.customer_column}'].nunique().reset_index(); df_chart.columns = ['{self.date_column}', 'Customer_Count']"
        
        elif entity_key == 'customer_id':
            # For customers: show product diversity
            return f"df_chart = df.groupby('{self.product_column}')['{self.revenue_column}'].sum().nlargest(10).reset_index()"
        
        else:
            # Default: revenue over time
            return f"df_chart = df.groupby(pd.Grouper(key='{self.date_column}', freq='ME'))['{self.revenue_column}'].sum().reset_index()"

    
    # ===========================
    # JSON EXTRACTION (keep existing methods)
    # ===========================
    
    def _extract_json_robust(self, content: str) -> str:
        """Multi-strategy JSON extraction"""
        
        if '```':
            try:
                json_part = content.split('```json').split('```')
                json.loads(json_part)
                return json_part
            except:
                pass
        
        if '```' in content:
            try:
                parts = content.split('```')
                for part in parts:
                    part = part.strip()
                    if part.startswith('json'):
                        part = part[4:].strip()
                    if part.startswith('{') and part.endswith('}'):
                        try:
                            json.loads(part)
                            return part
                        except:
                            continue
            except:
                pass
        
        try:
            start_idx = content.find('{')
            if start_idx == -1:
                return None
            
            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx == -1:
                return None
            
            json_part = content[start_idx:end_idx]
            json.loads(json_part)
            return json_part
        except:
            pass
        
        return None
    
    def _fix_truncated_json(self, content: str) -> str:
        """Fix truncated JSON"""
        try:
            start_idx = content.find('{')
            if start_idx == -1:
                return None
            
            json_part = content[start_idx:]
            
            open_braces = json_part.count('{')
            close_braces = json_part.count('}')
            open_brackets = json_part.count('[')
            close_brackets = json_part.count(']')
            
            last_comma = json_part.rfind(',')
            if last_comma > 0:
                after_comma = json_part[last_comma+1:].strip()
                if not after_comma.endswith('}') and not after_comma.endswith(']'):
                    json_part = json_part[:last_comma]
            
            json_part += ']' * (open_brackets - close_brackets)
            json_part += '}' * (open_braces - close_braces)
            
            json.loads(json_part)
            logger.info("✅ Fixed truncated JSON")
            return json_part
            
        except:
            return None
    
    # ===========================
    # TABLE & CHART GENERATION (keep existing with minor tweaks)
    # ===========================
    
    def _generate_aggregated_table(self, dashboard_plan: Dict[str, Any], df: pd.DataFrame, entities: dict):
        """Generate table"""
        table_spec = dashboard_plan.get('table_spec')
        if not table_spec or not table_spec.get('transformation_code'):
            return
        
        try:
            filtered_df = self._filter_dataframe(df, entities)
            local_namespace = {'df': filtered_df.copy(), 'pd': pd, 'np': np, 'datetime': datetime}
            exec(table_spec['transformation_code'], {}, local_namespace)
            df_table = local_namespace.get('df_table')
            
            if df_table is not None and isinstance(df_table, pd.DataFrame) and len(df_table) > 0:
                self.table_data = {'df': df_table.head(10), 'spec': table_spec}
                logger.info(f"[Table] ✅ {len(df_table)} rows")
        except Exception as e:
            logger.error(f"[Table] Failed: {e}")
            self.table_data = None
    
    def _preaggregate_chart_data(self, dashboard_plan: Dict[str, Any], df: pd.DataFrame, entities: dict):
        """Pre-aggregate chart data"""
        filtered_df = self._filter_dataframe(df, entities)
        
        for i, chart_config in enumerate(dashboard_plan.get('charts', [])):
            chart_key = f"chart_{i}"
            transformation_code = chart_config.get('data_transformation', '')
            
            if not transformation_code:
                self.chart_data_cache[chart_key] = None
                continue
            
            try:
                local_namespace = {'df': filtered_df.copy(), 'pd': pd, 'np': np, 'datetime': datetime}
                exec(transformation_code, {}, local_namespace)
                df_chart = local_namespace.get('df_chart')
                
                if df_chart is not None and isinstance(df_chart, pd.DataFrame) and len(df_chart) > 0:
                    x_col = chart_config.get('x_column')
                    y_col = chart_config.get('y_column')
                    
                    if x_col in df_chart.columns and y_col in df_chart.columns:
                        self.chart_data_cache[chart_key] = {'data': df_chart, 'config': chart_config}
                        logger.info(f"[Chart {i+1}] ✅ {len(df_chart)} rows")
                    else:
                        logger.error(f"[Chart {i+1}] Missing columns. Has: {list(df_chart.columns)}")
                        self.chart_data_cache[chart_key] = None
                else:
                    self.chart_data_cache[chart_key] = None
                    
            except Exception as e:
                logger.error(f"[Chart {i+1}] {e}")
                self.chart_data_cache[chart_key] = None
    
    def _generate_charts(self, dashboard_plan: Dict[str, Any], df: pd.DataFrame, entities: dict) -> List[Dict[str, Any]]:
        """Generate chart images"""
        charts = []
        
        for i, chart_config in enumerate(dashboard_plan.get('charts', [])):
            chart_key = f"chart_{i}"
            
            try:
                cached_data = self.chart_data_cache.get(chart_key)
                if not cached_data or cached_data.get('data') is None:
                    continue
                
                img_base64 = self._create_matplotlib_chart(cached_data, entities)
                if img_base64:
                    charts.append({
                        'image_base64': img_base64,
                        'title': chart_config.get('title', f'Chart {i+1}'),
                        'description': chart_config.get('description', ''),
                        'config': chart_config
                    })
            except Exception as e:
                logger.error(f"[Chart {i+1}] Failed: {e}")
        
        return charts
    
    def _create_matplotlib_chart(self, cached_data: dict, entities: dict) -> str:
        """Create chart with better error handling"""
        config = cached_data['config']
        df_chart = cached_data['data'].copy()
        
        chart_type = config.get('type', 'bar')
        x_col = config.get('x_column')
        y_col = config.get('y_column')
        title = config.get('title', 'Chart')
        
        try:
            # ✅ Validate data exists
            if len(df_chart) == 0:
                logger.error(f"Chart '{title}': No data to plot")
                return None
            
            # ✅ Validate columns exist
            if x_col not in df_chart.columns:
                logger.error(f"Chart '{title}': x_column '{x_col}' not in dataframe. Available: {list(df_chart.columns)}")
                return None
            
            if y_col not in df_chart.columns:
                logger.error(f"Chart '{title}': y_column '{y_col}' not in dataframe. Available: {list(df_chart.columns)}")
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            primary_color = '#667eea'
            secondary_color = '#764ba2'
            
            x_data = df_chart[x_col]
            y_data = df_chart[y_col]
            
            # ✅ Check for multi-index or grouping column
            grouping_col = config.get('color_column') or config.get('hue') or config.get('category')
            
            if grouping_col and grouping_col in df_chart.columns:
                # This is a grouped/multi-line chart
                logger.info(f"Chart '{title}': Detected grouped chart with column '{grouping_col}'")
                
                # For grouped line/area charts
                if chart_type in ['line', 'area']:
                    groups = df_chart[grouping_col].unique()
                    colors_list = plt.cm.Set2(np.linspace(0, 1, len(groups)))
                    
                    for i, group in enumerate(groups):
                        group_data = df_chart[df_chart[grouping_col] == group]
                        ax.plot(group_data[x_col], group_data[y_col], 
                            marker='o', linewidth=2, label=str(group), color=colors_list[i])
                    
                    ax.legend(loc='best')
                    ax.set_ylabel(y_col)
                    ax.grid(True, alpha=0.3)
                    
                    if pd.api.types.is_datetime64_any_dtype(df_chart[x_col]):
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                    
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            elif chart_type == 'bar':
                ax.bar(range(len(df_chart)), y_data, color=primary_color, alpha=0.8)
                ax.set_xticks(range(len(df_chart)))
                ax.set_xticklabels(x_data.astype(str), rotation=45, ha='right')
                ax.set_ylabel(y_col)
                ax.grid(axis='y', alpha=0.3)
            
            elif chart_type == 'line':
                ax.plot(x_data, y_data, marker='o', linewidth=2, markersize=6, color=primary_color)
                ax.set_ylabel(y_col)
                ax.grid(True, alpha=0.3)
                
                if pd.api.types.is_datetime64_any_dtype(x_data):
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                else:
                    if len(df_chart) < 20:
                        ax.set_xticks(range(len(df_chart)))
                        ax.set_xticklabels(x_data.astype(str), rotation=45, ha='right')
            
            elif chart_type == 'pie':
                colors_pie = plt.cm.Purples(np.linspace(0.3, 0.9, len(df_chart)))
                wedges, texts, autotexts = ax.pie(
                    y_data, 
                    labels=x_data.astype(str), 
                    autopct='%1.1f%%', 
                    colors=colors_pie, 
                    startangle=90
                )
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            
            elif chart_type == 'area':
                if pd.api.types.is_datetime64_any_dtype(x_data):
                    ax.fill_between(x_data, y_data, alpha=0.6, color=primary_color)
                    ax.plot(x_data, y_data, linewidth=2, color=secondary_color)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                else:
                    ax.fill_between(range(len(df_chart)), y_data, alpha=0.6, color=primary_color)
                    ax.plot(range(len(df_chart)), y_data, linewidth=2, color=secondary_color)
                    ax.set_xticks(range(len(df_chart)))
                    ax.set_xticklabels(x_data.astype(str), rotation=45, ha='right')
                
                ax.set_ylabel(y_col)
                ax.grid(True, alpha=0.3)
            
            elif chart_type == 'scatter':
                ax.scatter(x_data, y_data, alpha=0.6, s=50, color=primary_color)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.grid(True, alpha=0.3)

            elif chart_type == 'scatter':
                # ✅ NEW: Anomaly-aware scatter plot
                color_col = config.get('color_column')
                
                if color_col and color_col in df_chart.columns:
                    # Color by anomaly status
                    normal_mask = df_chart[color_col] == False
                    anomaly_mask = df_chart[color_col] == True
                    
                    # Plot normal points
                    if normal_mask.any():
                        ax.scatter(
                            df_chart.loc[normal_mask, x_col],
                            df_chart.loc[normal_mask, y_col],
                            alpha=0.6, s=50, color=primary_color, label='Normal'
                        )
                    
                    # Plot anomaly points
                    if anomaly_mask.any():
                        ax.scatter(
                            df_chart.loc[anomaly_mask, x_col],
                            df_chart.loc[anomaly_mask, y_col],
                            alpha=0.8, s=100, color='#e74c3c', marker='x', 
                            linewidths=3, label='Anomaly'
                        )
                    
                    ax.legend(loc='best')
                else:
                    # Regular scatter (existing code)
                    ax.scatter(x_data, y_data, alpha=0.6, s=50, color=primary_color)
                
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
            
            logger.info(f"Chart '{title}': ✅ Successfully rendered")
            return img_base64
            
        except Exception as e:
            logger.error(f"Chart '{title}' failed: {e}", exc_info=True)
            plt.close('all')
            return None


    
    # Keep your existing _create_table_html, _create_dashboard_html, _save_dashboard methods unchanged
    
    def _create_table_html(self) -> str:
        if not self.table_data:
            return ""
        
        df_table = self.table_data['df']
        spec = self.table_data['spec']
        title = spec.get('title', 'Data Table')
        description = spec.get('description', '')
        display_columns = spec.get('display_columns', list(df_table.columns))
        format_columns = spec.get('format_columns', {})
        
        df_display = df_table[display_columns].copy()
        df_styled = df_display.copy()
        
        for col, fmt in format_columns.items():
            if col in df_styled.columns:
                if '$' in fmt:
                    df_styled[col] = df_styled[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
                elif 'd' in fmt:
                    df_styled[col] = df_styled[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "")
        
        return f"""
        <div class="table-container">
            <div class="table-header">
                <h2>{title}</h2>
                <p class="table-description">{description}</p>
            </div>
            <div class="table-wrapper">
                {df_styled.to_html(index=False, classes='data-table', escape=False, border=0)}
            </div>
            <div class="table-footer">
                <span class="table-info">📊 Top {len(df_display)} entries</span>
            </div>
        </div>
        """
    
    def _create_dashboard_html(self, charts: List[Dict], dashboard_plan: Dict, query: str, 
                               entities: dict, insights: Dict) -> str:
        
        title = dashboard_plan.get('title', 'Sales Dashboard')
        description = dashboard_plan.get('description', '')
        executive_summary = insights.get('executive_summary', '')
        key_findings = insights.get('key_findings', [])
        recommendations = insights.get('recommendations', [])
        chart_insights = insights.get('chart_insights', [])
        
        table_html = self._create_table_html() if self.table_data else ""
        
        charts_html = ""
        for i, chart in enumerate(charts):
            insight = chart_insights[i] if i < len(chart_insights) else chart['description']
            charts_html += f"""
            <div class="chart-card">
                <div class="chart-header">
                    <h3>{chart['title']}</h3>
                    <p class="chart-description">{insight}</p>
                </div>
                <div class="chart-body">
                    <img src="data:image/png;base64,{chart['image_base64']}" alt="{chart['title']}" class="chart-image">
                </div>
            </div>
            """
        
        findings_html = "".join([f"<li>{f}</li>" for f in key_findings])
        recommendations_html = "".join([f"<li>{r}</li>" for r in recommendations])
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; text-align: center; }}
        .header h1 {{ font-size: 2.5rem; margin-bottom: 10px; }}
        .content {{ padding: 40px; }}
        .summary-section {{ background: #f8f9fa; padding: 30px; border-radius: 15px; margin-bottom: 30px; border-left: 5px solid #667eea; }}
        .summary-section h2 {{ color: #667eea; margin-bottom: 15px; }}
        .insights-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
        .insight-box {{ background: white; padding: 25px; border-radius: 12px; border: 2px solid #e0e0e0; }}
        .insight-box h3 {{ color: #667eea; margin-bottom: 15px; }}
        .insight-box ul {{ list-style: none; }}
        .insight-box li {{ padding: 8px 0; padding-left: 25px; position: relative; }}
        .insight-box li:before {{ content: "✓"; position: absolute; left: 0; color: #667eea; font-weight: bold; }}
        .table-container {{ margin: 30px 0; border-radius: 12px; overflow: hidden; border: 2px solid #e0e0e0; }}
        .table-header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; }}
        .table-wrapper {{ padding: 20px; }}
        .data-table {{ width: 100%; border-collapse: collapse; }}
        .data-table th {{ background: #f8f9fa; padding: 12px; text-align: left; font-weight: 600; color: #667eea; border-bottom: 2px solid #667eea; }}
        .data-table td {{ padding: 12px; border-bottom: 1px solid #e0e0e0; }}
        .data-table tr:hover {{ background: #f8f9fa; }}
        .chart-card {{ background: white; border-radius: 15px; padding: 25px; margin-bottom: 30px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .chart-header h3 {{ color: #333; margin-bottom: 10px; }}
        .chart-image {{ width: 100%; border-radius: 10px; }}
        .footer {{ background: #f8f9fa; padding: 20px; text-align: center; color: #666; }}
    </style>
</head>
<body>
    <div id="dashboard-main">
        <div class="container">
            <div class="header">
                <h1>📊 {title}</h1>
                <p>{description}</p>
            </div>
            <div class="content">
                <div class="summary-section">
                    <h2>Executive Summary</h2>
                    <p>{executive_summary}</p>
                </div>
                <div class="insights-grid">
                    <div class="insight-box">
                        <h3>🔍 Key Findings</h3>
                        <ul>{findings_html}</ul>
                    </div>
                    <div class="insight-box">
                        <h3>💡 Recommendations</h3>
                        <ul>{recommendations_html}</ul>
                    </div>
                </div>
                {table_html}
                <div class="charts-section">
                    <h2>📈 Visual Analytics</h2>
                    {charts_html}
                </div>
            </div>
            <div class="footer">
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def _save_dashboard(self, html_content: str, entities: dict, query: str) -> str:
        os.makedirs('dashboards', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        suffix = ""
        if entities.get('customer_id'):
            suffix = f"_cust_{entities['customer_id']}"
        elif entities.get('product_id'):
            suffix = f"_prod_{entities['product_id']}"
        
        filename = f"dashboard_{timestamp}{suffix}.html"
        filepath = os.path.join('dashboards', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"[Dashboard] Saved: {filepath}")
        return filepath
