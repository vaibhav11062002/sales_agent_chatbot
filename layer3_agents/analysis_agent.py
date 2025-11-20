import pandas as pd
import logging
from typing import Dict, Any
from data_connector import mcp_store

logger = logging.getLogger(__name__)

class AnalysisAgent:
    """Analysis Agent using MCP-backed data store"""
    
    def __init__(self):
        self.name = "AnalysisAgent"
    
    def execute(self, query: str, analysis_type: str = "summary", entities: dict = None) -> Dict[str, Any]:
        """
        Perform statistical analysis on sales data.
        
        Args:
            query: User's question
            analysis_type: Type of analysis (summary, aggregation, comparison)
            entities: Extracted entities like year, month, comparison flags
            
        Returns:
            Analysis results with metrics
        """
        logger.info(f"{self.name}: Performing {analysis_type} analysis")
        
        try:
            # Access shared DataFrame
            df = mcp_store.get_sales_data()
            
            # Filter based on entities from LLM
            if entities and 'year' in entities:
                year = entities['year']
                df = df[df['CreationDate'].dt.year == year]
                logger.info(f"Filtered data to year {year}: {len(df)} records")
            
            # Check context from other agents
            all_contexts = mcp_store.get_all_contexts()
            logger.info(f"Accessing contexts from {len(all_contexts)} other agents")
            
            # Perform analysis
            if analysis_type == "summary":
                results = self._summary_analysis(df, query, entities)
            elif analysis_type == "aggregation":
                results = self._aggregation_analysis(df, query, entities)
            else:
                results = self._summary_analysis(df, query, entities)
            
            # Share results
            mcp_store.update_agent_context(self.name, {
                "analysis_type": analysis_type,
                "results": results,
                "query": query,
                "filtered_by": entities
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            return {"status": "error", "message": str(e)}

    
    def _summary_analysis(self, df: pd.DataFrame, query: str, entities: dict = None) -> Dict[str, Any]:
        """Generate summary statistics"""
        try:
            results = {
                "total_sales": float(df['NetAmount'].sum()),
                "total_orders": len(df),
                "avg_order_value": float(df['NetAmount'].mean()),
                "unique_customers": int(df['SoldToParty'].nunique()),
                "unique_products": int(df['Product'].nunique()),
            }
            
            # Add year context if filtered
            if entities and 'year' in entities:
                results['year'] = entities['year']
                results['period'] = f"Year {entities['year']}"
            
            return {
                "status": "success",
                "analysis_type": "summary",
                "results": results
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _aggregation_analysis(self, df: pd.DataFrame, query: str, entities: dict = None) -> Dict[str, Any]:
        """Perform aggregations"""
        try:
            total_sales = float(df['NetAmount'].sum())
            
            results = {
                "total_sales": total_sales,
                "filtered_rows": len(df)
            }
            
            # Add year context
            if entities and 'year' in entities:
                results['year'] = entities['year']
                results['period'] = f"Year {entities['year']}"
            
            return {
                "status": "success",
                "analysis_type": "aggregation",
                "results": results
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}