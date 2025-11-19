import pandas as pd
import logging
from typing import Dict, Any
from data_connector import mcp_store

logger = logging.getLogger(__name__)

class AnalysisAgent:
    """Analysis Agent using MCP-backed data store"""
    
    def __init__(self):
        self.name = "AnalysisAgent"
    
    def execute(self, query: str, analysis_type: str = "summary") -> Dict[str, Any]:
        """Execute analysis using shared data store"""
        logger.info(f"{self.name}: Performing {analysis_type} analysis")
        
        try:
            # Access shared DataFrame
            df = mcp_store.get_sales_data()
            
            # Check context from other agents
            all_contexts = mcp_store.get_all_contexts()
            logger.info(f"Accessing contexts from {len(all_contexts)} other agents")
            
            # Perform analysis
            if analysis_type == "summary":
                results = self._summary_analysis(df, query)
            elif analysis_type == "aggregation":
                results = self._aggregation_analysis(df, query)
            else:
                results = self._summary_analysis(df, query)
            
            # Share results with other agents
            mcp_store.update_agent_context(self.name, {
                "analysis_type": analysis_type,
                "results": results,
                "query": query
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _summary_analysis(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Generate summary statistics"""
        try:
            results = {
                "total_sales": float(df['NetAmount'].sum()),
                "total_orders": len(df),
                "avg_order_value": float(df['NetAmount'].mean()),
                "unique_customers": int(df['SoldToParty'].nunique()),
            }
            
            return {
                "status": "success",
                "analysis_type": "summary",
                "results": results
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _aggregation_analysis(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Perform aggregations"""
        try:
            import re
            year_match = re.search(r'20\d{2}', query)
            
            if year_match:
                year = int(year_match.group())
                df = df[df['CreationDate'].dt.year == year]
            
            total_sales = float(df['NetAmount'].sum())
            
            return {
                "status": "success",
                "results": {"total_sales": total_sales, "filtered_rows": len(df)}
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
