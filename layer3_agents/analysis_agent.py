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
        
    def execute_comparison(self, query: str, years: list) -> Dict[str, Any]:
        """
        Compare sales across multiple years using MCP context first
        
        Args:
            query: User query
            years: List of years to compare
        """
        logger.info(f"{self.name}: Comparing years: {years}")
        
        try:
            # Check MCP context for existing data
            all_contexts = mcp_store.get_all_contexts()
            cached_data = {}
            
            for agent_name, context in all_contexts.items():
                agent_data = context.get('data', {})
                filtered_by = agent_data.get('filtered_by', {})
                if 'year' in filtered_by:
                    year = filtered_by['year']
                    if year in years:
                        logger.info(f"Found cached data for year {year} in {agent_name}")
                        cached_data[year] = agent_data.get('results', {}).get('results', {})
            
            # Get fresh data from mcp_store
            df = mcp_store.get_sales_data()
            comparison_results = {}
            
            for year in years:
                # Use cached if available
                if year in cached_data:
                    logger.info(f"Using cached data for year {year}")
                    comparison_results[f"year_{year}"] = cached_data[year]
                else:
                    # Compute fresh
                    logger.info(f"Computing fresh data for year {year}")
                    year_df = df[df['CreationDate'].dt.year == year]
                    comparison_results[f"year_{year}"] = {
                        "total_sales": float(year_df['NetAmount'].sum()),
                        "total_orders": len(year_df),
                        "avg_order_value": float(year_df['NetAmount'].mean()) if len(year_df) > 0 else 0
                    }
            
            # Calculate differences
            if len(years) == 2:
                sales_diff = (comparison_results[f"year_{years[1]}"]["total_sales"] - 
                            comparison_results[f"year_{years[0]}"]["total_sales"])
                growth_pct = (sales_diff / comparison_results[f"year_{years[0]}"]["total_sales"] * 100 
                            if comparison_results[f"year_{years[0]}"]["total_sales"] > 0 else 0)
                
                comparison_results["comparison"] = {
                    "sales_difference": sales_diff,
                    "growth_percentage": growth_pct,
                    "years_compared": years
                }
            
            # Store in MCP
            mcp_store.update_agent_context(self.name, {
                "analysis_type": "comparison",
                "results": comparison_results,
                "query": query,
                "filtered_by": {"years": years, "comparison": True}
            })
            
            return {
                "status": "success",
                "analysis_type": "comparison",
                "results": comparison_results
            }
        except Exception as e:
            logger.error(f"Error in comparison: {str(e)}")
            return {"status": "error", "message": str(e)}
