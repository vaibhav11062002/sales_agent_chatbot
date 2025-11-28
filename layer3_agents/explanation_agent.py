import logging
import google.generativeai as genai
from config import GEMINI_API_KEY
from data_connector import mcp_store

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


class ExplanationAgent:
    """Agent for generating natural language explanations using MCP-backed context"""
    
    def __init__(self):
        self.name = "ExplanationAgent"
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def _get_date_column(self, df) -> str:
        """Dynamically find the date column in the DataFrame"""
        date_candidates = ['Date', 'CreationDate', 'SalesDocumentDate', 'TransactionDate']
        
        for col in date_candidates:
            if col in df.columns:
                logger.info(f"Found date column: {col}")
                return col
        
        # If no match, try to find any column with 'date' in name
        for col in df.columns:
            if 'date' in col.lower():
                logger.info(f"Found date column by search: {col}")
                return col
        
        logger.warning("No date column found in DataFrame")
        return None
    
    def _get_revenue_column(self, df) -> str:
        """Dynamically find the revenue column in the DataFrame"""
        revenue_candidates = ['Revenue', 'NetAmount', 'Sales', 'TotalSales', 'Amount']
        
        for col in revenue_candidates:
            if col in df.columns:
                logger.info(f"Found revenue column: {col}")
                return col
        
        logger.warning("No revenue column found in DataFrame")
        return None
    
    def execute(self, query: str, analysis_results: dict = None) -> dict:
        """
        Generate natural language explanation of results
        Args:
            query: Original user query
            analysis_results: Results from current execution (optional)
        """
        logger.info(f"{self.name}: Generating explanation")
        
        try:
            # Get context from all agents via mcp_store
            all_contexts = mcp_store.get_all_contexts()
            
            # Get conversation history for better context
            conversation_history = mcp_store.conversation_history[-5:] if len(mcp_store.conversation_history) > 0 else []
            
            # Combine analysis results with agent contexts
            combined_results = {
                "current_results": analysis_results or {},
                "agent_contexts": all_contexts,
                "conversation_context": conversation_history
            }
            
            # Get data summary for context with dynamic column detection
            df = mcp_store.get_sales_data()
            
            # Find date column dynamically
            date_col = self._get_date_column(df)
            revenue_col = self._get_revenue_column(df)
            
            data_summary = {
                "total_records": len(df),
                "columns": len(df.columns)
            }
            
            # Add date range if date column exists
            if date_col and date_col in df.columns:
                try:
                    import pandas as pd
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    min_date = df[date_col].min()
                    max_date = df[date_col].max()
                    
                    if pd.notna(min_date) and pd.notna(max_date):
                        data_summary["date_range"] = f"{min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}"
                    else:
                        data_summary["date_range"] = "Date range unavailable"
                except Exception as e:
                    logger.warning(f"Could not parse date range: {e}")
                    data_summary["date_range"] = "Date range unavailable"
            else:
                data_summary["date_range"] = "Date column not found"
            
            # Add total sales if revenue column exists
            if revenue_col and revenue_col in df.columns:
                try:
                    total_sales = float(df[revenue_col].sum())
                    data_summary["total_sales"] = f"${total_sales:,.2f}"
                except Exception as e:
                    logger.warning(f"Could not calculate total sales: {e}")
                    data_summary["total_sales"] = "Total sales unavailable"
            else:
                data_summary["total_sales"] = "Revenue column not found"
            
            # Construct enhanced prompt
            prompt = f"""
You are a business analyst explaining sales data analysis results to executives.

Dataset Context:
- Total Records: {data_summary['total_records']:,}
- Total Columns: {data_summary['columns']}
- Date Range: {data_summary.get('date_range', 'N/A')}
- Total Sales: {data_summary.get('total_sales', 'N/A')}

User Query: {query}

Analysis Results:
{combined_results}

Instructions:
1. Provide a clear, executive-level explanation in 2-4 sentences
2. Focus on key insights and actionable takeaways
3. Use business language, not technical jargon
4. Highlight any trends, patterns, or anomalies
5. If relevant, reference previous conversation context

Generate a concise, insightful explanation:
"""
            
            response = self.model.generate_content(prompt)
            explanation = response.text
            
            # Store explanation in mcp_store for other agents
            mcp_store.update_agent_context(self.name, {
                "query": query,
                "explanation": explanation,
                "sources_used": list(all_contexts.keys())
            })
            
            return {
                "status": "success",
                "explanation": explanation,
                "message": "Explanation generated using AI and agent contexts"
            }
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}", exc_info=True)
            return {
                "status": "success",
                "explanation": self._fallback_explanation(analysis_results),
                "message": "Used fallback explanation due to error"
            }
    
    def _fallback_explanation(self, results: dict) -> str:
        """Generate simple explanation without LLM"""
        if not results:
            return "Analysis completed successfully."
        
        # Extract key metrics for fallback with flexible column names
        summary = []
        if isinstance(results, dict):
            # Check for various revenue field names
            revenue_keys = ['total_sales', 'Revenue', 'revenue', 'NetAmount', 'total_revenue']
            for key in revenue_keys:
                if key in results:
                    try:
                        value = float(results[key])
                        summary.append(f"Total sales: ${value:,.2f}")
                        break
                    except:
                        pass
            
            # Check for order count
            order_keys = ['total_orders', 'order_count', 'orders']
            for key in order_keys:
                if key in results:
                    try:
                        value = int(results[key])
                        summary.append(f"Total orders: {value:,}")
                        break
                    except:
                        pass
            
            # Check for forecasts
            if "forecasts" in results:
                try:
                    forecast_count = len(results['forecasts'])
                    summary.append(f"Forecast generated for {forecast_count} periods")
                except:
                    pass
        
        return ". ".join(summary) if summary else f"Results: {str(results)[:200]}..."
