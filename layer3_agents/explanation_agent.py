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
            
            # Get data summary for context
            df = mcp_store.get_sales_data()
            data_summary = {
                "total_records": len(df),
                "date_range": f"{df['CreationDate'].min().strftime('%Y-%m')} to {df['CreationDate'].max().strftime('%Y-%m')}",
                "total_sales": float(df['NetAmount'].sum())
            }
            
            # Construct enhanced prompt
            prompt = f"""
You are a business analyst explaining sales data analysis results to executives.

Dataset Context:
- Total Records: {data_summary['total_records']:,}
- Date Range: {data_summary['date_range']}
- Total Sales: ${data_summary['total_sales']:,.2f}

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
            logger.error(f"Error generating explanation: {str(e)}")
            return {
                "status": "success",
                "explanation": self._fallback_explanation(analysis_results),
                "message": "Used fallback explanation due to error"
            }
    
    def _fallback_explanation(self, results: dict) -> str:
        """Generate simple explanation without LLM"""
        if not results:
            return "Analysis completed successfully."
        
        # Extract key metrics for fallback
        summary = []
        if isinstance(results, dict):
            if "total_sales" in results:
                summary.append(f"Total sales: ${results['total_sales']:,.2f}")
            if "total_orders" in results:
                summary.append(f"Total orders: {results['total_orders']:,}")
            if "forecasts" in results:
                summary.append(f"Forecast generated for {len(results['forecasts'])} periods")
        
        return ". ".join(summary) if summary else f"Results: {str(results)[:200]}..."
