import pandas as pd
import logging
from typing import Dict, Any
from data_connector import mcp_store

logger = logging.getLogger(__name__)

# Gemini + LangChain agent imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
except ImportError:
    ChatGoogleGenerativeAI = None
    create_pandas_dataframe_agent = None

class AnalysisAgent:
    """Dynamic Analysis Agent powered by Gemini LLM for pandas."""

    def __init__(self):
        self.name = "AnalysisAgent"
        self.llm_agent = None
        try:
            if ChatGoogleGenerativeAI and create_pandas_dataframe_agent:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0,
                    convert_system_message_to_human=True
                )
                # Full DataFrame (all columns, all rows) for the agent
                df = mcp_store.get_sales_data()
                # Modern, strict, dynamic Gemini prefix:
                custom_prefix = """
You are working with a pandas dataframe called `df`.
This dataframe contains sales data from SAP HANA.

Instructions for analytics:
- Answer ALL user questions with efficient pandas code and business analytics output.
- For statistics (mean, median, std), use one-liner stats: e.g. df['NetAmount'].mean(), df['NetAmount'].median().
- For filtering (e.g., on product, customer), use .loc or boolean masks (e.g., df[df['Product']=="PC103"]).
- For groupby aggregations, always use groupby-agg, not loops.
- No iterative or looping code—just pandas/statements per question.
- For time-based analysis, always convert date columns first: pd.to_datetime(df['CreationDate']).
- For regression/correlation, use pandas functions or .corr().
- Your answer should always:
  - First show the full python code block (from import to answer definition).
  - Then output the answer as a concise bullet list or summary.
  - All numeric results must be clearly labeled and rounded to 2 decimals.
- If the user's question is ambiguous, assume best business intent and return as much relevant insight in minimal code.

If you cannot answer, return a python code comment and a user-friendly error.
"""
                self.llm_agent = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=True,
                    agent_type="zero-shot-react-description",
                    allow_dangerous_code=True,
                    agent_executor_kwargs={
                        "handle_parsing_errors": True
                    },
                    prefix=custom_prefix,
                    max_iterations=30,
                )
                logger.info("[AnalysisAgent] Dynamic Gemini LLM agent initialized on FULL dataframe")
        except Exception as e:
            logger.warning("[AnalysisAgent] Failed to set up LLM analytics agent: %s", e)
            self.llm_agent = None

    def execute(self, query: str, analysis_type: str = "summary", entities: dict = None) -> Dict[str, Any]:
        logger.info(f"[{self.name}] Received query: '{query}' | type={analysis_type} | entities={entities}")
        try:
            # Always LLM first for ANY analysis
            if self.llm_agent is not None:
                logger.info(f"[{self.name}] Using Gemini LLM agent for query.")
                llm_result = self._llm_analysis(query)
                logger.info(f"[{self.name}] LLM response: {str(llm_result)[:400]}")
                # Store in MCP context
                mcp_store.update_agent_context(self.name, {
                    "analysis_type": "llm_analysis",
                    "results": llm_result,
                    "query": query,
                    "filtered_by": entities
                })
                return llm_result

            # Only if LLM agent is entirely unavailable, fallback
            return self._summary_analysis(mcp_store.get_sales_data(), query, entities)
        except Exception as e:
            logger.error(f"[{self.name}] ERROR: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _llm_analysis(self, query: str) -> Dict[str, Any]:
        """
        Use LLM agent for analytics. Handles all queries, outputs code AND answer, always structured.
        """
        try:
            response = self.llm_agent.invoke(query)
            code_output = response.get('output', '')
            # Check if LLM was stopped or repeated
            if ('Agent stopped' in code_output) or (
                isinstance(code_output, str) and code_output.strip() == ''
            ):
                logger.warning("[AnalysisAgent] LLM agent stopped or blank. No answer.")
                return {
                    "status": "error",
                    "analysis_type": "llm_analysis",
                    "results": {
                        "error": "LLM agent stopped or exceeded iteration. Try rephrasing or seek admin help."
                    }
                }
            logger.info(f"[{self.name}] LLM RAW OUTPUT HEAD: {str(code_output)[:200]}")
            return {
                "status": "success",
                "analysis_type": "llm_analysis",
                "results": {
                    "llm_raw": code_output
                }
            }
        except Exception as e:
            logger.error(f"[AnalysisAgent] LLM analytics failed: {e}")
            return {
                "status": "error",
                "analysis_type": "llm_analysis",
                "results": {
                    "error": str(e)
                }
            }
    # (Classic stat methods kept only for fallback/legacy)
    def _summary_analysis(self, df: pd.DataFrame, query: str, entities: dict = None) -> Dict[str, Any]:
        try:
            results = {
                "total_sales": float(df['NetAmount'].sum()),
                "total_orders": len(df),
                "avg_order_value": float(df['NetAmount'].mean()),
                "unique_customers": int(df['SoldToParty'].nunique()),
                "unique_products": int(df['Product'].nunique()),
            }
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

    def execute_comparison(self, query: str, years: list) -> Dict[str, Any]:
        logger.info(f"{self.name}: Comparing years: {years}")
        try:
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
            df = mcp_store.get_sales_data()
            comparison_results = {}
            for year in years:
                if year in cached_data:
                    logger.info(f"Using cached data for year {year}")
                    comparison_results[f"year_{year}"] = cached_data[year]
                else:
                    logger.info(f"Computing fresh data for year {year}")
                    year_df = df[df['CreationDate'].dt.year == year]
                    comparison_results[f"year_{year}"] = {
                        "total_sales": float(year_df['NetAmount'].sum()),
                        "total_orders": len(year_df),
                        "avg_order_value": float(year_df['NetAmount'].mean()) if len(year_df) > 0 else 0
                    }
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
