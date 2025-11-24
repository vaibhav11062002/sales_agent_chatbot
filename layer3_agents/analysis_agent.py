import pandas as pd
import logging
from typing import Dict, Any
from data_connector import mcp_store
import os
import re
import json

# Set API key FIRST, before any imports
os.environ['GOOGLE_API_KEY'] = "AIzaSyBvGk-pDi2hqdq0CLSoKV2Sa8TH5IWShtE"

logger = logging.getLogger(__name__)

# Gemini + LangChain agent imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
    logger.info("✅ LangChain imports successful")
except ImportError as e:
    ChatGoogleGenerativeAI = None
    create_pandas_dataframe_agent = None
    logger.error(f"❌ LangChain import failed: {e}")


class AnalysisAgent:
    """Dynamic Analysis Agent powered by Gemini LLM for pandas with hallucination detection."""

    def __init__(self):
        self.name = "AnalysisAgent"
        self.llm_agent = None
        self.entity_extractor_llm = None
        
        # Check imports first
        if ChatGoogleGenerativeAI is None:
            logger.error("❌ ChatGoogleGenerativeAI not available - install: pip install langchain-google-genai")
            return
        
        if create_pandas_dataframe_agent is None:
            logger.error("❌ create_pandas_dataframe_agent not available - install: pip install langchain-experimental")
            return
        
        try:
            # Initialize Main LLM
            logger.info("🔄 Initializing Gemini LLM (gemini-2.5-flash)...")
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                convert_system_message_to_human=True
            )
            logger.info("✅ Main LLM initialized")
            
            # Initialize Entity Extractor LLM
            self.entity_extractor_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0
            )
            logger.info("✅ Entity extractor LLM initialized")
            
            # Load DataFrame
            logger.info("🔄 Loading sales data from mcp_store...")
            df = mcp_store.get_sales_data()
            logger.info(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Define agent prefix (simplified to avoid template variable issues)
            custom_prefix = """
You are a business analytics assistant working with a pandas DataFrame called `df`.
This DataFrame contains sales data from SAP HANA with these columns:
- CreationDate (datetime): Transaction date
- NetAmount (float): Revenue/sales amount
- OrderQuantity (float): Quantity sold
- TaxAmount (float): Tax amount
- CostAmount (float): Cost of goods sold
- SoldToParty (str): Customer ID
- Product (str): Product ID
- SalesDocument (str): Sales document number
- SalesOrganization (str): Sales organization ID

**CRITICAL RULES:**
1. For SPECIFIC customer queries: Filter first, then calculate
   Example: df[df['SoldToParty'] == '1002']['NetAmount'].sum()

2. For HIGHEST/TOP customer queries: Use groupby + idxmax()
   Example: df.groupby('SoldToParty')['NetAmount'].sum().idxmax()

3. Always PRINT the customer ID and use that EXACT value in Final Answer
   Example: print(f"Customer ID: 1002") then say "Customer: 1002"

4. DO NOT change customer IDs between execution and final answer

**Output Format:**
Final Answer:
* **Key Finding:** Value with units

**Remember:** Your Final Answer MUST match your execution output EXACTLY.
"""
            
            # Create pandas dataframe agent
            logger.info("🔄 Creating pandas dataframe agent...")
            
            self.llm_agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                agent_type="zero-shot-react-description",
                allow_dangerous_code=True,
                prefix=custom_prefix,
                max_iterations=10,
                agent_executor_kwargs={
                    "handle_parsing_errors": True
                }
            )
            
            logger.info("✅ [AnalysisAgent] Dynamic Gemini LLM agent initialized on FULL dataframe")
            logger.info(f"   └─ Model: gemini-2.5-flash")
            logger.info(f"   └─ DataFrame shape: {df.shape}")
            logger.info(f"   └─ Max iterations: 10")
            
        except Exception as e:
            logger.error(f"❌ [AnalysisAgent] Failed to set up LLM analytics agent: {e}", exc_info=True)
            self.llm_agent = None
            self.entity_extractor_llm = None

    # ===========================
    # DEDICATED CUSTOMER METHODS (100% ACCURATE)
    # ===========================
    
    def _get_customer_revenue(self, customer_id: str) -> Dict[str, Any]:
        """Get total revenue for specific customer - guaranteed accurate"""
        df = mcp_store.get_sales_data()
        customer_id = str(customer_id)
        
        logger.info(f"[CustomerRevenue] Querying revenue for customer: {customer_id}")
        
        customer_df = df[df['SoldToParty'] == customer_id]
        
        if len(customer_df) == 0:
            logger.warning(f"❌ No data found for customer {customer_id}")
            return {
                "customer_id": customer_id,
                "total_revenue": 0.0,
                "total_orders": 0,
                "avg_order_value": 0.0,
                "found": False
            }
        
        total_revenue = float(customer_df['NetAmount'].sum())
        total_orders = len(customer_df)
        avg_order = float(customer_df['NetAmount'].mean())
        
        logger.info(f"✅ Customer {customer_id}: ${total_revenue:,.2f} from {total_orders} orders")
        
        return {
            "customer_id": customer_id,
            "total_revenue": total_revenue,
            "total_orders": total_orders,
            "avg_order_value": avg_order,
            "found": True
        }
    
    def _get_top_customers_by_revenue(self, top_n: int = 10) -> pd.DataFrame:
        """Get top N customers by revenue - guaranteed accurate"""
        df = mcp_store.get_sales_data()
        
        logger.info(f"[TopCustomers] Getting top {top_n} customers by revenue...")
        
        customer_revenue = (
            df.groupby('SoldToParty')['NetAmount']
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        
        customer_revenue.columns = ['Customer', 'TotalRevenue']
        
        logger.info(f"✅ Top {top_n} customers:")
        for idx, row in customer_revenue.iterrows():
            logger.info(f"  {idx+1}. Customer {row['Customer']}: ${row['TotalRevenue']:,.2f}")
        
        return customer_revenue

    # ===========================
    # MAIN EXECUTION WITH FLEXIBLE PATTERN DETECTION
    # ===========================

    def execute(self, query: str, analysis_type: str = "summary", entities: dict = None) -> Dict[str, Any]:
        """Main execution with flexible pattern detection (no question word required)"""
        logger.info(f"[{self.name}] Received query: '{query}' | type={analysis_type} | entities={entities}")
        
        try:
            query_lower = query.lower()
            
            # Log pattern detection
            logger.info("="*60)
            logger.info("🔍 QUERY PATTERN DETECTION")
            logger.info("="*60)
            logger.info(f"Query: {query}")
            logger.info(f"  'customer' in query: {'customer' in query_lower}")
            logger.info(f"  'highest/max/top' in query: {any(word in query_lower for word in ['highest', 'max', 'maximum', 'top', 'most', 'best', 'largest', 'greatest'])}")
            logger.info(f"  'revenue/sales' in query: {('revenue' in query_lower or 'sales' in query_lower)}")
            
            # ✅ PATTERN 1: Specific customer revenue query
            customer_match = re.search(r'customer\s+(\d+)', query_lower)
            if customer_match and ('revenue' in query_lower or 'total' in query_lower or 'sales' in query_lower):
                customer_id = customer_match.group(1)
                logger.info(f"✅ PATTERN MATCHED: Specific customer revenue (Customer {customer_id})")
                logger.info("🎯 Using dedicated method (bypassing LLM)")
                logger.info("="*60)
                
                customer_data = self._get_customer_revenue(customer_id)
                
                if not customer_data['found']:
                    result_text = f"* **Customer:** {customer_id}\n* **Status:** No data found"
                else:
                    result_text = (
                        f"* **Customer:** {customer_id}\n"
                        f"* **Total Revenue:** ${customer_data['total_revenue']:,.2f}\n"
                        f"* **Total Orders:** {customer_data['total_orders']:,}\n"
                        f"* **Average Order Value:** ${customer_data['avg_order_value']:,.2f}"
                    )
                
                result = {
                    "status": "success",
                    "analysis_type": "customer_revenue",
                    "results": {"llm_raw": result_text}
                }
                
                extracted_entities = {
                    "customer_id": customer_id,
                    "metric": "revenue",
                    "metric_value": customer_data['total_revenue']
                }
                all_entities = {**(entities or {}), **extracted_entities}
                
                mcp_store.update_agent_context(self.name, {
                    "analysis_type": "customer_revenue",
                    "results": result,
                    "query": query,
                    "filtered_by": entities,
                    "extracted_entities": extracted_entities
                })
                
                mcp_store.update_dialogue_state(all_entities, query, result_text)
                
                return result
            
            # ✅ PATTERN 2: Top/highest/max customer query (FLEXIBLE - no question word required)
            if 'customer' in query_lower and \
               ('highest' in query_lower or 'most' in query_lower or 'top' in query_lower or \
                'max' in query_lower or 'maximum' in query_lower or 'largest' in query_lower or \
                'best' in query_lower or 'greatest' in query_lower) and \
               ('revenue' in query_lower or 'sales' in query_lower):
                logger.info(f"✅ PATTERN MATCHED: Top customer query (flexible match)")
                logger.info("🎯 Using dedicated method (bypassing LLM)")
                logger.info("="*60)
                
                top_customers = self._get_top_customers_by_revenue(top_n=1)
                
                if len(top_customers) > 0:
                    top_customer = top_customers.iloc[0]
                    customer_id = str(top_customer['Customer'])
                    revenue = float(top_customer['TotalRevenue'])
                    
                    result_text = (
                        f"* **Highest Revenue Customer:** {customer_id}\n"
                        f"* **Total Revenue:** ${revenue:,.2f}"
                    )
                    
                    result = {
                        "status": "success",
                        "analysis_type": "top_customer",
                        "results": {"llm_raw": result_text}
                    }
                    
                    extracted_entities = {
                        "customer_id": customer_id,
                        "metric": "revenue",
                        "metric_value": revenue,
                        "rank": "highest"
                    }
                    all_entities = {**(entities or {}), **extracted_entities}
                    
                    mcp_store.update_agent_context(self.name, {
                        "analysis_type": "top_customer",
                        "results": result,
                        "query": query,
                        "extracted_entities": extracted_entities
                    })
                    
                    mcp_store.update_dialogue_state(all_entities, query, result_text)
                    
                    return result
            
            # ✅ PATTERN 3: No pattern matched - use LLM agent
            logger.info("❌ PATTERN NOT MATCHED - Using LLM agent")
            logger.info("="*60)
            
            if self.llm_agent is not None:
                logger.info(f"[{self.name}] Using Gemini LLM agent for query.")
                llm_result = self._llm_analysis(query)
                logger.info(f"[{self.name}] LLM response status: {llm_result.get('status')}")
                
                llm_raw = llm_result.get('results', {}).get('llm_raw', '')
                clean_response = self._extract_clean_response(llm_raw)
                
                extracted_entities = self._extract_entities_dynamically(llm_raw, query)
                all_entities = {**(entities or {}), **extracted_entities}
                
                logger.info(f"[{self.name}] Extracted entities: {extracted_entities}")
                logger.info(f"[{self.name}] All entities: {all_entities}")
                
                mcp_store.update_agent_context(self.name, {
                    "analysis_type": "llm_analysis",
                    "results": llm_result,
                    "query": query,
                    "filtered_by": entities,
                    "extracted_entities": extracted_entities
                })
                
                mcp_store.update_dialogue_state(all_entities, query, clean_response)
                
                return llm_result

            # Fallback
            logger.info(f"[{self.name}] LLM agent unavailable, using fallback analysis")
            fallback_result = self._summary_analysis(mcp_store.get_sales_data(), query, entities)
            
            mcp_store.update_agent_context(self.name, {
                "analysis_type": "summary_fallback",
                "results": fallback_result,
                "query": query,
                "filtered_by": entities,
                "extracted_entities": entities or {}
            })
            
            mcp_store.update_dialogue_state(entities or {}, query, str(fallback_result))
            return fallback_result
            
        except Exception as e:
            logger.error(f"[{self.name}] ERROR: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _llm_analysis(self, query: str) -> Dict[str, Any]:
        """Use LLM agent with hallucination detection and correction"""
        import signal
        import platform
        import time
        
        logger.info("="*60)
        logger.info(f"[AnalysisAgent] Starting LLM analysis")
        logger.info("="*60)
        logger.info(f"📝 Query: '{query}'")
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Agent execution timeout")
        
        try:
            if platform.system() != 'Windows':
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)
            
            logger.info("🚀 Invoking LLM agent...")
            start_time = time.time()
            
            response = self.llm_agent.invoke(query)
            
            elapsed = time.time() - start_time
            logger.info("="*60)
            logger.info(f"✅ Response received successfully!")
            logger.info(f"⏱️ Elapsed time: {elapsed:.2f} seconds")
            logger.info("="*60)
            
            if platform.system() != 'Windows':
                signal.alarm(0)
            
            code_output = response.get('output', '')
            intermediate_steps = response.get('intermediate_steps', [])
            
            logger.info(f"📊 Intermediate steps: {len(intermediate_steps)}")
            
            # ✅ HALLUCINATION DETECTION: Parse actual execution output
            execution_data = {}
            
            if intermediate_steps:
                logger.info("🔍 Parsing intermediate steps for actual values...")
                for idx, step in enumerate(intermediate_steps):
                    if len(step) >= 2:
                        action, observation = step
                        observation_str = str(observation)
                        
                        logger.info(f"  Step {idx+1} observation: {observation_str[:200]}")
                        
                        # Extract customer ID from execution
                        customer_match = re.search(r'Customer ID:\s*(\d+)', observation_str)
                        if customer_match:
                            execution_data['customer_id'] = customer_match.group(1)
                            logger.info(f"    └─ Found customer_id in execution: {execution_data['customer_id']}")
                        
                        # Extract revenue value
                        revenue_match = re.search(r'Revenue:\s*([\d,]+\.?\d*)', observation_str)
                        if revenue_match:
                            execution_data['revenue'] = revenue_match.group(1).replace(',', '')
                            logger.info(f"    └─ Found revenue in execution: {execution_data['revenue']}")
            
            # ✅ HALLUCINATION CORRECTION: Compare with Final Answer
            if execution_data:
                logger.info("="*60)
                logger.info("🔍 HALLUCINATION CHECK")
                logger.info("="*60)
                logger.info(f"Execution data: {execution_data}")
                
                # Check customer ID in final answer
                if 'customer_id' in execution_data:
                    actual_customer = execution_data['customer_id']
                    
                    # Find customer ID in final answer
                    final_customer_matches = re.findall(r'Customer[:\s]+(\d+)', code_output)
                    
                    if final_customer_matches:
                        final_customer = final_customer_matches[0]
                        
                        if final_customer != actual_customer:
                            logger.error("🚨 HALLUCINATION DETECTED!")
                            logger.error(f"  Execution says: Customer {actual_customer}")
                            logger.error(f"  Final Answer says: Customer {final_customer}")
                            logger.error("  🔧 Correcting the hallucination...")
                            
                            # Replace wrong customer ID with correct one
                            code_output = re.sub(
                                rf'Customer[:\s]+{final_customer}',
                                f'Customer: {actual_customer}',
                                code_output
                            )
                            code_output = re.sub(
                                rf'Customer {final_customer}',
                                f'Customer {actual_customer}',
                                code_output
                            )
                            
                            logger.info(f"  ✅ Corrected to Customer {actual_customer}")
                        else:
                            logger.info("  ✅ No hallucination - customer ID matches")
                
                logger.info("="*60)
            
            # Validation checks
            if not code_output or code_output.strip() == '':
                return {
                    "status": "error",
                    "analysis_type": "llm_analysis",
                    "results": {"error": "Empty response from agent"}
                }
            
            if 'Agent stopped' in code_output:
                return {
                    "status": "error",
                    "analysis_type": "llm_analysis",
                    "results": {"error": "Agent stopped - query too complex"}
                }
            
            clean_output = code_output.replace("Final Answer:", "").strip()
            
            logger.info("="*60)
            logger.info("📋 FINAL OUTPUT (corrected):")
            logger.info("-"*60)
            logger.info(clean_output[:300])
            logger.info("-"*60)
            
            result = {
                "status": "success",
                "analysis_type": "llm_analysis",
                "results": {"llm_raw": clean_output}
            }
            
            logger.info("✅ LLM Analysis completed successfully!")
            logger.info("="*60)
            
            return result
            
        except TimeoutError:
            logger.error("❌ TIMEOUT ERROR")
            if platform.system() != 'Windows':
                signal.alarm(0)
            return {
                "status": "error",
                "analysis_type": "llm_analysis",
                "results": {"error": "Timeout after 60s"}
            }
            
        except Exception as e:
            logger.error(f"❌ EXCEPTION: {type(e).__name__}: {str(e)}", exc_info=True)
            if platform.system() != 'Windows':
                signal.alarm(0)
            
            error_msg = str(e).lower()
            if '429' in error_msg or 'rate limit' in error_msg:
                return {
                    "status": "error",
                    "analysis_type": "llm_analysis",
                    "results": {"error": "Rate limit exceeded - wait 60s"}
                }
            
            return {
                "status": "error",
                "analysis_type": "llm_analysis",
                "results": {"error": f"Analysis failed: {str(e)}"}
            }

    def _extract_entities_dynamically(self, llm_output: str, query: str) -> dict:
        """Use LLM to extract entities"""
        if not self.entity_extractor_llm:
            return self._extract_entities_regex_fallback(llm_output)
        
        try:
            prompt = f"""Extract entities from this query and result.

Query: "{query}"
Result: {llm_output[:1500]}

Extract: customer IDs, product IDs, years, metrics, etc.

Return JSON only:
{{"entity_type": "value", ...}}

Example: {{"customer_id": "1002", "metric": "revenue"}}"""

            response = self.entity_extractor_llm.invoke(prompt)
            content = response.content
            
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                entities_raw = json.loads(json_match.group())
                entities = {}
                for key, value in entities_raw.items():
                    normalized_key = key.lower().replace(' ', '_').replace('-', '_')
                    entities[normalized_key] = value
                
                logger.info(f"📌 LLM extracted {len(entities)} entities: {entities}")
                return entities
            else:
                return self._extract_entities_regex_fallback(llm_output)
            
        except Exception as e:
            logger.error(f"[Entity Extraction] Failed: {e}")
            return self._extract_entities_regex_fallback(llm_output)

    def _extract_entities_regex_fallback(self, llm_output: str) -> dict:
        """Regex fallback for entity extraction"""
        entities = {}
        
        customer_patterns = [
            r'Customer[:\s]+([A-Z0-9]+)',
            r'customer\s+(\d+)',
        ]
        for pattern in customer_patterns:
            match = re.search(pattern, llm_output, re.IGNORECASE)
            if match:
                entities['customer_id'] = match.group(1)
                break
        
        product_patterns = [r'Product[:\s]+([A-Z0-9]+)']
        for pattern in product_patterns:
            match = re.search(pattern, llm_output, re.IGNORECASE)
            if match:
                entities['product_id'] = match.group(1)
                break
        
        year_match = re.search(r'(20\d{2})', llm_output)
        if year_match:
            entities['year'] = int(year_match.group(1))
        
        metric_match = re.search(r'(revenue|sales|orders)', llm_output, re.IGNORECASE)
        if metric_match:
            entities['metric'] = metric_match.group(1).lower()
        
        return entities

    def _extract_clean_response(self, llm_raw: str) -> str:
        """Clean response for display"""
        cleaned = re.sub(r'``````', '', llm_raw, flags=re.DOTALL)
        cleaned = cleaned.replace("Final Answer:", "").strip()
        
        lines = cleaned.split('\n')
        answer_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('*') or stripped.startswith('-') or '**' in stripped:
                answer_lines.append(stripped)
            elif stripped and len(stripped) > 20:
                answer_lines.append(stripped)
        
        if answer_lines:
            return '\n'.join(answer_lines)
        
        return cleaned[:1000] if cleaned else llm_raw[:1000]

    def _summary_analysis(self, df: pd.DataFrame, query: str, entities: dict = None) -> Dict[str, Any]:
        """Fallback summary"""
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
            
            return {
                "status": "success",
                "analysis_type": "summary",
                "results": results
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def execute_comparison(self, query: str, years: list) -> Dict[str, Any]:
        """Execute comparison"""
        logger.info(f"{self.name}: Comparing years: {years}")
        
        try:
            df = mcp_store.get_sales_data()
            comparison_results = {}
            
            for year in years:
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
                "filtered_by": {"years": years}
            })
            
            return {
                "status": "success",
                "analysis_type": "comparison",
                "results": comparison_results
            }
            
        except Exception as e:
            logger.error(f"Comparison error: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


# ===========================
# TESTING
# ===========================
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    print("\n" + "="*60)
    print("TESTING ANALYSIS AGENT")
    print("="*60 + "\n")
    
    agent = AnalysisAgent()
    
    if agent.llm_agent:
        print("✅ LLM Agent initialized successfully!")
    else:
        print("❌ LLM Agent initialization failed!")
    
    print("\n" + "="*60)
    print("TOP 10 CUSTOMERS BY REVENUE (ACTUAL DATA)")
    print("="*60)
    
    df = mcp_store.get_sales_data()
    top_10 = df.groupby('SoldToParty')['NetAmount'].sum().sort_values(ascending=False).head(10)
    
    for idx, (customer, revenue) in enumerate(top_10.items(), 1):
        print(f"{idx:2d}. Customer {customer}: ${revenue:,.2f}")
    
    print("\n" + "="*60)
    print("TESTING PATTERN DETECTION")
    print("="*60)
    
    test_queries = [
        "which customer has highest revenue?",
        "customer with highest revenue?",
        "Highest Revenue Customer??",
        "show max revenue customer",
        "top customer by sales"
    ]
    
    for test_query in test_queries:
        print(f"\n📝 Query: {test_query}")
        result = agent.execute(test_query, "summary", {})
        print(f"✅ Result: {result.get('results', {}).get('llm_raw', '')[:100]}")
    
    print("\n" + "="*60 + "\n")
