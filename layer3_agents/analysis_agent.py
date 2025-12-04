import pandas as pd
import logging
from typing import Dict, Any
from data_connector import mcp_store
import os
import re
import json
from config import GEMINI_API_KEY


# Set API key FIRST, before any imports
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY


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
    """Dynamic Analysis Agent powered by Gemini LLM with anomaly detection support."""


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
            
            # Load DataFrame (check for anomaly-enriched data first)
            logger.info("🔄 Loading sales data from mcp_store...")
            df = self._get_dataframe_with_anomalies()
            logger.info(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Check if anomaly columns are present
            has_anomalies = 'is_anomaly' in df.columns
            if has_anomalies:
                logger.info(f"✅ Anomaly data available: {df['is_anomaly'].sum()} anomalies detected")
            
            # Define agent prefix with anomaly support
            custom_prefix = self._build_agent_prefix(has_anomalies)
            
            # Create pandas dataframe agent
            logger.info("🔄 Creating pandas dataframe agent...")
            
            self.llm_agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                agent_type="zero-shot-react-description",
                allow_dangerous_code=True,
                prefix=custom_prefix,
                max_iterations=3,
                early_stopping_method="force",
                return_intermediate_steps=True,
                agent_executor_kwargs={
                    "handle_parsing_errors": True
                }
            )
            
            logger.info("✅ [AnalysisAgent] Dynamic Gemini LLM agent initialized")
            logger.info(f"   └─ Model: gemini-2.5-flash")
            logger.info(f"   └─ DataFrame shape: {df.shape}")
            logger.info(f"   └─ Anomaly support: {has_anomalies}")
            logger.info(f"   └─ Max iterations: 3")
            
        except Exception as e:
            logger.error(f"❌ [AnalysisAgent] Failed to set up LLM analytics agent: {e}", exc_info=True)
            self.llm_agent = None
            self.entity_extractor_llm = None


    def _get_dataframe_with_anomalies(self) -> pd.DataFrame:
        """Get dataframe with anomaly flags if available, otherwise regular data"""
        
        # Try to get enriched data with anomaly flags
        df_with_anomalies = mcp_store.get_enriched_data('anomalies')
        
        if df_with_anomalies is not None and 'is_anomaly' in df_with_anomalies.columns:
            logger.info("✅ Using anomaly-enriched dataframe")
            return df_with_anomalies
        else:
            logger.info("ℹ️ No anomaly data available, using regular dataframe")
            return mcp_store.get_sales_data()


    def _build_agent_prefix(self, has_anomalies: bool) -> str:
        """Build agent prefix with optional anomaly instructions"""
        
        base_columns = """
- Date (datetime): Transaction date
- Revenue (float): Revenue/sales amount
- Volume (float): Quantity sold
- Customer (str): Customer ID
- Product (str): Product ID
- Sales Org (str): Sales organization
- COGS (float): Cost of goods sold"""


        anomaly_columns = """
- is_anomaly (bool): Whether this record is flagged as anomaly
- anomaly_score (float): Anomaly severity score (lower = more severe)
- anomaly_reason (str): Human-readable explanation of why it's anomalous"""


        anomaly_rules = """


**ANOMALY ANALYSIS RULES:**
1. To filter anomalies: df[df['is_anomaly'] == True]
2. To count anomalies by customer: df[df['is_anomaly']].groupby('Customer').size()
3. To get customer with most anomalies: df[df['is_anomaly']].groupby('Customer').size().idxmax()
4. Always include anomaly_reason when showing anomalies
5. Sort by anomaly_score (ascending) to show most severe first"""


        # ✅ INITIALIZE prefix HERE
        prefix = f"""
You are a business analytics assistant working with a pandas DataFrame called `df`.
This DataFrame contains sales data with these columns:
{base_columns}
"""


        # ✅ Conditionally add anomaly info
        if has_anomalies:
            prefix += anomaly_columns
            prefix += anomaly_rules


        # ✅ Add the rest of the instructions
        prefix += """


**YOUR TASK:**
Answer the user's question by writing Python code ONCE, then immediately provide the Final Answer.


**CRITICAL EXECUTION RULES:**
1. Write your Python code
2. Execute it ONCE
3. Read the result from the Observation
4. Immediately write "Thought: I now know the final answer"
5. Immediately write your Final Answer
6. STOP - DO NOT execute code again


**FORBIDDEN ACTIONS:**
- DO NOT run the same code multiple times
- DO NOT repeat actions
- DO NOT loop back after getting a result
- If you have an Observation with data, that means SUCCESS - stop and answer


**CORRECT FLOW (FOLLOW THIS EXACTLY):**
Thought: I need to find which customer has the least revenue
Action: python_repl_ast
Action Input:
customer_revenue = df.groupby('Customer')['Revenue'].sum()
min_customer = customer_revenue.idxmin()
min_value = customer_revenue.min()
result = f"Customer: {{min_customer}}, Revenue: {{min_value}}"
print(result)


Observation: Customer: 1001, Revenue: 42953211000


Thought: I now know the final answer. Both customer ID and revenue are in the observation.
Final Answer:
* **Customer with Least Revenue:** 1001
* **Total Revenue:** $42,953,211,000


**DATA QUERY RULES:**
1. For LOWEST/MINIMUM: Use .idxmin() and .min()
2. For HIGHEST/MAXIMUM: Use .idxmax() and .max()
3. For TOP N: Use .nlargest(n)
4. For BOTTOM N: Use .nsmallest(n)


**FINAL ANSWER FORMAT:**
Always format as:
* **Key Finding:** Value with units
* **Additional Info:** Supporting details


**REMEMBER:**
- One code execution = One result = One Final Answer
- After Observation, you MUST immediately provide Final Answer
- NO LOOPS, NO RETRIES, NO REPETITIONS
- If you see an Observation, your next step is ALWAYS "Thought: I now know the final answer"


**STRICT STOPPING CONDITION:**
Once you write "Final Answer:", YOU MUST STOP COMPLETELY.
Do not execute any more actions.
Do not think any more thoughts.
The conversation is OVER.
"""
        
        return prefix


    # ===========================
    # ANOMALY-SPECIFIC METHODS
    # ===========================
    
    def _handle_anomaly_query(self, query: str) -> dict:
        """Handle anomaly-specific queries with pattern matching"""
        
        anomaly_df = mcp_store.get_enriched_data('anomaly_records')
        
        if anomaly_df is None or len(anomaly_df) == 0:
            return {
                "status": "error",
                "message": "No anomalies detected yet. Run anomaly detection first.",
                "analysis_type": "anomaly",
                "results": {}
            }
        
        # ✅ DYNAMIC COLUMN DETECTION
        customer_col = 'Customer' if 'Customer' in anomaly_df.columns else 'SoldToParty'
        product_col = 'Product' if 'Product' in anomaly_df.columns else 'Material'
        date_col = 'Date' if 'Date' in anomaly_df.columns else 'CreationDate'
        
        query_lower = query.lower()
        
        # Pattern 1: Customer with most anomalies
        if 'customer' in query_lower and ('most' in query_lower or 'top' in query_lower):
            logger.info(f"🎯 Pattern: Customer with most anomalies")
            customer_counts = anomaly_df[customer_col].value_counts()
            top_customer = customer_counts.index[0]
            top_count = customer_counts.iloc[0]
            
            return {
                "status": "success",
                "analysis_type": "anomaly",
                "results": {
                    "llm_raw": f"**Customer with Most Anomalies:**\n* Customer ID: {top_customer}\n* Anomaly Count: {top_count}\n* Percentage: {(top_count/len(anomaly_df)*100):.1f}%"
                }
            }
        
        # Fallback: Use LLM agent
        logger.info(f"🎯 Using LLM agent for complex anomaly query")
        return None  # Will trigger LLM agent


    # ===========================
    # MAIN EXECUTION WITH PATTERN DETECTION
    # ===========================


    def execute(self, query: str, analysis_type: str = "summary", entities: dict = None) -> Dict[str, Any]:
        """Main execution with flexible pattern detection"""
        logger.info(f"[{self.name}] Received query: '{query}' | type={analysis_type} | entities={entities}")
        
        try:
            query_lower = query.lower()
            
            # Log pattern detection
            logger.info("="*60)
            logger.info("🔍 QUERY PATTERN DETECTION")
            logger.info("="*60)
            logger.info(f"Query: {query}")
            
            # ✅ PATTERN 0: Anomaly-related query
            if 'anomal' in query_lower:
                logger.info("✅ PATTERN MATCHED: Anomaly query")
                return self._handle_anomaly_query(query)
            
            # ✅ PATTERN 1: No pattern matched - use LLM agent
            logger.info("❌ PATTERN NOT MATCHED - Using LLM agent")
            logger.info("="*60)
            
            if self.llm_agent is not None:
                logger.info(f"[{self.name}] Using Gemini LLM agent for query.")
                # ✅ PASS ENTITIES to LLM analysis
                llm_result = self._llm_analysis(query, entities)
                logger.info(f"[{self.name}] LLM response status: {llm_result.get('status')}")
                
                llm_raw = llm_result.get('results', {}).get('llm_raw', '')
                clean_response = self._extract_clean_response(llm_raw)
                
                extracted_entities = self._extract_entities_dynamically(llm_raw, query)
                all_entities = {**(entities or {}), **extracted_entities}
                
                logger.info(f"[{self.name}] Extracted entities: {extracted_entities}")
                
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


    def _llm_analysis(self, query: str, entities: dict = None) -> Dict[str, Any]:
        """Use LLM agent with answer extraction from intermediate steps"""
        import signal
        import platform
        import time
        
        logger.info("="*60)
        logger.info(f"[AnalysisAgent] Starting LLM analysis")
        logger.info("="*60)
        logger.info(f"📝 Query: '{query}'")
        logger.info(f"📋 Entities: {entities}")
        
        # ✅ DETERMINE AGGREGATION LABEL FROM ENTITIES
        aggregation_type = (entities or {}).get('aggregation_type', 'total')
        
        # Map aggregation types to display labels
        label_map = {
            'highest': 'Highest',
            'most': 'Most',
            'maximum': 'Maximum',
            'max': 'Maximum',
            'top': 'Top',
            'least': 'Least',
            'lowest': 'Lowest',
            'minimum': 'Minimum',
            'min': 'Minimum',
            'bottom': 'Bottom',
            'total': 'Total'
        }
        
        aggregation_label = label_map.get(aggregation_type.lower(), aggregation_type.capitalize())
        logger.info(f"📊 Aggregation type: '{aggregation_type}' → Label: '{aggregation_label}'")
        
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
            
            # ✅ EXTRACT FROM INTERMEDIATE STEPS if no Final Answer
            if intermediate_steps and len(intermediate_steps) > 0:
                last_step = intermediate_steps[-1]
                if len(last_step) >= 2:
                    last_observation = str(last_step[1])
                    
                    logger.info(f"📝 Last observation: {last_observation[:200]}")
                    
                    # Check if we have both customer and revenue
                    if 'Customer:' in last_observation and 'Revenue:' in last_observation:
                        # Extract customer ID
                        customer_match = re.search(r'Customer:\s*(\d+)', last_observation)
                        revenue_match = re.search(r'Revenue:\s*([\d,\.]+)', last_observation)
                        
                        if customer_match and revenue_match:
                            customer_id = customer_match.group(1)
                            revenue_value = revenue_match.group(1).replace(',', '')
                            
                            # ✅ BUILD CLEAN OUTPUT WITH DYNAMIC LABEL
                            clean_output = (
                                f"* **Customer with {aggregation_label} Revenue:** {customer_id}\n"
                                f"* **Total Revenue:** ${float(revenue_value):,.2f}"
                            )
                            
                            logger.info(f"✅ Extracted answer from intermediate steps")
                            logger.info(f"   └─ Customer: {customer_id}")
                            logger.info(f"   └─ Revenue: ${float(revenue_value):,.2f}")
                            logger.info(f"   └─ Label: {aggregation_label}")
                            
                            return {
                                "status": "success",
                                "analysis_type": "llm_analysis",
                                "results": {"llm_raw": clean_output}
                            }
            
            # Fallback: use code_output
            if not code_output or code_output.strip() == '':
                return {
                    "status": "error",
                    "analysis_type": "llm_analysis",
                    "results": {"error": "Empty response from agent"}
                }
            
            clean_output = code_output.replace("Final Answer:", "").strip()
            
            logger.info("="*60)
            logger.info("📋 FINAL OUTPUT:")
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
        
        metric_match = re.search(r'(revenue|sales|orders|anomal)', llm_output, re.IGNORECASE)
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
                "total_sales": float(df['Revenue'].sum()) if 'Revenue' in df.columns else 0,
                "total_orders": len(df),
                "avg_order_value": float(df['Revenue'].mean()) if 'Revenue' in df.columns else 0,
                "unique_customers": int(df['Customer'].nunique()) if 'Customer' in df.columns else 0,
                "unique_products": int(df['Product'].nunique()) if 'Product' in df.columns else 0,
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
