import os
import logging
import json
import re
from typing import Dict, Any, List
from config import GEMINI_API_KEY




logger = logging.getLogger(__name__)




try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False




class IntelligentRouter:
    """LLM-based intent classifier and entity extractor"""
    
    def __init__(self):
        self.llm = None
        if LLM_AVAILABLE:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0,
                    api_key=GEMINI_API_KEY
                )
                logger.info(f"✅ IntelligentRouter initialized with model: gemini-2.5-flash")
            except Exception as e:
                logger.error(f"Failed to initialize LLM router: {e}")
    
    def route_query(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Route query using LLM-based classification with pronoun resolution"""
        logger.info(f"🔀 IntelligentRouter: Starting routing for query: '{query}'")
        
        if not self.llm:
            logger.warning("LLM not available, using fallback routing")
            return self._fallback_routing(query)
        
        try:
            # ✅ NEW: Get dialogue state for pronoun resolution
            from data_connector import mcp_store
            dialogue_state = mcp_store.get_current_dialogue_state()
            context_entities = dialogue_state.get('entities', {})
            context_stack = dialogue_state.get('context_stack', [])
            
            # ✅ NEW: Extract recent customer/product mentions for plural pronoun resolution
            recent_customers = []
            recent_products = []
            
            # Look at last 5 contexts for entity mentions
            for ctx in context_stack[-5:]:
                ctx_entities = ctx.get('entities', {})
                if ctx_entities:
                    # Collect customer IDs
                    if 'customer_id' in ctx_entities:
                        cust_id = str(ctx_entities['customer_id'])
                        if cust_id and cust_id not in recent_customers:
                            recent_customers.append(cust_id)
                    
                    # Collect product IDs
                    if 'product_id' in ctx_entities:
                        prod_id = str(ctx_entities['product_id'])
                        if prod_id and prod_id not in recent_products:
                            recent_products.append(prod_id)
            
            # Build conversation context
            context_str = ""
            if conversation_history:
                recent_turns = conversation_history[-3:]
                context_str = "\n".join([
                    f"{turn.get('role', 'unknown')}: {turn.get('content', '')[:100]}"
                    for turn in recent_turns
                ])
            
            # ✅ NEW: Add entity context for pronoun resolution with recent mentions
            entity_context_lines = []
            if context_entities:
                entity_context_lines.append("**Current Active Entities:**")
                for k, v in list(context_entities.items())[-5:]:
                    entity_context_lines.append(f"  - {k}: {v}")
            
            if recent_customers:
                entity_context_lines.append(f"\n**Recently Mentioned Customers (for plural pronouns):** {recent_customers}")
            
            if recent_products:
                entity_context_lines.append(f"**Recently Mentioned Products (for plural pronouns):** {recent_products}")
            
            if entity_context_lines:
                context_str += "\n\n" + "\n".join(entity_context_lines)
            
            # ✅ UPDATED: Routing prompt with enhanced pronoun resolution
            prompt = f"""You are an intent classifier for a sales analytics AI system. Analyze the user's query and classify the intent.



**User Query:** "{query}"



**Recent Conversation Context:**
{context_str if context_str else "No previous context"}



**SYSTEM CONTEXT:**
- **Anomaly detection has ALREADY been run at system startup**
- All data includes pre-detected anomalies with flags: `is_anomaly`, `anomaly_score`, `anomaly_reason`
- Users do NOT need to "detect" anomalies - they are already available



**PRONOUN RESOLUTION RULES:**

**Singular Pronouns (single entity):**
- "this customer", "that product", "for it", "about this" → resolve from Current Active Entities above
- Example: Active has {{"customer_id": "1002"}}, Query: "show data for this customer" → entities: {{"customer_id": "1002"}}

**Plural Pronouns (multiple entities as LIST):**
- "both the customers", "these customers", "those products", "all of them", "the two customers", "compare them"
- → Extract as LIST from Recently Mentioned Customers/Products above
- Example: Recently Mentioned: ['1002', '1001'], Query: "compare both the customers" → entities: {{"customer_id": ["1002", "1001"]}}
- Example: Recently Mentioned: ['1002', '1001'], Query: "compare the count of orders given by both the customers" → entities: {{"customer_id": ["1002", "1001"], "metric": "order_count"}}

**Comparison Keywords (always extract as LIST):**
- "compare", "between", "vs", "versus", "difference between"
- → ALWAYS return as LIST even if not explicitly mentioned: {{"customer_id": ["1002", "1001"]}}



**Available Intents:**



1. **dashboard** - User wants visualizations, charts, graphs, or dashboards
   - Keywords: chart, graph, visualization, plot, dashboard, visualize, show me visually, create a chart, build a dashboard
   - Examples: "create a chart", "show me a graph", "visualize the data", "build a dashboard", "can you chart this", "plot the trends"



2. **analysis** - Questions about metrics, calculations, aggregations, or data summaries (TEXT-BASED only, NO visuals)
   - Keywords: total, sum, average, count, how much, how many, show me the numbers, calculate, which, what, list, top, most, compare (without visuals)
   - Examples: "what are total sales", "calculate revenue", "show me top customers", "how many orders", "compare order counts"
   - **INCLUDES ALL ANOMALY QUERIES:** Since anomalies are pre-detected, ALL anomaly questions go here
     - "which customer has most anomalies?"
     - "show me anomaly breakdown"
     - "what product has highest anomalies?"
     - "detect anomalies" (they're already detected, just analyze them)
     - "are there anomalies?" (analyze existing anomalies)



3. **forecast** - Predictions or forecasting future values
   - Keywords: predict, forecast, next, future, will be, expected, projection
   - Examples: "predict next quarter", "forecast sales", "what will revenue be"



**🚨 CRITICAL: NO "anomaly" INTENT**
- There is NO separate "anomaly" intent
- ALL anomaly-related queries (detect, analyze, show, list, count) → use **"analysis"** intent
- Reason: Anomalies are pre-detected at startup, so all queries about them are analytical



**Other Rules:**
- If query contains "chart", "graph", "visualization", "plot", "dashboard", "visualize", "visual" → **dashboard intent**
- If query says "for the same" or "that" referring to previous data AND asks for visuals → **dashboard intent**
- Analysis intent is for TEXT/NUMBERS only, never for visual outputs



**🎯 CRITICAL: STANDARDIZED ENTITY KEY NAMES**



When extracting entities, use these EXACT key names (DO NOT make up new names):



**Required Entity Keys:**
- `customer_id` - Customer identifier (can be single "1001" OR list ["1002", "1001"] for comparisons/plurals)
- `product_id` - Product identifier (can be single "RE630" OR list ["RE630", "OX140"] for comparisons/plurals)
- `year` - Year filter (e.g., 2024, 2025)
- `metric` - Metric being queried (e.g., "revenue", "sales", "orders", "anomalies", "order_count")
- `aggregation_type` - **USE THIS FOR:** least, most, min, max, highest, lowest, top, bottom, best, worst
  - Examples: "least" → "aggregation_type": "least"
  - Examples: "most" → "aggregation_type": "most"
  - Examples: "highest" → "aggregation_type": "highest"
  - Examples: "minimum" → "aggregation_type": "minimum"



**🚨 FORBIDDEN Entity Keys (DO NOT USE):**
- ❌ ranking
- ❌ metric_qualifier
- ❌ aggregation
- ❌ qualifier
- ❌ comparison_type
- ❌ metric_level



**Examples of Correct Entity Extraction:**



Query: "which customer generated least revenue?"
✅ CORRECT: {{"metric": "revenue", "aggregation_type": "least"}}
❌ WRONG: {{"metric": "revenue", "ranking": "least"}}

Query: "show me the top product by sales"
✅ CORRECT: {{"metric": "sales", "aggregation_type": "top"}}
❌ WRONG: {{"metric": "sales", "metric_qualifier": "top"}}

Query: "compare the count of orders given by both the customers"
Recently Mentioned Customers: ['1002', '1001']
✅ CORRECT: {{"customer_id": ["1002", "1001"], "metric": "order_count"}}
❌ WRONG: {{"metric": "order_count"}} (missing customer_id list)

Query: "generate dashboard for both customers"
Recently Mentioned Customers: ['1002', '1001']
✅ CORRECT: intent="dashboard", entities={{"customer_id": ["1002", "1001"]}}

Query: "compare these two products"
Recently Mentioned Products: ['RE630', 'OX140']
✅ CORRECT: {{"product_id": ["RE630", "OX140"]}}

Query: "detect anomalies"
✅ CORRECT: intent = "analysis", entities = {{"metric": "anomalies"}}
❌ WRONG: intent = "anomaly"

Query: "which customer has most anomalies?"
✅ CORRECT: intent = "analysis", entities = {{"metric": "anomalies", "aggregation_type": "most"}}



**Response Format (STRICTLY JSON - NO MARKDOWN, NO TRAILING COMMAS):**
{{
  "intent": "dashboard",
  "entities": {{
    "customer_id": ["1002", "1001"],
    "year": 2024,
    "aggregation_type": "least"
  }},
  "confidence": 0.95,
  "reasoning": "Brief explanation of why this intent was chosen"
}}



**IMPORTANT JSON RULES:**
- Return ONLY valid JSON (no markdown code blocks like ```
- No trailing commas before closing braces or brackets
- All string values must be quoted
- Boolean values: true or false (lowercase, no quotes)
- Numbers: no quotes unless representing IDs (like customer_id)
- customer_id and product_id can be: single string "1002" OR list ["1002", "1001"]
- **ALWAYS use "aggregation_type" key for comparisons (least/most/min/max/highest/lowest)**
- **NEVER use "anomaly" as intent - use "analysis" for all anomaly queries**
- **For plural pronouns or comparisons, ALWAYS extract as LIST**



Analyze the query and respond with ONLY the JSON object."""




            logger.info("🤖 Calling Gemini LLM for routing decision...")
            response = self.llm.invoke(prompt)
            content = response.content
            logger.info(f"📥 LLM Response received (length: {len(content)} chars)")
            
            # Parse JSON from response
            routing_decision = self._parse_llm_response(content)
            
            if routing_decision:
                # ✅ SAFETY: Force anomaly intent to analysis (just in case LLM doesn't follow instructions)
                if routing_decision.get('intent') == 'anomaly':
                    logger.warning("⚠️ LLM returned 'anomaly' intent - auto-correcting to 'analysis'")
                    routing_decision['intent'] = 'analysis'
                    if 'metric' not in routing_decision.get('entities', {}):
                        routing_decision.setdefault('entities', {})['metric'] = 'anomalies'
                
                # ✅ NEW: Post-process to ensure plural pronouns are resolved
                resolved_entities = routing_decision.get('entities', {})
                query_lower = query.lower()
                
                # Check for plural pronouns that might not be resolved by LLM
                plural_keywords = ['both', 'these', 'those', 'all of them', 'the two', 'compare them']
                has_plural_pronoun = any(keyword in query_lower for keyword in plural_keywords)
                has_comparison = any(keyword in query_lower for keyword in ['compare', 'vs', 'versus', 'between'])
                
                # If plural pronoun/comparison detected but customer_id is not a list, fix it
                if (has_plural_pronoun or has_comparison) and 'customer' in query_lower:
                    if resolved_entities.get('customer_id') and not isinstance(resolved_entities['customer_id'], list):
                        # LLM gave single ID, but we need a list
                        if recent_customers and len(recent_customers) >= 2:
                            resolved_entities['customer_id'] = recent_customers[:2]  # Top 2 recent
                            logger.info(f"   🔧 Corrected customer_id to list: {resolved_entities['customer_id']}")
                    elif not resolved_entities.get('customer_id') and recent_customers and len(recent_customers) >= 2:
                        # LLM missed it entirely
                        resolved_entities['customer_id'] = recent_customers[:2]
                        logger.info(f"   🔧 Added missing customer_id list: {resolved_entities['customer_id']}")
                
                # Same for products
                if (has_plural_pronoun or has_comparison) and 'product' in query_lower:
                    if resolved_entities.get('product_id') and not isinstance(resolved_entities['product_id'], list):
                        if recent_products and len(recent_products) >= 2:
                            resolved_entities['product_id'] = recent_products[:2]
                            logger.info(f"   🔧 Corrected product_id to list: {resolved_entities['product_id']}")
                    elif not resolved_entities.get('product_id') and recent_products and len(recent_products) >= 2:
                        resolved_entities['product_id'] = recent_products[:2]
                        logger.info(f"   🔧 Added missing product_id list: {resolved_entities['product_id']}")
                
                # Singular pronoun fallback (if LLM missed it)
                if context_entities and not resolved_entities.get('customer_id') and 'customer_id' in context_entities:
                    if any(pronoun in query_lower for pronoun in ['this customer', 'that customer', 'the customer', 'for it', 'for them']):
                        resolved_entities['customer_id'] = context_entities['customer_id']
                        logger.info(f"   🔗 Resolved 'this customer' → customer_id: {context_entities['customer_id']}")
                
                routing_decision['entities'] = resolved_entities
                
                logger.info(f"🎯 ROUTING DECISION:")
                logger.info(f"   └─ Intent: {routing_decision['intent']}")
                logger.info(f"   └─ Entities: {routing_decision.get('entities', {})}")
                logger.info(f"   └─ Confidence: {routing_decision.get('confidence', 'N/A')}")
                logger.info(f"   └─ Reasoning: {routing_decision.get('reasoning', 'N/A')}")
                return routing_decision
            else:
                logger.warning("Failed to parse LLM response, using fallback")
                return self._fallback_routing(query)
                
        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            return self._fallback_routing(query)
    
    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """
        ✅ FIXED: Parse JSON from LLM response with robust error handling
        """
        # Strategy 1: Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.debug("Strategy 1 failed: Not direct JSON")
        
        # Strategy 2: Extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```')
        if json_match:
            try:
                json_str = json_match.group(1)
                # Clean trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                result = json.loads(json_str)
                logger.info("✅ Strategy 2 SUCCESS: Found JSON in code block")
                return result
            except json.JSONDecodeError as e:
                logger.debug(f"Strategy 2 failed: {e}")
        
        # Strategy 3: Find raw JSON object with nested support
        json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', content, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group()
                # Clean trailing commas before closing braces/brackets
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                result = json.loads(json_str)
                logger.info("✅ Strategy 3 SUCCESS: Found raw JSON object")
                return result
            except json.JSONDecodeError as e:
                logger.debug(f"Strategy 3 failed: {e}")
        
        # Strategy 4: Extract key fields manually as fallback
        try:
            intent_match = re.search(r'"intent":\s*"([^"]+)"', content)
            confidence_match = re.search(r'"confidence":\s*([0-9.]+)', content)
            reasoning_match = re.search(r'"reasoning":\s*"([^"]*)"', content)
            
            if intent_match:
                result = {
                    "intent": intent_match.group(1),
                    "entities": {},
                    "confidence": float(confidence_match.group(1)) if confidence_match else 0.7,
                    "reasoning": reasoning_match.group(1) if reasoning_match else "Extracted via regex"
                }
                
                # Try to extract entities object
                entities_match = re.search(r'"entities":\s*(\{[^}]*\})', content)
                if entities_match:
                    try:
                        entities_str = entities_match.group(1)
                        entities_str = re.sub(r',\s*}', '}', entities_str)
                        result["entities"] = json.loads(entities_str)
                    except:
                        pass
                
                logger.info("✅ Strategy 4 SUCCESS: Extracted key fields via regex")
                return result
        except Exception as e:
            logger.debug(f"Strategy 4 failed: {e}")
        
        logger.error("❌ All JSON extraction strategies failed")
        logger.debug(f"Raw LLM content: {content[:500]}")
        return None
    
    def _fallback_routing(self, query: str) -> Dict[str, Any]:
        """
        ✅ UPDATED: Keyword-based routing fallback - all anomaly queries go to analysis
        """
        query_lower = query.lower()
        
        # Dashboard detection
        dashboard_keywords = ['chart', 'graph', 'visualization', 'visualize', 'plot', 
                             'dashboard', 'visual', 'show visually', 'create a chart',
                             'build a dashboard', 'make a graph']
        
        if any(keyword in query_lower for keyword in dashboard_keywords):
            logger.info("Fallback: Detected DASHBOARD intent")
            return {
                "intent": "dashboard",
                "entities": {},
                "confidence": 0.6,
                "reasoning": "Keyword-based fallback routing (dashboard keywords detected)"
            }
        
        # Forecast detection
        forecast_keywords = ['forecast', 'predict', 'future', 'next', 'will be', 'projection']
        
        if any(keyword in query_lower for keyword in forecast_keywords):
            logger.info("Fallback: Detected FORECAST intent")
            return {
                "intent": "forecast",
                "entities": {},
                "confidence": 0.6,
                "reasoning": "Keyword-based fallback routing (forecast)"
            }
        
        # ✅ UPDATED: ALL anomaly queries go to ANALYSIS (no separate anomaly intent)
        has_anomaly_word = 'anomal' in query_lower or 'outlier' in query_lower
        
        if has_anomaly_word:
            logger.info("Fallback: Detected ANALYSIS intent (anomaly query)")
            entities = {'metric': 'anomalies'}
            
            # Try to extract aggregation type
            if any(word in query_lower for word in ['most', 'top', 'highest']):
                entities['aggregation_type'] = 'most'
            elif any(word in query_lower for word in ['least', 'bottom', 'lowest']):
                entities['aggregation_type'] = 'least'
            
            return {
                "intent": "analysis",
                "entities": entities,
                "confidence": 0.7,
                "reasoning": "Keyword-based fallback routing (anomaly analysis - anomalies pre-detected)"
            }
        
        # Default to analysis
        logger.info("Fallback: Detected ANALYSIS intent (default)")
        
        # Extract year if present
        entities = {}
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            entities['year'] = int(year_match.group(1))
        
        # ✅ NEW: Try to resolve pronouns from context even in fallback (including plural)
        try:
            from data_connector import mcp_store
            dialogue_state = mcp_store.get_current_dialogue_state()
            context_entities = dialogue_state.get('entities', {})
            context_stack = dialogue_state.get('context_stack', [])
            
            # Check for plural pronouns
            plural_keywords = ['both', 'these', 'those', 'all of them', 'the two', 'compare', 'them']
            has_plural = any(keyword in query_lower for keyword in plural_keywords)
            
            if has_plural and 'customer' in query_lower:
                # Extract recent customers for plural resolution
                recent_customers = []
                for ctx in context_stack[-5:]:
                    ctx_entities = ctx.get('entities', {})
                    if 'customer_id' in ctx_entities:
                        cust_id = str(ctx_entities['customer_id'])
                        if cust_id and cust_id not in recent_customers:
                            recent_customers.append(cust_id)
                
                if len(recent_customers) >= 2:
                    entities['customer_id'] = recent_customers[:2]
                    logger.info(f"   🔗 Fallback resolved plural pronoun → customer_id: {entities['customer_id']}")
            
            elif context_entities and any(pronoun in query_lower for pronoun in ['this', 'that', 'it']):
                # Singular pronoun resolution
                if 'customer_id' in context_entities and any(word in query_lower for word in ['customer', 'it', 'them']):
                    entities['customer_id'] = context_entities['customer_id']
                    logger.info(f"   🔗 Fallback resolved pronoun → customer_id: {context_entities['customer_id']}")
        except:
            pass
        
        return {
            "intent": "analysis",
            "entities": entities,
            "confidence": 0.6,
            "reasoning": "Keyword-based fallback routing (default to analysis)"
        }
