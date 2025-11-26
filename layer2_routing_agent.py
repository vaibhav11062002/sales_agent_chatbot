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
        """Route query using LLM-based classification"""
        logger.info(f"🔀 IntelligentRouter: Starting routing for query: '{query}'")
        
        if not self.llm:
            logger.warning("LLM not available, using fallback routing")
            return self._fallback_routing(query)
        
        try:
            # Build conversation context
            context_str = ""
            if conversation_history:
                recent_turns = conversation_history[-3:]
                context_str = "\n".join([
                    f"{turn.get('role', 'unknown')}: {turn.get('content', '')[:100]}"
                    for turn in recent_turns
                ])
            
            # Enhanced routing prompt with better dashboard detection
            prompt = f"""You are an intent classifier for a sales analytics AI system. Analyze the user's query and classify the intent.

**User Query:** "{query}"

**Recent Conversation Context:**
{context_str if context_str else "No previous context"}

**Available Intents:**

1. **dashboard** - User wants visualizations, charts, graphs, or dashboards
   - Keywords: chart, graph, visualization, plot, dashboard, visualize, show me visually, create a chart, build a dashboard
   - Examples: "create a chart", "show me a graph", "visualize the data", "build a dashboard", "can you chart this", "plot the trends"

2. **analysis** - Questions about metrics, calculations, aggregations, or data summaries (TEXT-BASED only, NO visuals)
   - Keywords: total, sum, average, count, how much, how many, show me the numbers, calculate
   - Examples: "what are total sales", "calculate revenue", "show me top customers", "how many orders"

3. **forecast** - Predictions or forecasting future values
   - Keywords: predict, forecast, next, future, will be, expected, projection
   - Examples: "predict next quarter", "forecast sales", "what will revenue be"

4. **anomaly** - Detecting unusual patterns or outliers
   - Keywords: anomaly, unusual, outlier, detect, abnormal, suspicious
   - Examples: "detect anomalies", "find unusual orders", "show outliers"

**CRITICAL RULES:**
- If query contains "chart", "graph", "visualization", "plot", "dashboard", "visualize", "visual" → **ALWAYS dashboard intent**
- If query says "for the same" or "that" referring to previous data AND asks for visuals → **dashboard intent**
- Analysis intent is for TEXT/NUMBERS only, never for visual outputs
- When in doubt between analysis and dashboard, choose dashboard if any visual keywords are present

**Response Format (JSON only, no markdown):**
{{
  "intent": "dashboard|analysis|forecast|anomaly",
  "entities": {{
    "year": 2024,
    "customer_id": "C12345",
    "product_id": "P567",
    "periods": 3,
    ...
  }},
  "confidence": 0.95,
  "reasoning": "Brief explanation of why this intent was chosen"
}}

Analyze the query and respond with JSON only."""

            logger.info("🤖 Calling Gemini LLM for routing decision...")
            response = self.llm.invoke(prompt)
            content = response.content
            logger.info(f"📥 LLM Response received (length: {len(content)} chars)")
            
            # Parse JSON from response
            routing_decision = self._parse_llm_response(content)
            
            if routing_decision:
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
        """Parse JSON from LLM response with multiple strategies"""
        
        # Strategy 1: Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.debug("Strategy 1 failed: Not direct JSON")
        
        # Strategy 2: Extract JSON from code blocks
        json_match = re.search(r'``````', content, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                logger.info("✅ Strategy 2 SUCCESS: Found JSON in code block")
                return result
            except json.JSONDecodeError:
                logger.debug("Strategy 2 failed: Invalid JSON in code block")
        
        # Strategy 3: Find raw JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                logger.info("✅ Strategy 3 SUCCESS: Found raw JSON object")
                return result
            except json.JSONDecodeError:
                logger.debug("Strategy 3 failed: Invalid JSON object")
        
        # Strategy 4: Extract key fields manually
        try:
            intent_match = re.search(r'"intent":\s*"([^"]+)"', content)
            entities_match = re.search(r'"entities":\s*(\{[^}]+\})', content)
            
            if intent_match:
                result = {
                    "intent": intent_match.group(1),
                    "entities": {},
                    "confidence": 0.7,
                    "reasoning": "Extracted via regex"
                }
                
                if entities_match:
                    try:
                        result["entities"] = json.loads(entities_match.group(1))
                    except:
                        pass
                
                logger.info("✅ Strategy 4 SUCCESS: Extracted key fields")
                return result
        except Exception as e:
            logger.debug(f"Strategy 4 failed: {e}")
        
        logger.error("❌ All JSON extraction strategies failed")
        return None
    
    def _fallback_routing(self, query: str) -> Dict[str, Any]:
        """Simple keyword-based routing fallback"""
        query_lower = query.lower()
        
        # ✅ ENHANCED: Better dashboard detection
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
        
        # Existing fallback logic
        forecast_keywords = ['forecast', 'predict', 'future', 'next', 'will be', 'projection']
        anomaly_keywords = ['anomaly', 'anomalies', 'outlier', 'unusual', 'detect', 'abnormal']
        
        if any(keyword in query_lower for keyword in forecast_keywords):
            intent = "forecast"
        elif any(keyword in query_lower for keyword in anomaly_keywords):
            intent = "anomaly"
        else:
            intent = "analysis"
        
        logger.info(f"Fallback: Detected {intent.upper()} intent")
        
        # Extract year if present
        entities = {}
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            entities['year'] = int(year_match.group(1))
        
        return {
            "intent": intent,
            "entities": entities,
            "confidence": 0.6,
            "reasoning": "Keyword-based fallback routing"
        }
