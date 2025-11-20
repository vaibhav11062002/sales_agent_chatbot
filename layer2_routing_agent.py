"""
LLM-based intelligent routing agent that uses Gemini to decide which agent to call
"""
import logging
import json
import re
import google.generativeai as genai
from config import GEMINI_API_KEY
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

class IntelligentRouter:
    """Uses LLM to intelligently route queries to appropriate agents"""
    
    def __init__(self):
        self.name = "IntelligentRouter"
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Define available agents as tools
        self.available_agents = {
            "analysis": {
                "name": "AnalysisAgent",
                "description": """Analyzes sales data and calculates metrics.
                
                Use this agent when the user asks for:
                - Total sales, revenue, or monetary amounts
                - Number of orders or transactions
                - Average order values
                - Customer counts or statistics
                - Product counts or statistics
                - Sales summaries or overviews
                - Comparisons between time periods
                - Growth rates or percentage changes
                - Trends over time
                - Year-over-year or period-over-period analysis
                
                Examples:
                - "What are total sales in 2024?"
                - "How many orders did we have?"
                - "Show me sales summary"
                - "Compare 2024 to 2025"
                - "What's the growth rate?"
                """,
                "parameters": ["year", "month", "period", "comparison"]
            },
            "forecast": {
                "name": "ForecastingAgent",
                "description": """Predicts future sales and creates forecasts.
                
                Use this agent when the user asks for:
                - Future predictions or projections
                - Next quarter/month/year forecasts
                - Expected sales
                - Upcoming trends
                - What will happen in the future
                - Predictive analysis
                
                Examples:
                - "Forecast next quarter sales"
                - "What will sales be next month?"
                - "Predict future revenue"
                - "Project next year's performance"
                """,
                "parameters": ["periods", "forecast_horizon"]
            },
            "anomaly": {
                "name": "AnomalyDetectionAgent",
                "description": """Detects unusual patterns and outliers in sales data.
                
                Use this agent when the user asks for:
                - Anomalies or outliers
                - Unusual transactions or orders
                - Strange patterns
                - Suspicious activity
                - Data quality issues
                - Extreme values
                - Abnormal behavior
                
                Examples:
                - "Detect any anomalies"
                - "Are there any unusual orders?"
                - "Find outliers in the data"
                - "Show me strange patterns"
                """,
                "parameters": ["threshold", "contamination"]
            }
        }
    
    def route_query(self, query: str, conversation_context: list = None) -> Dict[str, Any]:
        """
        Use LLM to determine which agent should handle the query
        
        Args:
            query: User's question
            conversation_context: Recent conversation history for context
            
        Returns:
            {
                "intent": "analysis" | "forecast" | "anomaly",
                "entities": {"year": 2024, ...},
                "reasoning": "Why this agent was chosen",
                "confidence": 0.95
            }
        """
        logger.info(f"{self.name}: Routing query: {query}")
        
        try:
            # Build context-aware prompt
            prompt = self._build_routing_prompt(query, conversation_context)
            
            # Call Gemini for reasoning
            response = self.model.generate_content(prompt)
            result_text = response.text
            
            logger.info(f"Raw LLM response: {result_text[:200]}...")
            
            # Parse JSON response with multiple strategies
            routing_decision = self._extract_json(result_text)
            
            if routing_decision is None:
                logger.warning("Could not parse LLM response, using fallback")
                return self._fallback_routing(query)
            
            # Validate intent
            if routing_decision.get("intent") not in self.available_agents:
                logger.warning(f"Invalid intent: {routing_decision.get('intent')}, defaulting to analysis")
                routing_decision["intent"] = "analysis"
            
            logger.info(f"Routing decision: {routing_decision['intent']} (confidence: {routing_decision.get('confidence', 'N/A')})")
            logger.info(f"Reasoning: {routing_decision.get('reasoning', 'N/A')}")
            
            return routing_decision
        
        except Exception as e:
            logger.error(f"Error in LLM routing: {str(e)}", exc_info=True)
            return self._fallback_routing(query)
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response with multiple strategies"""
        
        # Strategy 1: Try to find JSON in markdown code blocks
        code_block_pattern = r'``````'
        match = re.search(code_block_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Try to find raw JSON object
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
        
        # Strategy 3: Try parsing entire response as JSON
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Try cleaning and parsing
        try:
            cleaned = text.strip()
            # Remove markdown code blocks (backticks)
            cleaned = re.sub(r'`{3}(?:json)?', '', cleaned)
            cleaned = cleaned.strip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        logger.error(f"All JSON extraction strategies failed. Raw text: {text}")
        return None
    
    def _build_routing_prompt(self, query: str, conversation_context: list = None) -> str:
        """Build the prompt for LLM routing"""
        
        # Format agent descriptions
        agents_desc = "\n\n".join([
            f"**Agent: {agent_info['name']}**\n"
            f"Intent Key: '{key}'\n"
            f"{agent_info['description']}"
            for key, agent_info in self.available_agents.items()
        ])
        
        # Format conversation context
        context_str = ""
        if conversation_context and len(conversation_context) > 0:
            context_str = "\n\nRecent Conversation:\n" + "\n".join([
                f"- {turn.get('role', 'unknown')}: {str(turn.get('content', ''))[:100]}..."
                for turn in conversation_context[-3:]
            ])
        
        prompt = f"""You are an intelligent query router for a sales analytics system. Your job is to analyze user queries and determine which specialized agent should handle them.

Available Agents:
{agents_desc}

User Query: "{query}"
{context_str}

Your task:
1. Analyze the user's intent
2. Determine which agent is most appropriate
3. Extract any entities (years, months, comparison indicators, etc.)
4. Provide reasoning for your decision

IMPORTANT: Respond with ONLY a valid JSON object. Do not include any markdown, explanations, or extra text.

Required JSON format:
{{
    "intent": "analysis",
    "entities": {{"year": 2024, "comparison": true, "years": }},
    "reasoning": "Brief explanation of why this agent was chosen",
    "confidence": 0.95
}}

Rules:
- intent must be exactly one of: "analysis", "forecast", or "anomaly"
- If query mentions future/predictions/forecast → use "forecast"
- If query asks for calculations/totals/summaries/comparisons → use "analysis"  
- If query mentions anomalies/outliers/unusual → use "anomaly"
- Extract ALL years mentioned in the query
- If comparing periods, set "comparison": true and list years in "years" array
- confidence should be between 0.0 and 1.0

Respond ONLY with the JSON object, nothing else:"""
        
        return prompt
    
    def _fallback_routing(self, query: str) -> Dict[str, Any]:
        """Fallback to simple keyword matching if LLM fails"""
        logger.info("Using fallback keyword-based routing")
        
        query_lower = query.lower()
        
        # Simple keyword matching
        if any(word in query_lower for word in ["forecast", "predict", "future", "next"]):
            intent = "forecast"
        elif any(word in query_lower for word in ["anomaly", "unusual", "outlier", "strange", "detect"]):
            intent = "anomaly"
        else:
            intent = "analysis"
        
        # Extract year
        entities = {}
        year_matches = re.findall(r'20\d{2}', query)
        if year_matches:
            if len(year_matches) == 1:
                entities['year'] = int(year_matches)
            else:
                entities['years'] = [int(y) for y in year_matches]
                entities['comparison'] = True
        
        return {
            "intent": intent,
            "entities": entities,
            "reasoning": "Fallback keyword matching used due to LLM parsing error",
            "confidence": 0.5
        }
