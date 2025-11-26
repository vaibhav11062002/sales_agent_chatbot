from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from data_connector import mcp_store
from layer2_routing_agent import IntelligentRouter
from layer2_context_manager import ContextManager
import logging

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State passed between agents"""
    query: str
    intent: str
    entities: dict
    routing_reasoning: str
    from_cache: bool
    cached_response: dict
    analysis_result: dict
    forecast_result: dict
    anomaly_result: dict
    dashboard_result: dict
    explanation: str
    final_response: str

class AgentOrchestrator:
    """Layer 2: Orchestrates agent execution using LLM-based intelligent routing"""
    
    def __init__(self):
        # Import and initialize all specialized agents
        from layer3_agents.analysis_agent import AnalysisAgent
        from layer3_agents.forecasting_agent import ForecastingAgent
        from layer3_agents.anomaly_detection_agent import AnomalyDetectionAgent
        from layer3_agents.explanation_agent import ExplanationAgent
        from layer3_agents.dashboard_agent import DashboardAgent
        
        self.analysis_agent = AnalysisAgent()
        self.forecast_agent = ForecastingAgent()
        self.anomaly_agent = AnomalyDetectionAgent()
        self.explanation_agent = ExplanationAgent()
        self.dashboard_agent = DashboardAgent()
        
        # Initialize intelligent router and context manager
        self.router = IntelligentRouter()
        self.context_manager = ContextManager()
        
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow with context-first approach"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("check_context", self._check_context_first)
        workflow.add_node("route_query", self._route_query_with_llm)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("forecast", self._forecast_node)
        workflow.add_node("detect_anomalies", self._detect_anomalies_node)
        workflow.add_node("create_dashboard", self._create_dashboard_node)
        workflow.add_node("aggregate_results", self._aggregate_results)
        
        # Start with context check
        workflow.add_edge(START, "check_context")
        
        # If cached, skip to aggregation; otherwise route
        workflow.add_conditional_edges(
            "check_context",
            self._should_use_cache,
            {
                "use_cache": "aggregate_results",
                "compute": "route_query"
            }
        )
        
        # Route to specific agents based on intent
        workflow.add_conditional_edges(
            "route_query",
            self._route_to_agents,
            {
                "analysis": "analyze",
                "forecast": "forecast",
                "anomaly": "detect_anomalies",
                "dashboard": "create_dashboard"
            }
        )
        
        # All paths converge to aggregation
        workflow.add_edge("analyze", "aggregate_results")
        workflow.add_edge("forecast", "aggregate_results")
        workflow.add_edge("detect_anomalies", "aggregate_results")
        workflow.add_edge("create_dashboard", "aggregate_results")
        workflow.add_edge("aggregate_results", END)
        
        return workflow.compile()
    
    def _check_context_first(self, state: AgentState) -> AgentState:
        """FIRST: Check MCP context before routing to agents"""
        query = state["query"]
        
        logger.info(f"{'='*60}")
        logger.info(f"🚀 STEP 1: CHECK CONTEXT FIRST")
        logger.info(f"{'='*60}")
        logger.info(f"📝 Query: '{query}'")
        
        # Quick routing to get entities
        conversation_context = mcp_store.conversation_history[-5:] if hasattr(mcp_store, 'conversation_history') else []
        routing_decision = self.router.route_query(query, conversation_context)
        
        state["entities"] = routing_decision.get("entities", {})
        state["intent"] = routing_decision["intent"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 STEP 2: CHECKING MCP CACHE")
        logger.info(f"{'='*60}")
        
        # Skip cache check for dashboard queries (always fresh)
        if state["intent"] == "dashboard":
            logger.info(f"📊 Dashboard query detected - skipping cache (always generate fresh)")
            state["from_cache"] = False
            state["cached_response"] = {}
            return state
        
        # Check if answer exists in context
        cached_answer = self.context_manager.check_context_for_answer(query, state["entities"])
        
        if cached_answer:
            logger.info(f"\n{'='*60}")
            logger.info(f"✅ CACHE HIT - Skipping agent execution")
            logger.info(f"{'='*60}")
            logger.info(f"📦 Cached response length: {len(cached_answer.get('response', ''))} chars")
            state["from_cache"] = True
            state["cached_response"] = cached_answer
        else:
            logger.info(f"\n{'='*60}")
            logger.info(f"❌ CACHE MISS - Proceeding with agent execution")
            logger.info(f"{'='*60}")
            state["from_cache"] = False
            state["cached_response"] = {}
        
        return state
    
    def _should_use_cache(self, state: AgentState) -> str:
        """Decide whether to use cache or compute"""
        return "use_cache" if state.get("from_cache") else "compute"
    
    def _route_query_with_llm(self, state: AgentState) -> AgentState:
        """Use LLM to intelligently classify intent"""
        logger.info(f"LLM Routing Decision:")
        logger.info(f"  Intent: {state['intent']}")
        logger.info(f"  Entities: {state['entities']}")
        
        return state
    
    def _route_to_agents(self, state: AgentState) -> str:
        """Return the intent to determine which agent node to execute"""
        return state["intent"]
    
    # ============== AGENT NODE FUNCTIONS ==============
    
    def _analyze_node(self, state: AgentState) -> AgentState:
        """Execute Analysis Agent"""
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 EXECUTING: AnalysisAgent")
        logger.info(f"{'='*60}")
        
        entities = state.get("entities", {})
        logger.info(f"📋 Entities passed to agent: {entities}")
        
        # Determine analysis type
        analysis_type = "aggregation" if "total" in state["query"].lower() else "summary"
        logger.info(f"📈 Performing {analysis_type} analysis")
        
        # Execute analysis
        result = self.analysis_agent.execute(state["query"], analysis_type, entities=entities)
        
        state["analysis_result"] = result
        logger.info(f"✅ Analysis completed with status: {result.get('status')}")
        logger.info(f"{'='*60}\n")
        return state
    
    def _forecast_node(self, state: AgentState) -> AgentState:
        """Execute Forecasting Agent"""
        logger.info(f"\n{'='*60}")
        logger.info(f"📈 EXECUTING: ForecastingAgent")
        logger.info(f"{'='*60}")
        
        periods = state.get("entities", {}).get("periods", 3)
        logger.info(f"📋 Forecast periods: {periods}")
        
        result = self.forecast_agent.execute(state["query"], forecast_periods=periods)
        state["forecast_result"] = result
        
        logger.info(f"✅ Forecasting completed: {result.get('status')}")
        logger.info(f"{'='*60}\n")
        return state
    
    def _detect_anomalies_node(self, state: AgentState) -> AgentState:
        """Execute Anomaly Detection Agent"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 EXECUTING: AnomalyDetectionAgent")
        logger.info(f"{'='*60}")
        
        contamination = state.get("entities", {}).get("contamination", 0.05)
        logger.info(f"📋 Contamination threshold: {contamination}")
        
        result = self.anomaly_agent.execute(state["query"], contamination=contamination)
        state["anomaly_result"] = result
        
        logger.info(f"✅ Anomaly detection completed: {result.get('status')}")
        logger.info(f"{'='*60}\n")
        return state
    
    def _create_dashboard_node(self, state: AgentState) -> AgentState:
        """Execute Dashboard Agent with analysis agent reference"""
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 EXECUTING: DashboardAgent")
        logger.info(f"{'='*60}")
        
        entities = state.get("entities", {})
        logger.info(f"📋 Entities passed to agent: {entities}")
        
        # ✅ Pass analysis_agent reference to dashboard for insights
        result = self.dashboard_agent.execute(
            state["query"], 
            entities=entities,
            analysis_agent=self.analysis_agent  # ✅ NEW: Pass analysis agent
        )
        
        state["dashboard_result"] = result
        
        logger.info(f"✅ Dashboard created: {result.get('status')}")
        if result.get('status') == 'success':
            logger.info(f"   └─ Charts generated: {result.get('charts_generated', 0)}")
            logger.info(f"   └─ Output path: {result.get('output_path', 'N/A')}")
            logger.info(f"   └─ Dashboard plan: {result.get('dashboard_plan', {}).get('title', 'N/A')}")
        logger.info(f"{'='*60}\n")
        return state
    
    # ============== RESULT AGGREGATION ==============
    
    def _aggregate_results(self, state: AgentState) -> AgentState:
        """Aggregate results (from cache or agents) with explanation"""
        final_response = []
        
        # ✅ FIX: If from cache, structure it as analysis_result for explanation
        if state.get("from_cache"):
            cached = state.get("cached_response", {})
            if 'response' in cached:
                # Structure cached response as analysis result
                state["analysis_result"] = {
                    "status": "success",
                    "analysis_type": "cached_analysis",
                    "results": {"llm_raw": cached['response']}
                }
                logger.info("🔄 Cached response structured for explanation generation")
                # Don't return early - continue to explanation generation
        
        # Process Dashboard Results FIRST (if present)
        if state.get("dashboard_result"):
            dashboard = state["dashboard_result"]
            if dashboard.get("status") == "success":
                final_response.append(f"📊 **Dashboard Created Successfully!**\n\n")
                final_response.append(f"  • **Title:** {dashboard.get('dashboard_plan', {}).get('title', 'N/A')}\n")
                final_response.append(f"  • **Description:** {dashboard.get('dashboard_plan', {}).get('description', 'N/A')}\n")
                final_response.append(f"  • **Charts generated:** {dashboard.get('charts_generated', 0)}\n")
                final_response.append(f"  • **File saved:** `{dashboard.get('output_path', 'N/A')}`\n\n")
                final_response.append(f"💡 **Next Steps:**\n")
                final_response.append(f"  1. Open the dashboard file in your browser\n")
                final_response.append(f"  2. Explore interactive visualizations\n")
                final_response.append(f"  3. Use browser's print function to save as PDF\n")
                
                state["final_response"] = "".join(final_response)
                return state
            else:
                final_response.append(f"❌ **Dashboard Creation Failed:**\n")
                final_response.append(f"  • Error: {dashboard.get('message', 'Unknown error')}\n\n")
        
        # Process Analysis Results (including cached ones)
        if state.get("analysis_result"):
            results = state["analysis_result"].get("results", {})
            
            # Handle LLM raw responses
            if isinstance(results, dict) and 'llm_raw' in results:
                llm_raw = results['llm_raw']
                clean_response = self._extract_clean_llm_response(llm_raw)
                final_response.append(clean_response)
                final_response.append("\n\n")
            elif isinstance(results, dict):
                # Handle structured results
                if "total_sales" in results:
                    final_response.append(f"💰 **Total Sales:** ${results['total_sales']:,.2f}\n")
                if "total_orders" in results:
                    final_response.append(f"📦 **Total Orders:** {results['total_orders']:,}\n")
                if "avg_order_value" in results:
                    final_response.append(f"📊 **Average Order Value:** ${results['avg_order_value']:,.2f}\n")
                if "unique_customers" in results:
                    final_response.append(f"👥 **Unique Customers:** {results['unique_customers']:,}\n")
                if "unique_products" in results:
                    final_response.append(f"🏷️ **Unique Products:** {results['unique_products']:,}\n")
        
        # Process Forecast Results
        if state.get("forecast_result"):
            forecasts_result = state["forecast_result"]
            forecasts = forecasts_result.get("forecasts", [])
            if forecasts:
                final_response.append(f"\n🔮 **Forecast for next {len(forecasts)} periods:**\n")
                for f in forecasts:
                    final_response.append(
                        f"  • **{f.get('date', 'N/A')}:** ${f.get('forecasted_sales', 0):,.2f} "
                        f"(confidence: {f.get('confidence', 'N/A')})\n"
                    )
                
                final_response.append(f"\n**Model Used:** {forecasts_result.get('model_used', 'N/A')}\n")
                if 'model_params' in forecasts_result:
                    final_response.append(f"**Model Params:** {forecasts_result['model_params']}\n")
                if 'accuracy' in forecasts_result and isinstance(forecasts_result["accuracy"], dict):
                    acc = forecasts_result["accuracy"]
                    acc_str = ", ".join(f"{k}={v:.2f}" for k, v in acc.items())
                    final_response.append(f"**Accuracy:** {acc_str}\n")
                if 'message' in forecasts_result:
                    final_response.append(f"**Internal Message:** {forecasts_result['message']}\n")
                
                trend = forecasts_result.get("historical_trend")
                if trend:
                    trend_direction = "📈 upward" if trend > 0 else "📉 downward"
                    final_response.append(
                        f"\n**Historical Trend:** {trend_direction} trend "
                        f"(${abs(trend):,.2f}/month)\n"
                    )
        
        # Process Anomaly Results
        if state.get("anomaly_result"):
            anomalies = state["anomaly_result"].get("anomalies", {})
            if anomalies:
                final_response.append(f"\n⚠️ **Anomalies Detected:**\n")
                final_response.append(f"  • Total Anomalies: {anomalies.get('total_anomalies', 0):,}\n")
                final_response.append(f"  • Percentage: {anomalies.get('percentage', 0):.2f}%\n")
                final_response.append(f"  • Anomalous Sales Total: ${anomalies.get('anomaly_sales_total', 0):,.2f}\n")
                
                top_anomalies = anomalies.get('top_anomalies', [])
                if top_anomalies:
                    final_response.append(f"\n  **Top Anomalous Orders:**\n")
                    for anom in top_anomalies[:3]:
                        final_response.append(
                            f"    • Order **{anom.get('SalesDocument', 'N/A')}**: "
                            f"${anom.get('NetAmount', 0):,.2f}\n"
                        )
        
        # ✅ Generate AI Explanation for ALL queries (including cached ones)
        if state.get("intent") != "dashboard":
            try:
                # Add cache indicator to explanation context
                explanation_context = {
                    "analysis": state.get("analysis_result"),
                    "forecast": state.get("forecast_result"),
                    "anomaly": state.get("anomaly_result"),
                    "from_cache": state.get("from_cache", False)  # ✅ Pass cache status
                }
                
                explanation_result = self.explanation_agent.execute(
                    state["query"],
                    explanation_context
                )
                
                if explanation_result.get("status") == "success":
                    explanation = explanation_result.get("explanation", "")
                    if explanation:
                        # ✅ Add cache indicator if from cache
                        cache_prefix = "🔄 *(Retrieved from cache)* " if state.get("from_cache") else ""
                        final_response.append(f"\n---\n\n💡 **AI Insights:**\n\n{cache_prefix}{explanation}")
                        state["explanation"] = explanation
                        logger.info("AI explanation generated successfully" + (" (for cached result)" if state.get("from_cache") else ""))
            
            except Exception as e:
                logger.error(f"Error generating explanation: {str(e)}")
        
        # Finalize response
        state["final_response"] = "".join(final_response) if final_response else "No results available"
        return state

    
    def _extract_clean_llm_response(self, llm_raw: str) -> str:
        """Extract clean answer from LLM raw output"""
        import re
        
        # Remove code blocks
        cleaned = re.sub(r'``````', '', llm_raw, flags=re.DOTALL)
        
        # Extract bullet points and formatted text
        lines = cleaned.split('\n')
        answer_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('*') or stripped.startswith('-') or '**' in stripped:
                answer_lines.append(stripped)
            elif stripped and len(stripped) > 20 and not stripped.startswith('#'):
                answer_lines.append(stripped)
        
        if answer_lines:
            return '\n'.join(answer_lines)
        
        return cleaned.strip()[:1000] if cleaned.strip() else llm_raw[:500]
    
    def process_query(self, query: str) -> dict:
        """Main entry point for query processing with dialogue state tracking"""
        logger.info(f"{'='*60}")
        logger.info(f"🚀 NEW QUERY: '{query}'")
        logger.info(f"{'='*60}")
        
        # Ensure data loaded
        try:
            if mcp_store.sales_df is None:
                logger.info("Data not loaded, loading now...")
                mcp_store.load_sales_data()
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            return {
                "query": query,
                "intent": "error",
                "response": f"❌ Error loading data: {str(e)}",
                "from_cache": False
            }
        
        # Log current dialogue state BEFORE processing
        dialogue_state_before = mcp_store.get_current_dialogue_state()
        logger.info(f"💬 Dialogue state BEFORE query:")
        logger.info(f"   └─ Active entities: {len(dialogue_state_before.get('entities', {}))}")
        logger.info(f"   └─ Entity stack size: {len(dialogue_state_before.get('entity_stack', []))}")
        
        # Initialize state
        initial_state = {
            "query": query,
            "intent": "",
            "entities": {},
            "routing_reasoning": "",
            "from_cache": False,
            "cached_response": {},
            "analysis_result": {},
            "forecast_result": {},
            "anomaly_result": {},
            "dashboard_result": {},
            "explanation": "",
            "final_response": ""
        }
        
        # Execute workflow
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 EXECUTING LANGGRAPH WORKFLOW")
            logger.info(f"{'='*60}")
            
            result = self.workflow.invoke(initial_state)
            
            # Log dialogue state AFTER processing
            dialogue_state_after = mcp_store.get_current_dialogue_state()
            logger.info(f"\n{'='*60}")
            logger.info(f"💬 Dialogue state AFTER query:")
            logger.info(f"{'='*60}")
            logger.info(f"   └─ Active entities: {len(dialogue_state_after.get('entities', {}))}")
            logger.info(f"   └─ Entity stack size: {len(dialogue_state_after.get('entity_stack', []))}")
            
            # Log entity stack with details
            entity_stack = dialogue_state_after.get('entity_stack', [])
            if entity_stack:
                logger.info(f"   └─ Entity stack ({len(entity_stack)} items):")
                for i, entity in enumerate(entity_stack[:5], 1):
                    logger.info(f"      {i}. {entity['type']}: {entity['value']}")
            
            # Log cache performance
            cache_status = "✅ CACHE HIT" if result.get("from_cache", False) else "❌ CACHE MISS (computed)"
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 QUERY PROCESSING SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"   └─ Query: '{query}'")
            logger.info(f"   └─ Intent: {result['intent']}")
            logger.info(f"   └─ Cache: {cache_status}")
            logger.info(f"   └─ Response length: {len(result['final_response'])} chars")
            logger.info(f"{'='*60}\n")
            
            # Build response
            response = {
                "query": query,
                "intent": result["intent"],
                "response": result["final_response"],
                "from_cache": result.get("from_cache", False),
                "forecast_result": result.get("forecast_result", {}),
                "dashboard_result": result.get("dashboard_result", {}),
                "dialogue_state": {
                    "entities": dialogue_state_after.get('entities', {}),
                    "entity_stack_size": len(entity_stack)
                }
            }
            
            return response
        
        except Exception as e:
            logger.error(f"\n{'='*60}")
            logger.error(f"❌ ERROR IN WORKFLOW EXECUTION")
            logger.error(f"{'='*60}")
            logger.error(f"Query: '{query}'")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Full stack trace:", exc_info=True)
            logger.error(f"{'='*60}\n")
            
            return {
                "query": query,
                "intent": "error",
                "response": f"❌ Error processing query: {str(e)}",
                "from_cache": False,
                "error_type": type(e).__name__,
                "dashboard_result": {},
                "dialogue_state": None
            }
