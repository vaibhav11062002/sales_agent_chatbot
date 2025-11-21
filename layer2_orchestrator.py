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
        
        self.analysis_agent = AnalysisAgent()
        self.forecast_agent = ForecastingAgent()
        self.anomaly_agent = AnomalyDetectionAgent()
        self.explanation_agent = ExplanationAgent()
        
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
                "anomaly": "detect_anomalies"
            }
        )
        
        # All paths converge to aggregation
        workflow.add_edge("analyze", "aggregate_results")
        workflow.add_edge("forecast", "aggregate_results")
        workflow.add_edge("detect_anomalies", "aggregate_results")
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
        """Use LLM to intelligently classify intent (already done in check_context)"""
        # Entities already extracted in check_context_first
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
        
        # Handle comparison queries
        if entities.get('comparison') and 'years' in entities:
            logger.info(f"🔀 Detected comparison query for years: {entities['years']}")
            result = self.analysis_agent.execute_comparison(state["query"], entities['years'])
        else:
            analysis_type = "aggregation" if "total" in state["query"].lower() else "summary"
            logger.info(f"📈 Performing {analysis_type} analysis")
            result = self.analysis_agent.execute(state["query"], analysis_type, entities=entities)
        
        state["analysis_result"] = result
        logger.info(f"✅ Analysis completed with status: {result.get('status')}")
        logger.info(f"{'='*60}\n")
        return state
    
    def _forecast_node(self, state: AgentState) -> AgentState:
        """Execute Forecasting Agent"""
        periods = state.get("entities", {}).get("periods", 3)
        result = self.forecast_agent.execute(state["query"], forecast_periods=periods)
        state["forecast_result"] = result
        logger.info(f"Forecasting completed: {result.get('status')}")
        return state
    
    def _detect_anomalies_node(self, state: AgentState) -> AgentState:
        """Execute Anomaly Detection Agent"""
        contamination = state.get("entities", {}).get("contamination", 0.05)
        result = self.anomaly_agent.execute(state["query"], contamination=contamination)
        state["anomaly_result"] = result
        logger.info(f"Anomaly detection completed: {result.get('status')}")
        return state
    
    # ============== RESULT AGGREGATION ==============
    
    def _aggregate_results(self, state: AgentState) -> AgentState:
        """Aggregate results (from cache or agents)"""
        final_response = []
        
        # If from cache, use cached response
        if state.get("from_cache"):
            cached = state.get("cached_response", {})
            if 'response' in cached:
                state["final_response"] = "🔄 (From Cache)\n\n" + cached['response']
                return state
        
        # Process Analysis Results
        if state.get("analysis_result"):
            results = state["analysis_result"].get("results", {})
            if isinstance(results, dict):
                # Handle comparison results
                if 'comparison' in results:
                    comp = results['comparison']
                    final_response.append(f"📊 **Comparison Results:**\n")
                    final_response.append(f"  • Sales Difference: ${comp.get('sales_difference', 0):,.2f}\n")
                    final_response.append(f"  • Growth: {comp.get('growth_percentage', 0):.2f}%\n")
                    
                    # Safely iterate through year data
                    for year_key, year_data in results.items():
                        if year_key.startswith('year_') and isinstance(year_data, dict):
                            year = year_key.split('_')[1]
                            final_response.append(f"\n**Year {year}:**\n")
                            final_response.append(f"  • Total Sales: ${year_data.get('total_sales', 0):,.2f}\n")
                            
                            # Safely access optional fields
                            if 'total_orders' in year_data:
                                final_response.append(f"  • Total Orders: {year_data['total_orders']:,}\n")
                            if 'avg_order_value' in year_data:
                                final_response.append(f"  • Avg Order Value: ${year_data['avg_order_value']:,.2f}\n")
                else:
                    # Regular results
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
                # Show model used, params, accuracy, and custom message:
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
        
        # Generate AI Explanation
        try:
            explanation_result = self.explanation_agent.execute(
                state["query"],
                {
                    "analysis": state.get("analysis_result"),
                    "forecast": state.get("forecast_result"),
                    "anomaly": state.get("anomaly_result")
                }
            )
            
            if explanation_result.get("status") == "success":
                explanation = explanation_result.get("explanation", "")
                if explanation:
                    final_response.append(f"\n---\n\n💡 **AI Insights:**\n\n{explanation}")
                    state["explanation"] = explanation
                    logger.info("AI explanation generated successfully")
        
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
        
        # Finalize response
        state["final_response"] = "".join(final_response) if final_response else "No results available"
        return state

    
    def process_query(self, query: str) -> dict:
        """Main entry point"""
        logger.info(f"Processing query: {query}")
        
        # Ensure data loaded
        try:
            if mcp_store.sales_df is None:
                logger.info("Data not loaded, loading now...")
                mcp_store.load_sales_data()
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return {
                "query": query,
                "intent": "error",
                "response": f"❌ Error loading data: {str(e)}"
            }
        
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
            "explanation": "",
            "final_response": ""
        }
        
        # Execute workflow
        try:
            result = self.workflow.invoke(initial_state)
            
            return {
                "query": query,
                "intent": result["intent"],
                "response": result["final_response"],
                "from_cache": result.get("from_cache", False)
            }
        
        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}", exc_info=True)
            return {
                "query": query,
                "intent": "error",
                "response": f"❌ Error processing query: {str(e)}"
            }
