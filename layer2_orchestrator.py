from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from data_connector import mcp_store
from layer2_routing_agent import IntelligentRouter
import logging

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State passed between agents"""
    query: str
    intent: str
    entities: dict
    routing_reasoning: str
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
        
        # NEW: Initialize intelligent router
        self.router = IntelligentRouter()
        
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow with LLM-based routing"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("route_query", self._route_query_with_llm)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("forecast", self._forecast_node)
        workflow.add_node("detect_anomalies", self._detect_anomalies_node)
        workflow.add_node("aggregate_results", self._aggregate_results)
        
        # Define workflow edges
        workflow.add_edge(START, "route_query")
        
        # Conditional routing based on LLM decision
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
    
    def _route_query_with_llm(self, state: AgentState) -> AgentState:
        """
        Use LLM to intelligently classify intent and extract entities.
        This replaces manual keyword matching with AI reasoning.
        """
        query = state["query"]
        
        # Get conversation context from mcp_store
        conversation_context = mcp_store.conversation_history[-5:] if hasattr(mcp_store, 'conversation_history') else []
        
        # Use LLM router to make intelligent decision
        routing_decision = self.router.route_query(query, conversation_context)
        
        # Update state with LLM's decision
        state["intent"] = routing_decision["intent"]
        state["entities"] = routing_decision.get("entities", {})
        state["routing_reasoning"] = routing_decision.get("reasoning", "")
        
        logger.info(f"LLM Routing Decision:")
        logger.info(f"  Intent: {state['intent']}")
        logger.info(f"  Entities: {state['entities']}")
        logger.info(f"  Reasoning: {state['routing_reasoning']}")
        logger.info(f"  Confidence: {routing_decision.get('confidence', 'N/A')}")
        
        return state
    
    def _route_to_agents(self, state: AgentState) -> str:
        """Return the intent to determine which agent node to execute"""
        return state["intent"]
    
    # ============== AGENT NODE FUNCTIONS ==============
    
    def _analyze_node(self, state: AgentState) -> AgentState:
        """Execute Analysis Agent with entities from LLM"""
        analysis_type = "aggregation" if "total" in state["query"].lower() else "summary"
        
        # Pass entities extracted by LLM
        result = self.analysis_agent.execute(
            state["query"], 
            analysis_type,
            entities=state.get("entities", {})
        )
        
        state["analysis_result"] = result
        logger.info(f"Analysis completed: {result.get('status')}")
        return state
    
    def _forecast_node(self, state: AgentState) -> AgentState:
        """Execute Forecasting Agent"""
        # Extract forecast periods from entities if provided
        periods = state.get("entities", {}).get("periods", 3)
        
        result = self.forecast_agent.execute(state["query"], forecast_periods=periods)
        state["forecast_result"] = result
        logger.info(f"Forecasting completed: {result.get('status')}")
        return state
    
    def _detect_anomalies_node(self, state: AgentState) -> AgentState:
        """Execute Anomaly Detection Agent"""
        # Extract threshold from entities if provided
        contamination = state.get("entities", {}).get("contamination", 0.05)
        
        result = self.anomaly_agent.execute(state["query"], contamination=contamination)
        state["anomaly_result"] = result
        logger.info(f"Anomaly detection completed: {result.get('status')}")
        return state
    
    # ============== RESULT AGGREGATION ==============
    
    def _aggregate_results(self, state: AgentState) -> AgentState:
        """Aggregate results from all executed agents"""
        final_response = []
        
        # Add routing reasoning at the top (optional, for transparency)
        if state.get("routing_reasoning"):
            final_response.append(f"_🤖 Routing: {state['routing_reasoning']}_\n")
        
        # Process Analysis Results
        if state.get("analysis_result"):
            results = state["analysis_result"].get("results", {})
            if isinstance(results, dict):
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
            forecasts = state["forecast_result"].get("forecasts", [])
            if forecasts:
                final_response.append(f"\n🔮 **Forecast for next {len(forecasts)} periods:**\n")
                for f in forecasts:
                    final_response.append(
                        f"  • **{f['date']}:** ${f['forecasted_sales']:,.2f} "
                        f"(confidence: {f.get('confidence', 'N/A')})\n"
                    )
                
                trend = state["forecast_result"].get("historical_trend")
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
                            f"    • Order **{anom.get('SalesDocument')}**: "
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
    
    # ============== MAIN ENTRY POINT ==============
    
    def process_query(self, query: str) -> dict:
        """Main entry point for query processing"""
        logger.info(f"Processing query: {query}")
        
        # Ensure data is loaded
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
        
        # Initialize workflow state
        initial_state = {
            "query": query,
            "intent": "",
            "entities": {},
            "routing_reasoning": "",
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
                "explanation": result.get("explanation", ""),
                "routing_reasoning": result.get("routing_reasoning", "")
            }
        
        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}", exc_info=True)
            return {
                "query": query,
                "intent": "error",
                "response": f"❌ Error processing query: {str(e)}"
            }
