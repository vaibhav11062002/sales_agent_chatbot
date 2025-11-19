from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from data_connector import mcp_store
import logging

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State passed between agents"""
    query: str
    intent: str
    entities: dict
    analysis_result: dict
    forecast_result: dict
    anomaly_result: dict
    explanation: str
    final_response: str

class AgentOrchestrator:
    """Layer 2: Orchestrates agent execution based on query intent"""
    
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
        
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow with conditional routing"""
        workflow = StateGraph(AgentState)
        
        # Add nodes (each node is a step in the workflow)
        workflow.add_node("route_query", self._route_query)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("forecast", self._forecast_node)
        workflow.add_node("detect_anomalies", self._detect_anomalies_node)
        workflow.add_node("aggregate_results", self._aggregate_results)
        
        # Define workflow edges
        workflow.add_edge(START, "route_query")
        
        # Conditional routing based on intent
        workflow.add_conditional_edges(
            "route_query",
            self._route_to_agents,
            {
                "analysis": "analyze",
                "forecast": "forecast",
                "anomaly": "detect_anomalies",
                "summary": "analyze"
            }
        )
        
        # All paths converge to aggregation
        workflow.add_edge("analyze", "aggregate_results")
        workflow.add_edge("forecast", "aggregate_results")
        workflow.add_edge("detect_anomalies", "aggregate_results")
        workflow.add_edge("aggregate_results", END)
        
        return workflow.compile()
    
    def _route_query(self, state: AgentState) -> AgentState:
        """
        Classify user intent and extract entities.
        This determines which agent(s) to activate.
        """
        query = state["query"].lower()
        
        # Intent classification logic
        if any(word in query for word in ["forecast", "predict", "future", "next"]):
            intent = "forecast"
        elif any(word in query for word in ["anomaly", "unusual", "outlier", "strange", "detect"]):
            intent = "anomaly"
        elif any(word in query for word in ["total", "sum", "how much", "revenue"]):
            intent = "analysis"
        elif any(word in query for word in ["trend", "growth", "change"]):
            intent = "analysis"
        else:
            intent = "summary"
        
        # Entity extraction (extract year, month, etc.)
        import re
        entities = {}
        year_match = re.search(r'20\d{2}', query)
        if year_match:
            entities['year'] = int(year_match.group())
        
        state["intent"] = intent
        state["entities"] = entities
        
        logger.info(f"Query routed with intent: {intent}, entities: {entities}")
        return state
    
    def _route_to_agents(self, state: AgentState) -> str:
        """Return the intent to determine which agent node to execute"""
        return state["intent"]
    
    # ============== AGENT NODE FUNCTIONS ==============
    # These functions are REQUIRED - they're the bridge between
    # LangGraph orchestration and your actual agent classes
    
    def _analyze_node(self, state: AgentState) -> AgentState:
        """
        Execute Analysis Agent.
        This node function calls the actual AnalysisAgent class.
        """
        analysis_type = "aggregation" if "total" in state["query"].lower() else "summary"
        result = self.analysis_agent.execute(state["query"], analysis_type)
        state["analysis_result"] = result
        logger.info(f"Analysis completed: {result.get('status')}")
        return state
    
    def _forecast_node(self, state: AgentState) -> AgentState:
        """
        Execute Forecasting Agent.
        This node function calls the actual ForecastingAgent class.
        """
        result = self.forecast_agent.execute(state["query"])
        state["forecast_result"] = result
        logger.info(f"Forecasting completed: {result.get('status')}")
        return state
    
    def _detect_anomalies_node(self, state: AgentState) -> AgentState:
        """
        Execute Anomaly Detection Agent.
        This node function calls the actual AnomalyDetectionAgent class.
        """
        result = self.anomaly_agent.execute(state["query"])
        state["anomaly_result"] = result
        logger.info(f"Anomaly detection completed: {result.get('status')}")
        return state
    
    # ============== RESULT AGGREGATION ==============
    
    def _aggregate_results(self, state: AgentState) -> AgentState:
        """
        Aggregate results from all executed agents and generate explanation.
        This is where context from multiple agents gets combined.
        """
        final_response = []
        
        # Process Analysis Results
        if state.get("analysis_result"):
            results = state["analysis_result"].get("results", {})
            if isinstance(results, dict):
                if "total_sales" in results:
                    final_response.append(f"💰 **Total Sales:** ${results['total_sales']:,.2f}")
                if "total_orders" in results:
                    final_response.append(f"📦 **Total Orders:** {results['total_orders']:,}")
                if "avg_order_value" in results:
                    final_response.append(f"📊 **Average Order Value:** ${results['avg_order_value']:,.2f}")
                if "unique_customers" in results:
                    final_response.append(f"👥 **Unique Customers:** {results['unique_customers']:,}")
                if "unique_products" in results:
                    final_response.append(f"🏷️ **Unique Products:** {results['unique_products']:,}")
        
        # Process Forecast Results
        if state.get("forecast_result"):
            forecasts = state["forecast_result"].get("forecasts", [])
            if forecasts:
                final_response.append(f"\n🔮 **Forecast for next {len(forecasts)} periods:**")
                for f in forecasts:
                    final_response.append(
                        f"  • **{f['date']}:** ${f['forecasted_sales']:,.2f} "
                        f"(confidence: {f.get('confidence', 'N/A')})"
                    )
                
                trend = state["forecast_result"].get("historical_trend")
                if trend:
                    trend_direction = "📈 upward" if trend > 0 else "📉 downward"
                    final_response.append(
                        f"\n**Historical Trend:** {trend_direction} trend "
                        f"(${abs(trend):,.2f}/month)"
                    )
        
        # Process Anomaly Results
        if state.get("anomaly_result"):
            anomalies = state["anomaly_result"].get("anomalies", {})
            if anomalies:
                final_response.append(f"\n⚠️ **Anomalies Detected:**")
                final_response.append(f"  • Total Anomalies: {anomalies.get('total_anomalies', 0):,}")
                final_response.append(f"  • Percentage: {anomalies.get('percentage', 0):.2f}%")
                final_response.append(f"  • Anomalous Sales Total: ${anomalies.get('anomaly_sales_total', 0):,.2f}")
                
                top_anomalies = anomalies.get('top_anomalies', [])
                if top_anomalies:
                    final_response.append(f"\n  **Top Anomalous Orders:**")
                    for anom in top_anomalies[:3]:
                        final_response.append(
                            f"    • Order **{anom.get('SalesDocument')}**: "
                            f"${anom.get('NetAmount', 0):,.2f}"
                        )
        
        # Generate AI Explanation using ExplanationAgent
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
                    final_response.append(f"\n\n💡 **AI Insights:**\n{explanation}")
                    state["explanation"] = explanation
                    logger.info("AI explanation generated successfully")
        
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            # Continue without explanation - don't fail the whole query
        
        # Finalize response
        state["final_response"] = "\n".join(final_response) if final_response else "No results available"
        return state
    
    # ============== MAIN ENTRY POINT ==============
    
    def process_query(self, query: str) -> dict:
        """
        Main entry point for query processing.
        Ensures data is loaded, executes workflow, returns results.
        """
        logger.info(f"Processing query: {query}")
        
        # Ensure data is loaded in mcp_store
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
                "explanation": result.get("explanation", "")
            }
        
        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}", exc_info=True)
            return {
                "query": query,
                "intent": "error",
                "response": f"❌ Error processing query: {str(e)}"
            }
