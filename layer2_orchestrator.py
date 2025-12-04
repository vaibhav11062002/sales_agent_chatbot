from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from data_connector import mcp_store
from layer2_routing_agent import IntelligentRouter
from layer2_context_manager import ContextManager
import logging
import re




logger = logging.getLogger(__name__)





class AgentState(TypedDict):
    """State passed between agents"""
    query: str
    intent: str
    original_intent: str  # ✅ Track original intent before override
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
    semantic_context: str  # ✅ NEW: Semantic context from previous analysis





class AgentOrchestrator:
    """Layer 2: Orchestrates agent execution using LLM-based intelligent routing"""
    
    def __init__(self):
        logger.info("="*60)
        logger.info("🚀 ORCHESTRATOR INITIALIZATION STARTED")
        logger.info("="*60)
        
        # ✅ STEP 1: Load data FIRST (before initializing agents)
        self._ensure_data_loaded()
        
        # ✅ STEP 2: Import and initialize all specialized agents
        from layer3_agents.analysis_agent import AnalysisAgent
        from layer3_agents.forecasting_agent import ForecastingAgent
        from layer3_agents.anomaly_detection_agent import AnomalyDetectionAgent
        from layer3_agents.explanation_agent import ExplanationAgent
        from layer3_agents.dashboard_agent import DashboardAgent
        
        logger.info("📦 Initializing agents...")
        self.analysis_agent = AnalysisAgent()
        self.forecast_agent = ForecastingAgent()
        self.anomaly_agent = AnomalyDetectionAgent()
        self.explanation_agent = ExplanationAgent()
        self.dashboard_agent = DashboardAgent()
        logger.info("✅ All agents initialized")
        
        # ✅ STEP 3: Initialize intelligent router and context manager
        self.router = IntelligentRouter()
        self.context_manager = ContextManager()
        
        # ✅ STEP 4: Run anomaly detection at startup (data is now loaded)
        self._run_startup_anomaly_detection()
        
        # ✅ STEP 5: Build workflow
        self.workflow = self._build_workflow()
        
        logger.info("="*60)
        logger.info("✅ ORCHESTRATOR INITIALIZATION COMPLETE")
        logger.info("="*60)
    
    def _ensure_data_loaded(self):
        """Ensure data is loaded before agent initialization"""
        try:
            if mcp_store.sales_df is None:
                logger.info("📂 Data not loaded - loading sales data...")
                mcp_store.load_sales_data()
                df = mcp_store.get_sales_data()
                logger.info(f"✅ Data loaded successfully: {len(df):,} records")
            else:
                logger.info(f"✅ Data already loaded: {len(mcp_store.sales_df):,} records")
        except Exception as e:
            logger.error(f"❌ Error loading data during initialization: {e}", exc_info=True)
            # Don't fail initialization, but log the error
            logger.warning("⚠️ Continuing without data - will attempt to load on first query")
    
    def _run_startup_anomaly_detection(self):
        """Run anomaly detection once at application startup"""
        logger.info("="*60)
        logger.info("🚀 STARTUP: Running Global Anomaly Detection")
        logger.info("="*60)
        
        try:
            # Verify data is loaded
            if mcp_store.sales_df is None:
                logger.warning("⚠️ Data not loaded - skipping startup anomaly detection")
                logger.info("   └─ Will run on first anomaly query")
                logger.info("="*60)
                return
            
            # Check if anomalies already exist
            existing_anomalies = mcp_store.get_enriched_data('anomaly_records')
            
            if existing_anomalies is not None and len(existing_anomalies) > 0:
                logger.info(f"✅ Anomalies already detected: {len(existing_anomalies)} records")
                logger.info(f"   └─ Skipping detection (using cached anomalies)")
                logger.info("="*60)
                return
            
            # Run global anomaly detection (no filters)
            logger.info("🔍 Running anomaly detection on ALL records...")
            result = self.anomaly_agent.execute(
                query="startup_anomaly_detection",
                entities={},  # No filters - global detection
                contamination=0.05
            )
            
            if result.get("status") == "success":
                total_anomalies = result.get("total_anomalies", 0)
                total_records = result.get("total_records", 0)
                anomaly_rate = result.get("anomaly_rate", "0%")
                
                logger.info("="*60)
                logger.info("✅ STARTUP ANOMALY DETECTION COMPLETE")
                logger.info("="*60)
                logger.info(f"   └─ Total Records: {total_records:,}")
                logger.info(f"   └─ Anomalies Detected: {total_anomalies:,}")
                logger.info(f"   └─ Anomaly Rate: {anomaly_rate}")
                logger.info("="*60)
                
                # Update dialogue state
                mcp_store.update_dialogue_state({
                    'anomalies_detected': True,
                    'anomaly_count': total_anomalies
                }, query="", response="")
            else:
                logger.warning(f"⚠️ Anomaly detection failed: {result.get('message', 'Unknown error')}")
                logger.info("="*60)
        
        except Exception as e:
            logger.error(f"❌ Error in startup anomaly detection: {e}", exc_info=True)
            logger.info("="*60)
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow - anomaly detection runs at startup"""
        workflow = StateGraph(AgentState)
        
        # Add nodes (no detect_anomalies node - runs at startup)
        workflow.add_node("check_context", self._check_context_first)
        workflow.add_node("route_query", self._route_query_with_llm)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("forecast", self._forecast_node)
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
        
        # Route to specific agents based on intent (no anomaly node)
        workflow.add_conditional_edges(
            "route_query",
            self._route_to_agents,
            {
                "analysis": "analyze",
                "forecast": "forecast",
                "dashboard": "create_dashboard"
            }
        )
        
        # All paths converge to aggregation
        workflow.add_edge("analyze", "aggregate_results")
        workflow.add_edge("forecast", "aggregate_results")
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
        
        # Quick routing to get entities with conversation history
        conversation_context = mcp_store.conversation_history[-5:] if hasattr(mcp_store, 'conversation_history') else []
        routing_decision = self.router.route_query(query, conversation_context)
        
        state["entities"] = routing_decision.get("entities", {})
        state["intent"] = routing_decision["intent"]
        state["original_intent"] = routing_decision["intent"]  # ✅ Store original
        state["semantic_context"] = ""  # ✅ Initialize
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 STEP 2: CHECKING MCP CACHE")
        logger.info(f"{'='*60}")
        
        # Skip cache check for dashboard queries (always fresh)
        if state["intent"] == "dashboard":
            logger.info(f"📊 Dashboard query detected - skipping cache (always generate fresh)")
            state["from_cache"] = False
            state["cached_response"] = {}
            
            # ✅ NEW: Extract semantic context from previous analysis
            previous_analysis = mcp_store.get_agent_context('AnalysisAgent')
            if previous_analysis:
                prev_query = previous_analysis.get('query', '')
                prev_results = previous_analysis.get('results', {})
                
                # Build semantic context string
                semantic_parts = []
                
                # Add previous query
                if prev_query:
                    semantic_parts.append(f"Previous query: '{prev_query}'")
                
                # Extract key insights from LLM response
                if isinstance(prev_results, dict) and 'llm_raw' in prev_results:
                    llm_response = prev_results['llm_raw']
                    # Extract key findings
                    if 'lowest' in llm_response.lower():
                        semantic_parts.append("Context: This is about the LOWEST performing entity")
                    elif 'highest' in llm_response.lower():
                        semantic_parts.append("Context: This is about the HIGHEST performing entity")
                    
                    # Extract customer/product mentions
                    customer_match = re.search(r'Customer[:\s]+(\d+)', llm_response, re.IGNORECASE)
                    if customer_match:
                        semantic_parts.append(f"Focus: Customer {customer_match.group(1)}")
                
                state["semantic_context"] = " | ".join(semantic_parts)
                logger.info(f"📋 Semantic context extracted: {state['semantic_context']}")
            
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
        """
        ✅ SIMPLIFIED: Route to agents (anomaly detection already done at startup)
        All anomaly queries go to AnalysisAgent which uses anomaly-enriched data
        """
        query = state["query"]
        intent = state["intent"]
        
        # ✅ OVERRIDE: All anomaly queries go to AnalysisAgent (anomalies already detected)
        if intent == 'anomaly':
            logger.info(f"🔄 ROUTING OVERRIDE: Anomaly query → AnalysisAgent")
            logger.info(f"   └─ Reason: Anomalies pre-detected at startup")
            logger.info(f"   └─ AnalysisAgent will use anomaly-enriched data")
            
            state['intent'] = 'analysis'  # Override to analysis
            state['original_intent'] = 'anomaly'  # Track original
        
        logger.info(f"\nLLM Routing Decision:")
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
        
        # ✅ Check if this was originally an anomaly query
        if state.get("original_intent") == "anomaly":
            logger.info(f"🔍 Original intent was 'anomaly' - AnalysisAgent will use anomaly-enriched data")
        
        # Determine analysis type
        analysis_type = "aggregation" if "total" in state["query"].lower() else "summary"
        logger.info(f"📈 Performing {analysis_type} analysis")
        
        # Execute analysis
        result = self.analysis_agent.execute(state["query"], analysis_type, entities=entities)
        
        state["analysis_result"] = result
        
        # ✅ NEW: Store semantic context for dashboard
        if result.get("status") == "success":
            results_data = result.get("results", {})
            if isinstance(results_data, dict) and 'llm_raw' in results_data:
                state["semantic_context"] = results_data['llm_raw'][:500]  # First 500 chars
        
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
        """
        ⚠️ DEPRECATED: This node is no longer used in workflow
        Anomaly detection now runs at startup
        Keeping method for backward compatibility
        """
        logger.warning("⚠️ _detect_anomalies_node called but anomalies already detected at startup")
        logger.info("   └─ This node is deprecated and should not be called")
        
        state["anomaly_result"] = {
            "status": "success",
            "message": "Anomalies already detected at startup"
        }
        return state
    
    def _create_dashboard_node(self, state: AgentState) -> AgentState:
        """
        ✅ FIXED: Execute Dashboard Agent with enriched entities including aggregation_type
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 EXECUTING: DashboardAgent")
        logger.info(f"{'='*60}")
        
        entities = state.get("entities", {}).copy()
        
        # ✅ FIX: Merge aggregation_type from dialogue state if missing
        dialogue_state = mcp_store.get_current_dialogue_state()
        context_entities = dialogue_state.get('entities', {})
        
        # Add missing critical context entities
        if 'aggregation_type' in context_entities and 'aggregation_type' not in entities:
            entities['aggregation_type'] = context_entities['aggregation_type']
            logger.info(f"   🔗 Added aggregation_type from context: {context_entities['aggregation_type']}")
        
        if 'metric' in context_entities and 'metric' not in entities:
            entities['metric'] = context_entities['metric']
            logger.info(f"   🔗 Added metric from context: {context_entities['metric']}")
        
        logger.info(f"📋 Entities passed to agent: {entities}")
        
        # ✅ NEW: Enrich entities with semantic context
        enriched_entities = entities.copy()
        
        # Add semantic context from previous analysis
        if state.get("semantic_context"):
            enriched_entities['semantic_context'] = state["semantic_context"]
            logger.info(f"📋 Semantic context added: {state['semantic_context'][:100]}...")
        
        # Add previous query context
        previous_analysis = mcp_store.get_agent_context('AnalysisAgent')
        if previous_analysis:
            enriched_entities['previous_query'] = previous_analysis.get('query', '')
            enriched_entities['previous_results'] = previous_analysis.get('results', {})
            logger.info(f"📋 Previous query context added")
        
        # Execute dashboard with enriched context
        result = self.dashboard_agent.execute(
            state["query"], 
            entities=enriched_entities,
            analysis_agent=self.analysis_agent
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
            anomaly_result = state["anomaly_result"]
            
            if anomaly_result.get("status") == "success":
                final_response.append(f"\n⚠️ **Anomalies Detected:**\n")
                final_response.append(f"  • Total Records: {anomaly_result.get('total_records', 0):,}\n")
                final_response.append(f"  • Total Anomalies: {anomaly_result.get('total_anomalies', 0):,}\n")
                final_response.append(f"  • Anomaly Rate: {anomaly_result.get('anomaly_rate', 'N/A')}\n")
                
                # Get summary
                summary = anomaly_result.get('summary', {})
                if summary:
                    # Show category breakdown
                    by_category = summary.get('by_reason_category', {})
                    if by_category:
                        final_response.append(f"\n  **By Category:**\n")
                        for category, count in list(by_category.items())[:5]:
                            final_response.append(f"    • {category}: {count}\n")
                    
                    # Show top customers with anomalies
                    top_customers = summary.get('top_customers_with_anomalies', {})
                    if top_customers:
                        final_response.append(f"\n  **Top Customers with Anomalies:**\n")
                        for customer, count in list(top_customers.items())[:5]:
                            final_response.append(f"    • Customer {customer}: {count} anomalies\n")
                
                # Show sample anomalies
                anomalies_list = anomaly_result.get('anomalies', [])
                if anomalies_list and len(anomalies_list) > 0:
                    final_response.append(f"\n  **Sample Anomalies (Top 3 by severity):**\n")
                    for i, anom in enumerate(anomalies_list[:3], 1):
                        final_response.append(f"\n    {i}. ")
                        
                        # Customer
                        if 'Customer' in anom:
                            final_response.append(f"Customer: {anom.get('Customer', 'N/A')}, ")
                        elif 'SoldToParty' in anom:
                            final_response.append(f"Customer: {anom.get('SoldToParty', 'N/A')}, ")
                        
                        # Date
                        if 'Date' in anom:
                            final_response.append(f"Date: {anom.get('Date', 'N/A')}, ")
                        elif 'CreationDate' in anom:
                            final_response.append(f"Date: {anom.get('CreationDate', 'N/A')}, ")
                        
                        # Revenue
                        if 'Revenue' in anom:
                            final_response.append(f"Revenue: ${anom.get('Revenue', 0):,.2f}\n")
                        elif 'NetAmount' in anom:
                            final_response.append(f"Amount: ${anom.get('NetAmount', 0):,.2f}\n")
                        
                        # Reason
                        if 'anomaly_reason' in anom:
                            final_response.append(f"       Reason: {anom.get('anomaly_reason', 'N/A')}\n")
            
            elif anomaly_result.get("status") == "error":
                final_response.append(f"\n❌ **Anomaly Detection Failed:**\n")
                final_response.append(f"  • Error: {anomaly_result.get('message', 'Unknown error')}\n")
        
        # Generate AI Explanation with entities
        if state.get("intent") != "dashboard":
            try:
                explanation_context = {
                    "analysis": state.get("analysis_result"),
                    "forecast": state.get("forecast_result"),
                    "anomaly": state.get("anomaly_result"),
                    "from_cache": state.get("from_cache", False),
                    "entities": state.get("entities", {})
                }
                
                explanation_result = self.explanation_agent.execute(
                    state["query"],
                    explanation_context
                )
                
                if explanation_result.get("status") == "success":
                    explanation = explanation_result.get("explanation", "")
                    if explanation:
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
            "original_intent": "",
            "entities": {},
            "routing_reasoning": "",
            "from_cache": False,
            "cached_response": {},
            "analysis_result": {},
            "forecast_result": {},
            "anomaly_result": {},
            "dashboard_result": {},
            "explanation": "",
            "final_response": "",
            "semantic_context": ""  # ✅ Initialize
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
