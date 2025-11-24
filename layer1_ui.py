import streamlit as st
from layer2_orchestrator import AgentOrchestrator
from data_connector import mcp_store
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SalesAgentChatbot:
    """Layer 1: Streamlit UI for user interaction"""

    def __init__(self):
        self.orchestrator = AgentOrchestrator()
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'current_dashboard' not in st.session_state:
            st.session_state.current_dashboard = None
        if 'dashboard_html' not in st.session_state:
            st.session_state.dashboard_html = None

    def load_data(self):
        """Load data on first query"""
        if not st.session_state.data_loaded:
            with st.spinner("Loading sales data from SAP Datasphere..."):
                try:
                    mcp_store.load_sales_data()
                    st.session_state.data_loaded = True
                    df = mcp_store.get_sales_data()
                    st.success(f"✅ Data loaded: {len(df)} records")
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
                    return False
        return True

    def render_ui(self):
        """Render Streamlit UI"""
        st.set_page_config(
            page_title="Sales Analytics Agent",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🤖 Sales Analytics Agentic AI Chatbot")
        st.markdown("Ask questions about your sales data!")

        # Sidebar with info
        with st.sidebar:
            st.header("ℹ️ About")
            st.markdown("""
            This chatbot uses multi-agent AI to:
            - 📊 Analyze sales data
            - 📈 Forecast trends
            - 🔍 Detect anomalies
            - 📊 Create dashboards
            - 💡 Provide insights

            **Supported Queries:**
            - "What are total sales in 2024?"
            - "Show me sales trends"
            - "Forecast next quarter sales"
            - "Detect anomalies in orders"
            - "Create a dashboard for 2024"
            - "Build me visualizations"
            """)
            
            if st.session_state.data_loaded:
                st.success("✅ Data Loaded")
                try:
                    df = mcp_store.get_sales_data()
                    st.info(f"Records: {len(df):,}")
                    st.info(f"Agents Active: {len(mcp_store.get_all_contexts())}")
                except:
                    pass

            if st.button("🔄 Reload Data"):
                mcp_store.sales_df = None
                st.session_state.data_loaded = False
                st.rerun()
            
            # ✅ NEW: Dashboard navigation
            st.divider()
            st.header("📊 Dashboards")
            if st.session_state.current_dashboard:
                if st.button("🎨 View Current Dashboard", use_container_width=True):
                    st.switch_page("pages/dashboard_page.py")
            
            # List recent dashboards
            if os.path.exists('dashboards'):
                dashboards = sorted(
                    [f for f in os.listdir('dashboards') if f.endswith('.html')],
                    reverse=True
                )[:5]
                
                if dashboards:
                    st.markdown("**Recent Dashboards:**")
                    for dashboard in dashboards:
                        if st.button(f"📄 {dashboard[:20]}...", key=dashboard, use_container_width=True):
                            st.session_state.current_dashboard = os.path.join('dashboards', dashboard)
                            with open(st.session_state.current_dashboard, 'r', encoding='utf-8') as f:
                                st.session_state.dashboard_html = f.read()
                            st.switch_page("pages/dashboard_page.py")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # ✅ NEW: Show dashboard button if message contains dashboard result
                if message["role"] == "assistant" and "dashboard_path" in message:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("🎨 View Dashboard", key=f"view_{message['dashboard_path']}"):
                            st.session_state.current_dashboard = message['dashboard_path']
                            with open(message['dashboard_path'], 'r', encoding='utf-8') as f:
                                st.session_state.dashboard_html = f.read()
                            st.switch_page("pages/dashboard_page.py")

        # Chat input
        if prompt := st.chat_input("Ask about sales data..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Load data if needed
            if not self.load_data():
                return

            # Add to conversation history
            mcp_store.add_conversation_turn("user", prompt)

            # Process query with orchestrator
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = self.orchestrator.process_query(prompt)

                        # ✅ NEW: Check if dashboard was created
                        is_dashboard = result.get('intent') == 'dashboard'
                        dashboard_result = result.get('dashboard_result', {})
                        
                        if is_dashboard and dashboard_result.get('status') == 'success':
                            # Dashboard created successfully
                            dashboard_path = dashboard_result.get('output_path')
                            dashboard_html = dashboard_result.get('dashboard_html')
                            
                            # Store in session state
                            st.session_state.current_dashboard = dashboard_path
                            st.session_state.dashboard_html = dashboard_html
                            
                            # Display success message
                            st.success("✅ Dashboard created successfully!")
                            
                            response = f"""**Intent**: {result['intent']}\n\n"""
                            response += f"📊 **Dashboard Created!**\n\n"
                            response += f"  • Charts generated: {dashboard_result.get('charts_generated', 0)}\n"
                            response += f"  • Dashboard plan: {dashboard_result.get('dashboard_plan', {}).get('title', 'N/A')}\n"
                            response += f"  • File saved: `{dashboard_path}`\n\n"
                            
                            st.markdown(response)
                            
                            # ✅ Show navigation button
                            col1, col2, col3 = st.columns([1, 1, 2])
                            with col1:
                                if st.button("🎨 View Dashboard", key="view_new_dashboard", use_container_width=True):
                                    st.switch_page("pages/dashboard_page.py")
                            with col2:
                                if st.button("📥 Download HTML", key="download_dashboard", use_container_width=True):
                                    with open(dashboard_path, 'r', encoding='utf-8') as f:
                                        st.download_button(
                                            label="💾 Download",
                                            data=f.read(),
                                            file_name=os.path.basename(dashboard_path),
                                            mime="text/html"
                                        )
                            
                            # Add dashboard path to message for later reference
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "dashboard_path": dashboard_path
                            })
                        
                        else:
                            # Regular response (non-dashboard)
                            response = f"""**Intent**: {result['intent']}\n\n**Results:**\n{result['response']}"""

                            # Forecast details expander
                            forecast_result = result.get("forecast_result", {})
                            with st.expander("📈 Forecasting Agent Details", expanded=(forecast_result.get("model_used") is not None)):
                                if forecast_result:
                                    for k, v in forecast_result.items():
                                        if k == "forecasts":
                                            st.markdown("**Forecasts:**")
                                            for f in v:
                                                st.markdown(f"- **{f.get('date','N/A')}**: ${f.get('forecasted_sales',0):,.2f} _(confidence: {f.get('confidence','N/A')})_")
                                        elif isinstance(v, dict):
                                            st.markdown(f"**{k.capitalize()}:** " + ", ".join(f"{ik}={iv:.2f}" if isinstance(iv, (int,float)) else f"{ik}={iv}" for ik,iv in v.items()))
                                        elif isinstance(v, (float, int)):
                                            st.markdown(f"**{k.capitalize()}:** {v:,.2f}")
                                        else:
                                            st.markdown(f"**{k.capitalize()}:** {v}")

                            st.markdown(response)

                            # Add to chat history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response
                            })

                        # Add to conversation history
                        mcp_store.add_conversation_turn("assistant", result['response'])

                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        logger.error(error_msg, exc_info=True)

    def run(self):
        """Run the chatbot"""
        self.render_ui()


if __name__ == "__main__":
    chatbot = SalesAgentChatbot()
    chatbot.run()
