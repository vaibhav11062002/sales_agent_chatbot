import streamlit as st
from layer2_orchestrator import AgentOrchestrator
from data_connector import mcp_store
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Sales Analytics Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------- CUSTOM CSS (Matching Landing Page Theme) ----------
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global Styles */
* {
    font-family: 'Inter', sans-serif;
}

.main {
    background: #f8fafc;
}

.block-container {
    padding-top: 1rem;
    max-width: 1400px;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Back Button Styling */
.back-button-container {
    margin-bottom: 1.5rem;
}

.stButton > button[kind="secondary"] {
    background: white;
    color: #3b82f6;
    border: 2px solid #3b82f6;
    padding: 0.6rem 1.5rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.3s ease;
}

.stButton > button[kind="secondary"]:hover {
    background: #3b82f6;
    color: white;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}

/* Header Section */
.chat-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
    padding: 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
    box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2);
}

.chat-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}

.chat-header p {
    font-size: 1.1rem;
    opacity: 0.95;
    margin: 0;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid #e2e8f0;
}

section[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}

.sidebar-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 1rem;
}

/* Chat Messages */
.stChatMessage {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

/* Chat Input */
.stChatInput {
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    background: white;
}

.stChatInput:focus-within {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

/* Buttons */
.stButton > button {
    background: #3b82f6;
    color: white;
    border: none;
    padding: 0.6rem 1.5rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: #2563eb;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}

/* Success/Error Messages */
.stSuccess {
    background: #dcfce7;
    color: #166534;
    border: 1px solid #86efac;
    border-radius: 8px;
    padding: 1rem;
    font-weight: 500;
}

.stError {
    background: #fee2e2;
    color: #991b1b;
    border: 1px solid #fca5a5;
    border-radius: 8px;
    padding: 1rem;
    font-weight: 500;
}

.stInfo {
    background: #dbeafe;
    color: #1e40af;
    border: 1px solid #93c5fd;
    border-radius: 8px;
    padding: 0.8rem;
    font-weight: 500;
    font-size: 0.9rem;
}

/* Spinner */
.stSpinner > div {
    border-color: #3b82f6 transparent transparent transparent;
}

/* Expander */
.streamlit-expanderHeader {
    background: #f1f5f9;
    border-radius: 8px;
    font-weight: 600;
    color: #1e293b;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 1.5rem 0;
}

/* Markdown in chat */
.stMarkdown {
    color: #1e293b;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #1e293b;
    font-weight: 700;
}

.stMarkdown code {
    background: #f1f5f9;
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    font-size: 0.9em;
    color: #e11d48;
}

/* Dashboard Card */
.dashboard-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.dashboard-card h3 {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.5rem;
}

/* Responsive */
@media (max-width: 768px) {
    .chat-header h1 {
        font-size: 1.8rem;
    }
    .chat-header p {
        font-size: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)


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
            with st.spinner("🔄 Loading sales data from SAP Datasphere..."):
                try:
                    mcp_store.load_sales_data()
                    st.session_state.data_loaded = True
                    df = mcp_store.get_sales_data()
                    st.success(f"✅ Data loaded successfully: {len(df):,} records")
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
                    return False
        return True

    def render_ui(self):
        """Render Streamlit UI"""
        
        # Back to Home Button
        st.markdown('<div class="back-button-container">', unsafe_allow_html=True)
        if st.button("← Back to Home", key="back_home", type="secondary"):
            st.switch_page("pages/landing_page.py")
        st.markdown('</div>', unsafe_allow_html=True)

        # Header
        st.markdown("""
        <div class="chat-header">
            <h1>🤖 Sales Analytics AI Agent</h1>
            <p>Ask questions about your sales data and get intelligent insights powered by multi-agent AI</p>
        </div>
        """, unsafe_allow_html=True)

        # Sidebar
        with st.sidebar:
            # Back to Home in Sidebar too
            if st.button("🏠 Back to Home", key="sidebar_back_home", use_container_width=True):
                st.switch_page("pages/landing_page.py")
            
            st.divider()
            
            st.markdown('<h2 class="sidebar-header">ℹ️ About</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                <p style="margin: 0 0 0.8rem 0; font-weight: 600; color: #1e293b;">This chatbot uses multi-agent AI to:</p>
                <ul style="margin: 0; padding-left: 1.2rem; color: #64748b;">
                    <li>📊 Analyze sales data</li>
                    <li>📈 Forecast trends</li>
                    <li>🔍 Detect anomalies</li>
                    <li>📊 Create dashboards</li>
                    <li>💡 Provide insights</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 💬 Example Queries")
            st.markdown("""
            <div style="font-size: 0.9rem; color: #64748b; line-height: 1.6;">
                • "What are total sales in 2024?"<br>
                • "Show me sales trends"<br>
                • "Forecast next quarter sales"<br>
                • "Detect anomalies in orders"<br>
                • "Create a dashboard for 2024"<br>
                • "Build me visualizations"
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # Data Status
            st.markdown('<h3 class="sidebar-header">📊 Data Status</h3>', unsafe_allow_html=True)
            
            if st.session_state.data_loaded:
                st.success("✅ Data Loaded")
                try:
                    df = mcp_store.get_sales_data()
                    st.info(f"📁 Records: {len(df):,}")
                    st.info(f"🤖 Agents Active: {len(mcp_store.get_all_contexts())}")
                except:
                    pass
            else:
                st.warning("⏳ Data not loaded yet")

            if st.button("🔄 Reload Data", use_container_width=True):
                mcp_store.sales_df = None
                st.session_state.data_loaded = False
                st.rerun()
            
            st.divider()
            
            # Dashboard Navigation
            st.markdown('<h3 class="sidebar-header">📊 Dashboards</h3>', unsafe_allow_html=True)
            
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
                
                # Show dashboard button if message contains dashboard result
                if message["role"] == "assistant" and "dashboard_path" in message:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("🎨 View Dashboard", key=f"view_{message['dashboard_path']}"):
                            st.session_state.current_dashboard = message['dashboard_path']
                            with open(message['dashboard_path'], 'r', encoding='utf-8') as f:
                                st.session_state.dashboard_html = f.read()
                            st.switch_page("pages/dashboard_page.py")

        # Chat input
        if prompt := st.chat_input("💬 Ask about sales data..."):
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
                with st.spinner("🤔 Thinking..."):
                    try:
                        result = self.orchestrator.process_query(prompt)

                        # Check if dashboard was created
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
                            
                            st.markdown(f"""
                            <div class="dashboard-card">
                                <h3>📊 Dashboard Created!</h3>
                                <p><strong>Intent:</strong> {result['intent']}</p>
                                <p><strong>Charts generated:</strong> {dashboard_result.get('charts_generated', 0)}</p>
                                <p><strong>Title:</strong> {dashboard_result.get('dashboard_plan', {}).get('title', 'N/A')}</p>
                                <p><strong>File:</strong> <code>{dashboard_path}</code></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show navigation buttons
                            col1, col2, col3 = st.columns([1, 1, 2])
                            with col1:
                                if st.button("🎨 View Dashboard", key="view_new_dashboard", use_container_width=True):
                                    st.switch_page("pages/dashboard_page.py")
                            with col2:
                                if st.button("📥 Download", key="download_dashboard", use_container_width=True):
                                    with open(dashboard_path, 'r', encoding='utf-8') as f:
                                        st.download_button(
                                            label="💾 Download HTML",
                                            data=f.read(),
                                            file_name=os.path.basename(dashboard_path),
                                            mime="text/html",
                                            use_container_width=True
                                        )
                            
                            response = f"Dashboard created: {dashboard_path}"
                            
                            # Add dashboard path to message for later reference
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "dashboard_path": dashboard_path
                            })
                        
                        else:
                            # Regular response (non-dashboard)
                            response = f"**Intent:** `{result['intent']}`\n\n**Results:**\n\n{result['response']}"

                            # Forecast details expander
                            forecast_result = result.get("forecast_result", {})
                            if forecast_result and forecast_result.get("model_used"):
                                with st.expander("📈 Forecasting Agent Details", expanded=True):
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
