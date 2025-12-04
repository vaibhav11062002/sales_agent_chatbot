import streamlit as st
from layer2_orchestrator import AgentOrchestrator
from data_connector import mcp_store
import logging
import os
from datetime import datetime


if 'messages' not in st.session_state:
    st.session_state.messages = []


if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Sales Analytics Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------- CUSTOM CSS ----------
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


/* Dashboard Card */
.dashboard-card {
    background: white;
    border: 2px solid #3b82f6;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
}


.dashboard-card h3 {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.5rem;
}


/* Dashboard Container */
.dashboard-viewer {
    background: white;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
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
    """Layer 1: Streamlit UI for user interaction with integrated dashboard viewing"""


    def __init__(self):
        # ✅ Initialize orchestrator ONCE at app startup using session state
        if 'orchestrator' not in st.session_state:
            logger.info("="*60)
            logger.info("🚀 FIRST-TIME APP STARTUP: Initializing all agents...")
            logger.info("="*60)
            st.session_state.orchestrator = AgentOrchestrator()
            logger.info("✅ All agents initialized and ready!")
            logger.info("="*60)
        
        self.orchestrator = st.session_state.orchestrator
        self._initialize_session_state()


    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'show_dashboard' not in st.session_state:
            st.session_state.show_dashboard = {}
        if 'dashboard_files' not in st.session_state:
            st.session_state.dashboard_files = {}


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


    def render_dashboard_inline(self, dashboard_path, message_key):
        """Render dashboard inline in the chat"""
        try:
            with open(dashboard_path, 'r', encoding='utf-8') as f:
                dashboard_html = f.read()
            
            # Add dashboard wrapper with id for PDF export
            if '<div id="dashboard-main">' not in dashboard_html:
                dashboard_html = dashboard_html.replace('<body>', '<body><div id="dashboard-main">')
                dashboard_html = dashboard_html.replace('</body>', '</div></body>')
            
            # Inject PDF button + html2pdf.js if not present
            if 'html2pdf.bundle.min.js' not in dashboard_html:
                pdf_button_html = '''
                <button id="download-pdf-btn" style="position: fixed; top: 18px; right: 28px; z-index: 99999; 
                    background: #3b82f6; color: white; border: none; padding: 8px 16px; border-radius: 6px; 
                    font-weight: 600; cursor: pointer; box-shadow: 0 2px 8px rgba(59,130,246,0.3);">
                    📄 Save as PDF
                </button>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
                <script>
                document.getElementById('download-pdf-btn').onclick = function () {
                    var element = document.getElementById('dashboard-main');
                    var opt = {
                        margin: 0.3,
                        filename: 'dashboard_export.pdf',
                        image: { type: 'jpeg', quality: 0.98 },
                        html2canvas: { scale: 2, useCORS: true },
                        jsPDF: { unit: 'in', format: 'a4', orientation: 'portrait' }
                    };
                    html2pdf().set(opt).from(element).save();
                }
                </script>
                '''
                dashboard_html = dashboard_html.replace('<body>', '<body>' + pdf_button_html)
            
            # Display the dashboard
            st.markdown('<div class="dashboard-viewer">', unsafe_allow_html=True)
            st.components.v1.html(
                dashboard_html,
                height=800,
                scrolling=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download button
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                st.download_button(
                    label="📥 Download HTML",
                    data=dashboard_html,
                    file_name=os.path.basename(dashboard_path),
                    mime="text/html",
                    use_container_width=True,
                    key=f"download_{message_key}"
                )
            with col2:
                if st.button("🗑️ Hide Dashboard", key=f"hide_{message_key}", use_container_width=True):
                    st.session_state.show_dashboard[message_key] = False
                    st.rerun()
                    
        except Exception as e:
            st.error(f"❌ Error displaying dashboard: {str(e)}")


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
            <p>Ask questions about your sales data and get intelligent insights with inline dashboards</p>
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
                    <li>📊 Create inline dashboards</li>
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
                • "Which customer has most anomalies?"<br>
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
            
            # Dashboard History
            st.markdown('<h3 class="sidebar-header">📊 Dashboard History</h3>', unsafe_allow_html=True)
            
            if os.path.exists('dashboards'):
                dashboards = sorted(
                    [f for f in os.listdir('dashboards') if f.endswith('.html')],
                    key=lambda x: os.path.getmtime(os.path.join('dashboards', x)),
                    reverse=True
                )[:5]
                
                if dashboards:
                    st.markdown("**Recent Dashboards:**")
                    for idx, dashboard in enumerate(dashboards):
                        filepath = os.path.join('dashboards', dashboard)
                        file_time = os.path.getmtime(filepath)
                        time_str = datetime.fromtimestamp(file_time).strftime('%m/%d %H:%M')
                        display_name = dashboard.replace('dashboard_', '').replace('.html', '')[:15]
                        
                        if st.button(
                            f"📄 {display_name}... ({time_str})",
                            key=f"hist_dash_{idx}",
                            use_container_width=True,
                            help=f"Load: {dashboard}"
                        ):
                            # Add dashboard to chat
                            msg_key = f"history_{idx}_{int(datetime.now().timestamp())}"
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"📊 **Dashboard loaded from history:** `{dashboard}`",
                                "dashboard_path": filepath,
                                "message_key": msg_key
                            })
                            st.session_state.show_dashboard[msg_key] = True
                            st.session_state.dashboard_files[msg_key] = filepath
                            st.rerun()
                else:
                    st.info("No dashboards yet")
            else:
                st.info("No dashboards yet")


        # Display chat history
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show dashboard inline if it exists
                if message["role"] == "assistant" and "dashboard_path" in message:
                    message_key = message.get("message_key", f"msg_{idx}")
                    
                    # Initialize show state if not exists
                    if message_key not in st.session_state.show_dashboard:
                        st.session_state.show_dashboard[message_key] = True
                        st.session_state.dashboard_files[message_key] = message["dashboard_path"]
                    
                    # Show/Hide toggle
                    if st.session_state.show_dashboard.get(message_key, True):
                        self.render_dashboard_inline(message["dashboard_path"], message_key)
                    else:
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            if st.button("📊 Show Dashboard", key=f"show_{message_key}", use_container_width=True):
                                st.session_state.show_dashboard[message_key] = True
                                st.rerun()


        # Chat input
        if prompt := st.chat_input("💬 Ask about sales data or request dashboards..."):
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
                            
                            # Generate unique message key
                            message_key = f"dash_{len(st.session_state.messages)}_{int(datetime.now().timestamp())}"
                            
                            # Display success message
                            st.success("✅ Dashboard created successfully!")
                            
                            st.markdown(f"""
                            <div class="dashboard-card">
                                <h3>📊 Dashboard Generated!</h3>
                                <p><strong>Intent:</strong> {result['intent']}</p>
                                <p><strong>Charts:</strong> {dashboard_result.get('charts_generated', 0)}</p>
                                <p><strong>Title:</strong> {dashboard_result.get('dashboard_plan', {}).get('title', 'N/A')}</p>
                                <p><strong>File:</strong> <code>{os.path.basename(dashboard_path)}</code></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            response = f"📊 **Dashboard Created:** `{os.path.basename(dashboard_path)}`\n\nDashboard is displayed below. You can download it or hide it using the buttons."
                            
                            st.markdown(response)
                            
                            # Initialize dashboard state
                            st.session_state.show_dashboard[message_key] = True
                            st.session_state.dashboard_files[message_key] = dashboard_path
                            
                            # Display dashboard inline immediately
                            self.render_dashboard_inline(dashboard_path, message_key)
                            
                            # Add to chat history with dashboard info
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "dashboard_path": dashboard_path,
                                "message_key": message_key
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
