import streamlit as st
from layer2_orchestrator import AgentOrchestrator
from data_connector import mcp_store
import logging

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
            layout="wide"
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
            - 💡 Provide insights
            
            **Supported Queries:**
            - "What are total sales in 2024?"
            - "Show me sales trends"
            - "Forecast next quarter sales"
            - "Detect anomalies in orders"
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
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
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
                        
                        # Display response
                        response = f"""
**Intent**: {result['intent']}

**Results:**
{result['response']}
"""
                        st.markdown(response)
                        
                        # Add to conversation history
                        mcp_store.add_conversation_turn("assistant", response)
                        
                        # Add to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        
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
