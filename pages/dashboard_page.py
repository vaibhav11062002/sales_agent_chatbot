import streamlit as st
import os
from pathlib import Path

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="Dashboard Viewer",
    page_icon="📊",
    layout="wide"
)

# ===========================
# NAVIGATION HELPER
# ===========================
def go_back_to_chat():
    """Navigate back to main chat page"""
    # Clear dashboard from session
    st.session_state.pop('current_dashboard', None)
    st.session_state.pop('dashboard_html', None)
    
    # Set flag to trigger main page reload
    st.session_state['show_chat'] = True
    
    # Use JavaScript redirect as workaround
    st.markdown(
        """
        <script>
            window.parent.location.href = '/';
        </script>
        """,
        unsafe_allow_html=True
    )
    st.info("↩️ Redirecting to chat... Click here if not redirected automatically: [Back to Chat](/)")
    st.stop()

# ===========================
# CHECK DASHBOARD EXISTS
# ===========================
if 'dashboard_html' not in st.session_state or not st.session_state.dashboard_html:
    st.warning("⚠️ No dashboard to display. Please create a dashboard first.")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Back to Chat", use_container_width=True, type="primary"):
            go_back_to_chat()
    
    st.info("💡 **How to create a dashboard:**")
    st.markdown("""
    1. Go back to the chat
    2. Ask an analysis question (e.g., "What are total sales in 2024?")
    3. Then ask: "Create a dashboard for that" or "Build me a visualization"
    """)
    st.stop()

# ===========================
# HEADER
# ===========================
st.title("📊 Dashboard Viewer")
st.markdown("---")

# ===========================
# CONTROL PANEL
# ===========================
col1, col2, col3, col4 = st.columns([1.2, 1, 1, 2.8])

with col1:
    # Back to Chat Button
    if st.button("← Back to Chat", use_container_width=True, type="primary"):
        go_back_to_chat()

with col2:
    # Download HTML Button
    if st.session_state.current_dashboard:
        try:
            with open(st.session_state.current_dashboard, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            st.download_button(
                label="📥 HTML",
                data=html_content,
                file_name=os.path.basename(st.session_state.current_dashboard),
                mime="text/html",
                use_container_width=True,
                help="Download dashboard as HTML file"
            )
        except Exception as e:
            st.error(f"Error reading file: {e}")

with col3:
    # Save as PDF Button
    if st.button("📄 PDF", use_container_width=True, help="Generate and download PDF"):
        try:
            import pdfkit
            
            pdf_path = st.session_state.current_dashboard.replace('.html', '.pdf')
            
            # Configure pdfkit options
            options = {
                'page-size': 'A4',
                'margin-top': '0.5in',
                'margin-right': '0.5in',
                'margin-bottom': '0.5in',
                'margin-left': '0.5in',
                'encoding': "UTF-8",
                'enable-local-file-access': None,
                'no-stop-slow-scripts': None,
                'javascript-delay': 1000
            }
            
            with st.spinner("🔄 Generating PDF... This may take a moment..."):
                try:
                    pdfkit.from_file(
                        st.session_state.current_dashboard, 
                        pdf_path, 
                        options=options
                    )
                    st.success(f"✅ PDF generated successfully!")
                    
                    # Offer download
                    with open(pdf_path, 'rb') as f:
                        st.download_button(
                            label="💾 Download PDF",
                            data=f.read(),
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf",
                            help="Click to download the generated PDF"
                        )
                except Exception as pdf_error:
                    st.error(f"❌ PDF generation failed: {str(pdf_error)}")
                    st.info("💡 Try using your browser's print function (Ctrl+P / Cmd+P) and 'Save as PDF'")
        
        except ImportError:
            st.error("❌ **pdfkit not installed**")
            st.info("📦 Install with: `pip install pdfkit`")
            st.info("🔧 Also install wkhtmltopdf from: https://wkhtmltopdf.org/downloads.html")
            st.markdown("**Alternative:** Use browser's print function (Ctrl+P) → Save as PDF")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 **Tip:** Use browser's print function (Ctrl+P / Cmd+P) and select 'Save as PDF'")

with col4:
    # Display filename and metadata
    if st.session_state.current_dashboard:
        filename = os.path.basename(st.session_state.current_dashboard)
        file_size = os.path.getsize(st.session_state.current_dashboard) / 1024  # KB
        st.caption(f"📂 **{filename}** • {file_size:.1f} KB")

st.markdown("---")

# ===========================
# DASHBOARD DISPLAY
# ===========================
try:
    # Display the dashboard HTML in an iframe
    st.components.v1.html(
        st.session_state.dashboard_html, 
        height=800, 
        scrolling=True
    )
except Exception as e:
    st.error(f"❌ Error displaying dashboard: {str(e)}")
    st.info("💡 Try refreshing the page or creating a new dashboard")

# ===========================
# FOOTER
# ===========================
st.markdown("---")

# Tips and Instructions
with st.expander("💡 Tips & Shortcuts", expanded=False):
    st.markdown("""
    **Keyboard Shortcuts:**
    - `Ctrl+P` / `Cmd+P` - Print or save as PDF using browser
    - `Ctrl+F` / `Cmd+F` - Search within dashboard
    - `Ctrl+Mouse Wheel` - Zoom in/out
    
    **Dashboard Features:**
    - 🖱️ Hover over charts for detailed information
    - 📊 Click and drag on charts to zoom into specific areas
    - 💾 Download buttons provide both HTML and PDF formats
    - 🔄 Create new dashboards by asking in the chat
    
    **Troubleshooting:**
    - If dashboard doesn't load, go back to chat and try again
    - For PDF issues, use browser's print function as alternative
    - Clear browser cache if charts don't appear correctly
    """)

# Metadata footer
col_left, col_right = st.columns([3, 1])
with col_left:
    st.caption("🤖 Dashboard generated by Sales Agent AI • Powered by Gemini & Plotly")
with col_right:
    if st.button("🔄 Refresh", use_container_width=True, help="Reload dashboard"):
        st.rerun()

# ===========================
# SIDEBAR - Dashboard History
# ===========================
with st.sidebar:
    st.header("📊 Dashboard History")
    
    # Check for existing dashboards
    if os.path.exists('dashboards'):
        dashboards = sorted(
            [f for f in os.listdir('dashboards') if f.endswith('.html')],
            key=lambda x: os.path.getmtime(os.path.join('dashboards', x)),
            reverse=True
        )[:10]  # Show last 10 dashboards
        
        if dashboards:
            st.markdown(f"**Recent Dashboards** ({len(dashboards)})")
            
            for idx, dashboard in enumerate(dashboards, 1):
                filepath = os.path.join('dashboards', dashboard)
                file_time = os.path.getmtime(filepath)
                from datetime import datetime
                time_str = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M')
                
                # Parse dashboard name for display
                display_name = dashboard.replace('dashboard_', '').replace('.html', '')
                
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    if st.button(
                        f"{idx}. {display_name}", 
                        key=f"dash_{idx}",
                        use_container_width=True,
                        help=f"Created: {time_str}"
                    ):
                        # Load this dashboard
                        st.session_state.current_dashboard = filepath
                        with open(filepath, 'r', encoding='utf-8') as f:
                            st.session_state.dashboard_html = f.read()
                        st.rerun()
                
                with col_b:
                    # Delete button
                    if st.button("🗑️", key=f"del_{idx}", help="Delete this dashboard"):
                        try:
                            os.remove(filepath)
                            st.success("Deleted!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                
                st.caption(f"📅 {time_str}")
                if idx < len(dashboards):
                    st.markdown("---")
        else:
            st.info("No dashboards yet. Create one in the chat!")
    else:
        st.info("No dashboards yet. Create one in the chat!")
    
    st.markdown("---")
    
    # Quick Actions
    st.header("⚡ Quick Actions")
    
    if st.button("🆕 Create New Dashboard", use_container_width=True, type="primary"):
        go_back_to_chat()
    
    if st.button("📂 Open Dashboards Folder", use_container_width=True):
        import subprocess
        import platform
        
        dashboards_path = os.path.abspath('dashboards')
        
        try:
            if platform.system() == 'Windows':
                os.startfile(dashboards_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', dashboards_path])
            else:  # Linux
                subprocess.run(['xdg-open', dashboards_path])
            st.success("✅ Folder opened!")
        except Exception as e:
            st.error(f"❌ Could not open folder: {e}")
            st.info(f"📂 Path: {dashboards_path}")
    
    if st.button("🧹 Clear All Dashboards", use_container_width=True):
        if os.path.exists('dashboards'):
            files = [f for f in os.listdir('dashboards') if f.endswith('.html')]
            if files:
                if st.session_state.get('confirm_delete', False):
                    for f in files:
                        os.remove(os.path.join('dashboards', f))
                    st.success(f"✅ Deleted {len(files)} dashboards")
                    st.session_state.pop('confirm_delete')
                    st.rerun()
                else:
                    st.session_state['confirm_delete'] = True
                    st.warning("⚠️ Click again to confirm deletion")
            else:
                st.info("No dashboards to delete")
        else:
            st.info("No dashboards folder found")

# ===========================
# CUSTOM CSS
# ===========================
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Improve spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Dashboard iframe styling */
    iframe {
        border: 2px solid #f0f0f0;
        border-radius: 10px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        background-color: #f0f0f0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)
