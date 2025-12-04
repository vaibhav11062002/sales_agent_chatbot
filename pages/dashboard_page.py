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
# SESSION STATE INITIALIZATION
# ===========================
# Ensure critical session state persists across navigation
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'current_dashboard' not in st.session_state:
    st.session_state.current_dashboard = None

if 'dashboard_html' not in st.session_state:
    st.session_state.dashboard_html = None


# ===========================
# CHECK DASHBOARD EXISTS
# ===========================
if 'dashboard_html' not in st.session_state or not st.session_state.dashboard_html:
    st.warning("⚠️ No dashboard to display. Please create a dashboard first.")
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Back to Chat", use_container_width=True, type="primary"):
            st.switch_page("pages/layer1_ui.py")
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
    if st.button("← Back to Chat", use_container_width=True, type="primary"):
        # Simply navigate back - session state persists automatically
        st.switch_page("pages/layer1_ui.py")


with col2:
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
    st.info("📄 PDF (recommended: use the new dashboard PDF button below)")


with col4:
    if st.session_state.current_dashboard:
        filename = os.path.basename(st.session_state.current_dashboard)
        file_size = os.path.getsize(st.session_state.current_dashboard) / 1024  # KB
        st.caption(f"📂 **{filename}** • {file_size:.1f} KB")


st.markdown("---")


# ===========================
# DASHBOARD DISPLAY
# ===========================
try:
    # Add PDF button and scripts to dashboard HTML (if not already present)
    dashboard_html = st.session_state.dashboard_html
    # Ensure dashboard HTML wraps main content with id="dashboard-main"
    if '<div id="dashboard-main">' not in dashboard_html:
        dashboard_html = dashboard_html.replace('<body>', '<body><div id="dashboard-main">')
        dashboard_html = dashboard_html.replace('</body>', '</div></body>')
    # Inject PDF button + html2pdf.js if not present
    if 'html2pdf.bundle.min.js' not in dashboard_html:
        pdf_button_html = '''
        <button id="download-pdf-btn" style="position: fixed; top: 18px; right: 28px; z-index: 99999;">📄 Save as PDF</button>
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
        # Add just after <body> (<div id="dashboard-main"> is already added)
        dashboard_html = dashboard_html.replace('<body>', '<body>' + pdf_button_html)


    # Display the dashboard HTML
    st.components.v1.html(
        dashboard_html,
        height=800,
        scrolling=True
    )
except Exception as e:
    st.error(f"❌ Error displaying dashboard: {str(e)}")
    st.info("💡 Try refreshing the page or creating a new dashboard")


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
    - 📄 Use the new 'Save as PDF' button in the dashboard for instant PDF exports
    - 🔄 Create new dashboards by asking in the chat


    **PDF Generation:**
    - Uses client-side PDF export (button in dashboard)
    - Fallback: Use browser's print function (Ctrl+P) for PDF


    **Troubleshooting:**
    - If dashboard doesn't load, go back to chat and try again
    - For PDF issues, use browser's print function as alternative
    - Clear browser cache if charts don't appear correctly
    """)


# Metadata footer
col_left, col_right = st.columns([3, 1])
with col_left:
    st.caption("🤖 Dashboard generated by Sales Agent AI • Powered by Gemini & Matplotlib")
with col_right:
    if st.button("🔄 Refresh", use_container_width=True, help="Reload dashboard"):
        st.rerun()


# ===========================
# SIDEBAR - Dashboard History
# ===========================
with st.sidebar:
    st.header("📊 Dashboard History")
    
    # Back to Chat button in sidebar
    if st.button("🔙 Back to Chat", key="sidebar_back", use_container_width=True, type="primary"):
        st.switch_page("pages/layer1_ui.py")
    
    st.markdown("---")
    
    if os.path.exists('dashboards'):
        dashboards = sorted(
            [f for f in os.listdir('dashboards') if f.endswith('.html')],
            key=lambda x: os.path.getmtime(os.path.join('dashboards', x)),
            reverse=True
        )[:10]
        if dashboards:
            st.markdown(f"**Recent Dashboards** ({len(dashboards)})")
            for idx, dashboard in enumerate(dashboards, 1):
                filepath = os.path.join('dashboards', dashboard)
                file_time = os.path.getmtime(filepath)
                from datetime import datetime
                time_str = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M')
                display_name = dashboard.replace('dashboard_', '').replace('.html', '')
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    if st.button(
                        f"{idx}. {display_name}",
                        key=f"dash_{idx}",
                        use_container_width=True,
                        help=f"Created: {time_str}"
                    ):
                        st.session_state.current_dashboard = filepath
                        with open(filepath, 'r', encoding='utf-8') as f:
                            st.session_state.dashboard_html = f.read()
                        st.rerun()
                with col_b:
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
    st.header("⚡ Quick Actions")


    if st.button("🆕 Create New Dashboard", use_container_width=True, type="secondary"):
        st.switch_page("pages/layer1_ui.py")


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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    iframe {
        border: 2px solid #f0f0f0;
        border-radius: 10px;
    }
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    .streamlit-expanderHeader {
        font-weight: 600;
        background-color: #f0f0f0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)
