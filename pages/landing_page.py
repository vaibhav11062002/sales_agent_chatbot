import streamlit as st

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Enterprise Analytics AI Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- GLOBAL CSS (runs on every rerun) ----------
st.markdown(
    """
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global Styles */
* {
    font-family: 'Inter', sans-serif;
}

.main {
    background: #ffffff;
    padding: 0;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Hero Section */
.hero-section {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
    padding: 4rem 2rem;
    border-radius: 0;
    margin: -2rem -5rem 3rem -5rem;
    text-align: center;
    color: white;
    position: relative;
    overflow: hidden;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    opacity: 0.4;
}

.hero-content {
    position: relative;
    z-index: 1;
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 1.4rem;
    font-weight: 400;
    margin-bottom: 2rem;
    opacity: 0.95;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
}

/* Feature Pills */
.features-row {
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
    margin-top: 2rem;
}

.feature-pill {
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(10px);
    padding: 1rem 2rem;
    border-radius: 50px;
    display: inline-flex;
    align-items: center;
    gap: 0.75rem;
    font-weight: 500;
    border: 1px solid rgba(255, 255, 255, 0.3);
    transition: all 0.3s ease;
}

.feature-pill:hover {
    background: rgba(255, 255, 255, 0.3);
    transform: translateY(-2px);
}

.feature-icon {
    font-size: 1.5rem;
}

/* Section Header */
.section-header {
    text-align: center;
    margin: 4rem 0 3rem 0;
}

.section-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.5rem;
}

.section-subtitle {
    font-size: 1.1rem;
    color: #64748b;
    font-weight: 400;
}

/* Module Cards */
.module-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    height: 100%;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    margin-bottom: 1.5rem;
}

.module-icon {
    font-size: 3rem;
    margin-bottom: 1.5rem;
    display: block;
}

.module-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: #1e293b;
}

.module-description {
    font-size: 0.95rem;
    line-height: 1.6;
    color: #64748b;
    margin-bottom: 0;
    min-height: 60px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: #3b82f6;
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.3s ease;
    cursor: pointer;
}

.stButton > button:hover {
    background: #2563eb;
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(59, 130, 246, 0.3);
}

.stButton > button:active {
    transform: translateY(0);
}

/* Footer */
.footer {
    text-align: center;
    padding: 3rem 2rem 2rem 2rem;
    margin-top: 4rem;
    border-top: 1px solid #e2e8f0;
    color: #64748b;
}

.footer-content {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 2rem;
    flex-wrap: wrap;
    font-size: 0.95rem;
}

.footer-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Responsive Design */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2.5rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
    }
    .section-title {
        font-size: 2rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


def render_landing_page():
    # Hero Section
    st.markdown(
        """
    <div class="hero-section">
        <div class="hero-content">
            <h1 class="hero-title">Enterprise Analytics AI Platform</h1>
            <p class="hero-subtitle">Intelligent insights across your entire business ecosystem powered by advanced AI technology</p>
            <div class="features-row">
                <div class="feature-pill">
                    <span class="feature-icon">🤖</span>
                    <span>Multi-Agent AI</span>
                </div>
                <div class="feature-pill">
                    <span class="feature-icon">📊</span>
                    <span>Real-Time Analytics</span>
                </div>
                <div class="feature-pill">
                    <span class="feature-icon">🔮</span>
                    <span>Predictive Intelligence</span>
                </div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Section Header
    st.markdown(
        """
    <div class="section-header">
        <h2 class="section-title">Choose Your Business Module</h2>
        <p class="section-subtitle">Select a module to access powerful analytics and insights for your organization</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Row 1
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(
            """
        <div class="module-card">
            <div class="module-icon">📈</div>
            <h3 class="module-title">Sales & Development</h3>
            <p class="module-description">
                Comprehensive sales trends, accurate forecasting, deep customer insights, and pipeline analytics to drive revenue growth.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Module →", key="sales_launch"):
            st.switch_page("pages/layer1_ui.py")

    with col2:
        st.markdown(
            """
        <div class="module-card">
            <div class="module-icon">💰</div>
            <h3 class="module-title">Finance & Controlling</h3>
            <p class="module-description">
                Advanced financial planning, comprehensive P&L analysis, and detailed cost center performance insights.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Module →", key="finance_btn"):
            st.switch_page("pages/layer1_ui.py")

    with col3:
        st.markdown(
            """
        <div class="module-card">
            <div class="module-icon">📦</div>
            <h3 class="module-title">Inventory Management</h3>
            <p class="module-description">
                Smart stock optimization, intelligent demand planning, and comprehensive warehouse analytics.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Module →", key="inventory_btn"):
            st.switch_page("pages/layer1_ui.py")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2
    col4, col5, col6 = st.columns(3, gap="large")

    with col4:
        st.markdown(
            """
        <div class="module-card">
            <div class="module-icon">👥</div>
            <h3 class="module-title">Human Capital Management</h3>
            <p class="module-description">
                Comprehensive talent analytics, strategic workforce planning, and detailed performance insights.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Module →", key="hcm_btn"):
            st.switch_page("pages/layer1_ui.py")

    with col5:
        st.markdown(
            """
        <div class="module-card">
            <div class="module-icon">🏭</div>
            <h3 class="module-title">Production Planning</h3>
            <p class="module-description">
                Efficient capacity planning, optimized production scheduling, and advanced yield optimization.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Module →", key="production_btn"):
            st.switch_page("pages/layer1_ui.py")

    with col6:
        st.markdown(
            """
        <div class="module-card">
            <div class="module-icon">🔧</div>
            <h3 class="module-title">Maintenance & Quality</h3>
            <p class="module-description">
                Predictive maintenance scheduling, quality control analytics, and equipment performance monitoring.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Module →", key="maintenance_btn"):
            st.switch_page("pages/layer1_ui.py")

    # Footer
    st.markdown(
        """
    <div class="footer">
        <div class="footer-content">
            <div class="footer-item">
                <span>🚀</span>
                <span>Powered by Multi-Agent AI</span>
            </div>
            <div class="footer-item">
                <span>🔗</span>
                <span>SAP Datasphere Integration</span>
            </div>
            <div class="footer-item">
                <span>⚡</span>
                <span>Real-time Analytics</span>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render_landing_page()
