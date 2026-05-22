"""
styles.py - Custom CSS cho giao diện y tế chuyên nghiệp.
"""

import streamlit as st


def inject_custom_css():
    """Inject toàn bộ custom CSS vào trang Streamlit."""
    st.markdown("""
    <style>
    /* === Import Google Font === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* === Global === */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* === Header Section === */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 50%, #4a90d9 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(30, 58, 95, 0.3);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.85;
        margin: 0;
        font-weight: 400;
    }

    /* === Section Cards === */
    .section-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .section-card h3 {
        font-size: 1rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
        color: #e2e8f0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* === Feature Info Badge === */
    .feature-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* === Sidebar Styling === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }

    /* === Diagnose Button === */
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 50%, #f87171 100%);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        width: 100%;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.4);
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(220, 38, 38, 0.5);
    }

    /* === Model Info Cards in Sidebar === */
    .model-info-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .model-info-card .label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        font-weight: 600;
    }
    .model-info-card .value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #38bdf8;
    }

    /* === Status Indicator === */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        margin-right: 6px;
        animation: pulse-dot 2s infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    </style>
    """, unsafe_allow_html=True)
