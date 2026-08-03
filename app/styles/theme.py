"""Premium dashboard styling for the ATS UI.

The visual layer is intentionally isolated so the business logic keeps the same
behavior while the interface can evolve toward a polished enterprise product.
"""

from __future__ import annotations

PREMIUM_DASHBOARD_CSS = """
<style>
    :root {
        color-scheme: dark;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.18), transparent 22%),
            radial-gradient(circle at top right, rgba(16, 185, 129, 0.15), transparent 18%),
            linear-gradient(135deg, #020617, #0f172a 45%, #111827);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(2, 6, 23, 0.98));
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }
    .hero-shell {
        background: linear-gradient(135deg, rgba(6, 10, 24, 0.96), rgba(37, 99, 235, 0.9));
        border-radius: 24px;
        padding: 1.9rem;
        margin-bottom: 1.4rem;
        border: 1px solid rgba(96, 165, 250, 0.28);
        box-shadow: 0 12px 32px rgba(2, 8, 23, 0.45);
    }
    .hero-pill {
        display: inline-block;
        background: rgba(125, 211, 252, 0.16);
        color: #e0f2fe;
        border: 1px solid rgba(125, 211, 252, 0.4);
        border-radius: 999px;
        padding: 0.3rem 0.7rem;
        margin-bottom: 0.8rem;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }
    .hero-title {
        background: linear-gradient(90deg, #ffffff, #93c5fd, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }
    .hero-subtitle {
        color: #dbeafe;
        font-size: 1rem;
        margin-top: 0.7rem;
        line-height: 1.6;
        max-width: 760px;
    }
    .metric-card {
        background: linear-gradient(180deg, #0f172a, #111827);
        border: 1px solid rgba(96, 165, 250, 0.22);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.28);
        min-height: 130px;
        transition: transform 160ms ease, box-shadow 160ms ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 32px rgba(16, 185, 129, 0.12);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.9rem;
        font-weight: 900;
        margin-top: 0.45rem;
    }
    .metric-note {
        color: #cbd5e1;
        font-size: 0.82rem;
        margin-top: 0.3rem;
    }
    .panel-card {
        background: linear-gradient(180deg, #111827, #0f172a);
        border: 1px solid rgba(96, 165, 250, 0.18);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 10px 26px rgba(2, 8, 23, 0.3);
    }
    .panel-title {
        color: #f8fafc;
        font-size: 0.98rem;
        font-weight: 800;
        margin-bottom: 0.75rem;
    }
    .panel-soft {
        background: rgba(30, 41, 59, 0.92);
        border-radius: 14px;
        border: 1px solid rgba(96, 165, 250, 0.16);
        padding: 0.9rem 1rem;
        color: #dbeafe;
        line-height: 1.6;
    }
    .file-pill {
        display: inline-block;
        background: #38bdf8;
        color: #082f49;
        border-radius: 999px;
        padding: 0.25rem 0.65rem;
        font-size: 0.72rem;
        font-weight: 800;
        margin-right: 0.45rem;
        margin-bottom: 0.35rem;
    }
    .score-card {
        background: linear-gradient(180deg, rgba(17, 24, 39, 1), rgba(15, 23, 42, 1));
        border: 1px solid rgba(96, 165, 250, 0.18);
        border-radius: 18px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .score-name {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    .score-label {
        color: #93c5fd;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .workflow-card {
        background: linear-gradient(180deg, rgba(17, 24, 39, 1), rgba(14, 116, 144, 0.24));
        border: 1px solid rgba(96, 165, 250, 0.18);
        border-radius: 16px;
        padding: 0.9rem;
        color: #e0f2fe;
        text-align: center;
        font-weight: 700;
        min-height: 74px;
    }
    .workflow-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #38bdf8;
        font-size: 1.4rem;
        font-weight: 800;
    }
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(90deg, #2563eb, #38bdf8);
        color: white;
        border: none;
        font-weight: 800;
        height: 3rem;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #0ea5e9);
        color: white;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background: #0f172a;
        color: #f8fafc;
        border-radius: 12px;
        border: 1px solid rgba(96, 165, 250, 0.24);
    }
    .stFileUploader {
        background: rgba(15, 23, 42, 0.56);
        border-radius: 16px;
        padding: 0.4rem;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #22c55e, #38bdf8);
    }
    .sidebar .block-container {
        padding-top: 1.5rem;
    }
    .narrative-card {
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid rgba(96, 165, 250, 0.16);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        color: #dbeafe;
    }
    @media (max-width: 768px) {
        .hero-title { font-size: 1.8rem; }
    }
</style>
"""


def render_dashboard_theme() -> None:
    """Render the shared premium UI theme for the resume platform."""
    import streamlit as st

    st.markdown(PREMIUM_DASHBOARD_CSS, unsafe_allow_html=True)
