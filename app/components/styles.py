from __future__ import annotations

import streamlit as st


CSS = """
<style>
    :root {
        --bg: #07111f;
        --panel: #0b1729;
        --panel-2: #0f2238;
        --text: #edf6ff;
        --muted: #9fb4cc;
        --accent: #38bdf8;
        --accent-2: #8b5cf6;
        --success: #34d399;
        --warning: #f59e0b;
        --danger: #fb7185;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.15), transparent 28%),
            radial-gradient(circle at top right, rgba(139, 92, 246, 0.18), transparent 32%),
            linear-gradient(180deg, #020617, #07111f 60%, #020617);
        color: var(--text);
    }
    .hero-shell {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(37, 99, 235, 0.86));
        border: 1px solid rgba(125, 211, 252, 0.25);
        border-radius: 24px;
        padding: 1.9rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 28px rgba(2, 6, 23, 0.44);
    }
    .hero-pill {
        display: inline-block;
        background: rgba(96, 165, 250, 0.16);
        color: #dbeafe;
        border: 1px solid rgba(125, 211, 252, 0.3);
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        background: linear-gradient(90deg, #ffffff, #93c5fd, #d8b4fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 900;
        margin: 0 0 0.55rem 0;
    }
    .hero-subtitle {
        color: #dbeafe;
        max-width: 780px;
        line-height: 1.6;
        margin-bottom: 0.35rem;
    }
    .metric-card,
    .panel-card,
    .score-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(11, 23, 41, 0.98));
        border-radius: 18px;
        border: 1px solid rgba(96, 165, 250, 0.18);
        box-shadow: 0 10px 26px rgba(2, 6, 23, 0.35);
    }
    .metric-card {
        padding: 1rem 1.1rem;
        min-height: 132px;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 900;
        margin-top: 0.42rem;
    }
    .metric-note {
        color: #cbd5e1;
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }
    .panel-card {
        padding: 1rem;
        margin: 0.2rem 0 1rem 0;
    }
    .panel-title {
        color: #f8fafc;
        font-size: 0.98rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
    }
    .panel-soft {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(96, 165, 250, 0.16);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        color: #dbeafe;
        line-height: 1.6;
    }
    .file-pill {
        display: inline-block;
        background: #38bdf8;
        color: #082f49;
        border-radius: 999px;
        padding: 0.26rem 0.68rem;
        font-size: 0.72rem;
        font-weight: 800;
        margin: 0.15rem 0.4rem 0.15rem 0;
    }
    .score-card {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .score-name {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 800;
    }
    .score-label {
        color: #93c5fd;
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .workflow-card {
        background: linear-gradient(180deg, rgba(14, 116, 144, 0.28), rgba(15, 23, 42, 1));
        border: 1px solid rgba(96, 165, 250, 0.2);
        border-radius: 16px;
        padding: 0.9rem;
        color: #e0f2fe;
        text-align: center;
        font-weight: 700;
        min-height: 78px;
    }
    .workflow-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #38bdf8;
        font-size: 1.2rem;
        font-weight: 900;
        margin: 0.35rem 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(90deg, #2563eb, #38bdf8);
        color: #f8fafc;
        border: none;
        height: 3rem;
        font-weight: 800;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #0ea5e9);
        color: white;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        background: #0b1729;
        color: #f8fafc;
        border-radius: 12px;
        border: 1px solid rgba(96, 165, 250, 0.24);
    }
    .stFileUploader {
        background: rgba(15, 23, 42, 0.55);
        border-radius: 14px;
        padding: 0.4rem;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #22c55e, #38bdf8);
    }
    .sidebar .block-container {
        padding-top: 1.2rem;
    }
    .narrative-card {
        background: rgba(15, 23, 42, 0.9);
        color: #e0f2fe;
        border-radius: 16px;
        border: 1px solid rgba(96, 165, 250, 0.16);
        padding: 0.95rem 1rem;
    }
    .dark-overlay {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 18px;
        border: 1px solid rgba(96, 165, 250, 0.18);
    }
    @media (max-width: 768px) {
        .hero-title { font-size: 1.9rem; }
    }
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
