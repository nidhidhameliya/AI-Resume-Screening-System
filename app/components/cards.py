from __future__ import annotations

import streamlit as st


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-pill">AI / NLP Recruitment Platform</div>
            <h1 class="hero-title">AI Resume Intelligence Platform</h1>
            <div class="hero-subtitle">Analyze resumes using NLP-powered semantic matching.</div>
            <div class="hero-subtitle">A recruiter-grade AI experience for fast, explainable, screen-first hiring workflows.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics: dict[str, str]) -> None:
    metric_cols = st.columns(3)
    metric_items = [
        ("Total Resumes", metrics.get("candidates", "—"), "Visible resume files analyzed"),
        ("Average Match Score", metrics.get("avg_match", "—"), "Mean alignment against the role brief"),
        ("Processing Time", metrics.get("processing_time", "—"), "Per-run processing latency"),
    ]

    for col, (label, value, note) in zip(metric_cols, metric_items, strict=False):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
