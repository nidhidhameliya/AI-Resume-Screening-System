from __future__ import annotations

import streamlit as st


def render_pipeline() -> None:
    st.markdown(
        '<div class="panel-card"><div class="panel-title">AI Pipeline Visualization</div></div>',
        unsafe_allow_html=True,
    )
    steps = [
        "Resume Upload",
        "Text Extraction",
        "Preprocessing",
        "Embedding",
        "Similarity Matching",
        "Ranking",
    ]
    cols = st.columns(len(steps))
    for idx, step in enumerate(steps):
        with cols[idx]:
            st.markdown(f'<div class="workflow-card">{step}</div>', unsafe_allow_html=True)
            if idx < len(steps) - 1:
                st.markdown('<div class="workflow-arrow">↓</div>', unsafe_allow_html=True)
