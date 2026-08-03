from __future__ import annotations

import streamlit as st


def render_results_table(df) -> None:
    st.markdown('<div class="panel-card"><div class="panel-title">Candidate Ranking Dashboard</div></div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)


def render_score_visuals(df) -> None:
    st.markdown('<div class="panel-title">Candidate Score Visualization</div>', unsafe_allow_html=True)
    for _, row in df.sort_values("Match Score", ascending=False).iterrows():
        filled = int(round(row["Match Score"] / 100 * 20))
        bar = "█" * filled + "░" * (20 - filled)
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-name">{row['Candidate Name']}</div>
                <div class="score-label">Match Score</div>
                <div style="color: #f8fafc; font-size: 1.25rem; font-weight: 800; margin-top: 0.3rem;">{row['Match Score']}%</div>
                <div style="margin-top: 0.5rem;">{bar}</div>
                <div style="color: #cbd5e1; margin-top: 0.6rem;">Skills Found: {row['Skills Found']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
