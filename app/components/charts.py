from __future__ import annotations

import pandas as pd
import streamlit as st


def render_score_distribution(scores: pd.Series) -> None:
    st.bar_chart(scores)


def render_skill_frequency(skills: pd.Series) -> None:
    st.bar_chart(skills)


def render_candidate_count(total: int) -> None:
    st.metric("Total Candidates", total)


def render_average_score(score: float) -> None:
    st.metric("Average Match Score", f"{score:.2f}%")
