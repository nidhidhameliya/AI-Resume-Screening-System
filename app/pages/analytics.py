from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import render_average_score, render_candidate_count, render_score_distribution, render_skill_frequency


def render_analytics_page(history_df: pd.DataFrame) -> None:
    st.title("Recruiter Analytics")
    if history_df.empty:
        st.info("Run a screening to populate the analytics dashboard.")
        return

    scores = history_df["scores"].explode().astype(float)
    skills = pd.Series(
        " ".join(history_df["candidate_names"].tolist()).split(),
        name="skills",
    )

    render_score_distribution(scores)
    render_skill_frequency(skills.value_counts())
    render_candidate_count(len(history_df))
    render_average_score(scores.mean())
