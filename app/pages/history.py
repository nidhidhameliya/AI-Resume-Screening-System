from __future__ import annotations

import streamlit as st

from database.database import load_history


def render_history_page() -> None:
    st.title("Screening History")
    history_df = load_history()
    if history_df.empty:
        st.info("No screening sessions recorded yet.")
        return
    st.dataframe(history_df, use_container_width=True)
