from __future__ import annotations

import streamlit as st


def render_sidebar_navigation() -> str:
    with st.sidebar:
        st.title("Navigation")
        return st.radio("Choose a view", ["Dashboard", "History"], index=0)
