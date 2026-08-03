from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_upload_section() -> tuple[str, list, str]:
    left_col, right_col = st.columns([1.12, 0.88], gap="large")

    with left_col:
        st.markdown(
            '<div class="panel-card"><div class="panel-title">Job Description</div></div>',
            unsafe_allow_html=True,
        )
        job_description = st.text_area("", height=210, label_visibility="collapsed")

        st.markdown(
            '<div class="panel-card"><div class="panel-title">Job Requirements Information</div>'
            '<div class="panel-soft">Use the role brief and the uploaded resume content to evaluate alignment, skill fit, and screening readiness.</div></div>',
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            '<div class="panel-card"><div class="panel-title">Resume Upload</div></div>',
            unsafe_allow_html=True,
        )
        resume_files = st.file_uploader(
            "",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        st.markdown(
            '<div class="panel-card"><div class="panel-title">Uploaded File Information</div></div>',
            unsafe_allow_html=True,
        )
        if resume_files:
            file_badges = "".join(f"<span class='file-pill'>{Path(file.name).name}</span>" for file in resume_files)
            st.markdown(
                f"<div class='panel-soft'>{file_badges}<br><strong>{len(resume_files)}</strong> file(s) ready for analysis.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='panel-soft'>No PDFs selected yet. Upload one or more resumes to begin the evaluation workflow.</div>",
                unsafe_allow_html=True,
            )

    return job_description, resume_files, "Manual"
