from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from src.embedding import EmbeddingModel
from src.parser import extract_text_from_pdf
from src.preprocess import preprocess_text
from src.ranking import rank_candidates
from src.similarity import compute_similarity


st.set_page_config(page_title="AI Resume Screening System", layout="wide")
st.title("AI Resume Screening System")
st.write("Upload resumes and compare them against a job description.")

job_description = st.text_area("Job Description", height=180)
resume_files = st.file_uploader(
    "Upload candidate resumes (PDF)", type=["pdf"], accept_multiple_files=True
)

if st.button("Analyze", type="primary"):
    if not job_description.strip():
        st.error("Please enter a job description.")
        st.stop()

    if not resume_files:
        st.error("Please upload at least one resume.")
        st.stop()

    try:
        model = EmbeddingModel()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    processed_job = preprocess_text(job_description)
    resume_names: list[str] = []
    processed_resumes: list[str] = []

    for uploaded in resume_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded.getvalue())
            temp_path = Path(temp_file.name)

        try:
            raw_text = extract_text_from_pdf(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

        processed_text = preprocess_text(raw_text)
        if processed_text:
            resume_names.append(uploaded.name)
            processed_resumes.append(processed_text)

    if not processed_resumes:
        st.warning("No readable text found in uploaded resumes.")
        st.stop()

    vectors = model.encode([processed_job, *processed_resumes])
    job_vector = vectors[0].tolist()
    resume_vectors = [vector.tolist() for vector in vectors[1:]]

    scores = compute_similarity(job_vector, resume_vectors)
    ranked = rank_candidates(resume_names, scores)

    results = pd.DataFrame(
        {
            "Candidate": [row.candidate_name for row in ranked],
            "Match Score (%)": [round(row.score, 2) for row in ranked],
        }
    )

    st.subheader("Candidate Ranking")
    st.dataframe(results, use_container_width=True)
    st.bar_chart(results.set_index("Candidate"))
