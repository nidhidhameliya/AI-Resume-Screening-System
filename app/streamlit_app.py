from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from app.config import AppConfig, DEFAULT_EMAIL, DEFAULT_PASSWORD, ROLE_LIBRARY
from app.services import RecruitmentService
from app.styles import render_dashboard_theme
from src.embedding import EmbeddingModel
from src.parser import extract_text_from_pdf
from src.preprocess import preprocess_text
from src.ranking import rank_candidates
from src.similarity import compute_similarity


APP_CONFIG = AppConfig()
RECRUITMENT_SERVICE = RecruitmentService(APP_CONFIG.db_path)


def _init_history_db() -> None:
    RECRUITMENT_SERVICE.initialize_history_database()


def _save_screening_history(job_role: str, candidate_names: list[str], scores: list[float]) -> None:
    RECRUITMENT_SERVICE.save_screening_history(job_role, candidate_names, scores)


def _load_history() -> pd.DataFrame:
    return RECRUITMENT_SERVICE.load_history()


def _build_skill_profile(text: str) -> list[str]:
    return RECRUITMENT_SERVICE.build_skill_profile(text)


def _score_bar_text(score: float, width: int = 20) -> str:
    return RECRUITMENT_SERVICE.score_bar_text(score, width)


def _build_explainability(job_text: str, resume_text: str, score: float) -> tuple[list[str], list[str], list[str]]:
    return RECRUITMENT_SERVICE.build_explainability(job_text, resume_text, score)


def _safe_pdf_preview(uploaded_file) -> None:
    try:
        import fitz
    except ImportError:
        st.info("Install PyMuPDF to render inline PDF previews.")
        return

    try:
        doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        page_count = len(doc)
        st.caption(f"Page count: {page_count}")
        if page_count:
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            st.image(pix.tobytes("png"))
    except Exception as exc:
        st.warning(f"Preview unavailable: {exc}")


def _render_dashboard_metrics() -> None:
    st.markdown(
        """
        <style>
        .hero-shell {
            background: linear-gradient(135deg, rgba(6, 10, 24, 0.96), rgba(37, 99, 235, 0.9));
            border-radius: 24px;
            padding: 1.9rem;
            margin-bottom: 1.4rem;
            border: 1px solid rgba(96, 165, 250, 0.28);
            box-shadow: 0 12px 32px rgba(2, 8, 23, 0.45);
        }
        .hero-pill {
            display: inline-block;
            background: rgba(125, 211, 252, 0.16);
            color: #e0f2fe;
            border: 1px solid rgba(125, 211, 252, 0.4);
            border-radius: 999px;
            padding: 0.3rem 0.7rem;
            margin-bottom: 0.8rem;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }
        .hero-title {
            background: linear-gradient(90deg, #ffffff, #93c5fd, #c4b5fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0;
        }
        .hero-subtitle {
            color: #dbeafe;
            font-size: 1rem;
            margin-top: 0.7rem;
            line-height: 1.6;
            max-width: 760px;
        }
        .metric-card {
            background: linear-gradient(180deg, #0f172a, #111827);
            border: 1px solid rgba(96, 165, 250, 0.22);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.28);
            min-height: 130px;
        }
        .metric-label {
            color: #94a3b8;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .metric-value {
            color: #f8fafc;
            font-size: 1.9rem;
            font-weight: 900;
            margin-top: 0.45rem;
        }
        .metric-note {
            color: #cbd5e1;
            font-size: 0.82rem;
            margin-top: 0.3rem;
        }
        .panel-card {
            background: linear-gradient(180deg, #111827, #0f172a);
            border: 1px solid rgba(96, 165, 250, 0.18);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 10px 26px rgba(2, 8, 23, 0.3);
        }
        .panel-title {
            color: #f8fafc;
            font-size: 0.98rem;
            font-weight: 800;
            margin-bottom: 0.75rem;
        }
        .panel-soft {
            background: rgba(30, 41, 59, 0.92);
            border-radius: 14px;
            border: 1px solid rgba(96, 165, 250, 0.16);
            padding: 0.9rem 1rem;
            color: #dbeafe;
            line-height: 1.6;
        }
        .file-pill {
            display: inline-block;
            background: #38bdf8;
            color: #082f49;
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            font-size: 0.72rem;
            font-weight: 800;
            margin-right: 0.45rem;
            margin-bottom: 0.35rem;
        }
        .score-card {
            background: linear-gradient(180deg, rgba(17, 24, 39, 1), rgba(15, 23, 42, 1));
            border: 1px solid rgba(96, 165, 250, 0.18);
            border-radius: 18px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .score-name {
            color: #f8fafc;
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }
        .score-label {
            color: #93c5fd;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .workflow-card {
            background: linear-gradient(180deg, rgba(17, 24, 39, 1), rgba(14, 116, 144, 0.24));
            border: 1px solid rgba(96, 165, 250, 0.18);
            border-radius: 16px;
            padding: 0.9rem;
            color: #e0f2fe;
            text-align: center;
            font-weight: 700;
            min-height: 74px;
        }
        .workflow-arrow {
            display: flex;
            align-items: center;
            justify-content: center;
            color: #38bdf8;
            font-size: 1.4rem;
            font-weight: 800;
        }
        .stButton > button {
            width: 100%;
            border-radius: 12px;
            background: linear-gradient(90deg, #2563eb, #38bdf8);
            color: white;
            border: none;
            font-weight: 800;
            height: 3rem;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #1d4ed8, #0ea5e9);
            color: white;
        }
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select {
            background: #0f172a;
            color: #f8fafc;
            border-radius: 12px;
            border: 1px solid rgba(96, 165, 250, 0.24);
        }
        .stFileUploader {
            background: rgba(15, 23, 42, 0.56);
            border-radius: 16px;
            padding: 0.4rem;
        }
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #22c55e, #38bdf8);
        }
        .sidebar .block-container {
            padding-top: 1.5rem;
        }
        .narrative-card {
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid rgba(96, 165, 250, 0.16);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            color: #dbeafe;
        }
        @media (max-width: 768px) {
            .hero-title { font-size: 1.8rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-pill">AI / NLP Recruitment Platform</div>
            <h1 class="hero-title">AI Resume Intelligence Platform</h1>
            <div class="hero-subtitle">
                Capture, interpret, and rank candidate resumes using semantic intelligence for faster and cleaner hiring decisions.
            </div>
            <div class="hero-subtitle">
                A professional AI recruiting workflow designed for recruiter-facing screening, explainability, and candidate insight generation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "dashboard_metrics" not in st.session_state:
        st.session_state.dashboard_metrics = {
            "candidates": "—",
            "avg_match": "—",
            "processing_time": "—",
        }

    metric_cols = st.columns(3)
    metric_cards = [
        ("Total Candidates", st.session_state.dashboard_metrics["candidates"], "Visible resumés reviewed"),
        ("Average Match Score", st.session_state.dashboard_metrics["avg_match"], "Mean similarity across ranked candidates"),
        ("Processing Time", st.session_state.dashboard_metrics["processing_time"], "Analysis runtime for this run"),
    ]

    for column, (label, value, note) in zip(metric_cols, metric_cards, strict=False):
        with column:
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

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-title">AI Workflow Architecture</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    workflow_steps = [
        "Resume PDF Upload",
        "Text Extraction",
        "Text Preprocessing",
        "Embedding Generation",
        "Similarity Matching",
        "Candidate Ranking",
    ]
    cols = st.columns(len(workflow_steps))
    for idx, step in enumerate(workflow_steps):
        with cols[idx]:
            st.markdown(f'<div class="workflow-card">{step}</div>', unsafe_allow_html=True)
            if idx < len(workflow_steps) - 1:
                st.markdown('<div class="workflow-arrow">↓</div>', unsafe_allow_html=True)


st.set_page_config(page_title="AI Resume Screening System", layout="wide")
render_dashboard_theme()
_render_dashboard_metrics()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.sidebar.title("Authentication")
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if email == DEFAULT_EMAIL and password == DEFAULT_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials.")
    st.info("Use demo credentials: admin@resume.ai / admin123")
    st.stop()

with st.sidebar:
    st.title("Navigation")
    page = st.radio("Choose a view", ["Dashboard", "History"], index=0)

if page == "History":
    st.title("Screening History")
    history_df = _load_history()
    if history_df.empty:
        st.info("No screening sessions have been saved yet.")
    else:
        st.dataframe(history_df, use_container_width=True)
    st.stop()

st.title("AI Resume Screening System")
left_col, right_col = st.columns([1.12, 0.88], gap="large")

with left_col:
    st.markdown('<div class="panel-card"><div class="panel-title">Role Selection</div></div>', unsafe_allow_html=True)
    selected_role = st.selectbox("", ["Manual", *list(ROLE_LIBRARY)], index=0)
    if selected_role != "Manual":
        st.session_state.default_job_description = ROLE_LIBRARY[selected_role]
        job_description = st.text_area(
            "Job Description",
            value=ROLE_LIBRARY[selected_role],
            height=210,
        )
    else:
        job_description = st.text_area("Job Description", height=210)

    st.markdown(
        '<div class="panel-card" style="margin-top: 1rem;">'
        '<div class="panel-title">Job Requirements Information</div>'
        '<div class="panel-soft">Use the selected role and edit the job description to steer the matching workflow. '
        'The ranking engine uses the current text to compare against uploaded resumes.</div></div>',
        unsafe_allow_html=True,
    )

with right_col:
    st.markdown('<div class="panel-card"><div class="panel-title">Resume PDF Uploader</div></div>', unsafe_allow_html=True)
    resume_files = st.file_uploader("", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

    if resume_files:
        uploaded_list = []
        for uploaded in resume_files:
            uploaded_list.append(f"<span class='file-pill'>{uploaded.name}</span>")
        st.markdown(
            f"<div class='panel-soft'>{''.join(uploaded_list)}<br><strong>{len(resume_files)}</strong> file(s) selected for screening.</div>",
            unsafe_allow_html=True,
        )
        for uploaded in resume_files:
            _safe_pdf_preview(uploaded)
    else:
        st.markdown(
            "<div class='panel-soft'>No PDF resumes selected yet. Upload one or more files to preview and compare them.</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="panel-card" style="margin-top: 1rem;"><div class="panel-title">Resume Chat Assistant</div></div>', unsafe_allow_html=True)
    if resume_files:
        uploaded_resume_texts: list[str] = []
        for uploaded in resume_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded.getvalue())
                temp_path = Path(temp_file.name)
            try:
                raw_text = extract_text_from_pdf(temp_path)
            finally:
                temp_path.unlink(missing_ok=True)
            uploaded_resume_texts.append(raw_text)

        combined_resume_text = "\n".join(uploaded_resume_texts)
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_query = st.chat_input("Ask about the uploaded candidate(s)")
        if user_query:
            lower_query = user_query.lower()
            candidate_summary = ""
            if "nlp" in lower_query and "nlp" in combined_resume_text.lower():
                candidate_summary += "NLP experience is mentioned in the uploaded resume content."
            if "python" in lower_query and "python" in combined_resume_text.lower():
                candidate_summary += " Python expertise is present in the resume text."
            if "strongest skills" in lower_query:
                skills = _build_skill_profile(combined_resume_text)
                candidate_summary = "Strongest detected skills: " + ", ".join(skills[:5]) if skills else "No obvious skills were identified."
            if "why" in lower_query and "suitable" in lower_query:
                candidate_summary = "This candidate appears suitable because the uploaded resume shows role-relevant language and matching technical keywords."
            if not candidate_summary:
                candidate_summary = "The resume text contains relevant signals, but the query is not directly matched to a strong keyword pattern."

            st.session_state.chat_messages.append({"role": "user", "content": user_query})
            st.session_state.chat_messages.append({"role": "assistant", "content": candidate_summary})
            st.rerun()
    else:
        st.info("Upload resumes first to enable the lightweight resume Q&A assistant.")

    if st.button("Analyze", type="primary"):
        analysis_started_at = time.perf_counter()
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
        raw_resume_texts: list[str] = []

        for uploaded in resume_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded.getvalue())
                temp_path = Path(temp_file.name)
            try:
                raw_text = extract_text_from_pdf(temp_path)
            finally:
                temp_path.unlink(missing_ok=True)

            raw_resume_texts.append(raw_text)
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

        ranking_df = pd.DataFrame(
            {
                "Candidate Name": [row.candidate_name for row in ranked],
                "Match Score": [round(row.score, 2) for row in ranked],
            }
        )
        ranking_df["Rank"] = range(1, len(ranking_df) + 1)
        ranking_df["Status"] = ["Shortlisted" if score >= 85 else "Review" if score >= 70 else "Low Match" for score in ranking_df["Match Score"]]
        ranking_df["Skills Found"] = [", ".join(_build_skill_profile(text)[:5]) for text in raw_resume_texts]
        ranking_df = ranking_df[["Rank", "Candidate Name", "Match Score", "Skills Found", "Status"]]

        average_score = round(sum(row.score for row in ranked) / len(ranked), 2) if ranked else 0.0
        processing_time = round(time.perf_counter() - analysis_started_at, 2)

        st.session_state.dashboard_metrics = {
            "candidates": len(ranking_df),
            "avg_match": f"{average_score:.2f}%",
            "processing_time": f"{processing_time:.2f}s",
        }

        _save_screening_history(selected_role if selected_role != "Manual" else "Manual", list(ranking_df["Candidate Name"]), list(ranking_df["Match Score"]))

        st.success(f"Analysis complete. {len(ranking_df)} candidate(s) ranked successfully.")

        st.markdown('<div class="panel-card"><div class="panel-title">Candidate Ranking Dashboard</div></div>', unsafe_allow_html=True)
        st.dataframe(ranking_df.sort_values("Match Score", ascending=False), use_container_width=True)

        st.markdown('---')
        st.markdown('<div class="panel-title">Candidate Score Visualization</div>', unsafe_allow_html=True)
        for idx, row in ranking_df.sort_values("Match Score", ascending=False).iterrows():
            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-name">{row['Candidate Name']}</div>
                    <div class="score-label">Match Score</div>
                    <div style="color: #f8fafc; font-size: 1.25rem; font-weight: 800; margin-top: 0.25rem;">{row['Match Score']}%</div>
                    <div style="margin-top: 0.45rem;">{_score_bar_text(float(row['Match Score']))}</div>
                    <div style="margin-top: 0.65rem; color: #cbd5e1;">Skills Match: {row['Skills Found']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('---')
        st.markdown('<div class="panel-title">Explainability</div>', unsafe_allow_html=True)
        for idx, row in ranking_df.sort_values("Match Score", ascending=False).iterrows():
            score = float(row["Match Score"])
            candidate_text = raw_resume_texts[idx]
            matched_skills, lower_reasons, recommendations = _build_explainability(job_description, candidate_text, score)
            st.markdown(
                f"""
                <div class="narrative-card">
                    <strong>{row['Candidate Name']}</strong><br>
                    <div style="margin-top: 0.5rem; color: #93c5fd;">Why this candidate matched</div>
                    <ul>
                        {''.join(f'<li>{reason}</li>' for reason in matched_skills)}
                    </ul>
                    <div style="margin-top: 0.5rem; color: #fca5a5;">Reasons for lower score</div>
                    <ul>
                        {''.join(f'<li>{reason}</li>' for reason in lower_reasons)}
                    </ul>
                    <div style="margin-top: 0.5rem; color: #86efac;">Recommendations</div>
                    <ul>
                        {''.join(f'<li>{reason}</li>' for reason in recommendations)}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('---')
        st.markdown('<div class="panel-title">Recruiter Analytics</div>', unsafe_allow_html=True)
        analytics_df = ranking_df.copy()
        analytics_df["Match Score"] = analytics_df["Match Score"].astype(float)
        st.bar_chart(analytics_df.set_index("Candidate Name")["Match Score"])
        skill_counter = pd.Series(", ".join(analytics_df["Skills Found"].tolist()).split(", ")).value_counts()
        st.bar_chart(skill_counter)
        st.metric("Resumes Processed", len(ranking_df))
        st.metric("Average Match Score", f"{average_score:.2f}%")

        st.markdown('---')
        st.markdown('<div class="panel-title">Download Recruitment Reports</div>', unsafe_allow_html=True)
        report_df = ranking_df.copy()
        report_df["Skills Found"] = report_df["Skills Found"].astype(str)
        report_df["Missing Skills"] = [
            ", ".join(_build_skill_profile(job_description)) for _ in range(len(report_df))
        ]
        report_df["Recommendation"] = [
            "Focus on role-critical skills and hands-on project evidence." for _ in range(len(report_df))
        ]

        excel_data = report_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Excel Candidate Ranking Report",
            data=excel_data,
            file_name="candidate_ranking_report.csv",
            mime="text/csv",
        )

        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet

            report_buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            report_buffer.close()
            doc = SimpleDocTemplate(report_buffer.name, pagesize=letter)
            styles = getSampleStyleSheet()
            story = [Paragraph("Candidate Recruitment Report", styles["Title"]), Spacer(1, 18)]
            table_data = [["Candidate Name", "Match Score", "Skills", "Missing Skills", "Recommendation"]]
            for _, row in report_df.iterrows():
                table_data.append([
                    row["Candidate Name"],
                    f"{row['Match Score']}%",
                    row["Skills Found"],
                    row["Missing Skills"],
                    row["Recommendation"],
                ])
            table = Table(table_data)
            table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ])
            )
            story.append(table)
            doc.build(story)
            with open(report_buffer.name, "rb") as report_file:
                pdf_bytes = report_file.read()
            st.download_button(
                label="Download PDF Recruitment Report",
                data=pdf_bytes,
                file_name="recruitment_report.pdf",
                mime="application/pdf",
            )
            os.unlink(report_buffer.name)
        except ImportError:
            st.info("Install reportlab to enable PDF report download.")
