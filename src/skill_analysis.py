from __future__ import annotations

import re


def extract_skill_terms(text: str) -> list[str]:
    lowered = text.lower()
    skill_keywords = [
        "python",
        "sql",
        "machine learning",
        "ml",
        "numpy",
        "pandas",
        "nlp",
        "deep learning",
        "tensorflow",
        "pytorch",
        "azure",
        "aws",
        "cloud",
        "deployment",
        "power bi",
        "tableau",
        "api",
        "streamlit",
        "data visualization",
        "statistics",
        "model evaluation",
    ]
    return sorted({keyword for keyword in skill_keywords if keyword in lowered})


def analyze_skill_gap(job_description: str, resume_text: str) -> dict[str, list[str]]:
    job_skills = extract_skill_terms(job_description)
    resume_skills = extract_skill_terms(resume_text)
    matched_skills = [skill for skill in job_skills if skill in resume_skills]
    missing_skills = [skill for skill in job_skills if skill not in resume_skills]
    recommendations = [f"Strengthen {skill} readiness to match the requested role profile." for skill in missing_skills]
    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "recommendations": recommendations,
    }
