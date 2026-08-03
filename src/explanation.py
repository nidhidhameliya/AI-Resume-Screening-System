from __future__ import annotations


def generate_explanation(job_description: str, resume_text: str, score: float) -> dict[str, list[str]]:
    job_lower = job_description.lower()
    resume_lower = resume_text.lower()

    reasons = []
    missing = []

    if "python" in resume_lower:
        reasons.append("Python experience detected")
    if "machine learning" in resume_lower or "ml" in resume_lower:
        reasons.append("Machine learning projects detected")
    if "nlp" in resume_lower:
        reasons.append("NLP experience detected")
    if "transformers" in resume_lower:
        reasons.append("Transformers or LLM-related work detected")
    if not reasons:
        reasons.append("Role-specific signals were found in the uploaded resume content")

    if "cloud" in job_lower and "cloud" not in resume_lower:
        missing.append("Missing cloud experience")
    if "deployment" in job_lower and "deployment" not in resume_lower:
        missing.append("Missing deployment skills")
    if "aws" in job_lower and "aws" not in resume_lower:
        missing.append("Missing AWS exposure")
    if not missing:
        missing.append("No major missing skill signals were identified from the current brief")

    return {
        "why_matched": reasons,
        "reasons_for_lower_score": missing,
    }
