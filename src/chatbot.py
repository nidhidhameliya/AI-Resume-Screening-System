from __future__ import annotations


def answer_resume_question(resume_text: str, question: str) -> str:
    lowered = resume_text.lower()
    query = question.lower()

    if "nlp" in query and "nlp" in lowered:
        return "Yes, the resume content references NLP experience."
    if "python" in query and "python" in lowered:
        return "Yes, Python appears in the resume content."
    if "strongest" in query or "skills" in query:
        skills = [
            skill for skill in ["python", "machine learning", "nlp", "sql", "cloud", "deployment"]
            if skill in lowered
        ]
        return f"The strongest detected skill signals are: {', '.join(skills) if skills else 'no obvious matching keywords found'}"
    if "suitable" in query or "why" in query:
        return "The resume shows relevant technical signals and role-aligned language, making it a potentially strong candidate for the requested position."
    return "The uploaded resume context is limited, but relevant signal keywords were found in the extracted text."
