"""Reusable service logic for candidate ranking and explainability.

This module centralizes the score-building and history persistence behavior so
that the Streamlit UI can call structured business services instead of
embedding all workflow logic inline.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger("recruitment_service")


class RecruitmentService:
    """Business service for recruitment analysis workflow concerns."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def initialize_history_database(self) -> None:
        """Create the screening history storage if it is missing."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS screenings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    job_role TEXT NOT NULL,
                    candidate_names TEXT NOT NULL,
                    scores TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_screening_history(
        self,
        job_role: str,
        candidate_names: list[str],
        scores: list[float],
    ) -> None:
        """Persist analysis results for recruiter history tracking."""
        try:
            self.initialize_history_database()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO screenings (created_at, job_role, candidate_names, scores) VALUES (?, ?, ?, ?)",
                    (
                        now,
                        job_role,
                        " | ".join(candidate_names),
                        " | ".join(f"{score:.2f}%" for score in scores),
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.exception("Failed to store screening history: %s", exc)
            raise

    def load_history(self) -> pd.DataFrame:
        """Load screening history for the History page."""
        self.initialize_history_database()
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT created_at, job_role, candidate_names, scores FROM screenings ORDER BY id DESC",
                conn,
            )

    @staticmethod
    def build_skill_profile(text: str) -> list[str]:
        """Extract a normalized list of recruiter-relevant skills from text."""
        text_lower = text.lower()
        focus_keywords = [
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
        matched = [skill for skill in focus_keywords if skill in text_lower]
        return sorted(set(matched))

    @staticmethod
    def score_bar_text(score: float, width: int = 20) -> str:
        """Generate a compact bar-style matcher graph for UI display."""
        filled = max(0, min(width, int(round(score / 100 * width))))
        return "█" * filled + "░" * (width - filled)

    def build_explainability(
        self,
        job_text: str,
        resume_text: str,
        score: float,
    ) -> tuple[list[str], list[str], list[str]]:
        """Return concise explainability items for a candidate's score."""
        matched_skills = self.build_skill_profile(resume_text)
        job_keywords = self.build_skill_profile(job_text)
        matched = [skill for skill in job_keywords if skill in matched_skills]
        missing = [skill for skill in job_keywords if skill not in matched_skills]

        explanations: list[str] = []
        if "python" in matched:
            explanations.append("Python experience detected")
        if "machine learning" in matched or "ml" in matched:
            explanations.append("Machine learning projects found")
        if "nlp" in matched:
            explanations.append("NLP experience found")
        if not explanations:
            explanations.append("Core role-specific experience was found in the resume text")

        recommendations: list[str] = []
        for skill in missing[:3]:
            recommendations.append(f"Consider strengthening {skill} experience for this role")
        if not recommendations:
            recommendations.append("No major skill gap identified from the current job brief")

        if score < 80:
            lower_reason: list[str] = []
            if "cloud" in missing or "azure" in missing or "aws" in missing:
                lower_reason.append("Missing cloud experience")
            if "deployment" in missing:
                lower_reason.append("Missing deployment skills")
            if not lower_reason:
                lower_reason.append("Role-specific signal is weaker than expected from the resume content")
            return explanations, lower_reason, recommendations

        return explanations, [], recommendations
