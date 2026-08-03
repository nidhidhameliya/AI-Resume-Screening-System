"""Central configuration for the AI recruitment platform.

This module intentionally keeps the system configurable without introducing
new runtime dependencies. The goal is to preserve the current workflow while
providing a production-friendly seam for future expansion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final


ROLE_LIBRARY: Final[dict[str, str]] = {
    "Machine Learning Engineer": (
        "Develop and deploy machine learning models, optimize training pipelines, and collaborate with product teams. "
        "Strong emphasis on Python, ML fundamentals, model evaluation, and deployment practices."
    ),
    "Data Scientist": (
        "Analyze large datasets, develop predictive models, and communicate business insights. Skills include statistics, "
        "Python, experimentation, and data storytelling."
    ),
    "Data Analyst": (
        "Interpret business data, build dashboards, create reports, and support operational decision making. Skills include SQL, "
        "Excel, analytics, and data visualization."
    ),
    "AI Engineer": (
        "Build production-ready AI features, integrate NLP systems, and support model deployment and evaluation in software products."
    ),
    "Software Engineer": (
        "Design and build maintainable software systems. A strong mix of coding, testing, architecture, and problem solving is expected."
    ),
}

DEFAULT_EMAIL: Final[str] = "admin@resume.ai"
DEFAULT_PASSWORD: Final[str] = "admin123"


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration shared by the UI and services layer."""

    project_root: Path = Path(__file__).resolve().parents[2]
    default_email: str = DEFAULT_EMAIL
    default_password: str = DEFAULT_PASSWORD
    role_library: dict[str, str] = field(default_factory=lambda: dict(ROLE_LIBRARY))

    @property
    def db_path(self) -> Path:
        """Return the database location from the resolved project root."""
        return self.project_root / "screening_history.db"
