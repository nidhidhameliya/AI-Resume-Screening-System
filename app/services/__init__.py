"""Service layer for the recruitment dashboard.

The service package holds reusable business logic that keeps the Streamlit UI
entrypoint focused on presentation concerns rather than data workflow details.
"""

from .recruitment_service import RecruitmentService

__all__ = ["RecruitmentService"]
