"""Application configuration package.

This package centralizes environment and runtime settings for the
recruitment dashboard so the Streamlit entrypoint can stay slim and
maintainable.
"""

from .settings import AppConfig, DEFAULT_EMAIL, DEFAULT_PASSWORD, ROLE_LIBRARY

__all__ = [
    "AppConfig",
    "DEFAULT_EMAIL",
    "DEFAULT_PASSWORD",
    "ROLE_LIBRARY",
]
