from __future__ import annotations

import re
from typing import Iterable

DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with",
}


def _simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def preprocess_text(text: str, stop_words: Iterable[str] | None = None) -> str:
    """Normalize text for resume/job matching.

    The implementation intentionally avoids runtime corpus downloads so it can run in
    restricted environments.
    """
    words = set(stop_words or DEFAULT_STOPWORDS)
    tokens = _simple_tokenize(text)
    filtered = [token for token in tokens if token not in words]
    return " ".join(filtered)
