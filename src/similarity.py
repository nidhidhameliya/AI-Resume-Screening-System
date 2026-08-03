from __future__ import annotations

import math
from typing import Sequence


Vector = Sequence[float]


def _cosine_similarity(vec_a: Vector, vec_b: Vector) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same dimension.")

    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_similarity(job_vector: Vector, resume_vectors: Sequence[Vector]) -> list[float]:
    """Return cosine similarity scores as percentages (0-100)."""
    scores = [_cosine_similarity(job_vector, resume) * 100 for resume in resume_vectors]
    return [max(0.0, min(100.0, score)) for score in scores]
