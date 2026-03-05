import pytest

from src.similarity import compute_similarity


def test_compute_similarity_percentage_range() -> None:
    job = [1.0, 0.0]
    resumes = [[1.0, 0.0], [0.0, 1.0]]
    scores = compute_similarity(job, resumes)
    assert scores[0] > scores[1]
    assert 0 <= min(scores) <= max(scores) <= 100


def test_compute_similarity_raises_for_dimension_mismatch() -> None:
    with pytest.raises(ValueError):
        compute_similarity([1.0, 0.0], [[1.0, 0.0, 1.0]])
