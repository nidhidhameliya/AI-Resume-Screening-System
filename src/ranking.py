from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CandidateScore:
    candidate_name: str
    score: float


def rank_candidates(candidate_names: list[str], scores: list[float]) -> list[CandidateScore]:
    ranked = [
        CandidateScore(candidate_name=name, score=float(score))
        for name, score in zip(candidate_names, scores, strict=False)
    ]
    return sorted(ranked, key=lambda item: item.score, reverse=True)
