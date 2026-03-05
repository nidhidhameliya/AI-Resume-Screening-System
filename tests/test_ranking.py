from src.ranking import rank_candidates


def test_rank_candidates_orders_descending_scores() -> None:
    ranked = rank_candidates(["A", "B", "C"], [25.0, 85.0, 60.0])
    assert [r.candidate_name for r in ranked] == ["B", "C", "A"]
