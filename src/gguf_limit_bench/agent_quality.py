from __future__ import annotations

MIN_LIBRARIAN_RECOMMENDATION_PACKS = 3
MIN_LIBRARIAN_RECOMMENDATION_ATTEMPTS = 30


def librarian_agent_quality_gate(*, scored_pack_count: int, scored_attempt_count: int) -> str:
    if (
        scored_pack_count >= MIN_LIBRARIAN_RECOMMENDATION_PACKS
        and scored_attempt_count >= MIN_LIBRARIAN_RECOMMENDATION_ATTEMPTS
    ):
        return "recommendation_grade"
    return "weak_sample"


def is_recommendation_grade_librarian_sample(
    *, scored_pack_count: int, scored_attempt_count: int
) -> bool:
    return (
        librarian_agent_quality_gate(
            scored_pack_count=scored_pack_count,
            scored_attempt_count=scored_attempt_count,
        )
        == "recommendation_grade"
    )
