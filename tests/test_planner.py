from gguf_limit_bench.executive_planner import PlannerSuggestion, validate_planner_suggestion


def test_planner_rejects_self_promotion_without_metrics():
    suggestion = PlannerSuggestion(
        source_model="candidate.gguf",
        action="promote_self",
        reason="I am best",
        next_settings={},
        evidence={},
    )

    decision = validate_planner_suggestion(suggestion)

    assert decision.accepted is False
    assert "objective metrics" in decision.reason


def test_planner_accepts_safe_next_experiment_with_receipt_evidence():
    suggestion = PlannerSuggestion(
        source_model="candidate.gguf",
        action="try_next_settings",
        reason="128k passed, test q8 KV next",
        next_settings={"context_size": 131072, "k_cache": "q8_0", "v_cache": "q8_0"},
        evidence={"receipt_path": "runs/test", "score": 88.0},
    )

    decision = validate_planner_suggestion(suggestion)

    assert decision.accepted is True
