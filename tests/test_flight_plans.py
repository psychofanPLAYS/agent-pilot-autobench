import pytest

from gguf_limit_bench.flight_plans import (
    DEFAULT_FLIGHT_PLAN_ID,
    all_flight_plans,
    default_flight_plan,
    flight_plan_by_id,
    flight_plan_payloads,
    validate_flight_plan_modes,
)


def test_default_flight_plan_is_beginner_recommended():
    plan = default_flight_plan()

    assert plan.id == DEFAULT_FLIGHT_PLAN_ID
    assert plan.recommended is True
    assert plan.mode_id == "best_settings"
    assert plan.evidence_class == "recommendation"
    assert "preflight" in plan.workflow
    assert plan.start_label


def test_flight_plans_read_as_a_question_ladder():
    plans = all_flight_plans()

    assert [plan.id for plan in plans] == [
        "quick_check",
        "find_best_settings",
        "librarian_benchmark",
        "overnight_campaign",
    ]
    for plan in plans:
        assert plan.description, plan.id
        assert plan.evidence_goal.startswith("You get:"), plan.id
    assert flight_plan_by_id("quick_check").evidence_class == "speed_only"
    recommended = [plan for plan in plans if plan.recommended]
    assert len(recommended) == 1
    assert recommended[0].id == "find_best_settings"
    assert recommended[0].evidence_class == "recommendation"


def test_every_flight_plan_points_to_a_real_mode():
    validate_flight_plan_modes()

    ids = {plan.id for plan in all_flight_plans()}
    assert {"quick_check", "find_best_settings", "librarian_benchmark"} <= ids


def test_flight_plan_payload_is_plain_json_shape(tmp_path):
    payload = flight_plan_payloads(project_root=tmp_path)
    librarian = next(plan for plan in payload if plan["id"] == "librarian_benchmark")

    assert librarian["mode_id"] == "librarian_bench"
    assert librarian["evidence_class"] == "recommendation"
    assert librarian["score_contract"] == "agent_bench_score"
    assert librarian["workflow"] == ["preflight", "librarian-packs", "bias-checks", "report"]
    assert librarian["default_benchmark_suite_plan"] is None
    assert {
        "filename": "wiki-librarian-gemma4-26b-a4b-thinking.plan.json",
        "path": str(
            tmp_path / "benchmarks" / "plans" / "wiki-librarian-gemma4-26b-a4b-thinking.plan.json"
        ),
    } in librarian["suggested_benchmark_suite_plans"]
    assert {
        "filename": "wiki-librarian-qwen3-moe-thinking.plan.json",
        "path": str(
            tmp_path / "benchmarks" / "plans" / "wiki-librarian-qwen3-moe-thinking.plan.json"
        ),
    } in librarian["suggested_benchmark_suite_plans"]


def test_unknown_flight_plan_is_rejected():
    with pytest.raises(KeyError):
        flight_plan_by_id("does-not-exist")
