"""Turn autoresearch attempts into a flag recommendation (the buyer's answer)."""

from __future__ import annotations

from gguf_limit_bench.autoresearch import AttemptResult
from gguf_limit_bench.recommendation import (
    attempts_to_candidates,
    recommend_settings,
    render_recommendation_markdown,
)


def _ar(
    profile,
    *,
    ok=True,
    tps=0.0,
    acc=None,
    ctx=65536,
    completed=0,
    attempted=0,
    ttft=None,
    prompt_tps=0.0,
    failure="none",
):
    return AttemptResult(
        ok=ok,
        generation_tokens_per_second=tps,
        prompt_tokens_per_second=prompt_tps,
        ttft_ms=ttft,
        context_size=ctx,
        failure=failure,
        stdout="",
        stderr="",
        returncode=0,
        flag_profile=profile,
        simple_bench_accuracy=acc,
        completed_questions=completed,
        attempted_questions=attempted,
    )


def test_crashes_are_excluded_from_candidates():
    crash = _ar("L0", ok=False, failure="gpu_oom", completed=0)
    assert attempts_to_candidates([crash]) == []


def test_ok_and_partial_results_become_candidates():
    ok = _ar("L6", ok=True, tps=80.0, acc=0.6)
    partial = _ar(
        "L2",
        ok=False,
        tps=50.0,
        acc=0.4,
        completed=3,
        attempted=10,
        failure="budget_exhausted_partial",
    )
    candidates = attempts_to_candidates([ok, partial])
    assert {c.label for c in candidates} == {"L6", "L2"}


def test_partial_candidate_is_flagged_in_payload():
    partial = _ar(
        "L2",
        ok=False,
        tps=50.0,
        acc=0.4,
        completed=3,
        attempted=10,
        failure="budget_exhausted_partial",
    )
    (candidate,) = attempts_to_candidates([partial])
    assert candidate.payload["partial"] is True


def test_recommend_settings_is_accuracy_first_by_weight():
    a = _ar("A-accurate", tps=50.0, acc=0.8)
    b = _ar("B-fast", tps=90.0, acc=0.6)
    rec = recommend_settings([a, b], weights={"accuracy": 1.0, "tps": 0.0})
    assert rec.recommended is not None
    assert rec.recommended.label == "A-accurate"
    assert rec.considered == 2
    assert rec.total == 2


def test_recommend_settings_excludes_dominated_profiles():
    winner = _ar("win", tps=90.0, acc=0.8)
    dominated = _ar("lose", tps=40.0, acc=0.5)
    rec = recommend_settings([winner, dominated], weights={"accuracy": 1.0, "tps": 1.0})
    assert rec.recommended.label == "win"
    assert {c.label for c in rec.frontier} == {"win"}


def test_recommend_settings_no_usable_results_is_honest():
    rec = recommend_settings([_ar("L0", ok=False, failure="crash")])
    assert rec.recommended is None
    assert rec.considered == 0
    assert rec.total == 1
    assert "no usable" in rec.rationale.lower() or "crash" in rec.rationale.lower()


def test_render_markdown_names_the_recommended_profile():
    a = _ar("A-accurate", tps=50.0, acc=0.8)
    b = _ar("B-fast", tps=90.0, acc=0.6)
    rec = recommend_settings([a, b], weights={"accuracy": 1.0, "tps": 0.0})
    md = render_recommendation_markdown(rec)
    assert "A-accurate" in md
    assert "Tested" in md or "tested" in md


def test_render_markdown_handles_no_recommendation():
    rec = recommend_settings([_ar("L0", ok=False, failure="crash")])
    md = render_recommendation_markdown(rec)
    assert isinstance(md, str) and md.strip() != ""
