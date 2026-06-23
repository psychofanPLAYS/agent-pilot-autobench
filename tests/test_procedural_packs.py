"""RULER-style procedural long-context tasks (vendored generators, no deps).

These build synthetic, uncontaminated, length-controllable tasks: a model can
only answer by actually reading the long context, so they measure real
long-context capability (and its falloff vs ctx) instead of a memorizable fixture.
"""
from __future__ import annotations

import pytest

from gguf_limit_bench.answer_scoring import score_answer
from gguf_limit_bench.packs import AnswerType, QuestionPack, available_packs, load_pack
from gguf_limit_bench.procedural_packs import (
    approx_token_count,
    build_long_context_pack,
    generate_needle_single,
    generate_variable_tracking,
)


def test_approx_token_count_uses_four_chars_per_token():
    assert approx_token_count("") == 0
    assert approx_token_count("a" * 40) == 10


# --- needle-in-context ------------------------------------------------------

def test_needle_single_embeds_a_recoverable_code():
    q = generate_needle_single(target_tokens=300, depth_fraction=0.5, seed=1)
    # the answer code appears exactly once in the haystack
    assert q.prompt.count(q.answer) == 1
    assert q.answer.isalnum()


def test_needle_single_answer_scores_with_real_exact_scorer():
    q = generate_needle_single(target_tokens=300, depth_fraction=0.5, seed=1)
    good = f"After scanning the document, the code is clear.\nFinal Answer: {q.answer}"
    assert score_answer(good, q.answer, AnswerType.EXACT, q.accept) is True
    wrong = "Final Answer: ZZZZZZ"
    assert score_answer(wrong, q.answer, AnswerType.EXACT, q.accept) is False


def test_needle_single_reaches_target_length_band():
    target = 800
    q = generate_needle_single(target_tokens=target, depth_fraction=0.5, seed=2)
    tokens = approx_token_count(q.prompt)
    assert target * 0.8 <= tokens <= target * 1.6


def test_needle_depth_controls_position_in_context():
    early = generate_needle_single(target_tokens=600, depth_fraction=0.1, seed=3)
    late = generate_needle_single(target_tokens=600, depth_fraction=0.9, seed=3)
    assert early.prompt.index(early.answer) < late.prompt.index(late.answer)
    mid = generate_needle_single(target_tokens=600, depth_fraction=0.5, seed=3)
    rel = mid.prompt.index(mid.answer) / len(mid.prompt)
    assert 0.2 < rel < 0.85


def test_needle_single_is_reproducible_by_seed():
    a = generate_needle_single(target_tokens=300, depth_fraction=0.5, seed=7)
    b = generate_needle_single(target_tokens=300, depth_fraction=0.5, seed=7)
    c = generate_needle_single(target_tokens=300, depth_fraction=0.5, seed=8)
    assert (a.prompt, a.answer) == (b.prompt, b.answer)
    assert a.answer != c.answer


# --- variable tracking (multi-hop) ------------------------------------------

def test_variable_tracking_answer_is_the_resolved_value():
    q = generate_variable_tracking(target_tokens=400, hops=5, seed=4)
    assert q.answer.isdigit()
    good = f"Tracing the chain... Final Answer: {q.answer}"
    assert score_answer(good, q.answer, AnswerType.EXACT, q.accept) is True


def test_variable_tracking_chain_has_requested_hops():
    q = generate_variable_tracking(target_tokens=400, hops=5, seed=4)
    # the final variable name must be referenced in the prompt
    assert "VAR_5" in q.prompt
    assert "VAR_1" in q.prompt


def test_variable_tracking_is_reproducible_by_seed():
    a = generate_variable_tracking(target_tokens=400, hops=4, seed=11)
    b = generate_variable_tracking(target_tokens=400, hops=4, seed=11)
    assert (a.prompt, a.answer) == (b.prompt, b.answer)


# --- assembled pack ---------------------------------------------------------

def test_build_long_context_pack_shape():
    pack = build_long_context_pack(target_tokens=500, count=4, seed=5)
    assert isinstance(pack, QuestionPack)
    assert pack.answer_type is AnswerType.EXACT
    assert len(pack.questions) == 4
    assert pack.system_prompt.strip() != ""
    assert pack.pack_id == "ruler-longctx-500"
    # distinct question ids
    assert len({q.question_id for q in pack.questions}) == 4


def test_build_long_context_pack_answers_all_self_score():
    pack = build_long_context_pack(target_tokens=500, count=4, seed=5)
    for q in pack.questions:
        response = f"Final Answer: {q.answer}"
        assert score_answer(response, q.answer, pack.answer_type, q.accept) is True


# --- registry integration (load_pack / available_packs) ---------------------

def test_available_packs_includes_procedural_longctx_tiers():
    assert "ruler-longctx-65536" in available_packs()


def test_load_pack_generates_procedural_longctx_pack():
    pack = load_pack("ruler-longctx-2048")
    assert isinstance(pack, QuestionPack)
    assert pack.pack_id == "ruler-longctx-2048"
    assert pack.answer_type is AnswerType.EXACT
    assert len(pack.questions) >= 1


def test_load_pack_procedural_is_reproducible():
    a = load_pack("ruler-longctx-2048")
    b = load_pack("ruler-longctx-2048")
    assert [q.answer for q in a.questions] == [q.answer for q in b.questions]


def test_load_pack_unknown_still_raises_keyerror():
    with pytest.raises(KeyError):
        load_pack("definitely-not-a-pack")
