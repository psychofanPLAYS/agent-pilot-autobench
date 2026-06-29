"""Tests for the librarian ``write-entry`` job: TYPE classification and SLUG
formatting, both EXACT-scored and seed-deterministic.
"""

from __future__ import annotations

from gguf_limit_bench.answer_scoring import score_answer
from gguf_limit_bench.librarian._common import MEMORY_TYPES, kebab_slug
from gguf_limit_bench.librarian.write_entry import PACK_ID, build
from gguf_limit_bench.packs import AnswerType, QuestionPack


def _wrong_value(answer: str) -> str:
    """A clearly-wrong EXACT value distinct from ``answer`` (and any type word)."""
    if answer in MEMORY_TYPES:
        # Pick a different memory type so it is plausibly-shaped but wrong.
        return next(t for t in MEMORY_TYPES if t != answer)
    return "totally-wrong-slug-value"


def test_build_is_deterministic():
    a = build(0)
    b = build(0)
    triples_a = [(q.question_id, q.prompt, q.answer) for q in a.questions]
    triples_b = [(q.question_id, q.prompt, q.answer) for q in b.questions]
    assert triples_a == triples_b


def test_pack_shape():
    pack = build(0)
    assert isinstance(pack, QuestionPack)
    assert pack.pack_id == PACK_ID
    assert pack.answer_type is AnswerType.EXACT
    assert 10 <= len(pack.questions) <= 16
    ids = [q.question_id for q in pack.questions]
    assert len(ids) == len(set(ids))  # unique
    for q in pack.questions:
        assert "librarian" in q.tags
        assert "write-entry" in q.tags
        assert q.choices is None
        assert q.question_id == f"{PACK_ID}-s0-{pack.questions.index(q)}"


def test_pack_contains_both_kinds():
    pack = build(0)
    subtypes = {q.tags[-1] for q in pack.questions}
    assert "type" in subtypes
    assert "slug" in subtypes


def test_scorer_round_trip_correct_and_wrong():
    for seed in (0, 1, 2, 7):
        pack = build(seed)
        for q in pack.questions:
            good = f"Here is my reasoning.\nFinal Answer: {q.answer}"
            assert score_answer(good, q.answer, AnswerType.EXACT, q.accept) is True
            bad = f"Final Answer: {_wrong_value(q.answer)}"
            assert score_answer(bad, q.answer, AnswerType.EXACT, q.accept) is False


def test_gold_sanity():
    pack = build(0)
    for q in pack.questions:
        subtype = q.tags[-1]
        if subtype == "type":
            assert q.answer in MEMORY_TYPES
        else:
            assert subtype == "slug"
            # Recover the title from the prompt and recompute the slug.
            start = q.prompt.index("'") + 1
            end = q.prompt.index("'", start)
            title = q.prompt[start:end]
            assert q.answer == kebab_slug(title)
            assert q.answer == q.answer.lower()


def test_seeds_vary_selection():
    # Different seeds should not all produce the identical question set.
    sets = {tuple((q.prompt, q.answer) for q in build(s).questions) for s in range(6)}
    assert len(sets) > 1
