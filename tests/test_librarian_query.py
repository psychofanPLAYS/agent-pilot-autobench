"""Tests for the ``librarian-query`` deterministic QE/HyDE generator."""

from __future__ import annotations

import pytest

from gguf_limit_bench.answer_scoring import score_answer
from gguf_limit_bench.librarian import query
from gguf_limit_bench.librarian._common import LIBRARIAN_SYSTEM_PROMPT
from gguf_limit_bench.packs import AnswerType

_LETTERS = "ABCDEF"


def test_determinism() -> None:
    assert query.build(0) == query.build(0)


def test_determinism_across_seeds_differs() -> None:
    assert query.build(0) != query.build(1)


def test_pack_shape() -> None:
    pack = query.build(0)
    assert pack.pack_id == query.PACK_ID == "librarian-query"
    assert pack.tier == "librarian"
    assert pack.answer_type is AnswerType.MULTIPLE_CHOICE
    assert pack.system_prompt == LIBRARIAN_SYSTEM_PROMPT
    assert 10 <= len(pack.questions) <= 16


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_query_pack_measures_qe_payload_selection(seed: int) -> None:
    pack = query.build(seed)
    ids = [q.question_id for q in pack.questions]
    assert len(ids) == len(set(ids))
    for index, q in enumerate(pack.questions):
        assert q.question_id == f"librarian-query-s{seed}-{index}"
        assert q.answer_source == "librarian:query"
        assert q.choices is not None
        assert len(q.choices) == 4
        assert len(q.answer) == 1
        assert q.answer in "ABCD"
        assert "librarian" in q.tags
        assert "query" in q.tags
        assert any(tag.startswith("intent=") for tag in q.tags)
        assert "LEX:" in q.prompt
        assert "HYDE:" in q.prompt
        assert "Do not answer the user's question" in q.prompt


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_gold_choice_contains_lexical_vector_and_hyde_not_direct_answer(seed: int) -> None:
    pack = query.build(seed)
    for q in pack.questions:
        assert q.choices is not None
        gold = q.choices[_LETTERS.index(q.answer)]
        assert "LEX:" in gold
        assert "HYDE:" in gold
        assert "ANSWER:" not in gold
        assert "direct answer" not in gold.lower()
        assert (
            len(
                [
                    part
                    for part in gold.split("LEX:", 1)[1].split("HYDE:", 1)[0].split(",")
                    if part.strip()
                ]
            )
            >= 3
        )


def test_scorer_round_trip() -> None:
    pack = query.build(0)
    for q in pack.questions:
        assert q.choices is not None
        assert score_answer(f"Final Answer: {q.answer}", q.answer, AnswerType.MULTIPLE_CHOICE)
        for i in range(len(q.choices)):
            other = _LETTERS[i]
            if other == q.answer:
                continue
            assert not score_answer(f"Final Answer: {other}", q.answer, AnswerType.MULTIPLE_CHOICE)
            break
