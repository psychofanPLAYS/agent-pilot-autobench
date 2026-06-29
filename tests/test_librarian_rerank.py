"""Tests for the ``librarian-rerank`` deterministic generator."""

from __future__ import annotations

import pytest

from gguf_limit_bench.answer_scoring import score_answer
from gguf_limit_bench.librarian import rerank
from gguf_limit_bench.librarian._common import LIBRARIAN_SYSTEM_PROMPT
from gguf_limit_bench.packs import AnswerType

_LETTERS = "ABCDEF"


def test_determinism() -> None:
    """build(seed) twice returns byte-identical packs (same shuffles)."""
    assert rerank.build(0) == rerank.build(0)


def test_determinism_across_seeds_differs() -> None:
    """Different seeds should generally produce different packs."""
    assert rerank.build(0) != rerank.build(1)


def test_pack_shape() -> None:
    pack = rerank.build(0)
    assert pack.pack_id == rerank.PACK_ID == "librarian-rerank"
    assert pack.tier == "librarian"
    assert pack.answer_type is AnswerType.MULTIPLE_CHOICE
    assert pack.system_prompt == LIBRARIAN_SYSTEM_PROMPT
    assert 10 <= len(pack.questions) <= 16


def test_unique_ids() -> None:
    pack = rerank.build(0)
    ids = [q.question_id for q in pack.questions]
    assert len(ids) == len(set(ids))
    for i, q in enumerate(pack.questions):
        assert q.question_id == f"librarian-rerank-s0-{i}"


def test_tags_and_choices() -> None:
    pack = rerank.build(0)
    for q in pack.questions:
        assert "librarian" in q.tags
        assert "rerank" in q.tags
        assert q.choices is not None
        # Every question carries the abstention option, so k is one larger than
        # before: distractors (+ maybe the answer) + the None option.
        k = len(q.choices)
        assert k in (4, 5, 6)
        assert f"n_choices={k}" in q.tags
        assert rerank._NONE_OPTION in q.choices
        subtypes = {t.removeprefix("subtype=") for t in q.tags if t.startswith("subtype=")}
        assert subtypes <= {"answerable", "abstention"} and len(subtypes) == 1
        difficulties = {
            t.removeprefix("difficulty=") for t in q.tags if t.startswith("difficulty=")
        }
        assert difficulties <= {"medium", "adversarial"} and len(difficulties) == 1
        assert q.answer_source == "librarian:rerank"


def test_answers_in_range() -> None:
    pack = rerank.build(0)
    for q in pack.questions:
        assert q.choices is not None
        assert len(q.answer) == 1 and q.answer in _LETTERS
        idx = _LETTERS.index(q.answer)
        assert 0 <= idx < len(q.choices)


def test_scorer_round_trip() -> None:
    """Each gold answer scores True; any other letter scores False."""
    pack = rerank.build(0)
    for q in pack.questions:
        assert q.choices is not None
        assert score_answer(f"Final Answer: {q.answer}", q.answer, AnswerType.MULTIPLE_CHOICE)
        # A different in-range letter must score False.
        for i in range(len(q.choices)):
            other = _LETTERS[i]
            if other == q.answer:
                continue
            assert not score_answer(f"Final Answer: {other}", q.answer, AnswerType.MULTIPLE_CHOICE)
            break


def test_gold_sanity() -> None:
    """The gold choice is the planted correct snippet, or the abstention option."""
    correct_set = {item.correct for item in rerank._ITEM_BANK}
    for seed in (0, 1, 7, 42):
        pack = rerank.build(seed)
        for q in pack.questions:
            assert q.choices is not None
            gold_choice = q.choices[_LETTERS.index(q.answer)]
            is_abstention = "subtype=abstention" in q.tags
            if is_abstention:
                # No snippet answers: gold must be the None option.
                assert gold_choice == rerank._NONE_OPTION
            else:
                # The gold choice is one of the planted correct snippets.
                assert gold_choice in correct_set
            assert q.choices.count(gold_choice) == 1


def test_abstention_negative_controls_are_balanced() -> None:
    """Every pack has at least one abstention and one answerable question, and the
    abstention questions' gold is exactly the None-of-these option."""
    for seed in (0, 1, 2, 7, 42, 99):
        pack = rerank.build(seed)
        abstention = [q for q in pack.questions if "subtype=abstention" in q.tags]
        answerable = [q for q in pack.questions if "subtype=answerable" in q.tags]
        assert abstention, f"seed {seed}: no abstention negative controls"
        assert answerable, f"seed {seed}: no answerable questions"
        assert len(abstention) + len(answerable) == len(pack.questions)
        for q in abstention:
            assert q.choices is not None
            assert q.choices[_LETTERS.index(q.answer)] == rerank._NONE_OPTION
            # The planted correct snippet for this item is absent.
            assert not any(c in {it.correct for it in rerank._ITEM_BANK} for c in q.choices)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 17, 99])
def test_count_bounds_across_seeds(seed: int) -> None:
    pack = rerank.build(seed)
    assert 10 <= len(pack.questions) <= 16
