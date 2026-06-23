from __future__ import annotations

from gguf_limit_bench.question_selection import select_questions


def test_sequential_cursor_0_size_5():
    questions = list(range(10))
    result, new_cursor = select_questions(questions, size=5, mode="sequential", cursor=0)
    assert result == [0, 1, 2, 3, 4]
    assert new_cursor == 5


def test_sequential_cursor_8_wraps_around():
    questions = list(range(10))
    result, new_cursor = select_questions(questions, size=5, mode="sequential", cursor=8)
    assert result == [8, 9, 0, 1, 2]
    assert new_cursor == 3


def test_sequential_size_exceeds_length_returns_all_once():
    questions = list(range(10))
    result, new_cursor = select_questions(questions, size=99, mode="sequential", cursor=0)
    assert sorted(result) == list(range(10))
    assert len(result) == 10
    assert new_cursor == 0  # (0 + 10) % 10 == 0


def test_random_is_deterministic_with_seed():
    questions = list(range(20))
    result_a, _ = select_questions(questions, size=5, mode="random", seed=42)
    result_b, _ = select_questions(questions, size=5, mode="random", seed=42)
    assert result_a == result_b
    assert len(result_a) == 5
    assert len(set(result_a)) == 5  # distinct


def test_random_different_seeds_give_different_results():
    questions = list(range(20))
    result_42, _ = select_questions(questions, size=5, mode="random", seed=42)
    result_99, _ = select_questions(questions, size=5, mode="random", seed=99)
    # Different seeds should (almost certainly) produce different orderings
    assert result_42 != result_99


def test_random_size_exceeds_length_is_clamped():
    questions = list(range(5))
    result, _ = select_questions(questions, size=99, mode="random", seed=0)
    assert len(result) == 5
    assert sorted(result) == list(range(5))


def test_empty_list_returns_empty_and_cursor_zero():
    result, new_cursor = select_questions([], size=5, mode="sequential", cursor=0)
    assert result == []
    assert new_cursor == 0

    result2, new_cursor2 = select_questions([], size=5, mode="random", seed=42)
    assert result2 == []
    assert new_cursor2 == 0
