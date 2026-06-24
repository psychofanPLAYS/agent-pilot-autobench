from __future__ import annotations


from gguf_limit_bench.answer_scoring import (
    AnswerType,
    extract_answer,
    normalize_exact,
    score_answer,
)


# ---------------------------------------------------------------------------
# extract_answer — MC
# ---------------------------------------------------------------------------


def test_extract_answer_mc_final_answer_marker():
    assert extract_answer("Reasoning. Final Answer: C", AnswerType.MULTIPLE_CHOICE) == "C"


def test_extract_answer_mc_boxed():
    assert extract_answer(r"After work, \boxed{D}", AnswerType.MULTIPLE_CHOICE) == "D"


def test_extract_answer_mc_none_when_no_letter():
    assert extract_answer("I am not sure.", AnswerType.MULTIPLE_CHOICE) is None


def test_extract_answer_mc_bold_star():
    assert extract_answer("The answer is **E**", AnswerType.MULTIPLE_CHOICE) == "E"


def test_extract_answer_mc_letter_alone_on_line():
    assert extract_answer("lots of reasoning\n\nB", AnswerType.MULTIPLE_CHOICE) == "B"


# ---------------------------------------------------------------------------
# extract_answer — EXACT
# ---------------------------------------------------------------------------


def test_extract_answer_exact_after_final_answer():
    assert extract_answer("Final Answer: 3.", AnswerType.EXACT) == "3."


def test_extract_answer_exact_last_occurrence():
    text = "Final Answer: wrong\nMore thought.\nFinal Answer: correct"
    assert extract_answer(text, AnswerType.EXACT) == "correct"


def test_extract_answer_exact_none_when_absent():
    assert extract_answer("The answer is 42", AnswerType.EXACT) is None


# ---------------------------------------------------------------------------
# normalize_exact
# ---------------------------------------------------------------------------


def test_normalize_exact_number_word_three():
    assert normalize_exact("Three") == "3"


def test_normalize_exact_strips_trailing_punctuation():
    assert normalize_exact(" 9.9 ") == "9.9"


def test_normalize_exact_digit_to_word_and_back():
    # "9" → stays "9" (digits are canonical)
    assert normalize_exact("9") == "9"
    assert normalize_exact("nine") == "9"


def test_normalize_exact_collapses_internal_whitespace():
    assert normalize_exact("his  son") == "his son"


def test_normalize_exact_strips_surrounding_punctuation():
    assert normalize_exact('"hello"') == "hello"


# ---------------------------------------------------------------------------
# score_answer — MC
# ---------------------------------------------------------------------------


def test_score_answer_mc_correct_letter():
    assert score_answer("Final Answer: B", "B", AnswerType.MULTIPLE_CHOICE) is True


def test_score_answer_mc_wrong_letter():
    assert score_answer("Final Answer: A", "B", AnswerType.MULTIPLE_CHOICE) is False


def test_score_answer_mc_no_letter():
    assert score_answer("I don't know.", "B", AnswerType.MULTIPLE_CHOICE) is False


# ---------------------------------------------------------------------------
# score_answer — EXACT
# ---------------------------------------------------------------------------


def test_score_answer_exact_dot_stripped():
    assert score_answer("Final Answer: 3.", "3", AnswerType.EXACT) is True


def test_score_answer_exact_wrong():
    assert score_answer("Final Answer: 4", "3", AnswerType.EXACT) is False


def test_score_answer_exact_accept_variant():
    assert (
        score_answer("Final Answer: his son", "the son", AnswerType.EXACT, accept=("his son",))
        is True
    )


def test_score_answer_exact_phrase_containment_in_response():
    # "his son" should be found as a whitespace-bounded substring in the full response
    response = "The person is his son. Final Answer: his son"
    assert score_answer(response, "his son", AnswerType.EXACT) is True


def test_score_answer_exact_no_final_answer_false():
    assert score_answer("The answer is 3", "3", AnswerType.EXACT) is False


def test_score_answer_exact_number_word_match():
    assert score_answer("Final Answer: three", "3", AnswerType.EXACT) is True
