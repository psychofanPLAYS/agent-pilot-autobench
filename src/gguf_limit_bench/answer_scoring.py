"""Answer extraction and scoring for QuestionPack responses."""

from __future__ import annotations

import re

from gguf_limit_bench.packs import AnswerType

# ---------------------------------------------------------------------------
# Number-word <-> digit mapping (zero..twenty)
# ---------------------------------------------------------------------------

_WORD_TO_DIGIT: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
_DIGIT_TO_WORD: dict[str, str] = {v: k for k, v in _WORD_TO_DIGIT.items()}

# Regex that matches any number word (whole word, case-insensitive)
_NUMBER_WORDS_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _WORD_TO_DIGIT) + r")\b",
    flags=re.IGNORECASE,
)

# Hardened MC extraction patterns (reused from simple_bench.py logic)
_MC_PRIORITY_PATTERNS = [
    re.compile(r"final\s*answer\s*(?:is)?\s*[:\-=]?[\s*()]*([A-F])\b", re.IGNORECASE),
    re.compile(r"\\boxed\{[\s*()]*([A-F])\b", re.IGNORECASE),
    re.compile(r"\banswer\s+is[\s*()]*:?[\s*()]*([A-F])\b", re.IGNORECASE),
    re.compile(r"\banswer\s*[:\-=][\s*()]*([A-F])\b", re.IGNORECASE),
    re.compile(r"\boption\s+[\s*()]*([A-F])\b", re.IGNORECASE),
]
_MC_LINE_FALLBACK = re.compile(r"(?m)^[\s*()]*([A-F])[\s*().]*$")

# Exact answer: text after the LAST "Final Answer:" up to end of that line
_EXACT_PATTERN = re.compile(r"(?i)final\s*answer\s*:\s*(.+)")


def extract_answer(text: str, answer_type: AnswerType) -> str | None:
    """Extract the answer from a model response.

    For MULTIPLE_CHOICE: returns an uppercase A-F letter, or None.
    For EXACT: returns the text after the last "Final Answer:" on its line, stripped,
               or None if no such marker is present.
    """
    if not text:
        return None

    if answer_type is AnswerType.MULTIPLE_CHOICE:
        for pattern in _MC_PRIORITY_PATTERNS:
            matches = list(pattern.finditer(text))
            if matches:
                return matches[-1].group(1).upper()
        line_matches = list(_MC_LINE_FALLBACK.finditer(text))
        if line_matches:
            return line_matches[-1].group(1).upper()
        return None

    # EXACT: last "Final Answer: ..." line
    matches = list(_EXACT_PATTERN.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def normalize_exact(s: str) -> str:
    """Normalize an exact answer string for comparison.

    Steps:
    1. Lowercase.
    2. Strip surrounding whitespace and punctuation (.,!?;:'").
    3. Map number words (zero..twenty) to digits.
    4. Collapse internal whitespace to single spaces.
    """
    s = s.lower().strip()
    # Strip surrounding punctuation
    s = s.strip(".,!?;:'\"")
    s = s.strip()
    # Map number words to digits
    s = _NUMBER_WORDS_PATTERN.sub(lambda m: _WORD_TO_DIGIT[m.group(1).lower()], s)
    # Collapse internal whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def score_answer(
    response: str,
    expected: str,
    answer_type: AnswerType,
    accept: tuple[str, ...] = (),
) -> bool:
    """Score a model response against an expected answer.

    MC: True iff extract_answer(response, MC) == expected (case-insensitive).
    EXACT:
      - Extract candidate from "Final Answer:" line; if absent, return False.
      - Build expected_set = {normalize_exact(expected)} | {normalize_exact(a) for a in accept}.
      - Return True if:
          * normalize_exact(candidate) in expected_set, OR
          * any value in expected_set appears as a whitespace-bounded token/substring
            within normalize_exact(response).
    """
    if answer_type is AnswerType.MULTIPLE_CHOICE:
        extracted = extract_answer(response, AnswerType.MULTIPLE_CHOICE)
        if extracted is None:
            return False
        return extracted.upper() == expected.strip().upper()

    # EXACT scoring
    candidate = extract_answer(response, AnswerType.EXACT)
    if candidate is None:
        return False

    norm_candidate = normalize_exact(candidate)
    expected_set = {normalize_exact(expected)} | {normalize_exact(a) for a in accept}

    if norm_candidate in expected_set:
        return True

    # Phrase-containment fallback: look in the full normalized response
    norm_response = normalize_exact(response)
    for exp_val in expected_set:
        if not exp_val:
            continue
        # Whitespace-bounded substring search
        pattern = r"(?<!\w)" + re.escape(exp_val) + r"(?!\w)"
        if re.search(pattern, norm_response):
            return True

    return False
