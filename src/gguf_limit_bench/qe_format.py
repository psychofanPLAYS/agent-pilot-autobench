"""Query-expansion response format scoring.

QE models should expand retrieval intent, not answer the user's question. This
module scores that contract without needing a judge model.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_LEX_LABEL = r"LEX|Lex search terms|Lex entry"
_HYDE_LABEL = r"HYDE|Hyde document|Hyde entry"
_STOP_LABEL = rf"{_LEX_LABEL}|{_HYDE_LABEL}|ANSWER|FINAL ANSWER"
_LEX_RE = re.compile(rf"(?im)^\s*({_LEX_LABEL})\s*:\s*(.+?)\s*$")
_HYDE_RE = re.compile(rf"(?ims)^\s*({_HYDE_LABEL})\s*:\s*(.+?)(?=^\s*(?:{_STOP_LABEL})\s*:|\Z)")
_STRICT_HYDE_RE = re.compile(r"(?im)^\s*HYDE\s*:")
_DIRECT_ANSWER_RE = re.compile(
    r"(?im)^\s*(?:ANSWER|FINAL\s+ANSWER)\s*:|\b(?:the\s+answer\s+is|you\s+should\s+use)\b"
)


@dataclass(frozen=True)
class QeFormatAssessment:
    """Deterministic assessment of one QE model response."""

    format_ok: bool
    lex_terms: tuple[str, ...]
    hyde: str
    answered_question: bool
    issues: tuple[str, ...]
    score: float


def assess_qe_response(
    response: str, *, min_lex_terms: int = 1, max_lex_terms: int = 3
) -> QeFormatAssessment:
    """Score one QE response for the APB QE contract.

    Valid output has:
    - a ``LEX:`` line with 1-3 comma/semicolon-separated terms
    - exactly one ``HYDE:`` section
    - no direct-answer markers
    """
    text = response or ""
    lex_terms = _extract_lex_terms(text)
    hyde_entries = [(match.group(1), match.group(2).strip()) for match in _HYDE_RE.finditer(text)]
    hyde_matches = [entry for _label, entry in hyde_entries]
    answered_question = bool(_DIRECT_ANSWER_RE.search(text))
    issues: list[str] = []
    if _has_noncanonical_labels(text):
        issues.append("noncanonical_labels")

    if len(lex_terms) < min_lex_terms:
        issues.append("missing_lex")
    if len(lex_terms) > max_lex_terms:
        issues.append("too_many_lex_terms")
    if not hyde_matches or not hyde_matches[0]:
        issues.append("missing_hyde")
    if len(list(_STRICT_HYDE_RE.finditer(text))) > 1:
        issues.append("multiple_hyde_sections")
    if answered_question:
        issues.append("direct_answer")

    score = _format_score(issues)
    warning_issues = {"noncanonical_labels", "multiple_hyde_sections"}
    critical_issues = [issue for issue in issues if issue not in warning_issues]
    return QeFormatAssessment(
        format_ok=not critical_issues,
        lex_terms=tuple(lex_terms),
        hyde=hyde_matches[-1] if hyde_matches else "",
        answered_question=answered_question,
        issues=tuple(issues),
        score=score,
    )


def summarize_qe_assessments(assessments: Iterable[QeFormatAssessment]) -> dict[str, object]:
    """Aggregate QE format assessments for receipts and reports."""
    rows = list(assessments)
    attempts = len(rows)
    valid = sum(1 for row in rows if row.format_ok)
    direct_answer_count = sum(1 for row in rows if row.answered_question)
    issue_counts: dict[str, int] = {}
    for row in rows:
        for issue in row.issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    score = sum(row.score for row in rows) / attempts if attempts else 0.0
    return {
        "attempts": attempts,
        "valid": valid,
        "format_rate": valid / attempts if attempts else 0.0,
        "direct_answer_count": direct_answer_count,
        "issue_counts": issue_counts,
        "score": score,
    }


def _extract_lex_terms(text: str) -> list[str]:
    matches = list(_LEX_RE.finditer(text))
    if not matches:
        return []
    raw = matches[-1].group(2)
    return [
        term.strip().strip("-* ") for term in re.split(r"[,;|]", raw) if term.strip().strip("-* ")
    ]


def _has_noncanonical_labels(text: str) -> bool:
    for match in list(_LEX_RE.finditer(text)) + list(_HYDE_RE.finditer(text)):
        label = match.group(1)
        if label not in {"LEX", "HYDE"}:
            return True
    return False


def _format_score(issues: list[str]) -> float:
    if not issues:
        return 1.0
    penalties = {
        "missing_lex": 0.35,
        "too_many_lex_terms": 0.20,
        "missing_hyde": 0.35,
        "multiple_hyde_sections": 0.20,
        "direct_answer": 0.45,
        "noncanonical_labels": 0.05,
    }
    penalty = sum(penalties.get(issue, 0.10) for issue in issues)
    return max(0.0, 1.0 - penalty)
