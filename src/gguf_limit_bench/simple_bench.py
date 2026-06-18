from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re


def _default_asset_path(filename: str) -> Path:
    """Resolve data shipped inside both editable installs and built wheels."""
    return Path(__file__).resolve().parent / "data" / filename


DEFAULT_SIMPLE_BENCH_PATH = _default_asset_path("simple_bench_public.json")
DEFAULT_SIMPLE_BENCH_SYSTEM_PROMPT = _default_asset_path("system_prompt.txt")


@dataclass(frozen=True)
class SimpleBenchQuestion:
    question_id: int | str
    prompt: str
    answer: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SimpleBenchQuestionResult:
    question_id: int | str
    expected_answer: str
    predicted_answer: str | None
    correct: bool
    ttft_ms: float | None
    tokens_per_second: float
    generated_tokens: int
    output_chars: int
    prompt_chars: int
    response: str
    failure: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SimpleBenchBatchResult:
    ok: bool
    score: float
    accuracy: float
    correct: int
    total: int
    median_tps: float
    min_tps: float
    median_ttft_ms: float | None
    results: list[SimpleBenchQuestionResult]
    failure: str = "none"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["results"] = [result.to_dict() for result in self.results]
        return payload


def load_simple_bench_questions(path: Path | None = None) -> list[SimpleBenchQuestion]:
    source = path or DEFAULT_SIMPLE_BENCH_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"SimpleBench dataset must be a JSON object: {source}")
    rows = payload.get("eval_data")
    if not isinstance(rows, list):
        raise ValueError(f"SimpleBench eval_data must be a list: {source}")

    questions: list[SimpleBenchQuestion] = []
    question_ids: set[int | str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"SimpleBench row {index} must be an object")
        missing = [key for key in ("question_id", "prompt", "answer") if key not in row]
        if missing:
            raise ValueError(f"SimpleBench row {index} is missing: {', '.join(missing)}")

        question_id = row["question_id"]
        if not isinstance(question_id, int | str) or isinstance(question_id, bool):
            raise ValueError(f"SimpleBench row {index} question_id must be an integer or string")
        if isinstance(question_id, str) and not question_id.strip():
            raise ValueError(f"SimpleBench row {index} question_id must be non-empty")
        if question_id in question_ids:
            raise ValueError(f"SimpleBench duplicate question_id {question_id}")

        prompt = str(row["prompt"]).strip()
        if not prompt:
            raise ValueError(f"SimpleBench row {index} must have a non-empty prompt")
        answer = str(row["answer"]).strip().upper()
        if answer not in "ABCDEF" or len(answer) != 1:
            raise ValueError(f"SimpleBench row {index} must have an answer A-F")

        question_ids.add(question_id)
        questions.append(SimpleBenchQuestion(question_id=question_id, prompt=prompt, answer=answer))
    if not questions:
        raise ValueError(f"No SimpleBench questions found in {source}")
    return questions


def load_simple_bench_system_prompt(path: Path | None = None) -> str:
    source = path or DEFAULT_SIMPLE_BENCH_SYSTEM_PROMPT
    if not source.exists():
        if path is not None:
            raise FileNotFoundError(f"SimpleBench system prompt was not found: {source}")
        return (
            "You are an expert at reasoning. Think briefly, then end with "
            "Final Answer: X where X is A, B, C, D, E, or F."
        )
    return source.read_text(encoding="utf-8").strip()


def simple_bench_prompt(system_prompt: str, question: SimpleBenchQuestion) -> str:
    return f"{system_prompt.strip()}\n\nQuestion:\n{question.prompt.strip()}\n"


def extract_final_answer(text: str) -> str | None:
    patterns = [
        r"final\s+answer\s*[:\-]\s*([A-F])\b",
        r"\banswer\s*[:\-]\s*([A-F])\b",
        r"\boption\s+([A-F])\b",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper()
    return None


def combine_simple_bench_results(
    results: list[SimpleBenchQuestionResult],
) -> SimpleBenchBatchResult:
    total = len(results)
    correct = sum(1 for result in results if result.correct)
    accuracy = correct / total if total else 0.0
    tps_values = [
        result.tokens_per_second
        for result in results
        if result.tokens_per_second > 0 and result.output_chars > 0
    ]
    ttft_values = [result.ttft_ms for result in results if result.ttft_ms is not None]
    median_tps = _median(tps_values) or 0.0
    min_tps = min(tps_values) if tps_values else 0.0
    median_ttft = _median(ttft_values)
    # Accuracy wins lexicographically. The bounded speed term is always smaller
    # than the score change from one additional correct answer.
    speed_tiebreaker = (2.0 / math.pi) * math.atan(max(0.0, median_tps))
    score = accuracy * 1000.0 + speed_tiebreaker * (1000.0 / (total + 1))
    failures = [result.failure for result in results if result.failure != "none"]
    return SimpleBenchBatchResult(
        ok=total > 0 and not failures,
        score=score,
        accuracy=accuracy,
        correct=correct,
        total=total,
        median_tps=median_tps,
        min_tps=min_tps,
        median_ttft_ms=median_ttft,
        results=results,
        failure=";".join(failures) if failures else "none",
    )


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0
