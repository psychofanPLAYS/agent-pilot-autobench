"""Pack runner: ask QuestionPack questions with forced-final follow-up.

This module provides :func:`run_pack_questions`, which runs a list of
:class:`~gguf_limit_bench.packs.PackQuestion` items against a running
llama.cpp-compatible chat server and classifies each answer as
``"correct"``, ``"wrong"``, or ``"incomplete"`` using one forced-final
follow-up turn when the first response contains no extractable answer.

Back-compat note
----------------
The existing :mod:`gguf_limit_bench.simple_bench_runner` public API is
**not touched**.  ``LlamaServerSimpleBenchAttemptRunner`` and
``measure_simple_bench_completion`` remain the canonical entry-points for
the simple-bench path; callers such as ``autoresearch.py`` and ``cli.py``
continue to import from there unchanged.
"""

from __future__ import annotations

import json
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from gguf_limit_bench.answer_scoring import extract_answer, score_answer
from gguf_limit_bench.packs import PackQuestion, QuestionPack
from gguf_limit_bench.server_probe import iter_llama_completion_stream_events
from gguf_limit_bench.simple_bench import (
    SimpleBenchBatchResult,
    SimpleBenchQuestionResult,
)

_FORCED_FINAL_INSTRUCTION = "Reply with ONLY your final answer in the form 'Final Answer: X'."
_FORCED_FINAL_MAX_TOKENS = 64

# 0 (or any value <= 0) means "do not cap the answer" — let a reasoning model
# think for as long as it needs and stop on its own. The per-request timeout is
# the only bound, so a model that never stops still can't hang the run.
UNLIMITED_THINKING = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pack_questions(
    *,
    pack: QuestionPack,
    questions: list[PackQuestion],
    answer_max_tokens: int = UNLIMITED_THINKING,
    base_url: str,
    timeout_seconds: int = 600,
) -> SimpleBenchBatchResult:
    """Run *questions* from *pack* and return a scored batch result.

    Parameters
    ----------
    pack:
        The :class:`~gguf_limit_bench.packs.QuestionPack` that owns the
        questions (provides ``answer_type`` and ``system_prompt``).
    questions:
        The subset of questions to run (typically the full
        ``pack.questions`` or a sampled slice).
    answer_max_tokens:
        Token budget for the primary response.  Defaults to unlimited
        (``UNLIMITED_THINKING``) so reasoning models are never truncated
        mid-thought; the per-request timeout is the only bound.
    base_url:
        Base URL of the running llama-server, e.g.
        ``"http://127.0.0.1:8080"``.
    timeout_seconds:
        Per-request HTTP timeout.
    """
    results: list[SimpleBenchQuestionResult] = []

    for question in questions:
        result = _run_one_question(
            pack=pack,
            question=question,
            answer_max_tokens=answer_max_tokens,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        results.append(result)

    return _aggregate(results)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_one_question(
    *,
    pack: QuestionPack,
    question: PackQuestion,
    answer_max_tokens: int,
    base_url: str,
    timeout_seconds: int,
) -> SimpleBenchQuestionResult:
    """Run a single question with at most one forced-final follow-up."""
    system_prompt = pack.system_prompt
    answer_type = pack.answer_type

    # ---- Primary turn ------------------------------------------------
    primary_text, ttft_ms, tps, prompt_tps, generated_tokens = _chat(
        base_url=base_url,
        system_prompt=system_prompt,
        user_content=question.prompt,
        max_tokens=answer_max_tokens,
        timeout_seconds=timeout_seconds,
    )

    extracted = extract_answer(primary_text, answer_type)

    # ---- Forced-final follow-up (if needed) --------------------------
    if extracted is None:
        followup_user = f"{primary_text}\n\n{_FORCED_FINAL_INSTRUCTION}"
        followup_text, _, _, _, _ = _chat(
            base_url=base_url,
            system_prompt=system_prompt,
            user_content=followup_user,
            max_tokens=_FORCED_FINAL_MAX_TOKENS,
            timeout_seconds=timeout_seconds,
        )
        extracted = extract_answer(followup_text, answer_type)

    # ---- Classify outcome --------------------------------------------
    if extracted is None:
        outcome = "incomplete"
        correct = False
    elif score_answer(
        f"Final Answer: {extracted}",
        question.answer,
        answer_type,
        accept=question.accept,
    ):
        outcome = "correct"
        correct = True
    else:
        outcome = "wrong"
        correct = False

    return SimpleBenchQuestionResult(
        question_id=question.question_id,
        expected_answer=question.answer,
        predicted_answer=extracted,
        correct=correct,
        ttft_ms=ttft_ms,
        tokens_per_second=tps,
        generated_tokens=generated_tokens,
        output_chars=len(primary_text),
        prompt_chars=len(system_prompt) + len(question.prompt),
        response=primary_text,
        prompt_tokens_per_second=prompt_tps,
        failure="none",
        outcome=outcome,
    )


def _chat(
    *,
    base_url: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    timeout_seconds: int,
) -> tuple[str, float | None, float, float, int]:
    """Send a single chat completion request and return
    ``(response_text, ttft_ms, tps, prompt_tps, generated_tokens)``.

    On network error returns empty text with zero metrics.
    """
    body: dict[str, object] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": True,
        "temperature": 0,
    }
    if max_tokens > 0:
        body["max_tokens"] = max_tokens
    else:
        # llama.cpp: n_predict = -1 means generate until EOS (let it think).
        body["n_predict"] = -1
    payload = json.dumps(body).encode("utf-8")
    request = Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    first_token_at: float | None = None
    content_parts: list[str] = []
    generated_tokens = 0
    fallback_chunks = 0
    server_tps: float | None = None
    server_prompt_tps: float | None = None
    usage_tokens: int | None = None

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            for event in iter_llama_completion_stream_events(response):
                timings = event.get("timings")
                if isinstance(timings, dict):
                    if timings.get("predicted_per_second") is not None:
                        server_tps = float(timings["predicted_per_second"])
                    if timings.get("prompt_per_second") is not None:
                        server_prompt_tps = float(timings["prompt_per_second"])
                usage = event.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                    usage_tokens = int(usage["completion_tokens"])
                parts_this_event: list[str] = []
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    for field in ("reasoning_content", "content"):
                        value = delta.get(field, "")
                        if isinstance(value, str):
                            parts_this_event.append(value)
                chunk = "".join(parts_this_event)
                # Count tokens via the tokens list if available, else 0
                token_count = _event_token_count(event)
                if token_count <= 0 and not chunk:
                    continue
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                generated_tokens += token_count
                fallback_chunks += 1
                content_parts.append(chunk)
    except (OSError, URLError):
        return "", None, 0.0, 0.0, 0

    finished = time.perf_counter()
    response_text = "".join(content_parts)

    if first_token_at is None:
        return response_text, None, 0.0, server_prompt_tps or 0.0, 0

    measured = usage_tokens or generated_tokens or fallback_chunks
    gen_seconds = max(finished - first_token_at, 0.001)
    tps = server_tps or (measured / gen_seconds)
    ttft_ms = (first_token_at - started) * 1000.0
    return response_text, ttft_ms, tps, server_prompt_tps or 0.0, measured


def _event_token_count(event: dict) -> int:
    tokens = event.get("tokens")
    if isinstance(tokens, list):
        return len(tokens)
    return 0


def _aggregate(results: list[SimpleBenchQuestionResult]) -> SimpleBenchBatchResult:
    """Build a :class:`SimpleBenchBatchResult` with outcome-aware fields."""
    import math

    total = len(results)
    correct_count = sum(1 for r in results if r.correct)
    incomplete_count = sum(1 for r in results if r.outcome == "incomplete")
    answered = total - incomplete_count  # correct + wrong
    accuracy = correct_count / total if total else 0.0
    completion_rate = answered / total if total else 0.0

    tps_values = [
        r.tokens_per_second for r in results if r.tokens_per_second > 0 and r.output_chars > 0
    ]
    prompt_tps_values = [
        r.prompt_tokens_per_second for r in results if r.prompt_tokens_per_second > 0
    ]
    ttft_values = [r.ttft_ms for r in results if r.ttft_ms is not None]

    median_tps = _median(tps_values) or 0.0
    min_tps = min(tps_values) if tps_values else 0.0
    median_ttft = _median(ttft_values)

    speed_tiebreaker = (2.0 / math.pi) * math.atan(max(0.0, median_tps))
    score = accuracy * 1000.0 + speed_tiebreaker * (1000.0 / (total + 1))

    failures = [r.failure for r in results if r.failure != "none"]

    return SimpleBenchBatchResult(
        ok=total > 0 and not failures,
        score=score,
        accuracy=accuracy,
        correct=correct_count,
        total=total,
        median_tps=median_tps,
        min_tps=min_tps,
        median_ttft_ms=median_ttft,
        results=results,
        median_prompt_tps=_median(prompt_tps_values) or 0.0,
        failure=";".join(failures) if failures else "none",
        incomplete=incomplete_count,
        completion_rate=completion_rate,
    )


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
