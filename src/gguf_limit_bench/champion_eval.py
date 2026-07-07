"""champion_eval: run question packs on the champion settings after autoresearch.

Provides :func:`evaluate_champion_packs`, which:

1. Launches ONE llama-server on *best_settings* (via
   :func:`~gguf_limit_bench.server_session.llama_server_session`).
2. For every requested pack ID:
   - loads the pack,
   - reads the per-(model, pack) selection cursor from the state DB,
   - selects questions with :func:`~gguf_limit_bench.question_selection.select_questions`,
   - runs them with :func:`~gguf_limit_bench.pack_runner.run_pack_questions`,
   - advances the cursor in the state DB,
   - records every question attempt.
3. Assembles a results payload with
   :func:`~gguf_limit_bench.results_report.build_results_payload` and writes it
   to *run_dir* with :func:`~gguf_limit_bench.results_report.write_results`.

The server is always torn down (``finally``).  Callers should wrap the call in
their own ``try/except`` so a failure here does not abort the outer run.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from gguf_limit_bench.autoresearch import AutoresearchSettings
from gguf_limit_bench.gpu_profiles import recommended_always_on
from gguf_limit_bench.librarian.preflight import (
    PREFLIGHT_FAILURE_CLASS,
    run_librarian_preflight,
    write_preflight_receipt,
)
from gguf_limit_bench.pack_runner import run_pack_questions
from gguf_limit_bench.packs import DEFAULT_PACKS, load_pack
from gguf_limit_bench.question_selection import select_questions
from gguf_limit_bench.results_report import build_results_payload, write_results
from gguf_limit_bench.server_session import llama_server_session
from gguf_limit_bench.state_db import (
    _init_connection,
    get_selection_cursor,
    record_question_attempt,
    set_selection_cursor,
)

_log = logging.getLogger(__name__)


def evaluate_champion_packs(
    *,
    model: Path,
    llama_server: Path,
    best_settings: AutoresearchSettings,
    run_dir: Path,
    pack_ids: tuple[str, ...] = DEFAULT_PACKS,
    sample_size: int = 5,
    selection: str = "sequential",
    seed: int | None = None,
    state_db_path: Path | None = None,
    gpu_name: str = "",
    timeout_seconds: int = 600,
    repeats: int = 3,
    budget_seconds: float | None = None,
) -> None:
    """Evaluate *pack_ids* on the best champion settings and write results.json.

    Parameters
    ----------
    model:
        Path to the GGUF model file (used as the model_key in the state DB).
    llama_server:
        Path to the llama-server executable.
    best_settings:
        The winning :class:`~gguf_limit_bench.autoresearch.AutoresearchSettings`
        from the autoresearch loop.
    run_dir:
        Receipt directory — ``results.json`` and ``results.md`` are written here.
    pack_ids:
        Question-pack IDs to evaluate.  Defaults to ``DEFAULT_PACKS``.
    sample_size:
        How many questions to select from each pack.
    selection:
        Selection mode: ``"sequential"`` (default) or ``"random"``.
    seed:
        RNG seed for ``"random"`` selection; ignored for ``"sequential"``.
    state_db_path:
        Path to the SQLite state DB.  When *None* an in-memory DB is used
        (cursors are not persisted across runs).
    gpu_name:
        GPU name string for flag recommendations (e.g. ``"RTX 4090"``).
    timeout_seconds:
        Server startup timeout forwarded to
        :func:`~gguf_limit_bench.server_session.llama_server_session`.
    repeats:
        How many times to run the selected question slice. The selection cursor
        advances once per pack, so repeats measure variance without consuming
        additional question-bank positions.
    budget_seconds:
        Optional wall-clock budget for the whole evaluation, measured from
        server-ready. Questions/repeats/packs that do not start before the
        budget runs out are skipped and the results payload is marked
        ``budget_exhausted`` — a partial-but-honest result instead of an
        unbounded run. ``None`` means no budget.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    model_key = model.name

    # Open (or create) the state-DB connection.
    if state_db_path is not None:
        state_db_path = Path(state_db_path)
        state_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(state_db_path)
    else:
        conn = sqlite3.connect(":memory:")
    _init_connection(conn)

    pack_dicts: list[dict] = []
    try:
        with llama_server_session(
            llama_server=llama_server,
            model=model,
            settings=best_settings,
            log_dir=run_dir,
            timeout_seconds=timeout_seconds,
        ) as base_url:
            if _is_librarian_eval(pack_ids):
                preflight = run_librarian_preflight(
                    model=model,
                    settings=best_settings,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                )
                write_preflight_receipt(run_dir, preflight)
                if not preflight.ok:
                    pack_dicts = [
                        _preflight_failed_pack_dict(pack_id, preflight.failure)
                        for pack_id in pack_ids
                    ]
                    return _write_results(
                        run_dir=run_dir,
                        model=model,
                        selection=selection,
                        seed=seed,
                        sample_size=sample_size,
                        gpu_name=gpu_name,
                        pack_dicts=pack_dicts,
                        repeats=repeats,
                    )
            deadline = None if budget_seconds is None else time.monotonic() + budget_seconds
            budget_exhausted = False
            for pack_id in pack_ids:
                if deadline is not None and time.monotonic() >= deadline:
                    budget_exhausted = True
                    pack_dicts.append(_budget_skipped_pack_dict(pack_id))
                    continue
                pack_dict = _eval_one_pack(
                    conn=conn,
                    base_url=base_url,
                    pack_id=pack_id,
                    model_key=model_key,
                    sample_size=sample_size,
                    selection=selection,
                    seed=seed,
                    repeats=repeats,
                    deadline=deadline,
                )
                pack_dicts.append(pack_dict)
    finally:
        conn.close()

    _write_results(
        run_dir=run_dir,
        model=model,
        selection=selection,
        seed=seed,
        sample_size=sample_size,
        gpu_name=gpu_name,
        pack_dicts=pack_dicts,
        repeats=repeats,
        budget_exhausted=budget_exhausted,
    )


def _write_results(
    *,
    run_dir: Path,
    model: Path,
    selection: str,
    seed: int | None,
    sample_size: int,
    gpu_name: str,
    pack_dicts: list[dict],
    repeats: int,
    budget_exhausted: bool = False,
) -> None:
    recommended_flags = list(recommended_always_on(gpu_name))
    payload = build_results_payload(
        model=str(model),
        selection_mode=selection,
        selection_seed=seed,
        sample_size=sample_size,
        gpu=gpu_name,
        recommended_flags=recommended_flags,
        packs=pack_dicts,
    )
    payload["repeats"] = repeats
    payload["score_contract"] = (
        "agent_bench_score = accuracy * completion_rate over scored attempts"
    )
    if budget_exhausted:
        payload["budget_exhausted"] = True
    write_results(run_dir, payload)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _eval_one_pack(
    *,
    conn: sqlite3.Connection,
    base_url: str,
    pack_id: str,
    model_key: str,
    sample_size: int,
    selection: str,
    seed: int | None,
    repeats: int,
    deadline: float | None = None,
) -> dict:
    """Run a single pack and return a pack dict suitable for build_results_payload."""
    try:
        pack = load_pack(pack_id)
    except KeyError:
        _log.warning("champion_eval: unknown pack %r — skipping", pack_id)
        return _empty_pack_dict(pack_id)

    cursor = get_selection_cursor(conn, model_key=model_key, pack_id=pack_id)
    questions_list = list(pack.questions)
    chosen, next_cursor = select_questions(
        questions_list,
        size=sample_size,
        mode=selection,
        seed=seed,
        cursor=cursor,
    )

    batch = run_pack_questions(
        pack=pack,
        questions=chosen,
        base_url=base_url,
        repeats=repeats,
        deadline_monotonic=deadline,
    )

    set_selection_cursor(conn, model_key=model_key, pack_id=pack_id, cursor=next_cursor)

    ts = datetime.now(timezone.utc).isoformat()
    for result in batch.results:
        record_question_attempt(
            conn,
            model_key=model_key,
            pack_id=pack_id,
            question_id=str(result.question_id),
            outcome=result.outcome
            if result.outcome is not None
            else ("correct" if result.correct else "wrong"),
            ts=ts,
        )

    # Build the per-pack dict for build_results_payload.
    # We look up the original question prompt from the pack questions by question_id.
    chosen_by_id = {q.question_id: q for q in chosen}
    question_dicts = []
    for result in batch.results:
        original = chosen_by_id.get(result.question_id)
        question_dicts.append(
            {
                "question_id": result.question_id,
                "prompt": original.prompt if original is not None else "",
                "expected": result.expected_answer,
                "predicted": result.predicted_answer,
                "outcome": result.outcome
                if result.outcome is not None
                else ("correct" if result.correct else "wrong"),
            }
        )

    incomplete = sum(1 for r in batch.results if getattr(r, "outcome", None) == "incomplete")
    wrong = sum(
        1 for r in batch.results if not r.correct and getattr(r, "outcome", None) != "incomplete"
    )

    return {
        "pack_id": pack_id,
        "tier": pack.tier,
        "repeats": repeats,
        "asked": len(chosen),
        "correct": batch.correct,
        "wrong": wrong,
        "incomplete": incomplete,
        "completion_rate": batch.completion_rate,
        "accuracy": batch.accuracy,
        "median_tps": batch.median_tps,
        "median_ttft_ms": batch.median_ttft_ms,
        "questions": question_dicts,
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _budget_skipped_pack_dict(pack_id: str) -> dict:
    """Pack dict for a pack that never started because the eval budget ran out."""
    payload = _empty_pack_dict(pack_id)
    payload["status"] = "skipped_budget_exhausted"
    return payload


def _empty_pack_dict(pack_id: str) -> dict:
    """Return a no-result pack dict when the pack cannot be loaded."""
    return {
        "pack_id": pack_id,
        "tier": "unknown",
        "asked": 0,
        "correct": 0,
        "wrong": 0,
        "incomplete": 0,
        "accuracy": 0.0,
        "median_tps": 0.0,
        "median_ttft_ms": None,
        "questions": [],
    }


def _is_librarian_eval(pack_ids: tuple[str, ...]) -> bool:
    return any(pack_id.startswith("librarian-") for pack_id in pack_ids)


def _preflight_failed_pack_dict(pack_id: str, failure: str) -> dict:
    payload = _empty_pack_dict(pack_id)
    payload.update(
        {
            "tier": "librarian",
            "failure_class": PREFLIGHT_FAILURE_CLASS,
            "failure": failure,
            "status": "preflight_fail",
        }
    )
    return payload
