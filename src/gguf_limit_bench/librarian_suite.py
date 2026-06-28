from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gguf_limit_bench.autoresearch import AutoresearchSettings
from gguf_limit_bench.librarian.preflight import (
    PREFLIGHT_FAILURE_CLASS,
    run_librarian_preflight,
    write_preflight_receipt,
)
from gguf_limit_bench.librarian.registry import LIBRARIAN_PACK_IDS
from gguf_limit_bench.pack_runner import run_pack_questions
from gguf_limit_bench.packs import load_pack
from gguf_limit_bench.results_report import render_results_markdown


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run librarian question packs against a live llama.cpp chat endpoint."
    )
    parser.add_argument("--model", required=True, help="Model label to record in receipts.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--score-out", required=True, type=Path)
    parser.add_argument("--settings-json", default="{}", help="Plan metadata copied into summary.")
    parser.add_argument("--sample-size", type=int, default=0, help="0 means full pack.")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--pack", action="append", dest="packs", help="Pack id; repeatable.")
    args = parser.parse_args(argv)

    settings = _load_settings(args.settings_json)
    pack_ids = tuple(args.packs or LIBRARIAN_PACK_IDS)
    summary = run_librarian_suite(
        model=args.model,
        base_url=args.base_url,
        out_dir=args.out_dir,
        pack_ids=pack_ids,
        sample_size=args.sample_size,
        timeout_seconds=args.timeout_seconds,
        settings=settings,
    )

    args.score_out.parent.mkdir(parents=True, exist_ok=True)
    args.score_out.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"librarian_bench_score": summary["librarian_bench_score"]}))


def run_librarian_suite(
    *,
    model: str,
    base_url: str,
    out_dir: Path,
    pack_ids: tuple[str, ...],
    sample_size: int = 0,
    timeout_seconds: int = 600,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_summaries: list[dict[str, Any]] = []
    settings_payload = settings or {}
    preflight = run_librarian_preflight(
        model=Path(model),
        settings=_autoresearch_settings_from_payload(settings_payload),
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    write_preflight_receipt(out_dir, preflight)
    if not preflight.ok:
        pack_summaries = [
            _preflight_failed_pack_summary(pack_id, preflight.failure) for pack_id in pack_ids
        ]
        summary = _suite_summary(
            model=model,
            base_url=base_url,
            settings=settings_payload,
            pack_summaries=pack_summaries,
            failure_class=PREFLIGHT_FAILURE_CLASS,
            failure=preflight.failure,
        )
        _write_suite_receipts(out_dir, summary)
        return summary

    for pack_id in pack_ids:
        pack = load_pack(pack_id)
        questions = list(pack.questions)
        if sample_size > 0:
            questions = questions[:sample_size]
        batch = run_pack_questions(
            pack=pack,
            questions=questions,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        prompts_by_id = {str(question.question_id): question.prompt for question in questions}
        pack_summary = _pack_summary(pack_id, pack.tier, batch, prompts_by_id)
        pack_summaries.append(pack_summary)
        (out_dir / f"{_safe_id(pack_id)}.json").write_text(
            json.dumps(pack_summary, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    summary = _suite_summary(
        model=model,
        base_url=base_url,
        settings=settings_payload,
        pack_summaries=pack_summaries,
    )
    _write_suite_receipts(out_dir, summary)
    return summary


def _write_suite_receipts(out_dir: Path, summary: dict[str, Any]) -> None:
    (out_dir / "librarian-suite-summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (out_dir / "librarian-suite.tsv").write_text(_summary_tsv(summary), encoding="utf-8")
    (out_dir / "librarian-suite.md").write_text(_summary_markdown(summary), encoding="utf-8")


def _load_settings(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--settings-json must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--settings-json must be a JSON object.")
    return payload


def _pack_summary(
    pack_id: str, tier: str, batch, prompts_by_id: dict[str, str]
) -> dict[str, Any]:
    wrong = sum(
        1 for result in batch.results if not result.correct and result.outcome != "incomplete"
    )
    return {
        "pack_id": pack_id,
        "tier": tier,
        "asked": batch.total,
        "correct": batch.correct,
        "wrong": wrong,
        "incomplete": batch.incomplete,
        "accuracy": batch.accuracy,
        "completion_rate": batch.completion_rate,
        "median_tps": batch.median_tps,
        "median_prompt_tps": batch.median_prompt_tps,
        "median_ttft_ms": batch.median_ttft_ms,
        "score": batch.score,
        "questions": [
            {
                "question_id": result.question_id,
                "prompt": prompts_by_id.get(str(result.question_id), ""),
                "expected": result.expected_answer,
                "predicted": result.predicted_answer,
                "outcome": result.outcome,
                "ttft_ms": result.ttft_ms,
                "tokens_per_second": result.tokens_per_second,
            }
            for result in batch.results
        ],
    }


def _preflight_failed_pack_summary(pack_id: str, failure: str) -> dict[str, Any]:
    return {
        "pack_id": pack_id,
        "tier": "librarian",
        "status": "preflight_fail",
        "failure_class": PREFLIGHT_FAILURE_CLASS,
        "failure": failure,
        "asked": 0,
        "correct": 0,
        "wrong": 0,
        "incomplete": 0,
        "accuracy": 0.0,
        "completion_rate": 0.0,
        "median_tps": 0.0,
        "median_prompt_tps": 0.0,
        "median_ttft_ms": None,
        "score": 0.0,
        "questions": [],
    }


def _suite_summary(
    *,
    model: str,
    base_url: str,
    settings: dict[str, Any],
    pack_summaries: list[dict[str, Any]],
    failure_class: str = "none",
    failure: str = "none",
) -> dict[str, Any]:
    asked = sum(int(pack["asked"]) for pack in pack_summaries)
    correct = sum(int(pack["correct"]) for pack in pack_summaries)
    incomplete = sum(int(pack["incomplete"]) for pack in pack_summaries)
    accuracy = correct / asked if asked else 0.0
    completion_rate = (asked - incomplete) / asked if asked else 0.0
    librarian_bench_score = accuracy * completion_rate
    return {
        "model": model,
        "base_url": base_url,
        "settings": settings,
        "packs": pack_summaries,
        "asked": asked,
        "correct": correct,
        "incomplete": incomplete,
        "accuracy": accuracy,
        "completion_rate": completion_rate,
        "librarian_bench_score": librarian_bench_score,
        "agent_bench_score": librarian_bench_score,
        "score": librarian_bench_score,
        "failure_class": failure_class,
        "failure": failure,
        "status": "preflight_fail" if failure_class == PREFLIGHT_FAILURE_CLASS else "scored",
    }


def _summary_tsv(summary: dict[str, Any]) -> str:
    lines = [
        "pack_id\tstatus\tfailure_class\tfailure\tasked\tcorrect\twrong\tincomplete\taccuracy\tcompletion_rate\tmedian_tps\tmedian_ttft_ms"
    ]
    for pack in summary["packs"]:
        lines.append(
            "\t".join(
                [
                    str(pack["pack_id"]),
                    str(pack.get("status", "scored")),
                    str(pack.get("failure_class", "none")),
                    str(pack.get("failure", "none")),
                    str(pack["asked"]),
                    str(pack["correct"]),
                    str(pack["wrong"]),
                    str(pack["incomplete"]),
                    _fmt_float(pack["accuracy"]),
                    _fmt_float(pack["completion_rate"]),
                    _fmt_float(pack["median_tps"]),
                    "" if pack["median_ttft_ms"] is None else _fmt_float(pack["median_ttft_ms"]),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _summary_markdown(summary: dict[str, Any]) -> str:
    payload = {
        "model": summary["model"],
        "selection_mode": "plan",
        "selection_seed": None,
        "sample_size": summary["asked"],
        "gpu": "",
        "recommended_flags": [],
        "packs": summary["packs"],
    }
    header = [
        f"# Librarian Suite: {summary['model']}",
        "",
        f"- Score: `{_fmt_float(summary['librarian_bench_score'])}`",
        f"- Accuracy: `{summary['correct']}/{summary['asked']}`",
        f"- Completion: `{_fmt_float(summary['completion_rate'])}`",
        "",
    ]
    return "\n".join(header) + render_results_markdown(payload)


def _fmt_float(value: float) -> str:
    return f"{float(value):.6f}"


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in value)[:80]


def _autoresearch_settings_from_payload(settings: dict[str, Any]) -> AutoresearchSettings:
    extras = settings.get("extra_server_args", ())
    if isinstance(extras, str):
        extras = tuple(extras.split())
    elif isinstance(extras, list | tuple):
        extras = tuple(str(item) for item in extras)
    else:
        extras = ()
    return AutoresearchSettings(extra_server_args=extras)


if __name__ == "__main__":
    main()
