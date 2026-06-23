"""Results report: payload assembly, markdown rendering, and file writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------


def build_results_payload(
    model: str,
    selection_mode: str,
    selection_seed: int | None,
    sample_size: int,
    gpu: str,
    recommended_flags: list[str],
    packs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble and return a JSON-serialisable results payload dict.

    The returned dict round-trips through json.dumps / json.loads without loss.
    """
    payload: dict[str, Any] = {
        "model": str(model),
        "selection_mode": str(selection_mode),
        "selection_seed": selection_seed,
        "sample_size": int(sample_size),
        "gpu": str(gpu),
        "recommended_flags": [str(f) for f in recommended_flags],
        "packs": [_normalise_pack(p) for p in packs],
    }
    # Validate round-trip (catches non-serialisable values early)
    json.dumps(payload)
    return payload


def _normalise_pack(pack: dict[str, Any]) -> dict[str, Any]:
    questions = [
        {
            "question_id": q["question_id"],
            "prompt": str(q.get("prompt", "")),
            "expected": str(q.get("expected", "")),
            "predicted": q.get("predicted"),  # may be None
            "outcome": str(q.get("outcome", "")),
        }
        for q in pack.get("questions", [])
    ]
    return {
        "pack_id": str(pack["pack_id"]),
        "tier": str(pack.get("tier", "")),
        "asked": int(pack.get("asked", 0)),
        "correct": int(pack.get("correct", 0)),
        "wrong": int(pack.get("wrong", 0)),
        "incomplete": int(pack.get("incomplete", 0)),
        "accuracy": float(pack.get("accuracy", 0.0)),
        "median_tps": float(pack.get("median_tps", 0.0)),
        "median_ttft_ms": _float_or_none(pack.get("median_ttft_ms")),
        "questions": questions,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

_PROMPT_TRUNCATE = 60


def render_results_markdown(payload: dict[str, Any]) -> str:
    """Render a human-readable markdown report from a results payload."""
    lines: list[str] = []

    # Title
    lines.append(f"# Results: {payload['model']}")
    lines.append("")

    # Summary line
    flags_str = " ".join(payload.get("recommended_flags", [])) or "(none)"
    seed_str = str(payload.get("selection_seed")) if payload.get("selection_seed") is not None else "n/a"
    lines.append(
        f"**GPU:** {payload['gpu']}  "
        f"**Mode:** {payload['selection_mode']} (seed {seed_str})  "
        f"**Sample:** {payload['sample_size']}  "
        f"**Flags:** `{flags_str}`"
    )
    lines.append("")

    for pack in payload.get("packs", []):
        pack_id = pack["pack_id"]
        tier = pack["tier"]
        asked = pack["asked"]
        correct = pack["correct"]
        incomplete = pack["incomplete"]

        # Pack header: e.g.  ## easy-gotcha  easy  score 1/2 (incomplete 1)
        header = f"## {pack_id}  [{tier}]  score {correct}/{asked}"
        if incomplete > 0:
            header += f"  (incomplete {incomplete})"
        lines.append(header)
        lines.append("")

        # Per-question table
        lines.append("| question_id | prompt | expected | predicted | outcome |")
        lines.append("| --- | --- | --- | --- | --- |")
        for q in pack.get("questions", []):
            prompt_short = str(q["prompt"])[:_PROMPT_TRUNCATE]
            if len(str(q["prompt"])) > _PROMPT_TRUNCATE:
                prompt_short += "…"
            predicted_str = str(q["predicted"]) if q["predicted"] is not None else "(none)"
            lines.append(
                f"| {q['question_id']} | {prompt_short} | {q['expected']} "
                f"| {predicted_str} | {q['outcome']} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File writer
# ---------------------------------------------------------------------------


def write_results(run_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    """Write results.json and results.md into *run_dir*; return both paths."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "results.json"
    md_path = run_dir / "results.md"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_results_markdown(payload), encoding="utf-8")

    return json_path, md_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)
