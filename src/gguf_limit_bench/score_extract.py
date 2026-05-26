from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SCORE_KEYS = (
    "agent_bench_score",
    "score",
    "accuracy",
    "overall_accuracy",
    "acc_norm",
    "acc",
    "pass_rate",
    "exact_match",
)


def extract_score_from_json_files(
    root: Path,
    *,
    keys: tuple[str, ...] = DEFAULT_SCORE_KEYS,
    exclude_names: tuple[str, ...] = ("score.json",),
) -> float:
    grouped: dict[str, list[float]] = {key: [] for key in keys}
    for path in sorted(root.rglob("*.json")):
        if path.name in exclude_names:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        _collect_scores(payload, grouped)

    for key in keys:
        scores = grouped[key]
        if scores:
            return sum(scores) / len(scores)
    raise ValueError(f"No numeric score found under {root}")


def _collect_scores(payload: Any, grouped: dict[str, list[float]]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = key.split(",", 1)[0]
            if normalized in grouped:
                if _is_number(value):
                    grouped[normalized].append(float(value))
                elif isinstance(value, dict):
                    nested_value = value.get("value")
                    if _is_number(nested_value):
                        grouped[normalized].append(float(nested_value))
                    _collect_scores(value, grouped)
                else:
                    _collect_scores(value, grouped)
            else:
                _collect_scores(value, grouped)
        return
    if isinstance(payload, list):
        for item in payload:
            _collect_scores(item, grouped)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a numeric benchmark score from JSON files.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--keys",
        default=",".join(DEFAULT_SCORE_KEYS),
        help="Comma-separated score-key priority order.",
    )
    args = parser.parse_args()

    keys = tuple(key.strip() for key in args.keys.split(",") if key.strip())
    score = extract_score_from_json_files(args.root, keys=keys)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"score": score}, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"score": score}, ensure_ascii=True))


if __name__ == "__main__":
    main()
