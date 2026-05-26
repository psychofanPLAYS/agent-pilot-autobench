from __future__ import annotations

from pathlib import Path

from gguf_limit_bench.score_extract import extract_score_from_json_files


def extract_inspect_score(log_dir: Path) -> float:
    return extract_score_from_json_files(log_dir, keys=("accuracy", "score", "acc"))


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Extract a numeric score from Inspect JSON logs.")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    score = extract_inspect_score(args.log_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"score": score}, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"score": score}, ensure_ascii=True))


if __name__ == "__main__":
    main()
