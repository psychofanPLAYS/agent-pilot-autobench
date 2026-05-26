import json

from gguf_limit_bench.score_extract import extract_score_from_json_files


def test_extract_score_from_lm_eval_style_metric_keys(tmp_path):
    result_dir = tmp_path / "lm-eval"
    result_dir.mkdir()
    (result_dir / "results.json").write_text(
        json.dumps({"results": {"arc_easy": {"acc,none": 0.55, "acc_norm,none": 0.61}}}),
        encoding="utf-8",
    )

    assert (
        extract_score_from_json_files(
            result_dir,
            keys=("acc_norm", "acc", "accuracy", "score"),
        )
        == 0.61
    )


def test_extract_score_ignores_boolean_metric_metadata(tmp_path):
    result_dir = tmp_path / "lm-eval"
    result_dir.mkdir()
    (result_dir / "results.json").write_text(
        json.dumps(
            {
                "results": {
                    "gsm8k_cot_zeroshot": {
                        "exact_match,strict-match": 0.0,
                        "exact_match,flexible-extract": 0.0,
                    }
                },
                "higher_is_better": {"gsm8k_cot_zeroshot": {"exact_match": True}},
            }
        ),
        encoding="utf-8",
    )

    assert extract_score_from_json_files(result_dir, keys=("exact_match", "score")) == 0.0
