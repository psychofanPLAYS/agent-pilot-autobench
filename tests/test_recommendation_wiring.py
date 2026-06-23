"""The autoresearch run must emit a flag recommendation, not just a leaderboard."""
from __future__ import annotations

import json
from pathlib import Path

from gguf_limit_bench.autoresearch import AttemptResult, AutoresearchLoop, AutoresearchSettings


def test_autoresearch_run_writes_flag_recommendation(tmp_path):
    profiles = {"A-accurate": (0.8, 50.0), "B-fast": (0.6, 90.0)}
    sequence = (
        AutoresearchSettings(profile_name="A-accurate", context_size=65536),
        AutoresearchSettings(profile_name="B-fast", context_size=65536),
    )

    def fake_runner(settings: AutoresearchSettings) -> AttemptResult:
        accuracy, tps = profiles[settings.profile_name]
        return AttemptResult(
            ok=True,
            generation_tokens_per_second=tps,
            prompt_tokens_per_second=800.0,
            ttft_ms=100.0,
            context_size=settings.context_size,
            failure="none",
            stdout="{}",
            stderr="",
            returncode=0,
            flag_profile=settings.profile_name,
            simple_bench_accuracy=accuracy,
        )

    loop = AutoresearchLoop(
        model=Path("G:/AI/models/Qwen3-Test-Q4_K_M.gguf"),
        runs_root=tmp_path,
        attempt_runner=fake_runner,
        budget_seconds=60,
        parallel_max=4,
        candidate_sequence=sequence,
    )
    receipt = loop.run()

    assert (receipt.path / "recommendation.md").exists()
    rec = json.loads((receipt.path / "recommendation.json").read_text(encoding="utf-8"))
    # accuracy-first default => the accurate profile wins even though it is slower
    assert rec["recommended"] == "A-accurate"
    assert rec["considered"] == 2
    assert rec["total"] == 2
    assert "A-accurate" in (receipt.path / "recommendation.md").read_text(encoding="utf-8")
