from pathlib import Path


def test_required_benchmark_suite_phase_is_documented():
    doc = Path("docs/BENCHMARK-SUITE-PHASE.md").read_text(encoding="utf-8")

    required_terms = [
        "Karpathy Autoresearch Contract",
        "lm-evaluation-harness",
        "BFCL",
        "SWE-bench",
        "tau-bench",
        "runs\\benchmark-suite.tsv",
        "runs\\agentic-suite.tsv",
        "runs\\agent-bench-score.tsv",
        "runs\\autoresearch-attempts.tsv",
        "inspect-ai",
        "keep",
        "discard",
        "crash",
        "production-ready",
    ]

    for term in required_terms:
        assert term in doc


def test_public_docs_block_production_ready_claim_until_suite_exists():
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            Path("README.md"),
            Path("docs/AUTORESEARCH-PROGRAM.md"),
            Path("docs/COMMAND-BOARD.md"),
            Path("docs/IMPLEMENTATION-PLAN.md"),
        ]
    )

    assert "docs\\BENCHMARK-SUITE-PHASE.md" in docs
    assert "runs\\benchmark-suite.tsv" in docs
    assert "runs\\agentic-suite.tsv" in docs
    assert "runs\\agent-bench-score.tsv" in docs
    assert "not production-ready" in docs or "before production-ready" in docs
    assert "keep/discard/crash" in docs


def test_current_status_vocabulary_is_canonical_in_docs():
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            Path("README.md"),
            Path("docs/AUTORESEARCH-PROGRAM.md"),
            Path("docs/BENCHMARK-SUITE-PHASE.md"),
        ]
    )

    for status in [
        "slow",
        "speed_only",
        "serving_measured",
        "context_unproven",
        "workflow_unproven",
        "workflow_weak",
        "workflow_smoke",
    ]:
        assert status in docs

    assert "agent_ready" not in Path("README.md").read_text(encoding="utf-8")
