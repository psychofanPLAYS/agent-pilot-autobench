from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gguf_limit_bench.hf_catalog import HubRecord
from gguf_limit_bench.hf_model_match import (
    build_match_queries,
    resolve_hf_model_match,
    score_match_record,
)


@dataclass(frozen=True)
class SearchResult:
    id: str


def record(repo_id: str, *, filename_verified: bool, base_models: tuple[str, ...] = ()):
    return HubRecord(
        repo_id=repo_id,
        filename="Qwen3-4B-Q4_K_M.gguf",
        revision="abc",
        retrieved_at="2026-06-29T00:00:00+00:00",
        last_modified=None,
        pipeline_tag="text-generation",
        library_name="gguf",
        license="apache-2.0",
        base_models=base_models,
        datasets=(),
        filename_verified=filename_verified,
        identity_confidence="verified" if filename_verified else "candidate",
        document_confidence="verified",
        readme="# card",
    )


class FakeMatchHub:
    def __init__(self) -> None:
        self.records = {
            "Qwen/Qwen3-4B": record(
                "Qwen/Qwen3-4B",
                filename_verified=False,
            ),
            "unsloth/Qwen3-4B-GGUF": record(
                "unsloth/Qwen3-4B-GGUF",
                filename_verified=True,
                base_models=("Qwen/Qwen3-4B",),
            ),
        }
        self.queries: list[str] = []

    def search_models(self, query: str, limit: int):
        self.queries.append(query)
        return [SearchResult("Qwen/Qwen3-4B"), SearchResult("unsloth/Qwen3-4B-GGUF")]

    def fetch(self, repo_id: str, filename: str) -> HubRecord:
        assert filename == "Qwen3-4B-Q4_K_M.gguf"
        return self.records[repo_id]


def test_match_queries_learn_from_folder_filename_and_source_repo():
    queries = build_match_queries(
        model_path=Path("G:/models/LM_Studio-gguf/unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"),
        filename="Qwen3-4B-Q4_K_M.gguf",
        source_repo_id="unsloth/Qwen3-4B-GGUF",
    )

    assert "unsloth/Qwen3-4B-GGUF" in queries
    assert "Qwen3-4B-GGUF" in queries
    assert "Qwen3-4B-Q4_K_M.gguf" in queries
    assert "Qwen3-4B" in queries


def test_matcher_prefers_verified_quant_repo_over_base_publisher():
    hub = FakeMatchHub()

    decision, selected = resolve_hf_model_match(
        hub=hub,
        model_path=Path("G:/models/Qwen3-4B-Q4_K_M.gguf"),
        filename="Qwen3-4B-Q4_K_M.gguf",
        source_repo_id="Qwen/Qwen3-4B",
    )

    assert selected is not None
    assert selected.repo_id == "unsloth/Qwen3-4B-GGUF"
    assert decision.selected_repo_id == "unsloth/Qwen3-4B-GGUF"
    assert decision.source_repo_id == "Qwen/Qwen3-4B"
    assert decision.confidence == "verified"
    assert decision.candidates[0].repo_id == "unsloth/Qwen3-4B-GGUF"
    assert "gguf_filename_verified" in decision.candidates[0].reasons


def test_official_base_publisher_is_not_treated_as_equal_to_quant_repo():
    base = score_match_record(
        record("Qwen/Qwen3-4B", filename_verified=False),
        model_path=Path("G:/models/Qwen3-4B-Q4_K_M.gguf"),
        filename="Qwen3-4B-Q4_K_M.gguf",
        source_repo_id="Qwen/Qwen3-4B",
    )
    quant = score_match_record(
        record(
            "unsloth/Qwen3-4B-GGUF",
            filename_verified=True,
            base_models=("Qwen/Qwen3-4B",),
        ),
        model_path=Path("G:/models/Qwen3-4B-Q4_K_M.gguf"),
        filename="Qwen3-4B-Q4_K_M.gguf",
        source_repo_id="Qwen/Qwen3-4B",
    )

    assert quant.score > base.score
