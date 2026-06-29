from __future__ import annotations

import json
from pathlib import Path

from gguf_limit_bench.discovery import ModelInfo
from gguf_limit_bench.hf_catalog import HubRecord
from gguf_limit_bench.model_catalog import ModelCatalog, write_catalog
from gguf_limit_bench.model_identity import (
    IdentityConfidence,
    ModelIdentity,
)
from gguf_limit_bench.runtime_capabilities import parse_llama_help


README = """
```bash
llama-server -m model.gguf --spec-type draft-mtp --spec-draft-n-max 3 --temp 1.0
```
"""


class FakeHub:
    def __init__(self, *, fail_repo: str | None = None) -> None:
        self.fail_repo = fail_repo
        self.calls: list[tuple[str, str]] = []

    def fetch(self, repo_id: str, filename: str) -> HubRecord:
        self.calls.append((repo_id, filename))
        if repo_id == self.fail_repo:
            raise OSError("Hub unavailable")
        return HubRecord(
            repo_id=repo_id,
            filename=filename,
            revision="abc123",
            retrieved_at="2026-06-19T00:00:00+00:00",
            last_modified="2026-06-15T00:00:00+00:00",
            pipeline_tag="text-generation",
            library_name="gguf",
            license="apache-2.0",
            base_models=("Qwen/Qwen3.6-27B",),
            datasets=(),
            filename_verified=True,
            identity_confidence="verified",
            document_confidence="verified",
            readme=README,
            auxiliary_files={
                "generation_config.json": '{"top_p": 0.8}',
                "tokenizer_config.json": '{"chat_template": "template"}',
            },
        )


class SearchableFakeHub(FakeHub):
    def search_models(self, query: str, limit: int):
        return [{"id": "bytkim/Qwen3.6-27B-MTP-GGUF"}]


def model_info(name: str = "Qwen3.6-27B-MTP-Q5_K_M.gguf") -> ModelInfo:
    return ModelInfo(
        path=Path("G:/models/bytkim/Qwen3.6-27B-MTP-GGUF") / name,
        name=name,
        family="qwen",
        parameters="27B",
        quant="Q5_K_M",
        size_bytes=20_000_000_000,
        has_mtp=True,
        identity=ModelIdentity(
            repo_id="bytkim/Qwen3.6-27B-MTP-GGUF",
            filename=name,
            confidence=IdentityConfidence.CANDIDATE,
            source="lm_studio_path",
        ),
    )


def test_catalog_keeps_three_confidence_dimensions(tmp_path):
    catalog = ModelCatalog(cache_root=tmp_path, hub=FakeHub()).build([model_info()], enrich=True)

    entry = catalog.entries[0]

    assert entry.identity_confidence == "verified"
    assert entry.document_confidence == "verified"
    assert entry.recommendations[0].confidence == "publisher_claim"
    assert entry.recommendations[0].key == "spec_type"
    assert catalog.network_used is True


def test_catalog_keeps_other_models_when_one_enrichment_fails(tmp_path):
    failed = model_info("failed-Q5_K_M.gguf")
    failed = ModelInfo(
        **{
            **failed.__dict__,
            "identity": ModelIdentity(
                repo_id="bytkim/failing-repo",
                filename=failed.name,
                confidence=IdentityConfidence.CANDIDATE,
                source="lm_studio_path",
            ),
        }
    )
    good = model_info("good-Q5_K_M.gguf")

    catalog = ModelCatalog(
        cache_root=tmp_path,
        hub=FakeHub(fail_repo="bytkim/failing-repo"),
    ).build([failed, good], enrich=True)

    assert len(catalog.entries) == 2
    assert catalog.entries[0].errors or catalog.entries[1].errors
    assert any(not entry.errors for entry in catalog.entries)


def test_catalog_exports_stable_json_and_markdown(tmp_path):
    snapshot = ModelCatalog(cache_root=tmp_path, hub=FakeHub()).build([model_info()], enrich=True)

    paths = write_catalog(snapshot, tmp_path / "exports")
    payload = json.loads(paths.json.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["entries"][0]["repo_id"] == "bytkim/Qwen3.6-27B-MTP-GGUF"
    markdown = paths.markdown.read_text(encoding="utf-8")
    assert "Identity confidence" in markdown
    assert "publisher_claim" in markdown
    recommendations = json.loads(paths.recommendations.read_text(encoding="utf-8"))
    entry = recommendations["entries"][0]
    assert entry["model"]["repo_id"] == "bytkim/Qwen3.6-27B-MTP-GGUF"
    assert entry["values"]["spec_type"] == "draft-mtp"
    assert entry["recommendations"]
    matches = json.loads(paths.matches.read_text(encoding="utf-8"))
    assert matches["entries"][0]["selected_repo_id"] == "bytkim/Qwen3.6-27B-MTP-GGUF"
    assert matches["entries"][0]["candidates"]


def test_local_only_catalog_never_calls_hub(tmp_path):
    hub = FakeHub()

    snapshot = ModelCatalog(cache_root=tmp_path, hub=hub).build([model_info()], enrich=False)

    assert hub.calls == []
    assert snapshot.network_used is False
    assert snapshot.entries[0].document_confidence == "unavailable"


def test_catalog_enrich_searches_even_when_filename_has_no_repo_identity(tmp_path):
    model = ModelInfo(
        path=Path("G:/models/Qwen3.6-27B-MTP-Q5_K_M.gguf"),
        name="Qwen3.6-27B-MTP-Q5_K_M.gguf",
        family="qwen",
        parameters="27B",
        quant="Q5_K_M",
        size_bytes=20_000_000_000,
        has_mtp=True,
        identity=ModelIdentity(
            repo_id=None,
            filename="Qwen3.6-27B-MTP-Q5_K_M.gguf",
            confidence=IdentityConfidence.UNRESOLVED,
            source="filename",
        ),
    )

    snapshot = ModelCatalog(cache_root=tmp_path, hub=SearchableFakeHub()).build(
        [model],
        enrich=True,
    )
    entry = snapshot.entries[0]

    assert entry.repo_id == "bytkim/Qwen3.6-27B-MTP-GGUF"
    assert entry.source_repo_id is None
    assert entry.match_confidence in {"strong", "verified"}
    assert entry.match_candidates


def test_catalog_promotes_only_locally_supported_recommendations(tmp_path):
    capabilities = parse_llama_help(
        "version: 9596 (18ef86ece)",
        """
--spec-type none,draft-mtp              speculative decoding type
--spec-draft-n-max N                    number of draft tokens
--temp N                                sampling temperature
""",
    )

    snapshot = ModelCatalog(
        cache_root=tmp_path,
        hub=FakeHub(),
        capabilities=capabilities,
    ).build([model_info()], enrich=True)
    recommendations = {item.key: item for item in snapshot.entries[0].recommendations}

    assert recommendations["spec_type"].confidence == "locally_validated"
    assert recommendations["spec_draft_n_max"].local_validation == "supported"
    assert recommendations["temperature"].confidence == "locally_validated"
