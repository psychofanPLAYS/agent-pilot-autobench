from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from gguf_limit_bench.hf_catalog import HubCatalog


FIXTURES = Path(__file__).parent / "fixtures" / "hf"
PI_REPO = "bytkim/Qwen3.6-27B-MTP-pi-reasoning-GGUF"
PI_FILENAME = "Qwen3.6-27B-MTP-pi-reasoning-Q5_K_M.gguf"


@dataclass(frozen=True)
class FakeSibling:
    rfilename: str


@dataclass(frozen=True)
class FakeModelInfo:
    id: str
    sha: str
    siblings: list[FakeSibling]
    pipeline_tag: str
    library_name: str
    card_data: dict
    last_modified: str


class FakeHubGateway:
    def __init__(self, info: FakeModelInfo, readme: str) -> None:
        self.info = info
        self.readme = readme
        self.card_revisions: list[str] = []

    def model_info(self, repo_id: str) -> FakeModelInfo:
        assert repo_id == self.info.id
        return self.info

    def model_card(self, repo_id: str, revision: str) -> str:
        assert repo_id == self.info.id
        self.card_revisions.append(revision)
        return self.readme


def fake_gateway() -> FakeHubGateway:
    payload = json.loads((FIXTURES / "pi-reasoning-model-info.json").read_text(encoding="utf-8"))
    info = FakeModelInfo(
        id=payload["id"],
        sha=payload["sha"],
        siblings=[FakeSibling(row["rfilename"]) for row in payload["siblings"]],
        pipeline_tag=payload["pipeline_tag"],
        library_name=payload["library_name"],
        card_data=payload["cardData"],
        last_modified=payload["lastModified"],
    )
    readme = (FIXTURES / "pi-reasoning-readme.md").read_text(encoding="utf-8")
    return FakeHubGateway(info, readme)


def test_fetch_pins_model_card_to_returned_revision(tmp_path):
    gateway = fake_gateway()

    record = HubCatalog(gateway=gateway, cache_root=tmp_path).fetch(PI_REPO, PI_FILENAME)

    assert record.revision == "cc2a865cdb2dd229fbd0ee89d00f9dcb3db0c3bf"
    assert record.filename_verified is True
    assert record.document_confidence == "verified"
    assert gateway.card_revisions == [record.revision]
    record_path = (
        tmp_path / "bytkim--Qwen3.6-27B-MTP-pi-reasoning-GGUF" / record.revision / "record.json"
    )
    assert record_path.exists()


def test_offline_load_uses_latest_pinned_cache(tmp_path):
    online = HubCatalog(gateway=fake_gateway(), cache_root=tmp_path)
    cached = online.fetch(PI_REPO, PI_FILENAME)

    record = HubCatalog(gateway=None, cache_root=tmp_path, offline=True).load(PI_REPO)

    assert record.revision == cached.revision
    assert record.document_confidence == "cached"
    assert record.readme.startswith("---")


def test_missing_filename_does_not_produce_verified_identity(tmp_path):
    gateway = fake_gateway()

    record = HubCatalog(gateway=gateway, cache_root=tmp_path).fetch(
        PI_REPO, "not-in-repository.gguf"
    )

    assert record.filename_verified is False
    assert record.identity_confidence == "candidate"


def test_offline_load_without_cache_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="No cached Hugging Face evidence"):
        HubCatalog(gateway=None, cache_root=tmp_path, offline=True).load(PI_REPO)


def test_missing_readme_is_partial_document_evidence(tmp_path):
    gateway = fake_gateway()
    gateway.readme = ""

    record = HubCatalog(gateway=gateway, cache_root=tmp_path).fetch(PI_REPO, PI_FILENAME)

    assert record.document_confidence == "partial"


def test_refresh_failure_falls_back_to_existing_cache(tmp_path):
    online = HubCatalog(gateway=fake_gateway(), cache_root=tmp_path)
    cached = online.fetch(PI_REPO, PI_FILENAME)

    class FailingGateway:
        def model_info(self, repo_id: str):
            raise OSError("network unavailable")

        def model_card(self, repo_id: str, revision: str) -> str:
            raise AssertionError("model_card should not be reached")

    refreshed = HubCatalog(gateway=FailingGateway(), cache_root=tmp_path).fetch(
        PI_REPO, PI_FILENAME
    )

    assert refreshed.revision == cached.revision
    assert refreshed.document_confidence == "cached"


def test_refresh_failure_without_cache_preserves_network_error(tmp_path):
    class FailingGateway:
        def model_info(self, repo_id: str):
            raise OSError("network unavailable")

        def model_card(self, repo_id: str, revision: str) -> str:
            raise AssertionError("model_card should not be reached")

    with pytest.raises(OSError, match="network unavailable"):
        HubCatalog(gateway=FailingGateway(), cache_root=tmp_path).fetch(PI_REPO, PI_FILENAME)
