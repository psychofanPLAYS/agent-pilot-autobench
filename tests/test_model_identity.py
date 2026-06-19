from pathlib import Path
import subprocess

import pytest

from gguf_limit_bench.model_identity import (
    IdentityConfidence,
    parse_lm_studio_inventory,
    read_lm_studio_inventory,
    resolve_path_identity,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_resolve_lm_studio_layout_to_exact_repo_and_file():
    path = Path(
        r"G:\AI\models\LM_Studio-gguf\bytkim\Qwen3.6-27B-MTP-pi-reasoning-GGUF"
        r"\Qwen3.6-27B-MTP-pi-reasoning-Q5_K_M.gguf"
    )

    identity = resolve_path_identity(path)

    assert identity.repo_id == "bytkim/Qwen3.6-27B-MTP-pi-reasoning-GGUF"
    assert identity.filename == path.name
    assert identity.confidence is IdentityConfidence.CANDIDATE
    assert identity.source == "lm_studio_path"


def test_plain_filename_stays_unresolved():
    identity = resolve_path_identity(Path("Qwen3.6-27B-Q5_K_M.gguf"))

    assert identity.repo_id is None
    assert identity.confidence is IdentityConfidence.UNRESOLVED


def test_parse_lm_studio_inventory_preserves_exact_indexed_identifier():
    payload = (FIXTURES / "lms-models.json").read_text(encoding="utf-8")

    inventory = parse_lm_studio_inventory(payload)
    pi = inventory.models["qwen3.6-27b-mtp-pi-reasoning"]

    assert pi.repo_id == "bytkim/Qwen3.6-27B-MTP-pi-reasoning-GGUF"
    assert pi.filename == "Qwen3.6-27B-MTP-pi-reasoning-Q5_K_M.gguf"
    assert pi.max_context_length == 262144
    assert pi.trained_for_tool_use is True
    assert inventory.diagnostics == ()


def test_parse_lm_studio_inventory_reports_invalid_rows_without_aborting():
    inventory = parse_lm_studio_inventory('[{"modelKey": "broken"}, 4]')

    assert inventory.models == {}
    assert len(inventory.diagnostics) == 2


def test_read_lm_studio_inventory_uses_explicit_command(monkeypatch):
    observed: list[object] = []

    def fake_run(command, **kwargs):
        observed.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")

    monkeypatch.setattr("gguf_limit_bench.model_identity.subprocess.run", fake_run)

    assert read_lm_studio_inventory(("fake-lms", "models")) == "[]"
    assert observed[0][0] == ("fake-lms", "models")


def test_read_lm_studio_inventory_surfaces_cli_failure(monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="service unavailable")

    monkeypatch.setattr("gguf_limit_bench.model_identity.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="service unavailable"):
        read_lm_studio_inventory()
