import json
from pathlib import Path

from typer.testing import CliRunner

from gguf_limit_bench.cli import app
from gguf_limit_bench.model_catalog import CatalogEntry, CatalogSnapshot, write_catalog
from gguf_limit_bench.model_recommendations import Recommendation


runner = CliRunner()


def create_model(root: Path) -> Path:
    folder = root / "LM_Studio-gguf" / "bytkim" / "Qwen3.6-27B-MTP-GGUF"
    folder.mkdir(parents=True)
    model = folder / "Qwen3.6-27B-MTP-Q5_K_M.gguf"
    model.write_bytes(b"GGUF-test")
    return model


def test_models_scan_is_local_only_and_writes_catalog(tmp_path, monkeypatch):
    model = create_model(tmp_path / "models")

    def forbidden_gateway(*args, **kwargs):
        raise AssertionError("local scan must not construct a Hugging Face gateway")

    monkeypatch.setattr("gguf_limit_bench.cli.HuggingFaceGateway", forbidden_gateway)
    result = runner.invoke(
        app,
        [
            "models",
            "scan",
            "--model-root",
            str(tmp_path / "models"),
            "--cache-root",
            str(tmp_path / "catalog"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["network_used"] is False
    assert payload["entries"][0]["local_path"] == str(model)
    assert (tmp_path / "catalog" / "catalog.json").exists()


def test_models_enrich_offline_does_not_construct_network_gateway(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()

    def forbidden_gateway(*args, **kwargs):
        raise AssertionError("offline enrichment must not construct a network gateway")

    monkeypatch.setattr("gguf_limit_bench.cli.HuggingFaceGateway", forbidden_gateway)
    result = runner.invoke(
        app,
        [
            "models",
            "enrich",
            "--offline",
            "--model-root",
            str(tmp_path / "models"),
            "--cache-root",
            str(tmp_path / "catalog"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["network_used"] is False


def test_models_enrich_can_validate_against_selected_llama_binary(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    observed: list[Path] = []

    def fake_inspect(path: Path):
        observed.append(path)
        return None

    monkeypatch.setattr("gguf_limit_bench.cli.inspect_llama_executable", fake_inspect)
    result = runner.invoke(
        app,
        [
            "models",
            "enrich",
            "--offline",
            "--model-root",
            str(tmp_path / "models"),
            "--cache-root",
            str(tmp_path / "catalog"),
            "--llama-server",
            str(tmp_path / "llama-server.exe"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed == [tmp_path / "llama-server.exe"]


def test_models_command_group_exposes_product_commands():
    result = runner.invoke(app, ["models", "--help"])

    assert result.exit_code == 0
    for command in ("scan", "enrich", "list", "show", "recommendations", "export"):
        assert command in result.output


def test_models_scan_prints_recommendations_database_path(tmp_path):
    create_model(tmp_path / "models")

    result = runner.invoke(
        app,
        [
            "models",
            "scan",
            "--model-root",
            str(tmp_path / "models"),
            "--cache-root",
            str(tmp_path / "catalog"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Recommendations DB:" in result.output
    assert (tmp_path / "catalog" / "recommendations.json").exists()


def test_models_show_reads_persisted_catalog(tmp_path):
    model = create_model(tmp_path / "models")
    scan = runner.invoke(
        app,
        [
            "models",
            "scan",
            "--model-root",
            str(tmp_path / "models"),
            "--cache-root",
            str(tmp_path / "catalog"),
        ],
    )
    assert scan.exit_code == 0, scan.output

    shown = runner.invoke(
        app,
        [
            "models",
            "show",
            model.name,
            "--cache-root",
            str(tmp_path / "catalog"),
            "--json",
        ],
    )

    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.stdout)["name"] == model.name


def test_models_recommendations_serializes_evidence(tmp_path):
    entry = CatalogEntry(
        local_path="G:/models/model.gguf",
        name="model.gguf",
        family="qwen",
        parameters="27B",
        quant="Q5_K_M",
        size_bytes=20,
        is_moe=False,
        has_mtp=True,
        vision_mmproj=None,
        repo_id="owner/repo",
        hub_filename="model.gguf",
        revision="abc",
        identity_confidence="verified",
        document_confidence="verified",
        license="apache-2.0",
        base_models=(),
        datasets=(),
        recommendations=(
            Recommendation(
                key="temperature",
                value=0.7,
                confidence="publisher_claim",
                source_url="https://huggingface.co/owner/repo",
                revision="abc",
                evidence="--temp 0.7",
                parser="fenced_llama_server_command",
            ),
        ),
    )
    write_catalog(
        CatalogSnapshot(1, "2026-06-19T00:00:00Z", str(tmp_path), False, (entry,)),
        tmp_path,
    )

    result = runner.invoke(
        app,
        [
            "models",
            "recommendations",
            "model.gguf",
            "--cache-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)[0]["confidence"] == "publisher_claim"
