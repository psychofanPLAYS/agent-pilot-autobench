import json

from typer.testing import CliRunner

from gguf_limit_bench.cli import app
from gguf_limit_bench.doctor import build_doctor_report


runner = CliRunner()


def test_doctor_report_marks_existing_and_missing_paths(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    runs = tmp_path / "runs"

    report = build_doctor_report(
        model_roots=[models, tmp_path / "missing-models"],
        llama_bench=tmp_path / "missing" / "llama-bench.exe",
        llama_cli=tmp_path / "missing" / "llama-cli.exe",
        runs_root=runs,
    )

    statuses = {check.name: check.status for check in report.checks}

    assert statuses[f"model root: {models}"] == "ok"
    assert statuses[f"model root: {tmp_path / 'missing-models'}"] == "missing"
    assert statuses["llama-bench"] == "missing"
    assert statuses["llama-cli"] == "missing"
    assert statuses["runs root"] == "ok"
    assert runs.exists()


def test_doctor_command_is_non_strict_by_default(tmp_path):
    result = runner.invoke(
        app,
        [
            "doctor",
            "--root",
            str(tmp_path / "missing-models"),
            "--llama-bench",
            str(tmp_path / "missing" / "llama-bench.exe"),
            "--llama-cli",
            str(tmp_path / "missing" / "llama-cli.exe"),
            "--runs-root",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0
    assert "missing" in result.output
    assert "Use --strict" in result.output


def test_doctor_command_strict_fails_when_required_paths_are_missing(tmp_path):
    result = runner.invoke(
        app,
        [
            "doctor",
            "--root",
            str(tmp_path / "missing-models"),
            "--llama-bench",
            str(tmp_path / "missing" / "llama-bench.exe"),
            "--llama-cli",
            str(tmp_path / "missing" / "llama-cli.exe"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--strict",
        ],
    )

    assert result.exit_code == 1
    assert "Required checks failed" in result.output


def test_doctor_command_can_emit_json(tmp_path):
    result = runner.invoke(
        app,
        [
            "doctor",
            "--root",
            str(tmp_path / "models"),
            "--llama-bench",
            str(tmp_path / "missing" / "llama-bench.exe"),
            "--llama-cli",
            str(tmp_path / "missing" / "llama-cli.exe"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--json-out",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)

    assert payload["ready"] is False
    assert payload["checks"][0]["name"].startswith("model root")


def test_doctor_command_strict_json_keeps_stdout_machine_readable(tmp_path):
    result = runner.invoke(
        app,
        [
            "doctor",
            "--root",
            str(tmp_path / "missing-models"),
            "--llama-bench",
            str(tmp_path / "missing" / "llama-bench.exe"),
            "--llama-cli",
            str(tmp_path / "missing" / "llama-cli.exe"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--strict",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ready"] is False
    assert "Required checks failed" in result.stderr
