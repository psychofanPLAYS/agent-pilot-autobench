import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from gguf_limit_bench import runtime_capabilities
from gguf_limit_bench.autoresearch import AttemptResult
from gguf_limit_bench.cli import _run_one_autoresearch, _write_flag_ladder_dry_run
from gguf_limit_bench.flag_ladder import build_flag_ladder_plan
from gguf_limit_bench.runtime_capabilities import parse_llama_help


HELP = """
--spec-draft-n-max N number of tokens to draft
--spec-type none,draft-simple,draft-mtp
--draft-max N the argument has been removed
"""


def test_removed_alias_is_rejected_and_native_mtp_is_supported():
    capabilities = parse_llama_help("version: 9596 (18ef86ece)", HELP)

    assert capabilities.supports("--spec-type")
    assert capabilities.supports("--spec-draft-n-max")
    assert not capabilities.supports("--draft-max")
    assert capabilities.version == "9596"
    assert capabilities.build == 9596
    assert capabilities.commit == "18ef86ece"
    assert capabilities.help_sha256 == hashlib.sha256(HELP.encode("utf-8")).hexdigest()


def test_help_description_substrings_are_not_treated_as_supported_options():
    capabilities = parse_llama_help("version: b9596", "--help describe --imaginary-only")

    assert capabilities.supports("--help")
    assert not capabilities.supports("--imaginary-only")


def test_mtp_plan_is_marked_unsupported_when_runtime_lacks_native_options():
    capabilities = parse_llama_help("version: b9596", "--model FNAME\n--draft-max removed")

    plan = build_flag_ladder_plan(
        llama_server=Path("llama-server.exe"),
        model=Path("Qwen-MTP.gguf"),
        host="127.0.0.1",
        port=6939,
        context_size=4096,
        parallel_max=4,
        enable_mtp=True,
        runtime_capabilities=capabilities,
    )

    mtp_row = next(row for row in plan if row["name"].startswith("MTP-"))
    assert mtp_row["supported"] is False
    assert mtp_row["command"] is None
    assert "--spec-type" in mtp_row["unsupported_reason"]


def test_collector_uses_bounded_non_serving_introspection_and_combines_stderr(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[-1] == "--version":
            return SimpleNamespace(returncode=0, stdout="version: 9596 (18ef86ece)\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr=HELP)

    monkeypatch.setattr(runtime_capabilities.subprocess, "run", fake_run)

    capabilities = runtime_capabilities.collect_llama_capabilities(
        Path("llama-server.exe"), timeout_seconds=2
    )

    assert [call[0][-1] for call in calls] == ["--version", "--help"]
    assert all(call[1]["shell"] is False for call in calls)
    assert all(call[1]["timeout"] == 2 for call in calls)
    assert capabilities.supports("--spec-type")
    assert capabilities.introspection_ok is True


def test_nonzero_introspection_degrades_to_unknown_capabilities(monkeypatch):
    monkeypatch.setattr(
        runtime_capabilities.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="--spec-type draft-mtp\n--spec-draft-n-max N",
            stderr="failed",
        ),
    )

    capabilities = runtime_capabilities.collect_llama_capabilities(Path("missing.exe"))

    assert capabilities.introspection_ok is False
    assert not capabilities.supports("--spec-type")


def test_missing_binary_degrades_to_unknown_capabilities(monkeypatch):
    def missing_binary(command, **kwargs):
        raise FileNotFoundError("llama-server.exe not found")

    monkeypatch.setattr(runtime_capabilities.subprocess, "run", missing_binary)

    capabilities = runtime_capabilities.collect_llama_capabilities(Path("missing.exe"))

    assert capabilities.introspection_ok is False
    assert "FileNotFoundError" in capabilities.introspection_error
    assert not capabilities.supports("--spec-type")


def test_real_dry_run_path_records_capabilities_and_renders_unsupported_mtp(tmp_path):
    capabilities = parse_llama_help("version: b9596", "--model FNAME")

    receipt = _write_flag_ladder_dry_run(
        model=Path("Qwen-MTP.gguf"),
        llama_server=Path("llama-server.exe"),
        runs_root=tmp_path,
        context_size=4096,
        parallel_max=4,
        extra_server_args=(),
        enable_mtp=True,
        capability_collector=lambda path: capabilities,
    )

    payload = json.loads((receipt.path / "flag-ladder-plan.json").read_text(encoding="utf-8"))
    mtp_row = next(row for row in payload["profiles"] if row["name"].startswith("MTP-"))
    markdown = (receipt.path / "flag-ladder-plan.md").read_text(encoding="utf-8")

    assert mtp_row["supported"] is False
    assert mtp_row["command"] is None
    assert payload["runtime_capabilities"]["version"] == "b9596"
    assert payload["runtime_capabilities"]["help_sha256"] == capabilities.help_sha256
    assert "Unsupported" in markdown
    assert "runtime lacks required options" in markdown


def test_live_flag_ladder_skips_unsupported_mtp_before_runner(tmp_path):
    capabilities = parse_llama_help("version: b9596", "--model FNAME")
    seen_profiles = []

    def fake_runner(settings):
        seen_profiles.append(settings.profile_name)
        return AttemptResult(
            ok=True,
            generation_tokens_per_second=1.0,
            prompt_tokens_per_second=1.0,
            ttft_ms=None,
            context_size=settings.context_size,
            failure="none",
            stdout="",
            stderr="",
            returncode=0,
        )

    receipt = _run_one_autoresearch(
        model=Path("Qwen-MTP.gguf"),
        llama_bench=Path("llama-bench.exe"),
        llama_cli=Path("llama-cli.exe"),
        llama_server=Path("llama-server.exe"),
        runs_root=tmp_path,
        budget_seconds=60,
        parallel_max=4,
        max_attempts=None,
        learning=False,
        workflow_eval=False,
        ttft_probe=False,
        flag_ladder=True,
        enable_mtp=True,
        capability_collector=lambda path: capabilities,
        flag_ladder_attempt_runner=fake_runner,
    )

    assert seen_profiles
    assert not any(profile.startswith("MTP-") for profile in seen_profiles)
    events = (receipt.path / "events.jsonl").read_text(encoding="utf-8")
    assert "flag_ladder_profiles_skipped" in events
    assert "MTP-draft-3" in events
