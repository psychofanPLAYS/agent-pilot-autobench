from pathlib import Path
import subprocess

from gguf_limit_bench.runtime_capabilities import (
    inspect_llama_executable,
    parse_llama_help,
    validate_flag_names,
)


VERSION = """version: 9596 (18ef86ece)
built with Clang 20.1.8 for Windows x86_64
"""

HELP = """
-c, --ctx-size N                     size of the prompt context
--spec-draft-n-max N                 number of tokens to draft
--spec-draft-p-min P                 minimum speculative probability
--spec-type none,draft-simple,draft-mtp
--draft, --draft-n, --draft-max N    the argument has been removed. use --spec-draft-n-max
--draft-min, --draft-n-min N         the argument has been removed. use --spec-draft-n-min
"""


def test_removed_alias_is_rejected_and_native_mtp_is_supported():
    capabilities = parse_llama_help(VERSION, HELP)

    assert capabilities.version == 9596
    assert capabilities.commit == "18ef86ece"
    assert capabilities.supports("--spec-type")
    assert capabilities.supports("--spec-draft-n-max")
    assert not capabilities.supports("--draft-max")
    assert capabilities.is_removed("--draft-max")


def test_capability_snapshot_has_stable_help_digest():
    first = parse_llama_help(VERSION, HELP)
    second = parse_llama_help(VERSION, HELP)

    assert first.help_sha256 == second.help_sha256
    assert len(first.help_sha256) == 64


def test_flag_validation_separates_supported_removed_and_unknown():
    capabilities = parse_llama_help(VERSION, HELP)

    validation = validate_flag_names(
        ("--spec-type", "--spec-draft-n-max", "--draft-max", "--future-flag"),
        capabilities,
    )

    assert validation.supported == ("--spec-type", "--spec-draft-n-max")
    assert validation.removed == ("--draft-max",)
    assert validation.unsupported == ("--future-flag",)


def test_inspect_llama_executable_runs_version_and_help_only():
    calls: list[list[str]] = []

    def fake_runner(command, **kwargs):
        calls.append(command)
        stdout = VERSION if command[-1] == "--version" else HELP
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    capabilities = inspect_llama_executable(Path("llama-server.exe"), runner=fake_runner)

    assert capabilities.version == 9596
    assert calls == [
        ["llama-server.exe", "--version"],
        ["llama-server.exe", "--help"],
    ]
