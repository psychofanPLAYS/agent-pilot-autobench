from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import subprocess
from typing import Any, Callable


_LONG_FLAG_RE = re.compile(r"--[a-z][a-z0-9-]*")
_VERSION_RE = re.compile(r"version:\s*(\d+)\s*\(([^)]+)\)", re.I)


@dataclass(frozen=True)
class LlamaCapabilities:
    version: int | None
    commit: str | None
    supported_flags: frozenset[str]
    removed_flags: frozenset[str]
    help_sha256: str

    def supports(self, flag: str) -> bool:
        return flag in self.supported_flags and flag not in self.removed_flags

    def is_removed(self, flag: str) -> bool:
        return flag in self.removed_flags


@dataclass(frozen=True)
class FlagValidation:
    supported: tuple[str, ...]
    removed: tuple[str, ...]
    unsupported: tuple[str, ...]


def parse_llama_help(version_output: str, help_output: str) -> LlamaCapabilities:
    version_match = _VERSION_RE.search(version_output)
    supported: set[str] = set()
    removed: set[str] = set()
    for line in help_output.splitlines():
        option_column = re.split(r"\s{2,}", line.strip(), maxsplit=1)[0]
        flags = set(_LONG_FLAG_RE.findall(option_column))
        if "argument has been removed" in line.lower():
            removed.update(flags)
        else:
            supported.update(flags)
    return LlamaCapabilities(
        version=int(version_match.group(1)) if version_match else None,
        commit=version_match.group(2) if version_match else None,
        supported_flags=frozenset(supported),
        removed_flags=frozenset(removed),
        help_sha256=hashlib.sha256(help_output.encode("utf-8")).hexdigest(),
    )


def validate_flag_names(
    flags: tuple[str, ...] | list[str],
    capabilities: LlamaCapabilities,
) -> FlagValidation:
    supported: list[str] = []
    removed: list[str] = []
    unsupported: list[str] = []
    for flag in flags:
        if capabilities.is_removed(flag):
            removed.append(flag)
        elif capabilities.supports(flag):
            supported.append(flag)
        else:
            unsupported.append(flag)
    return FlagValidation(tuple(supported), tuple(removed), tuple(unsupported))


def inspect_llama_executable(
    executable: Path,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> LlamaCapabilities:
    outputs: list[str] = []
    for argument in ("--version", "--help"):
        command = [str(executable), argument]
        completed = runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "unknown error").strip()
            raise RuntimeError(f"llama.cpp {argument} failed: {detail}")
        outputs.append(completed.stdout or completed.stderr or "")
    return parse_llama_help(outputs[0], outputs[1])
