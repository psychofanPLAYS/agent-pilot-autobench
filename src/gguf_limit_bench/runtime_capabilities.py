from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import subprocess


_OPTION = re.compile(r"^--?[A-Za-z0-9][A-Za-z0-9-]*$")
_REMOVED = re.compile(r"\bremoved\b", re.IGNORECASE)


@dataclass(frozen=True)
class LlamaRuntimeCapabilities:
    version: str | None
    build: int | None
    commit: str | None
    supported_options: frozenset[str]
    removed_options: frozenset[str]
    help_sha256: str
    introspection_ok: bool = True
    introspection_error: str | None = None

    def supports(self, option: str) -> bool:
        return option in self.supported_options and option not in self.removed_options

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "build": self.build,
            "commit": self.commit,
            "supported_options": sorted(self.supported_options),
            "removed_options": sorted(self.removed_options),
            "help_sha256": self.help_sha256,
            "introspection_ok": self.introspection_ok,
            "introspection_error": self.introspection_error,
        }


def parse_llama_help(version_text: str, help_text: str) -> LlamaRuntimeCapabilities:
    version, build, commit = _parse_version(version_text)
    supported: set[str] = set()
    removed: set[str] = set()

    for line in help_text.splitlines():
        options = _option_column(line)
        if not options:
            continue
        if _REMOVED.search(line):
            removed.update(options)
        else:
            supported.update(options)

    supported.difference_update(removed)
    return LlamaRuntimeCapabilities(
        version=version,
        build=build,
        commit=commit,
        supported_options=frozenset(supported),
        removed_options=frozenset(removed),
        help_sha256=hashlib.sha256(help_text.encode("utf-8")).hexdigest(),
    )


def collect_llama_capabilities(
    executable: Path, timeout_seconds: float = 5.0
) -> LlamaRuntimeCapabilities:
    try:
        version_result = _run_introspection(executable, "--version", timeout_seconds)
        help_result = _run_introspection(executable, "--help", timeout_seconds)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _unknown_capabilities(f"{type(exc).__name__}: {exc}")

    version_text = _combined_output(version_result.stdout, version_result.stderr)
    help_text = _combined_output(help_result.stdout, help_result.stderr)
    if version_result.returncode != 0 or help_result.returncode != 0:
        return _unknown_capabilities(
            "llama.cpp introspection failed "
            f"(--version={version_result.returncode}, --help={help_result.returncode})",
            version_text=version_text,
            help_text=help_text,
        )
    return parse_llama_help(version_text, help_text)


def _run_introspection(executable: Path, option: str, timeout_seconds: float):
    return subprocess.run(
        [str(executable), option],
        shell=False,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout_seconds,
    )


def _combined_output(stdout: str, stderr: str) -> str:
    return "\n".join(part.strip("\n") for part in (stdout, stderr) if part).strip("\n")


def _unknown_capabilities(
    error: str, *, version_text: str = "", help_text: str = ""
) -> LlamaRuntimeCapabilities:
    parsed = parse_llama_help(version_text, help_text)
    return LlamaRuntimeCapabilities(
        version=parsed.version,
        build=parsed.build,
        commit=parsed.commit,
        supported_options=frozenset(),
        removed_options=parsed.removed_options,
        help_sha256=parsed.help_sha256,
        introspection_ok=False,
        introspection_error=error,
    )


def _option_column(line: str) -> tuple[str, ...]:
    tokens = line.lstrip().split()
    options: list[str] = []
    for token in tokens:
        candidate = token.rstrip(",")
        if not _OPTION.fullmatch(candidate):
            break
        options.append(candidate)
    return tuple(options)


def _parse_version(version_text: str) -> tuple[str | None, int | None, str | None]:
    version_match = re.search(r"\bversion\s*:\s*(\S+)", version_text, re.IGNORECASE)
    version = version_match.group(1) if version_match else None
    build_match = re.search(r"\bbuild\s*[:=]?\s*b?(\d+)\b", version_text, re.IGNORECASE)
    if build_match is None and version is not None:
        build_match = re.fullmatch(r"b?(\d+)", version)
    build = int(build_match.group(1)) if build_match else None
    commit_match = re.search(r"\(([0-9a-f]{7,40})\)", version_text, re.IGNORECASE)
    if commit_match is None:
        commit_match = re.search(
            r"\bcommit\s*[:=]\s*([0-9a-f]{7,40})\b", version_text, re.IGNORECASE
        )
    commit = commit_match.group(1) if commit_match else None
    return version, build, commit
