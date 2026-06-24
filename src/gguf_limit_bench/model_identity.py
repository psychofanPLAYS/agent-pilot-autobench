from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import PurePath, PurePosixPath, PureWindowsPath
import re
import subprocess


_REPO_SEGMENT_RE = re.compile(r"[A-Za-z0-9._-]{1,96}\Z")


class IdentityConfidence(StrEnum):
    VERIFIED = "verified"
    STRONG = "strong"
    CANDIDATE = "candidate"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ModelIdentity:
    repo_id: str | None
    filename: str
    confidence: IdentityConfidence
    source: str
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class LmStudioModelEvidence:
    model_key: str
    repo_id: str
    filename: str
    architecture: str | None
    quantization: str | None
    max_context_length: int | None
    trained_for_tool_use: bool | None


@dataclass(frozen=True)
class LmStudioInventory:
    models: dict[str, LmStudioModelEvidence]
    diagnostics: tuple[str, ...] = ()


def resolve_path_identity(path: PurePath) -> ModelIdentity:
    """Resolve identity evidence encoded in a known local model path."""
    path_text = str(path)
    parsed_path = _path_with_intended_flavour(path)
    parts = parsed_path.parts
    filename = parsed_path.name
    marker_indexes = (
        index for index in range(len(parts) - 1, -1, -1) if parts[index].lower() == "lm_studio-gguf"
    )
    for marker_index in marker_indexes:
        if marker_index != len(parts) - 4:
            continue
        publisher, repo = parts[marker_index + 1 : marker_index + 3]
        if not all(_is_safe_repo_segment(segment) for segment in (publisher, repo)):
            continue
        if len(f"{publisher}/{repo}") > 96:
            continue
        if filename in {"", ".", ".."}:
            continue
        return ModelIdentity(
            repo_id=f"{publisher}/{repo}",
            filename=filename,
            confidence=IdentityConfidence.CANDIDATE,
            source="lm_studio_path",
            evidence=(path_text,),
        )

    return ModelIdentity(
        repo_id=None,
        filename=filename,
        confidence=IdentityConfidence.UNRESOLVED,
        source="filename",
    )


def parse_lm_studio_inventory(payload: str) -> LmStudioInventory:
    rows = json.loads(payload)
    if not isinstance(rows, list):
        raise ValueError("LM Studio inventory must be a JSON list")

    models: dict[str, LmStudioModelEvidence] = {}
    diagnostics: list[str] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            diagnostics.append(f"row {index}: expected an object")
            continue
        model_key = row.get("modelKey")
        identifier = row.get("indexedModelIdentifier")
        if not isinstance(model_key, str) or not model_key.strip():
            diagnostics.append(f"row {index}: missing modelKey")
            continue
        if not isinstance(identifier, str) or len(identifier.split("/", 2)) != 3:
            diagnostics.append(f"row {index}: invalid indexedModelIdentifier")
            continue

        owner, repository, filename = identifier.split("/", 2)
        quantization = row.get("quantization")
        quantization_name = (
            str(quantization["name"])
            if isinstance(quantization, dict) and quantization.get("name")
            else None
        )
        context = row.get("maxContextLength")
        tool_use = row.get("trainedForToolUse")
        models[model_key] = LmStudioModelEvidence(
            model_key=model_key,
            repo_id=f"{owner}/{repository}",
            filename=filename,
            architecture=str(row["architecture"]) if row.get("architecture") else None,
            quantization=quantization_name,
            max_context_length=context
            if isinstance(context, int) and not isinstance(context, bool)
            else None,
            trained_for_tool_use=tool_use if isinstance(tool_use, bool) else None,
        )
    return LmStudioInventory(models=models, diagnostics=tuple(diagnostics))


def read_lm_studio_inventory(
    command: Sequence[str] = ("lms", "ls", "--llm", "--json"),
) -> str:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "LM Studio inventory failed")
    return completed.stdout


def _path_with_intended_flavour(path: PurePath) -> PurePath:
    """Interpret clearly Windows-shaped paths without rewriting POSIX filenames."""
    path_text = str(path)
    if isinstance(path, PureWindowsPath):
        return path
    if type(path) is PurePosixPath:
        return path
    if re.match(r"^[A-Za-z]:[\\/]", path_text) or path_text.startswith((r"\\", "//")):
        return PureWindowsPath(path_text)
    return path


def _is_safe_repo_segment(segment: str) -> bool:
    """Apply a deterministic conservative subset of Hub repo-id segment rules."""
    return bool(
        segment
        and segment not in {".", ".."}
        and _REPO_SEGMENT_RE.fullmatch(segment)
        and not segment.startswith((".", "-"))
        and not segment.endswith((".", "-"))
        and ".." not in segment
        and "--" not in segment
    )
