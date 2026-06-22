from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePath, PurePosixPath, PureWindowsPath
import re


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
