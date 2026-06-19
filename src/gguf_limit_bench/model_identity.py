from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


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


def resolve_path_identity(path: Path) -> ModelIdentity:
    parts = path.parts
    lowered = [part.lower() for part in parts]
    if "lm_studio-gguf" in lowered:
        index = lowered.index("lm_studio-gguf")
        if len(parts) >= index + 4:
            repo_id = f"{parts[index + 1]}/{parts[index + 2]}"
            return ModelIdentity(
                repo_id=repo_id,
                filename=path.name,
                confidence=IdentityConfidence.CANDIDATE,
                source="lm_studio_path",
                evidence=(str(path),),
            )
    return ModelIdentity(
        repo_id=None,
        filename=path.name,
        confidence=IdentityConfidence.UNRESOLVED,
        source="filename",
        evidence=(str(path),),
    )
