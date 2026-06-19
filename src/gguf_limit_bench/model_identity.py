from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path
import subprocess


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
