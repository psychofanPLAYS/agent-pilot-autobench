from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from gguf_limit_bench.model_identity import ModelIdentity, resolve_path_identity


QUANT_RE = re.compile(
    r"(IQ\d_[A-Z0-9]+|Q\d_[A-Z0-9]+(?:_[A-Z0-9]+)?|MXFP4_MOE|TQ\d_\dS)",
    re.I,
)
PARAM_RE = re.compile(r"(\d+(?:\.\d+)?B(?:-A\d+B?)?)", re.I)

# Markers for GGUF files that are NOT chat/generative LLMs and must never enter the
# benchmark candidate set. Serving an embedding/reranker/query-expansion model through
# llama-server chat-completions would produce degenerate output (or fail outright); the
# autobench must reject them at discovery rather than "score" them as if they were LLMs.
# - mmproj / imatrix: vision projector + importance-matrix calibration sidecars.
# - embedding / reranker: retrieval models, not text generators.
# - query-expansion: a narrow search-rewrite fine-tune, not a general chat model.
_NON_GENERATIVE_MARKERS = (
    "mmproj",
    "imatrix",
    "embedding",
    "embed-",
    "reranker",
    "rerank",
    "query-expansion",
    "query_expansion",
)


def is_non_generative_gguf(path: Path) -> bool:
    """True if *path* looks like a non-chat GGUF (embedding/reranker/QE/imatrix/mmproj).

    Used both to keep these out of model discovery and to keep them from ever
    appearing as a champion in the leaderboard (which scans historical receipts).
    """
    haystack = str(path).lower()
    return any(marker in haystack for marker in _NON_GENERATIVE_MARKERS)


def _is_non_generative(path: Path) -> bool:  # backwards-compatible alias
    return is_non_generative_gguf(path)


@dataclass(frozen=True)
class ModelInfo:
    path: Path
    name: str
    family: str = "unknown"
    parameters: str = "unknown"
    quant: str = "unknown"
    size_bytes: int = 0
    is_moe: bool = False
    has_mtp: bool = False
    vision_mmproj: Path | None = None
    identity: ModelIdentity | None = None

    @property
    def has_vision(self) -> bool:
        return self.vision_mmproj is not None

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024**3)


def parse_model_name(path: Path) -> ModelInfo:
    name = path.name
    lowered = name.lower()
    if "qwen" in lowered or "qwopus" in lowered:
        family = "qwen"
    elif "gemma" in lowered:
        family = "gemma"
    elif "llama" in lowered:
        family = "llama"
    else:
        family = "unknown"
    quant = _quant_label(name)
    param_match = PARAM_RE.search(name)
    parameters = param_match.group(1).upper().replace("A3B", "A3B") if param_match else "unknown"
    is_moe = "moe" in lowered or bool(re.search(r"-a\d+b", lowered))
    has_mtp = "mtp" in lowered
    return ModelInfo(
        path=path,
        name=name,
        family=family,
        parameters=parameters,
        quant=quant,
        is_moe=is_moe,
        has_mtp=has_mtp,
    )


def discover_models(roots: list[Path]) -> list[ModelInfo]:
    models: list[ModelInfo] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.gguf")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if _is_non_generative(path):
                continue
            info = parse_model_name(path)
            mmproj = _find_mmproj(path.parent)
            models.append(
                ModelInfo(
                    path=path,
                    name=info.name,
                    family=info.family,
                    parameters=info.parameters,
                    quant=info.quant,
                    size_bytes=path.stat().st_size,
                    is_moe=info.is_moe,
                    has_mtp=info.has_mtp,
                    vision_mmproj=mmproj,
                    identity=resolve_path_identity(path),
                )
            )
    return sorted(models, key=lambda model: (-model.size_bytes, str(model.path).lower()))


def _find_mmproj(folder: Path) -> Path | None:
    mmprojs = sorted(path for path in folder.glob("*.gguf") if "mmproj" in path.name.lower())
    return mmprojs[0] if mmprojs else None


def _quant_label(name: str) -> str:
    quant_match = QUANT_RE.search(name)
    if quant_match:
        return quant_match.group(1).upper()
    apex_tier_match = re.search(r"\bAPEX(?:[-_. ]+MTP)?[-_. ]+I[-_. ]+([A-Z]+)", name, re.I)
    if apex_tier_match:
        return f"APEX-I-{apex_tier_match.group(1).upper()}"
    apex_tier_match = re.search(r"\bAPEX(?:[-_. ]+MTP)?[-_. ]+([A-Z]+)", name, re.I)
    if apex_tier_match and apex_tier_match.group(1).casefold() not in {"mtp", "gguf"}:
        return f"APEX-{apex_tier_match.group(1).upper()}"
    if re.search(r"\bAPEX\b", name, re.I):
        return "APEX"
    if re.search(r"\bQAT\b", name, re.I):
        return "QAT"
    return "unknown"
