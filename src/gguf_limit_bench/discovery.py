from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from gguf_limit_bench.model_identity import ModelIdentity, resolve_path_identity


QUANT_RE = re.compile(r"(IQ\d_[A-Z]+|Q\d_[A-Z](?:_[A-Z]+)?|Q8_0|MXFP4_MOE|TQ\d_\dS)", re.I)
PARAM_RE = re.compile(r"(\d+(?:\.\d+)?B(?:-A\d+B?)?)", re.I)


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
    family = "qwen" if "qwen" in lowered else "llama" if "llama" in lowered else "unknown"
    quant_match = QUANT_RE.search(name)
    param_match = PARAM_RE.search(name)
    parameters = param_match.group(1).upper().replace("A3B", "A3B") if param_match else "unknown"
    is_moe = "moe" in lowered or bool(re.search(r"-a\d+b", lowered))
    has_mtp = "mtp" in lowered
    return ModelInfo(
        path=path,
        name=name,
        family=family,
        parameters=parameters,
        quant=quant_match.group(1).upper() if quant_match else "unknown",
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
            if "mmproj" in path.name.lower():
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
