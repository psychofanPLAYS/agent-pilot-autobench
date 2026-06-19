from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Protocol

from gguf_limit_bench.discovery import ModelInfo
from gguf_limit_bench.hf_catalog import HubRecord
from gguf_limit_bench.model_recommendations import (
    Recommendation,
    RecommendationSource,
    extract_recommendations,
    validate_recommendations,
)
from gguf_limit_bench.runtime_capabilities import LlamaCapabilities


class HubFetcher(Protocol):
    def fetch(self, repo_id: str, filename: str) -> HubRecord: ...


@dataclass(frozen=True)
class CatalogEntry:
    local_path: str
    name: str
    family: str
    parameters: str
    quant: str
    size_bytes: int
    is_moe: bool
    has_mtp: bool
    vision_mmproj: str | None
    repo_id: str | None
    hub_filename: str
    revision: str | None
    identity_confidence: str
    document_confidence: str
    license: str | None
    base_models: tuple[str, ...]
    datasets: tuple[str, ...]
    recommendations: tuple[Recommendation, ...]
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["recommendations"] = [asdict(item) for item in self.recommendations]
        return payload


@dataclass(frozen=True)
class CatalogSnapshot:
    schema_version: int
    generated_at: str
    cache_root: str
    network_used: bool
    entries: tuple[CatalogEntry, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "cache_root": self.cache_root,
            "network_used": self.network_used,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class CatalogPaths:
    json: Path
    markdown: Path


class ModelCatalog:
    def __init__(
        self,
        *,
        cache_root: Path,
        hub: HubFetcher | None = None,
        capabilities: LlamaCapabilities | None = None,
    ) -> None:
        self.cache_root = cache_root
        self.hub = hub
        self.capabilities = capabilities

    def build(self, models: list[ModelInfo], *, enrich: bool = False) -> CatalogSnapshot:
        entries = [self._entry(model, enrich=enrich) for model in models]
        entries.sort(key=lambda item: ((item.repo_id or "~").lower(), item.local_path.lower()))
        return CatalogSnapshot(
            schema_version=1,
            generated_at=datetime.now(UTC).isoformat(),
            cache_root=str(self.cache_root),
            network_used=bool(enrich and self.hub is not None),
            entries=tuple(entries),
        )

    def _entry(self, model: ModelInfo, *, enrich: bool) -> CatalogEntry:
        identity = model.identity
        repo_id = identity.repo_id if identity else None
        hub_filename = identity.filename if identity else model.name
        identity_confidence = identity.confidence.value if identity else "unresolved"
        document_confidence = "unavailable"
        revision: str | None = None
        license_name: str | None = None
        base_models: tuple[str, ...] = ()
        datasets: tuple[str, ...] = ()
        recommendations: tuple[Recommendation, ...] = ()
        errors: list[str] = []

        if enrich and repo_id and self.hub is not None:
            try:
                record = self.hub.fetch(repo_id, hub_filename)
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
            else:
                identity_confidence = record.identity_confidence
                document_confidence = record.document_confidence
                revision = record.revision
                license_name = record.license
                base_models = record.base_models
                datasets = record.datasets
                recommendations = extract_recommendations(
                    record.readme,
                    source=RecommendationSource(
                        url=(
                            f"https://huggingface.co/{record.repo_id}/blob/"
                            f"{record.revision}/README.md"
                        ),
                        revision=record.revision,
                    ),
                )
                if self.capabilities is not None:
                    recommendations = validate_recommendations(
                        recommendations,
                        self.capabilities,
                    )

        return CatalogEntry(
            local_path=str(model.path),
            name=model.name,
            family=model.family,
            parameters=model.parameters,
            quant=model.quant,
            size_bytes=model.size_bytes,
            is_moe=model.is_moe,
            has_mtp=model.has_mtp,
            vision_mmproj=str(model.vision_mmproj) if model.vision_mmproj else None,
            repo_id=repo_id,
            hub_filename=hub_filename,
            revision=revision,
            identity_confidence=identity_confidence,
            document_confidence=document_confidence,
            license=license_name,
            base_models=base_models,
            datasets=datasets,
            recommendations=recommendations,
            errors=tuple(errors),
        )


def write_catalog(snapshot: CatalogSnapshot, output_dir: Path) -> CatalogPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "catalog.json"
    markdown_path = output_dir / "catalog.md"
    _atomic_write(
        json_path,
        json.dumps(snapshot.to_dict(), indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(markdown_path, _catalog_markdown(snapshot))
    return CatalogPaths(json=json_path, markdown=markdown_path)


def _catalog_markdown(snapshot: CatalogSnapshot) -> str:
    lines = [
        "# PilotBENCHY Model Catalog",
        "",
        f"Generated: `{snapshot.generated_at}`",
        f"Network used: `{'yes' if snapshot.network_used else 'no'}`",
        "",
        "| Model | Repository | Quant | Identity confidence | Document confidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in snapshot.entries:
        lines.append(
            f"| `{entry.name}` | `{entry.repo_id or 'unresolved'}` | `{entry.quant}` | "
            f"`{entry.identity_confidence}` | `{entry.document_confidence}` |"
        )
    for entry in snapshot.entries:
        lines.extend(["", f"## {entry.name}", ""])
        if entry.errors:
            lines.append(f"Errors: `{' | '.join(entry.errors)}`")
        if not entry.recommendations:
            lines.append("Recommendations: none retrieved.")
        else:
            lines.append("Recommendations:")
            for item in entry.recommendations:
                lines.append(
                    f"- `{item.key}={item.value}` — `{item.confidence}`"
                    f"{' (conflicted)' if item.conflicted else ''}"
                )
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
