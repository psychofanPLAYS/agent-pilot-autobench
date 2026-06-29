from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Protocol

from gguf_limit_bench.discovery import ModelInfo
from gguf_limit_bench.hf_catalog import HubRecord
from gguf_limit_bench.hf_model_match import MatchDecision, resolve_hf_model_match
from gguf_limit_bench.model_recommendations import (
    Recommendation,
    RecommendationSource,
    extract_recommendations,
    recommendation_values,
    validate_recommendations,
)
from gguf_limit_bench.runtime_capabilities import LlamaCapabilities


class HubFetcher(Protocol):
    def fetch(self, repo_id: str, filename: str) -> HubRecord: ...

    def search_models(self, query: str, limit: int) -> list[object]: ...


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
    source_repo_id: str | None = None
    match_confidence: str = "unresolved"
    match_candidates: tuple[dict[str, object], ...] = ()

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
    recommendations: Path
    matches: Path


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
            network_used=bool(
                enrich and self.hub is not None and not getattr(self.hub, "offline", False)
            ),
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
        match_decision: MatchDecision | None = None

        if enrich and self.hub is not None:
            try:
                match_decision, record = resolve_hf_model_match(
                    hub=self.hub,
                    model_path=model.path,
                    filename=hub_filename,
                    source_repo_id=repo_id,
                )
                if record is None:
                    raise FileNotFoundError(f"No Hugging Face match found for {hub_filename}")
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
            else:
                repo_id = record.repo_id
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
                    auxiliary_files=record.auxiliary_files,
                )
                if self.capabilities is not None:
                    recommendations = validate_recommendations(
                        recommendations,
                        self.capabilities,
                    )
            if match_decision is not None:
                errors.extend(match_decision.errors)

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
            source_repo_id=identity.repo_id if identity else None,
            match_confidence=match_decision.confidence if match_decision else identity_confidence,
            match_candidates=tuple(candidate.to_dict() for candidate in match_decision.candidates)
            if match_decision
            else (),
        )


def write_catalog(snapshot: CatalogSnapshot, output_dir: Path) -> CatalogPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "catalog.json"
    markdown_path = output_dir / "catalog.md"
    recommendations_path = output_dir / "recommendations.json"
    matches_path = output_dir / "hf-match-decisions.json"
    _atomic_write(
        json_path,
        json.dumps(snapshot.to_dict(), indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(markdown_path, _catalog_markdown(snapshot))
    _atomic_write(
        recommendations_path,
        json.dumps(_recommendation_database(snapshot), indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(
        matches_path,
        json.dumps(_match_database(snapshot), indent=2, sort_keys=True) + "\n",
    )
    return CatalogPaths(
        json=json_path,
        markdown=markdown_path,
        recommendations=recommendations_path,
        matches=matches_path,
    )


def load_catalog(cache_root: Path) -> CatalogSnapshot:
    path = cache_root / "catalog.json"
    if not path.exists():
        raise FileNotFoundError(f"PilotBENCHY catalog was not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = []
    for row in payload.get("entries", []):
        recommendations = tuple(Recommendation(**item) for item in row.get("recommendations", []))
        entries.append(
            CatalogEntry(
                local_path=str(row["local_path"]),
                name=str(row["name"]),
                family=str(row["family"]),
                parameters=str(row["parameters"]),
                quant=str(row["quant"]),
                size_bytes=int(row["size_bytes"]),
                is_moe=bool(row["is_moe"]),
                has_mtp=bool(row["has_mtp"]),
                vision_mmproj=row.get("vision_mmproj"),
                repo_id=row.get("repo_id"),
                hub_filename=str(row["hub_filename"]),
                revision=row.get("revision"),
                identity_confidence=str(row["identity_confidence"]),
                document_confidence=str(row["document_confidence"]),
                license=row.get("license"),
                base_models=tuple(row.get("base_models", [])),
                datasets=tuple(row.get("datasets", [])),
                recommendations=recommendations,
                errors=tuple(row.get("errors", [])),
                source_repo_id=row.get("source_repo_id"),
                match_confidence=str(row.get("match_confidence", row["identity_confidence"])),
                match_candidates=tuple(row.get("match_candidates", [])),
            )
        )
    return CatalogSnapshot(
        schema_version=int(payload["schema_version"]),
        generated_at=str(payload["generated_at"]),
        cache_root=str(payload["cache_root"]),
        network_used=bool(payload["network_used"]),
        entries=tuple(entries),
    )


def find_catalog_entry(snapshot: CatalogSnapshot, selector: str) -> CatalogEntry:
    normalized = selector.casefold()
    matches = [
        entry
        for entry in snapshot.entries
        if normalized
        in {
            entry.name.casefold(),
            entry.local_path.casefold(),
            (entry.repo_id or "").casefold(),
        }
    ]
    if not matches:
        raise KeyError(f"Model was not found in the catalog: {selector}")
    if len(matches) > 1:
        raise KeyError(f"Model selector is ambiguous: {selector}")
    return matches[0]


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


def _recommendation_database(snapshot: CatalogSnapshot) -> dict:
    entries = []
    for entry in snapshot.entries:
        values = recommendation_values(entry.recommendations)
        entries.append(
            {
                "model": {
                    "name": entry.name,
                    "local_path": entry.local_path,
                    "repo_id": entry.repo_id,
                    "hub_filename": entry.hub_filename,
                    "family": entry.family,
                    "parameters": entry.parameters,
                    "quant": entry.quant,
                    "has_mtp": entry.has_mtp,
                    "is_moe": entry.is_moe,
                },
                "identity_confidence": entry.identity_confidence,
                "document_confidence": entry.document_confidence,
                "revision": entry.revision,
                "license": entry.license,
                "base_models": list(entry.base_models),
                "datasets": list(entry.datasets),
                "values": values,
                "recommendations": [asdict(item) for item in entry.recommendations],
                "source_repo_id": entry.source_repo_id,
                "match_confidence": entry.match_confidence,
                "match_candidates": list(entry.match_candidates),
                "errors": list(entry.errors),
            }
        )
    return {
        "schema_version": 1,
        "generated_at": snapshot.generated_at,
        "network_used": snapshot.network_used,
        "source_catalog_schema_version": snapshot.schema_version,
        "entries": entries,
    }


def _match_database(snapshot: CatalogSnapshot) -> dict:
    return {
        "schema_version": 1,
        "generated_at": snapshot.generated_at,
        "network_used": snapshot.network_used,
        "entries": [
            {
                "name": entry.name,
                "local_path": entry.local_path,
                "source_repo_id": entry.source_repo_id,
                "selected_repo_id": entry.repo_id,
                "match_confidence": entry.match_confidence,
                "hub_filename": entry.hub_filename,
                "candidates": list(entry.match_candidates),
                "errors": list(entry.errors),
            }
            for entry in snapshot.entries
        ],
    }


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
