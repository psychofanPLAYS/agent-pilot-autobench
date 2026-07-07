from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Protocol

from gguf_limit_bench.gguf_metadata import read_gguf_metadata
from gguf_limit_bench.hf_catalog import HubRecord


class MatchHub(Protocol):
    def fetch(self, repo_id: str, filename: str) -> HubRecord: ...

    def search_models(self, query: str, limit: int) -> list[Any]: ...


@dataclass(frozen=True)
class MatchCandidate:
    repo_id: str
    score: int
    reasons: tuple[str, ...]
    filename_verified: bool = False
    base_models: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MatchDecision:
    source_repo_id: str | None
    selected_repo_id: str | None
    confidence: str
    filename: str
    queries: tuple[str, ...]
    candidates: tuple[MatchCandidate, ...]
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return payload


def resolve_hf_model_match(
    *,
    hub: MatchHub,
    model_path: Path,
    filename: str,
    source_repo_id: str | None,
    search_limit: int = 8,
    max_fetches: int = 8,
) -> tuple[MatchDecision, HubRecord | None]:
    queries = build_match_queries(
        model_path=model_path,
        filename=filename,
        source_repo_id=source_repo_id,
    )
    repo_ids: list[str] = []
    if source_repo_id:
        repo_ids.append(source_repo_id)

    errors: list[str] = []
    search_models = getattr(hub, "search_models", None)
    if callable(search_models):
        for query in queries:
            try:
                results = search_models(query, search_limit)
            except Exception as error:
                errors.append(f"search {query!r}: {type(error).__name__}: {error}")
                continue
            repo_ids.extend(_repo_id_from_search_result(result) for result in results)

    records: list[HubRecord] = []
    for repo_id in _unique_text(repo_ids):
        if len(records) >= max_fetches:
            break
        try:
            records.append(hub.fetch(repo_id, filename))
        except Exception as error:
            errors.append(f"fetch {repo_id}: {type(error).__name__}: {error}")

    candidates = tuple(
        sorted(
            (
                score_match_record(
                    record,
                    model_path=model_path,
                    filename=filename,
                    source_repo_id=source_repo_id,
                )
                for record in records
            ),
            key=lambda candidate: (-candidate.score, candidate.repo_id.casefold()),
        )
    )
    selected = candidates[0] if candidates else None
    selected_record = next(
        (record for record in records if selected and record.repo_id == selected.repo_id),
        None,
    )
    decision = MatchDecision(
        source_repo_id=source_repo_id,
        selected_repo_id=selected.repo_id if selected else None,
        confidence=_confidence(selected),
        filename=filename,
        queries=queries,
        candidates=candidates,
        errors=tuple(errors),
    )
    return decision, selected_record


def build_match_queries(
    *,
    model_path: Path,
    filename: str,
    source_repo_id: str | None,
) -> tuple[str, ...]:
    terms: list[str] = []
    if source_repo_id:
        terms.append(source_repo_id)
        terms.append(source_repo_id.split("/", 1)[-1])
    terms.append(model_path.parent.name)
    terms.append(filename)
    terms.append(_strip_quant_suffix(Path(filename).stem))
    terms.extend(_metadata_terms(model_path))
    return tuple(_unique_text(term for term in terms if _useful_query(term)))


def score_match_record(
    record: HubRecord,
    *,
    model_path: Path,
    filename: str,
    source_repo_id: str | None,
) -> MatchCandidate:
    score = 0
    reasons: list[str] = []
    repo_norm = _normalize(record.repo_id)
    repo_name_norm = _normalize(record.repo_id.split("/", 1)[-1])
    filename_norm = _normalize(filename)
    filename_core = _normalize(_strip_quant_suffix(Path(filename).stem))
    folder_norm = _normalize(model_path.parent.name)

    if record.repo_id == source_repo_id:
        score += 35
        reasons.append("local_path_repo")
    if record.filename_verified:
        score += 120
        reasons.append("gguf_filename_verified")
    if filename_core and (filename_core in repo_norm or repo_name_norm in filename_core):
        score += 55
        reasons.append("repo_name_matches_filename_core")
    if folder_norm and (folder_norm in repo_norm or repo_name_norm in folder_norm):
        score += 45
        reasons.append("repo_name_matches_folder")
    if "gguf" in repo_name_norm:
        score += 20
        reasons.append("repo_is_gguf_distribution")
    if any(
        _base_model_matches_filename(base_model, filename_norm) for base_model in record.base_models
    ):
        score += 20
        reasons.append("base_model_matches_filename")
    if record.document_confidence in {"verified", "cached"}:
        score += 10
        reasons.append("model_card_available")
    if record.repo_id and record.repo_id.split("/", 1)[0].casefold() == "qwen":
        score -= 20
        reasons.append("official_base_publisher_penalty")

    return MatchCandidate(
        repo_id=record.repo_id,
        score=score,
        reasons=tuple(reasons),
        filename_verified=record.filename_verified,
        base_models=record.base_models,
    )


def _repo_id_from_search_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(result.get("id") or result.get("modelId") or "")
    return str(getattr(result, "id", "") or getattr(result, "modelId", ""))


def _confidence(candidate: MatchCandidate | None) -> str:
    if candidate is None:
        return "unresolved"
    if candidate.filename_verified and candidate.score >= 150:
        return "verified"
    if candidate.score >= 100:
        return "strong"
    return "candidate"


def _metadata_terms(model_path: Path) -> tuple[str, ...]:
    try:
        metadata = read_gguf_metadata(model_path)
    except Exception:
        return ()
    keys = (
        "general.name",
        "general.basename",
        "general.finetune",
        "general.architecture",
        "tokenizer.ggml.model",
    )
    return tuple(str(metadata[key]) for key in keys if isinstance(metadata.get(key), str))


def _strip_quant_suffix(value: str) -> str:
    value = re.sub(r"\.(?:gguf|safetensors|bin)$", "", value, flags=re.I)
    value = re.sub(
        r"[-_. ](?:UD-)?(?:IQ\d_[A-Z]+|Q\d_[A-Z](?:_[A-Z]+)?|Q8_0|MXFP4_MOE|TQ\d_\dS)(?:[-_. ].*)?$",
        "",
        value,
        flags=re.I,
    )
    return value


def _base_model_matches_filename(base_model: str, filename_norm: str) -> bool:
    base_name = base_model.split("/", 1)[-1]
    base_norm = _normalize(base_name)
    return bool(base_norm and base_norm in filename_norm)


def _useful_query(value: str) -> bool:
    normalized = _normalize(value)
    return len(normalized) >= 4 and normalized not in {"gguf", "model", "models"}


def _unique_text(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        clean = str(value).strip()
        key = clean.casefold()
        if not clean or key in seen:
            continue
        seen.add(key)
        unique.append(clean)
    return tuple(unique)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())
