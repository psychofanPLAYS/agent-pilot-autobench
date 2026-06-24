from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Protocol

try:
    from huggingface_hub.errors import HfHubHTTPError
except ModuleNotFoundError:

    class HfHubHTTPError(OSError):  # type: ignore[no-redef]
        pass


class HubGateway(Protocol):
    def model_info(self, repo_id: str) -> Any: ...

    def model_card(self, repo_id: str, revision: str) -> str: ...


class HuggingFaceGateway:
    def __init__(self, cache_dir: Path | None = None) -> None:
        from huggingface_hub import HfApi

        self._api = HfApi()
        self._cache_dir = cache_dir

    def model_info(self, repo_id: str) -> Any:
        return self._api.model_info(repo_id, files_metadata=True)

    def model_card(self, repo_id: str, revision: str) -> str:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                revision=revision,
                cache_dir=self._cache_dir,
            )
        except EntryNotFoundError:
            return ""
        return Path(path).read_text(encoding="utf-8")


@dataclass(frozen=True)
class HubRecord:
    repo_id: str
    filename: str
    revision: str
    retrieved_at: str
    last_modified: str | None
    pipeline_tag: str | None
    library_name: str | None
    license: str | None
    base_models: tuple[str, ...]
    datasets: tuple[str, ...]
    filename_verified: bool
    identity_confidence: str
    document_confidence: str
    readme: str

    def to_dict(self, *, include_readme: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if not include_readme:
            payload.pop("readme")
        return payload


class HubCatalog:
    def __init__(
        self,
        *,
        gateway: HubGateway | None,
        cache_root: Path,
        offline: bool = False,
    ) -> None:
        self.gateway = gateway
        self.cache_root = cache_root
        self.offline = offline

    def fetch(self, repo_id: str, filename: str) -> HubRecord:
        if self.offline or self.gateway is None:
            return self.load(repo_id)

        try:
            return self._fetch_online(repo_id, filename)
        except (OSError, TimeoutError, HfHubHTTPError) as error:
            try:
                return self.load(repo_id)
            except FileNotFoundError:
                raise error

    def _fetch_online(self, repo_id: str, filename: str) -> HubRecord:
        assert self.gateway is not None

        info = self.gateway.model_info(repo_id)
        revision = str(_attribute(info, "sha") or "").strip()
        if not revision:
            raise ValueError(f"Hugging Face model info has no revision SHA: {repo_id}")
        readme = self.gateway.model_card(repo_id, revision)
        filenames = {
            str(name)
            for sibling in (_attribute(info, "siblings") or [])
            if (name := _attribute(sibling, "rfilename"))
        }
        filename_verified = filename in filenames
        card_data = _card_data_dict(_attribute(info, "card_data"))
        record = HubRecord(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            retrieved_at=datetime.now(UTC).isoformat(),
            last_modified=_optional_text(_attribute(info, "last_modified")),
            pipeline_tag=_optional_text(_attribute(info, "pipeline_tag")),
            library_name=_optional_text(_attribute(info, "library_name")),
            license=_optional_text(card_data.get("license")),
            base_models=_string_tuple(card_data.get("base_model")),
            datasets=_string_tuple(card_data.get("datasets")),
            filename_verified=filename_verified,
            identity_confidence="verified" if filename_verified else "candidate",
            document_confidence="verified" if readme.strip() else "partial",
            readme=readme,
        )
        self._write(record)
        return record

    def load(self, repo_id: str) -> HubRecord:
        repo_root = self.cache_root / _repo_cache_name(repo_id)
        candidates = sorted(
            repo_root.glob("*/record.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No cached Hugging Face evidence for {repo_id}")
        record_path = candidates[0]
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        readme_path = record_path.with_name("README.md")
        readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
        return HubRecord(
            repo_id=str(payload["repo_id"]),
            filename=str(payload["filename"]),
            revision=str(payload["revision"]),
            retrieved_at=str(payload["retrieved_at"]),
            last_modified=_optional_text(payload.get("last_modified")),
            pipeline_tag=_optional_text(payload.get("pipeline_tag")),
            library_name=_optional_text(payload.get("library_name")),
            license=_optional_text(payload.get("license")),
            base_models=_string_tuple(payload.get("base_models")),
            datasets=_string_tuple(payload.get("datasets")),
            filename_verified=bool(payload["filename_verified"]),
            identity_confidence=str(payload["identity_confidence"]),
            document_confidence="cached" if readme else "partial",
            readme=readme,
        )

    def _write(self, record: HubRecord) -> None:
        target = self.cache_root / _repo_cache_name(record.repo_id) / record.revision
        target.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            target / "record.json",
            json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
        )
        _atomic_write_text(target / "README.md", record.readme)


def _attribute(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _card_data_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        return result if isinstance(result, dict) else {}
    return {}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    return (str(value),)


def _optional_text(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _repo_cache_name(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
