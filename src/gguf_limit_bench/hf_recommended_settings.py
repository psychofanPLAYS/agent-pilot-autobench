from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any


_DATA_ROOT = Path(__file__).resolve().parent / "data" / "model_recs"
_SAMPLER_FLAG_BY_KEY = {
    "temperature": "--temp",
    "top_p": "--top-p",
    "top_k": "--top-k",
    "min_p": "--min-p",
    "presence_penalty": "--presence-penalty",
    "repetition_penalty": "--repeat-penalty",
}


def recommended_sampler_presets(model: Path) -> tuple[dict[str, Any], ...]:
    payload = _load_recommendation_payload(model)
    if payload is None:
        return ()
    sampling = payload.get("recommended_sampling")
    if not isinstance(sampling, dict):
        return ()
    presets: list[dict[str, Any]] = []
    canonical = sampling.get("canonical_preset")
    for name, values in sampling.items():
        if isinstance(values, dict):
            presets.append({"name": str(name), "values": _sampler_values(values)})
    top_level = _sampler_values(sampling)
    if top_level:
        top_name = str(canonical or "hf_primary")
        if not any(item["name"] == top_name for item in presets):
            presets.insert(0, {"name": top_name, "values": top_level})
    return tuple(item for item in presets if item["values"])


def recommended_sampler_flags(model: Path) -> tuple[str, ...]:
    presets = recommended_sampler_presets(model)
    if not presets:
        return ()
    return sampler_flags_from_values(presets[0]["values"])


def sampler_flags_from_values(values: dict[str, Any]) -> tuple[str, ...]:
    args: list[str] = []
    for key, flag in _SAMPLER_FLAG_BY_KEY.items():
        value = values.get(key)
        if value is None:
            continue
        args.extend([flag, _format_value(value)])
    return tuple(args)


def sampling_payload_from_server_args(args: tuple[str, ...]) -> dict[str, object]:
    values: dict[str, object] = {}
    index = 0
    while index < len(args):
        flag, inline = _split_flag(args[index])
        key = _key_for_flag(flag)
        if key is None:
            index += 1
            continue
        raw_value = inline
        if raw_value is None and index + 1 < len(args):
            raw_value = args[index + 1]
            index += 1
        if raw_value is not None:
            values[key] = _coerce_sampler_value(key, raw_value)
        index += 1
    return values


def _load_recommendation_payload(model: Path) -> dict[str, Any] | None:
    model_key = _normalize(model.stem)
    if not _DATA_ROOT.is_dir():
        return None
    for path in sorted(_DATA_ROOT.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        aliases = [payload.get("model_family"), payload.get("base_repo_id")]
        aliases.extend(payload.get("aliases") or [])
        aliases.extend(payload.get("matched_local_dirs") or [])
        normalized_aliases = [_normalize(str(alias)) for alias in aliases if alias]
        if any(
            alias and (alias in model_key or model_key in alias) for alias in normalized_aliases
        ):
            return payload
    return None


def _sampler_values(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: raw[key] for key in _SAMPLER_FLAG_BY_KEY if raw.get(key) is not None}


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _split_flag(token: str) -> tuple[str, str | None]:
    if token.startswith("-") and "=" in token:
        flag, inline = token.split("=", 1)
        return flag, inline
    return token, None


def _key_for_flag(flag: str) -> str | None:
    for key, candidate in _SAMPLER_FLAG_BY_KEY.items():
        if candidate == flag:
            return "repeat_penalty" if key == "repetition_penalty" else key
    return None


def _coerce_sampler_value(key: str, raw_value: str) -> object:
    if key == "top_k":
        return int(raw_value)
    return float(raw_value)
