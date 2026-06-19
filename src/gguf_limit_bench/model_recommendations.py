from __future__ import annotations

from dataclasses import dataclass, replace
import re
import shlex
from typing import Any, Callable


@dataclass(frozen=True)
class RecommendationSource:
    url: str
    revision: str


@dataclass(frozen=True)
class Recommendation:
    key: str
    value: str | int | float | bool
    confidence: str
    source_url: str
    revision: str
    evidence: str
    parser: str
    local_validation: str = "not_checked"
    conflicted: bool = False


_VALUE_FLAGS: dict[str, tuple[str, Callable[[str], Any]]] = {
    "--temp": ("temperature", float),
    "--top-p": ("top_p", float),
    "--top-k": ("top_k", int),
    "--min-p": ("min_p", float),
    "--presence-penalty": ("presence_penalty", float),
    "-c": ("context_size", int),
    "--ctx-size": ("context_size", int),
    "-np": ("parallel", int),
    "--parallel": ("parallel", int),
    "-ngl": ("gpu_layers", int),
    "--gpu-layers": ("gpu_layers", int),
    "-b": ("batch_size", int),
    "--batch-size": ("batch_size", int),
    "-ub": ("ubatch_size", int),
    "--ubatch-size": ("ubatch_size", int),
    "--cache-type-k": ("cache_type_k", str),
    "--cache-type-v": ("cache_type_v", str),
    "--spec-type": ("spec_type", str),
    "--spec-draft-n-max": ("spec_draft_n_max", int),
    "--spec-draft-n-min": ("spec_draft_n_min", int),
    "--spec-draft-p-min": ("spec_draft_p_min", float),
}

_BOOLEAN_FLAGS = {
    "--jinja": ("jinja", True),
    "-fa": ("flash_attention", True),
    "--flash-attn": ("flash_attention", True),
    "--kv-unified": ("kv_unified", True),
}

_CODE_BLOCK_RE = re.compile(r"```(?:bash|sh|shell|powershell)?\s*\n(.*?)```", re.I | re.S)


def extract_recommendations(
    readme: str,
    *,
    source: RecommendationSource,
) -> tuple[Recommendation, ...]:
    recommendations: list[Recommendation] = []
    for match in _CODE_BLOCK_RE.finditer(readme):
        block = match.group(1).strip()
        if "llama-server" not in block:
            continue
        recommendations.extend(_parse_command_block(block, source))

    values_by_key: dict[str, set[str]] = {}
    for item in recommendations:
        values_by_key.setdefault(item.key, set()).add(repr(item.value))
    conflicted_keys = {key for key, values in values_by_key.items() if len(values) > 1}
    return tuple(replace(item, conflicted=item.key in conflicted_keys) for item in recommendations)


def recommendation_values(
    recommendations: tuple[Recommendation, ...] | list[Recommendation],
) -> dict[str, str | int | float | bool]:
    values: dict[str, str | int | float | bool] = {}
    for item in recommendations:
        if not item.conflicted:
            values[item.key] = item.value
    return values


def _parse_command_block(
    block: str,
    source: RecommendationSource,
) -> list[Recommendation]:
    normalized = block.replace("\\\n", " ")
    try:
        tokens = shlex.split(normalized, posix=True)
    except ValueError:
        return []

    recommendations: list[Recommendation] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        flag, inline_value = _split_flag(token)
        if flag in _BOOLEAN_FLAGS:
            key, value = _BOOLEAN_FLAGS[flag]
            recommendations.append(_recommendation(key, value, flag, source))
            index += 1
            continue
        if flag in _VALUE_FLAGS:
            raw_value = inline_value
            if raw_value is None and index + 1 < len(tokens):
                raw_value = tokens[index + 1]
                index += 1
            if raw_value is not None:
                key, converter = _VALUE_FLAGS[flag]
                try:
                    value = converter(raw_value)
                except ValueError:
                    pass
                else:
                    recommendations.append(
                        _recommendation(key, value, f"{flag} {raw_value}", source)
                    )
            index += 1
            continue
        if token.startswith("-") and inline_value is None:
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
                index += 1
        index += 1
    return recommendations


def _split_flag(token: str) -> tuple[str, str | None]:
    if token.startswith("-") and "=" in token:
        flag, value = token.split("=", 1)
        return flag, value
    return token, None


def _recommendation(
    key: str,
    value: str | int | float | bool,
    evidence: str,
    source: RecommendationSource,
) -> Recommendation:
    return Recommendation(
        key=key,
        value=value,
        confidence="publisher_claim",
        source_url=source.url,
        revision=source.revision,
        evidence=evidence,
        parser="fenced_llama_server_command",
    )
