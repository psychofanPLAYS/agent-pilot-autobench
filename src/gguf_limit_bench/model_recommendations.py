from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
import shlex
from typing import Any, Callable

from gguf_limit_bench.runtime_capabilities import LlamaCapabilities


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

_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", re.I | re.S)
_PROSE_PATTERNS: dict[str, tuple[re.Pattern[str], Callable[[str], Any]]] = {
    "temperature": (re.compile(r"\btemp(?:erature)?\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", re.I), float),
    "top_p": (re.compile(r"\btop[_ -]?p\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", re.I), float),
    "top_k": (re.compile(r"\btop[_ -]?k\b\s*[:=]?\s*([0-9]+)", re.I), int),
    "min_p": (re.compile(r"\bmin[_ -]?p\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", re.I), float),
    "presence_penalty": (
        re.compile(r"\bpresence[_ -]?penalty\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", re.I),
        float,
    ),
    "repetition_penalty": (
        re.compile(r"\brep(?:etition)?[_ -]?penalty\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", re.I),
        float,
    ),
}
_PROSE_CONTEXT_RE = re.compile(
    r"\b(?:context|ctx|sequence length|max_position_embeddings)\b[^.\n]{0,80}?"
    r"([0-9][0-9_, ]*)(?:\s*[kK])?\b",
    re.I,
)
_RECOMMENDATION_HINT_RE = re.compile(
    r"\b(recommend|recommended|suggest|use|should|default|sampling|inference|generation|llama\.cpp|llama-server)\b",
    re.I,
)
_CONFIG_KEYS: dict[str, tuple[str, Callable[[Any], Any]]] = {
    "temperature": ("temperature", float),
    "top_p": ("top_p", float),
    "top_k": ("top_k", int),
    "min_p": ("min_p", float),
    "presence_penalty": ("presence_penalty", float),
    "repetition_penalty": ("repetition_penalty", float),
    "max_position_embeddings": ("context_size", int),
    "model_max_length": ("context_size", int),
}


def extract_recommendations(
    readme: str,
    *,
    source: RecommendationSource,
    auxiliary_files: dict[str, str] | None = None,
) -> tuple[Recommendation, ...]:
    recommendations: list[Recommendation] = []
    for match in _CODE_BLOCK_RE.finditer(readme):
        block = match.group(1).strip()
        if "llama-server" not in block:
            continue
        recommendations.extend(_parse_command_block(block, source))
    recommendations.extend(_parse_recommendation_prose(readme, source))
    recommendations.extend(_parse_auxiliary_files(auxiliary_files or {}, source))

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


def validate_recommendations(
    recommendations: tuple[Recommendation, ...],
    capabilities: LlamaCapabilities,
) -> tuple[Recommendation, ...]:
    validated: list[Recommendation] = []
    for item in recommendations:
        flag = item.evidence.split(maxsplit=1)[0]
        if not flag.startswith("-"):
            validated.append(item)
            continue
        if capabilities.supports(flag):
            validated.append(
                replace(
                    item,
                    confidence="locally_validated",
                    local_validation="supported",
                )
            )
        elif capabilities.is_removed(flag):
            validated.append(replace(item, confidence="rejected", local_validation="removed"))
        else:
            validated.append(replace(item, confidence="rejected", local_validation="unsupported"))
    return tuple(validated)


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
    *,
    confidence: str = "publisher_claim",
    parser: str = "fenced_llama_server_command",
) -> Recommendation:
    return Recommendation(
        key=key,
        value=value,
        confidence=confidence,
        source_url=source.url,
        revision=source.revision,
        evidence=evidence,
        parser=parser,
    )


def _parse_recommendation_prose(readme: str, source: RecommendationSource) -> list[Recommendation]:
    recommendations: list[Recommendation] = []
    for paragraph in _candidate_prose_blocks(readme):
        for key, (pattern, converter) in _PROSE_PATTERNS.items():
            match = pattern.search(paragraph)
            if match is None:
                continue
            try:
                value = converter(match.group(1).replace(",", ""))
            except ValueError:
                continue
            recommendations.append(
                _recommendation(
                    key,
                    value,
                    _evidence_excerpt(paragraph),
                    source,
                    confidence="publisher_claim",
                    parser="recommended_settings_prose",
                )
            )
        context_match = _PROSE_CONTEXT_RE.search(paragraph)
        if context_match is not None:
            value = _parse_context_number(context_match.group(1), paragraph)
            if value is not None:
                recommendations.append(
                    _recommendation(
                        "context_size",
                        value,
                        _evidence_excerpt(paragraph),
                        source,
                        confidence="publisher_claim",
                        parser="recommended_settings_prose",
                    )
                )
    return recommendations


def _candidate_prose_blocks(readme: str) -> list[str]:
    without_code = _CODE_BLOCK_RE.sub("\n", readme)
    blocks = re.split(r"\n\s*\n|(?<=\.)\s+(?=[A-Z])", without_code)
    return [
        " ".join(block.split())
        for block in blocks
        if len(block.strip()) >= 12 and _RECOMMENDATION_HINT_RE.search(block)
    ]


def _parse_auxiliary_files(
    auxiliary_files: dict[str, str],
    source: RecommendationSource,
) -> list[Recommendation]:
    recommendations: list[Recommendation] = []
    for filename, content in sorted(auxiliary_files.items()):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for source_key, (key, converter) in _CONFIG_KEYS.items():
            if source_key not in payload:
                continue
            try:
                value = converter(payload[source_key])
            except (TypeError, ValueError):
                continue
            recommendations.append(
                _recommendation(
                    key,
                    value,
                    f"{filename}:{source_key}",
                    source,
                    confidence="hub_config_default",
                    parser="hub_config_json",
                )
            )
        chat_template = payload.get("chat_template")
        if isinstance(chat_template, str) and chat_template.strip():
            recommendations.append(
                _recommendation(
                    "jinja",
                    True,
                    f"{filename}:chat_template",
                    source,
                    confidence="hub_config_default",
                    parser="hub_config_json",
                )
            )
    return recommendations


def _parse_context_number(raw_value: str, paragraph: str) -> int | None:
    compact = raw_value.replace(",", "").replace("_", "").replace(" ", "")
    if not compact.isdigit():
        return None
    value = int(compact)
    suffix_match = re.search(re.escape(raw_value) + r"\s*[kK]\b", paragraph)
    if suffix_match and value < 1000:
        return value * 1024
    return value


def _evidence_excerpt(text: str, *, limit: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."
