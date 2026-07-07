from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from gguf_limit_bench.autoresearch import AutoresearchSettings
from gguf_limit_bench.discovery import is_non_generative_gguf, parse_model_name
from gguf_limit_bench.gpu_profiles import recommended_always_on, recommended_parallel
from gguf_limit_bench.hf_recommended_settings import recommended_sampler_flags
from gguf_limit_bench.server_probe import build_llama_server_command
from gguf_limit_bench.template_recommend import merge_flags, recommended_model_flags


STANDARD_AGENT_CONTEXT = 131_072
DAVID_LONG_CONTEXT_TARGET = 200_000
FULL_NATIVE_CONTEXT = 262_144
QE_CONTEXT = 20_480
_PROFILE_MANAGED_OPTIONS = frozenset(
    {
        "--flash-attn",
        "--kv-unified",
        "--cache-type-k",
        "--cache-type-v",
        "--gpu-layers",
    }
)
_PROFILE_MANAGED_OPTIONS_WITH_VALUES = frozenset(
    {"--flash-attn", "--cache-type-k", "--cache-type-v", "--gpu-layers"}
)


@dataclass(frozen=True)
class FlagRecommendationOutputs:
    json_path: Path
    markdown_path: Path


def write_flag_recommendations(
    *,
    model: Path,
    llama_server: Path,
    output_dir: Path,
    gpu_name: str,
    search_roots: tuple[Path, ...] = (),
    host: str = "127.0.0.1",
    port: int = 8080,
) -> FlagRecommendationOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_flag_recommendations(
        model=model,
        llama_server=llama_server,
        gpu_name=gpu_name,
        search_roots=search_roots,
        host=host,
        port=port,
    )
    json_path = output_dir / "flag-recommendations.json"
    markdown_path = output_dir / "flag-recommendations.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return FlagRecommendationOutputs(json_path=json_path, markdown_path=markdown_path)


def build_flag_recommendations(
    *,
    model: Path,
    llama_server: Path,
    gpu_name: str,
    search_roots: tuple[Path, ...] = (),
    host: str = "127.0.0.1",
    port: int = 8080,
) -> dict[str, Any]:
    info = parse_model_name(model)
    lane_type = "query_expansion" if is_non_generative_gguf(model) else "chat_agent"
    profiles = (
        _qe_profiles(model, llama_server, host, port)
        if lane_type == "query_expansion"
        else _chat_profiles(model, llama_server, gpu_name, search_roots, host, port)
    )
    return {
        "schema_version": 1,
        "model": str(model),
        "model_name": model.name,
        "family": info.family,
        "parameters": info.parameters,
        "quant": info.quant,
        "lane_type": lane_type,
        "gpu_name": gpu_name or "unknown",
        "profiles": profiles,
        "notes": _notes(lane_type),
    }


def _chat_profiles(
    model: Path,
    llama_server: Path,
    gpu_name: str,
    search_roots: tuple[Path, ...],
    host: str,
    port: int,
) -> list[dict[str, Any]]:
    max_context = _max_native_context(model)
    standard_context = min(STANDARD_AGENT_CONTEXT, max_context)
    long_context = min(DAVID_LONG_CONTEXT_TARGET, max_context)
    over_context = max_context
    is_mtp = "mtp" in model.name.lower()
    parallel = 1 if is_mtp else recommended_parallel(gpu_name)
    bare_flags = _chat_correctness_flags(model, search_roots)
    common_flags = _chat_common_flags(model, gpu_name, search_roots)
    rows = [
        _profile(
            profile_id="bare_minimum",
            label="Bare minimum",
            recommendation="Correct template, full GPU offload, and 128k-class context.",
            settings=AutoresearchSettings(
                profile_name="bare_minimum",
                context_size=standard_context,
                parallel=1,
                gpu_layers=99,
                batch_size=1024,
                ubatch_size=512,
                flash_attention=True,
                kv_unified=False,
                cont_batching=False,
                extra_server_args=bare_flags,
            ),
            llama_server=llama_server,
            model=model,
            host=host,
            port=port,
        ),
        _profile(
            profile_id="standard",
            label="Standard",
            recommendation="Default agent serving profile: 128k context, q8_0 KV, metrics, slots.",
            settings=AutoresearchSettings(
                profile_name="standard",
                context_size=standard_context,
                parallel=min(parallel, 4),
                gpu_layers=99,
                batch_size=2048,
                ubatch_size=512,
                flash_attention=True,
                kv_unified=True,
                cache_type_k="q8_0",
                cache_type_v="q8_0",
                extra_server_args=common_flags,
            ),
            llama_server=llama_server,
            model=model,
            host=host,
            port=port,
        ),
        _profile(
            profile_id="long_agent",
            label="Long agent",
            recommendation=_long_agent_recommendation(model),
            settings=AutoresearchSettings(
                profile_name="long_agent",
                context_size=long_context,
                parallel=1,
                gpu_layers=99,
                batch_size=2048,
                ubatch_size=512,
                flash_attention=True,
                kv_unified=True,
                cache_type_k="q8_0",
                cache_type_v="q8_0",
                extra_server_args=common_flags,
            ),
            llama_server=llama_server,
            model=model,
            host=host,
            port=port,
        ),
        _profile(
            profile_id="over_the_top",
            label="Over-the-top",
            recommendation="Full native context target. Use only after fit/fill benchmark receipts prove it.",
            settings=AutoresearchSettings(
                profile_name="over_the_top",
                context_size=over_context,
                parallel=1,
                gpu_layers=99,
                batch_size=1024,
                ubatch_size=512,
                flash_attention=True,
                kv_unified=True,
                cache_type_k="q8_0",
                cache_type_v="q8_0",
                extra_server_args=common_flags,
            ),
            llama_server=llama_server,
            model=model,
            host=host,
            port=port,
        ),
    ]
    return rows


def _qe_profiles(model: Path, llama_server: Path, host: str, port: int) -> list[dict[str, Any]]:
    settings = AutoresearchSettings(
        profile_name="qe_standard",
        context_size=QE_CONTEXT,
        parallel=1,
        gpu_layers=99,
        batch_size=1024,
        ubatch_size=512,
        flash_attention=True,
        kv_unified=False,
        cache_type_k="q4_0",
        cache_type_v="q4_0",
        extra_server_args=("--jinja",),
    )
    return [
        _profile(
            profile_id="qe_standard",
            label="QE standard",
            recommendation=(
                "Query-expansion helper profile: 20k context and q4_0 KV. "
                "Do not use as a chat-answering lane."
            ),
            settings=settings,
            llama_server=llama_server,
            model=model,
            host=host,
            port=port,
        )
    ]


def _profile(
    *,
    profile_id: str,
    label: str,
    recommendation: str,
    settings: AutoresearchSettings,
    llama_server: Path,
    model: Path,
    host: str,
    port: int,
) -> dict[str, Any]:
    _require_paired_kv_cache(settings)
    command = build_llama_server_command(
        llama_server=llama_server,
        model=model,
        settings=settings,
        host=host,
        port=port,
    )
    return {
        "id": profile_id,
        "label": label,
        "recommendation": recommendation,
        "context_size": settings.context_size,
        "parallel": settings.parallel,
        "kv_cache": {
            "k": settings.cache_type_k or "runtime_default",
            "v": settings.cache_type_v or "runtime_default",
            "unified": settings.kv_unified,
        },
        "settings": settings.to_dict(),
        "command": command,
        "command_text": _shell_join(command),
    }


def _require_paired_kv_cache(settings: AutoresearchSettings) -> None:
    if settings.cache_type_k is None and settings.cache_type_v is None:
        return
    if settings.cache_type_k != settings.cache_type_v:
        raise ValueError(
            "flag recommendations require paired K/V cache types; "
            f"got K={settings.cache_type_k!r}, V={settings.cache_type_v!r} "
            f"for profile {settings.profile_name!r}"
        )


def _chat_common_flags(
    model: Path, gpu_name: str, search_roots: tuple[Path, ...]
) -> tuple[str, ...]:
    flags = _chat_correctness_flags(model, search_roots)
    gpu_flags = _drop_options(recommended_always_on(gpu_name), _PROFILE_MANAGED_OPTIONS)
    flags = merge_flags(gpu_flags, flags)
    return flags


def _drop_options(args: tuple[str, ...], options: frozenset[str]) -> tuple[str, ...]:
    kept: list[str] = []
    index = 0
    while index < len(args):
        flag = args[index]
        option = flag.split("=", 1)[0]
        if option in options:
            has_separate_value = (
                option in _PROFILE_MANAGED_OPTIONS_WITH_VALUES
                and "=" not in flag
                and index + 1 < len(args)
            )
            index += 2 if has_separate_value else 1
            continue
        kept.append(flag)
        index += 1
    return tuple(kept)


def _long_agent_recommendation(model: Path) -> str:
    if parse_model_name(model).family == "gemma":
        return (
            "David target profile: 200k context with matched q8_0/q8_0 KV cache. "
            "Keep K and V cache types paired; Gemma long-context stability should be "
            "handled with cache/checkpoint flags and repeat proof, not mixed KV types."
        )
    return "David target profile: 200k context with matched q8_0/q8_0 KV; expect slower decode as KV fills."


def _chat_correctness_flags(model: Path, search_roots: tuple[Path, ...]) -> tuple[str, ...]:
    flags: tuple[str, ...] = ()
    flags = merge_flags(flags, recommended_model_flags(model, search_roots=search_roots))
    flags = merge_flags(flags, recommended_sampler_flags(model))
    if parse_model_name(model).family == "gemma":
        flags = merge_flags(flags, ("--cache-ram", "0", "--ctx-checkpoints", "0"))
    if "mtp" in model.name.lower():
        flags = merge_flags(flags, ("--spec-type", "draft-mtp", "--spec-draft-n-max", "2"))
    return flags


def _max_native_context(model: Path) -> int:
    info = parse_model_name(model)
    params = info.parameters.upper()
    if params.startswith("2B") or params.startswith("4B"):
        return STANDARD_AGENT_CONTEXT
    if info.family in {"qwen", "gemma"}:
        return FULL_NATIVE_CONTEXT
    return STANDARD_AGENT_CONTEXT


def _notes(lane_type: str) -> list[str]:
    if lane_type == "query_expansion":
        return [
            "QE models should emit retrieval payloads, not answer user questions.",
            "Use apb qe-format and apb qe-results before promoting this helper lane.",
        ]
    return [
        "128k is the minimum serious 2026 agent-context target in this recommendation ladder.",
        "200k and 262k modes must still be proven on the local machine with fit/fill receipts.",
        "K/V cache types are intentionally paired; q8_0/q8_0 KV is selected for quality and stable bookkeeping.",
        "Gemma profiles disable llama.cpp prompt-cache RAM and context checkpoints because current Gemma 4 CUDA/SWA builds can crash during long cached workflows with those defaults enabled.",
    ]


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Flag Recommendations: {payload['model_name']}",
        "",
        f"- Lane type: `{payload['lane_type']}`",
        f"- GPU: `{payload['gpu_name']}`",
        "",
        "## Profiles",
        "",
        "| Mode | Context | KV | Parallel | Recommendation |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for profile in payload["profiles"]:
        kv = profile["kv_cache"]
        lines.append(
            f"| {profile['label']} | {profile['context_size']} | "
            f"K={kv['k']}, V={kv['v']}, unified={kv['unified']} | "
            f"{profile['parallel']} | {profile['recommendation']} |"
        )
    lines.extend(["", "## Commands", ""])
    for profile in payload["profiles"]:
        lines.extend(
            [
                f"### {profile['label']}",
                "",
                "```powershell",
                profile["command_text"],
                "```",
                "",
            ]
        )
    if payload.get("notes"):
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in payload["notes"])
        lines.append("")
    return "\n".join(lines)


def _shell_join(command: list[str]) -> str:
    return " ".join(_quote_powershell_token(token) for token in command)


def _quote_powershell_token(token: str) -> str:
    if not token:
        return "''"
    if not re.search(r"[\s{};`'\"]", token):
        return token
    return "'" + token.replace("'", "''") + "'"
