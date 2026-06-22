from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gguf_limit_bench.autoresearch import AutoresearchSettings
from gguf_limit_bench.runtime_capabilities import LlamaRuntimeCapabilities


MANAGED_SERVER_ARGS = frozenset({"--host", "--port", "--model", "-m"})
MTP_REQUIRED_OPTIONS = ("--spec-type", "--spec-draft-n-max")


@dataclass(frozen=True)
class FlagLadderProfile:
    name: str
    settings: AutoresearchSettings
    hypothesis: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "hypothesis": self.hypothesis,
            "settings": self.settings.to_dict(),
        }


def build_core_flag_ladder(
    *,
    context_size: int = 4096,
    parallel_max: int = 6,
    extra_server_args: tuple[str, ...] = (),
    enable_mtp: bool = False,
) -> tuple[AutoresearchSettings, ...]:
    base = AutoresearchSettings(
        profile_name="L0-baseline",
        context_size=context_size,
        parallel=1,
        gpu_layers=99,
        batch_size=2048,
        ubatch_size=512,
        flash_attention=True,
        kv_unified=False,
        cont_batching=True,
        extra_server_args=extra_server_args,
    )
    stripped = _copy(
        base,
        profile_name="Lmin-stripped",
        flash_attention=False,
        cont_batching=False,
    )
    # Speed flags are measured at single stream (parallel=1) so the comparison is a
    # clean single-stream tok/s ranking, not confounded by concurrency.
    comparison_base = _copy(base, profile_name="L2-kv-unified", kv_unified=True)
    q8_profile = _copy(
        comparison_base,
        profile_name="L6-q8-kv",
        cache_type_k="q8_0",
        cache_type_v="q8_0",
    )
    ladder = [
        stripped,
        base,
        comparison_base,
        _copy(
            comparison_base,
            profile_name="L3-ram-cache",
            cache_ram_mb=22_528,
        ),
        _copy(
            comparison_base,
            profile_name="L4-cache-reuse",
            cache_reuse=512,
        ),
        _copy(
            comparison_base,
            profile_name="L5-checkpoints",
            ctx_checkpoints=16,
        ),
        q8_profile,
    ]
    for threads in (12, 16, 24, 32):
        ladder.append(
            _copy(
                q8_profile,
                profile_name=f"T{threads}-threads",
                threads=threads,
                threads_batch=threads,
            )
        )
    if enable_mtp:
        ladder.append(
            _copy(
                comparison_base,
                profile_name="MTP-draft-3",
                spec_type="draft-mtp",
                spec_draft_n_max=3,
            )
        )
    # Parallel adds concurrent-request CAPABILITY, not single-stream speed, so it is
    # tested LAST on the kv-unified base. Slot counts mirror a server that allows one
    # heavy plus two light requests (1+2) without choking the GPU.
    for slots in (2, 3):
        if slots <= max(1, parallel_max):
            ladder.append(_copy(comparison_base, profile_name=f"Lpar-{slots}", parallel=slots))
    return tuple(ladder)


def profile_descriptions(
    *,
    context_size: int = 4096,
    parallel_max: int = 6,
    extra_server_args: tuple[str, ...] = (),
    enable_mtp: bool = False,
) -> tuple[FlagLadderProfile, ...]:
    hypotheses = {
        "Lmin-stripped": "Fewest flags: no flash-attn, no continuous batching. "
        "Does removing flags help or hurt speed?",
        "L0-baseline": "Plain llama-server baseline for comparison.",
        "L2-kv-unified": "Measure unified KV behavior against baseline (single stream).",
        "L3-ram-cache": "Measure prompt RAM cache overhead against L2.",
        "L4-cache-reuse": "Measure cache reuse overhead against L2.",
        "L5-checkpoints": "Measure context checkpoint overhead against L2.",
        "L6-q8-kv": "Measure q8 KV quality/speed/memory tradeoff against L2.",
    }
    profiles = []
    for settings in build_core_flag_ladder(
        context_size=context_size,
        parallel_max=parallel_max,
        extra_server_args=extra_server_args,
        enable_mtp=enable_mtp,
    ):
        profiles.append(
            FlagLadderProfile(
                name=settings.profile_name,
                settings=settings,
                hypothesis=hypotheses.get(
                    settings.profile_name,
                    (
                        "Measure native MTP self-draft overhead and throughput."
                        if settings.profile_name.startswith("MTP-")
                        else (
                            "Concurrent-request capability (parallel slots), tested last."
                            if settings.profile_name.startswith("Lpar-")
                            else "Thread sweep over the best q8 KV profile."
                        )
                    ),
                ),
            )
        )
    return tuple(profiles)


def llama_server_args_for_settings(settings: AutoresearchSettings) -> list[str]:
    validate_native_spec_settings(settings)
    args: list[str] = []
    if settings.cont_batching:
        args.append("--cont-batching")
    if settings.kv_unified:
        args.append("--kv-unified")
    if settings.cache_ram_mb is not None:
        args.extend(["--cache-ram", str(settings.cache_ram_mb)])
    if settings.cache_idle_slots:
        args.append("--cache-idle-slots")
    if settings.cache_reuse is not None:
        args.extend(["--cache-reuse", str(settings.cache_reuse)])
    if settings.ctx_checkpoints is not None:
        args.extend(["--ctx-checkpoints", str(settings.ctx_checkpoints)])
    if settings.checkpoint_min_step is not None:
        args.extend(["--checkpoint-min-step", str(settings.checkpoint_min_step)])
    if settings.cache_type_k is not None:
        args.extend(["--cache-type-k", settings.cache_type_k])
    if settings.cache_type_v is not None:
        args.extend(["--cache-type-v", settings.cache_type_v])
    if settings.threads is not None:
        args.extend(["--threads", str(settings.threads)])
    if settings.threads_batch is not None:
        args.extend(["--threads-batch", str(settings.threads_batch)])
    if settings.spec_type is not None:
        args.extend(["--spec-type", settings.spec_type])
    if settings.spec_draft_n_max is not None:
        args.extend(["--spec-draft-n-max", str(settings.spec_draft_n_max)])
    if settings.spec_draft_n_min is not None:
        args.extend(["--spec-draft-n-min", str(settings.spec_draft_n_min)])
    if settings.spec_draft_p_min is not None:
        args.extend(["--spec-draft-p-min", str(settings.spec_draft_p_min)])
    args.extend(settings.extra_server_args)
    return args


def validate_native_spec_settings(settings: AutoresearchSettings) -> None:
    n_max = settings.spec_draft_n_max
    n_min = settings.spec_draft_n_min
    p_min = settings.spec_draft_p_min
    if settings.spec_type == "draft-mtp" and n_max is not None and not 1 <= n_max <= 4:
        raise ValueError("draft-mtp spec_draft_n_max must be between 1 and 4")
    if n_min is not None and n_min < 0:
        raise ValueError("spec_draft_n_min must be nonnegative")
    if n_min is not None and n_max is not None and n_min > n_max:
        raise ValueError("spec_draft_n_min cannot exceed spec_draft_n_max")
    if p_min is not None and not 0 <= p_min <= 1:
        raise ValueError("spec_draft_p_min must be between 0 and 1")


def validate_extra_server_args(args: tuple[str, ...]) -> tuple[str, ...]:
    for argument in args:
        option = argument.split("=", 1)[0]
        if option in MANAGED_SERVER_ARGS:
            raise ValueError(f"{option} is managed by Agent Pilot Autobench")
    return args


def filter_unsupported_profiles(
    profiles: tuple[AutoresearchSettings, ...],
    runtime_capabilities: LlamaRuntimeCapabilities,
) -> tuple[tuple[AutoresearchSettings, ...], tuple[dict, ...]]:
    supported = []
    skipped = []
    for settings in profiles:
        missing_options = (
            tuple(
                option
                for option in MTP_REQUIRED_OPTIONS
                if not runtime_capabilities.supports(option)
            )
            if settings.spec_type == "draft-mtp"
            else ()
        )
        if missing_options:
            skipped.append(
                {
                    "profile": settings.profile_name,
                    "reason": "runtime lacks required options: " + ", ".join(missing_options),
                    "runtime_version": runtime_capabilities.version,
                    "runtime_help_sha256": runtime_capabilities.help_sha256,
                    "runtime_introspection_ok": runtime_capabilities.introspection_ok,
                }
            )
        else:
            supported.append(settings)
    return tuple(supported), tuple(skipped)


def build_flag_ladder_plan(
    *,
    llama_server: Path,
    model: Path,
    host: str,
    port: int,
    context_size: int,
    parallel_max: int,
    extra_server_args: tuple[str, ...] = (),
    enable_mtp: bool = False,
    runtime_capabilities: LlamaRuntimeCapabilities | None = None,
) -> list[dict]:
    from gguf_limit_bench.server_probe import build_llama_server_command

    rows = []
    for profile in profile_descriptions(
        context_size=context_size,
        parallel_max=parallel_max,
        extra_server_args=extra_server_args,
        enable_mtp=enable_mtp,
    ):
        missing_options = (
            tuple(
                option
                for option in MTP_REQUIRED_OPTIONS
                if not runtime_capabilities.supports(option)
            )
            if runtime_capabilities is not None and profile.name.startswith("MTP-")
            else ()
        )
        supported = not missing_options
        rows.append(
            {
                **profile.to_dict(),
                "supported": supported,
                "unsupported_reason": (
                    "runtime lacks required options: " + ", ".join(missing_options)
                    if missing_options
                    else None
                ),
                "command": (
                    build_llama_server_command(
                        llama_server=llama_server,
                        model=model,
                        settings=profile.settings,
                        host=host,
                        port=port,
                    )
                    if supported
                    else None
                ),
            }
        )
    return rows


def _copy(settings: AutoresearchSettings, **changes) -> AutoresearchSettings:
    payload = settings.to_dict()
    payload.update(changes)
    extra_args = payload.get("extra_server_args", ())
    if isinstance(extra_args, list):
        payload["extra_server_args"] = tuple(extra_args)
    return AutoresearchSettings(**payload)
