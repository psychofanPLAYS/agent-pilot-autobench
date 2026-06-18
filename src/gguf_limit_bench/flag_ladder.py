from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gguf_limit_bench.autoresearch import AutoresearchSettings


MANAGED_SERVER_ARGS = frozenset({"--host", "--port", "--model", "-m"})


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
    parallel = max(1, min(parallel_max, 6))
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
    comparison_base = _copy(base, profile_name="L2-kv-unified", parallel=parallel, kv_unified=True)
    ladder = [
        base,
        _copy(base, profile_name="L1-parallel", parallel=parallel),
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
        _copy(
            comparison_base,
            profile_name="L6-q8-kv",
            cache_type_k="q8_0",
            cache_type_v="q8_0",
        ),
    ]
    for threads in (12, 16, 24, 32):
        ladder.append(
            _copy(
                comparison_base,
                profile_name=f"T{threads}-threads",
                threads=threads,
                threads_batch=threads,
            )
        )
    if enable_mtp:
        mtp_base = comparison_base
        for draft_max in (8, 16, 32):
            ladder.append(
                _copy(
                    mtp_base,
                    profile_name=f"MTP-draft-{draft_max}",
                    draft_max=draft_max,
                    draft_min=0,
                    draft_p_min=0.75,
                )
            )
    return tuple(ladder)


def profile_descriptions(
    *,
    context_size: int = 4096,
    parallel_max: int = 6,
    extra_server_args: tuple[str, ...] = (),
    enable_mtp: bool = False,
) -> tuple[FlagLadderProfile, ...]:
    hypotheses = {
        "L0-baseline": "Plain llama-server baseline for comparison.",
        "L1-parallel": "Measure throughput cost/benefit from parallel slots.",
        "L2-kv-unified": "Measure unified KV behavior against baseline.",
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
                        else "Thread sweep over the best q8 KV profile."
                    ),
                ),
            )
        )
    return tuple(profiles)


def llama_server_args_for_settings(settings: AutoresearchSettings) -> list[str]:
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
    if settings.draft_max is not None:
        args.extend(["--draft-max", str(settings.draft_max)])
    if settings.draft_min is not None:
        args.extend(["--draft-min", str(settings.draft_min)])
    if settings.draft_p_min is not None:
        args.extend(["--draft-p-min", str(settings.draft_p_min)])
    args.extend(settings.extra_server_args)
    return args


def validate_extra_server_args(args: tuple[str, ...]) -> tuple[str, ...]:
    for argument in args:
        option = argument.split("=", 1)[0]
        if option in MANAGED_SERVER_ARGS:
            raise ValueError(f"{option} is managed by Agent Pilot Autobench")
    return args


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
) -> list[dict]:
    from gguf_limit_bench.server_probe import build_llama_server_command

    rows = []
    for profile in profile_descriptions(
        context_size=context_size,
        parallel_max=parallel_max,
        extra_server_args=extra_server_args,
        enable_mtp=enable_mtp,
    ):
        rows.append(
            {
                **profile.to_dict(),
                "command": build_llama_server_command(
                    llama_server=llama_server,
                    model=model,
                    settings=profile.settings,
                    host=host,
                    port=port,
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
