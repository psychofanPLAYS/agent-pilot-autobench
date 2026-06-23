"""Find the largest context a model can actually serve — safely, from the bottom up.

Rules (from how people really run local models):

* **Start useful and climb.** Begin at 32k and work up by 32k. 16k remains the
  absolute floor when a caller explicitly asks for it, not the default target.
* **q8_0 KV cache is the default.** Nobody benchmarks KV at f16, so the search
  uses q8_0 cache unless told otherwise.
* **An OOM is data, not a crash.** When a tier fails to allocate VRAM we record
  it, note the context that broke, and stop climbing (optionally refining the
  boundary just below it) instead of letting the run die.
* **Don't even try the impossible.** If a VRAM estimate says a tier cannot fit,
  skip launching it — but a tier that *should* fit is still confirmed by a real
  launch, because the estimate is only an upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from gguf_limit_bench.context_search import ContextLimitPlanner
from gguf_limit_bench.programs import (
    FIT_ASCENT_STEP,
    FIT_BACKOFF_STEP,
    FIT_REFINE_STEP,
    FIT_START_CONTEXT_SIZE,
    MIN_SERIOUS_CONTEXT_SIZE,
)
from gguf_limit_bench.oom import is_oom_failure, oom_failure_label

DEFAULT_MIN_CONTEXT = FIT_START_CONTEXT_SIZE
DEFAULT_KV_CACHE_TYPE = "q8_0"


@dataclass(frozen=True)
class LaunchOutcome:
    """Result of trying to bring up the server at one context size."""

    ok: bool
    stderr: str = ""
    returncode: int | None = None
    detail: str = ""


@dataclass
class ContextAttempt:
    context_size: int
    ok: bool
    outcome: str  # "passed" | "oom" | "failed" | "skipped_vram"
    note: str = ""


@dataclass
class ContextLimitResult:
    max_context: int | None
    attempts: list[ContextAttempt] = field(default_factory=list)

    @property
    def hit_oom(self) -> bool:
        return any(a.outcome == "oom" for a in self.attempts)


def ascending_ladder(min_context: int, max_context: int) -> list[int]:
    """The fit ladder, ascending, within [min, max].

    The useful default start is 32k. If the caller explicitly asks for 16k,
    include it as a floor probe before the 32k-step sequence.
    """
    start = max(min_context, MIN_SERIOUS_CONTEXT_SIZE)
    tiers: list[int] = []
    if start < FIT_START_CONTEXT_SIZE:
        tiers.append(start)
        current = FIT_START_CONTEXT_SIZE
    else:
        current = _round_up_to_step(start, FIT_ASCENT_STEP)
    while current <= max_context:
        tiers.append(current)
        current += FIT_ASCENT_STEP
    return tiers or [start]


def find_context_limit(
    attempt: Callable[[int], LaunchOutcome],
    *,
    min_context: int = DEFAULT_MIN_CONTEXT,
    max_context: int = 262_144,
    fits_vram: Callable[[int], bool] | None = None,
    refine: bool = True,
    log: Callable[[str], None] | None = None,
) -> ContextLimitResult:
    """Climb the context ladder until a tier OOMs (or we run out of tiers).

    Parameters
    ----------
    attempt:
        Brings the server up at a context size and returns a :class:`LaunchOutcome`.
        The caller wires this to a real llama-server launch (with q8_0 KV).
    fits_vram:
        Optional pre-flight predicate; when it returns False for a tier we skip
        the launch (the tier is recorded as ``skipped_vram``) and stop climbing,
        since every larger tier needs even more memory.
    refine:
        After an OOM, binary-search the gap between the last pass and the OOM to
        recover a little more usable context.
    """
    emit = log or (lambda _message: None)
    planner = ContextLimitPlanner()
    attempts: list[ContextAttempt] = []
    tiers = ascending_ladder(min_context, max_context)

    first_oom: int | None = None
    for tier in tiers:
        if fits_vram is not None and not fits_vram(tier):
            emit(f"skip {tier // 1024}k: VRAM estimate says it will not fit")
            attempts.append(
                ContextAttempt(tier, ok=False, outcome="skipped_vram", note="vram estimate")
            )
            break

        emit(f"try {tier // 1024}k context...")
        outcome = attempt(tier)
        if outcome.ok:
            planner.record(tier, True)
            attempts.append(ContextAttempt(tier, ok=True, outcome="passed"))
            emit(f"  {tier // 1024}k OK")
            continue

        if is_oom_failure(outcome.stderr, outcome.returncode):
            planner.record(tier, False)
            note = oom_failure_label(tier)
            attempts.append(ContextAttempt(tier, ok=False, outcome="oom", note=note))
            emit(f"  {tier // 1024}k OOM — noted, backing off")
            first_oom = tier
            break

        # Non-OOM failure (timeout, bad flag, crash): record and stop, but don't
        # claim a memory ceiling.
        planner.record(tier, False)
        attempts.append(
            ContextAttempt(tier, ok=False, outcome="failed", note=outcome.detail or "launch failed")
        )
        emit(f"  {tier // 1024}k failed: {outcome.detail or 'launch failed'}")
        break

    if refine and first_oom is not None:
        _refine_boundary(attempt, planner, attempts, first_oom, emit)

    return ContextLimitResult(max_context=planner.max_passing_context, attempts=attempts)


def _refine_boundary(
    attempt: Callable[[int], LaunchOutcome],
    planner: ContextLimitPlanner,
    attempts: list[ContextAttempt],
    first_oom: int,
    emit: Callable[[str], None],
) -> None:
    last_pass = planner.max_passing_context
    if last_pass is None:
        return  # even the smallest tier OOMed; nothing to refine
    probes: list[int] = []
    backoff_probe = first_oom - FIT_BACKOFF_STEP
    if last_pass < backoff_probe < first_oom:
        probes.append(backoff_probe)
    probe = last_pass + FIT_REFINE_STEP
    while probe < first_oom:
        if probe not in probes:
            probes.append(probe)
        probe += FIT_REFINE_STEP

    for probe in probes:
        if probe <= (planner.max_passing_context or 0):
            continue
        emit(f"refine: try {probe // 1024}k context...")
        outcome = attempt(probe)
        if outcome.ok:
            planner.record(probe, True)
            attempts.append(ContextAttempt(probe, ok=True, outcome="passed", note="refined"))
            emit(f"  {probe // 1024}k OK")
        else:
            ok_oom = is_oom_failure(outcome.stderr, outcome.returncode)
            planner.record(probe, False)
            attempts.append(
                ContextAttempt(
                    probe,
                    ok=False,
                    outcome="oom" if ok_oom else "failed",
                    note=oom_failure_label(probe) if ok_oom else (outcome.detail or "failed"),
                )
            )
            emit(f"  {probe // 1024}k {'OOM' if ok_oom else 'failed'}")
            if ok_oom and probe == backoff_probe:
                continue
            break


def _round_up_to_step(value: int, step: int) -> int:
    return ((value + step - 1) // step) * step
