"""Find the largest context a model can actually serve — safely, from the bottom up.

Rules (from how people really run local models):

* **Start small and climb.** Begin at 16k and work up the ladder; never launch a
  huge context first and hope.
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

from gguf_limit_bench.context_search import (
    ContextLimitPlanner,
    context_ladder,
    refine_context_boundary,
)
from gguf_limit_bench.oom import is_oom_failure, oom_failure_label

DEFAULT_MIN_CONTEXT = 16_384
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
    """The 16k-and-up context ladder, ascending, within [min, max]."""
    tiers = [tier for tier in context_ladder(max_context) if tier >= min_context]
    return tiers or [min_context]


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
    for probe in refine_context_boundary(last_pass, first_oom):
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
            break
