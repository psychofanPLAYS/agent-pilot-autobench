from gguf_limit_bench.context_limit import (
    LaunchOutcome,
    ascending_ladder,
    find_context_limit,
)

_OOM_STDERR = "cudaMalloc failed: out of memory"


def test_ladder_starts_at_16k_and_ascends():
    ladder = ascending_ladder(16_384, 262_144)
    assert ladder[0] == 16_384
    assert ladder == sorted(ladder)
    assert ladder[-1] == 262_144
    assert all(tier >= 16_384 for tier in ladder)


def test_climbs_until_oom_then_backs_off_and_records():
    # Server OOMs at any context above 64k.
    def attempt(ctx):
        if ctx > 65_536:
            return LaunchOutcome(ok=False, stderr=_OOM_STDERR)
        return LaunchOutcome(ok=True)

    result = find_context_limit(attempt, min_context=16_384, max_context=262_144, refine=False)

    assert result.hit_oom is True
    assert result.max_context == 65_536  # largest that passed
    outcomes = [(a.context_size, a.outcome) for a in result.attempts]
    # It climbed 16k,32k,64k (pass) then 128k (oom) and stopped — did not try 256k.
    assert (65_536, "passed") in outcomes
    assert (131_072, "oom") in outcomes
    assert all(a.context_size <= 131_072 for a in result.attempts)


def test_all_tiers_pass_returns_max():
    result = find_context_limit(lambda ctx: LaunchOutcome(ok=True), max_context=262_144)
    assert result.max_context == 262_144
    assert result.hit_oom is False


def test_vram_guard_skips_and_stops_before_launch():
    launched: list[int] = []

    def attempt(ctx):
        launched.append(ctx)
        return LaunchOutcome(ok=True)

    # VRAM estimate says nothing above 32k fits: 64k+ never launched.
    result = find_context_limit(
        attempt,
        min_context=16_384,
        max_context=262_144,
        fits_vram=lambda ctx: ctx <= 32_768,
    )

    assert max(launched) == 32_768
    assert result.max_context == 32_768
    assert any(a.outcome == "skipped_vram" for a in result.attempts)


def test_refine_recovers_more_context_between_pass_and_oom():
    # Passes <= 96k, OOMs above. Ladder jumps 64k -> 128k; refinement probes 96k.
    def attempt(ctx):
        return LaunchOutcome(ok=True) if ctx <= 98_304 else LaunchOutcome(ok=False, stderr=_OOM_STDERR)

    result = find_context_limit(
        attempt, min_context=16_384, max_context=262_144, refine=True
    )

    assert result.hit_oom is True
    # Without refine the answer would be 64k; refinement should find 96k (98304).
    assert result.max_context == 98_304


def test_non_oom_failure_stops_without_claiming_memory_ceiling():
    def attempt(ctx):
        if ctx == 32_768:
            return LaunchOutcome(ok=False, stderr="some flag error", detail="bad flag")
        return LaunchOutcome(ok=True)

    result = find_context_limit(attempt, min_context=16_384, max_context=262_144, refine=False)

    assert result.hit_oom is False
    assert any(a.outcome == "failed" for a in result.attempts)
    assert result.max_context == 16_384
