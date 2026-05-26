from __future__ import annotations

from dataclasses import dataclass, field


def context_ladder(max_context: int = 262_144) -> list[int]:
    ladder = [4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 163_840, 196_608, 229_376, 262_144]
    return [value for value in ladder if value <= max_context]


def refine_context_boundary(last_pass: int, first_fail: int) -> list[int]:
    probes: list[int] = []
    if last_pass == 65_536 and first_fail == 131_072:
        probes.append(98_304)
    midpoint = _round_to_4k((last_pass + first_fail) // 2)
    if last_pass < midpoint < first_fail and midpoint not in probes:
        probes.append(midpoint)
    for step in (16_384, 8_192, 4_096):
        candidate = first_fail - step
        if last_pass < candidate < first_fail and candidate not in probes:
            probes.append(candidate)
    return probes


def _round_to_4k(value: int) -> int:
    return max(4_096, round(value / 4_096) * 4_096)


@dataclass
class ContextLimitPlanner:
    results: dict[int, bool] = field(default_factory=dict)

    def record(self, context_size: int, ok: bool) -> None:
        self.results[context_size] = ok

    @property
    def max_passing_context(self) -> int | None:
        passing = [ctx for ctx, ok in self.results.items() if ok]
        return max(passing) if passing else None

    @property
    def first_failing_context(self) -> int | None:
        failing = [ctx for ctx, ok in self.results.items() if not ok]
        return min(failing) if failing else None

    def verdict(self) -> str:
        if self.max_passing_context is None:
            return "failed"
        if self.first_failing_context is None:
            return "candidate"
        return "needs_refinement"
