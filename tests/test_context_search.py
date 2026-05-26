from gguf_limit_bench.context_search import (
    ContextLimitPlanner,
    context_ladder,
    refine_context_boundary,
)


def test_context_ladder_matches_qwen36_campaign_shape():
    assert context_ladder() == [
        4_096,
        8_192,
        16_384,
        32_768,
        65_536,
        131_072,
        163_840,
        196_608,
        229_376,
        262_144,
    ]


def test_context_boundary_refinement_uses_98k_after_64k_to_128k_failure():
    assert refine_context_boundary(last_pass=65_536, first_fail=131_072)[:4] == [
        98_304,
        114_688,
        122_880,
        126_976,
    ]


def test_context_limit_planner_records_max_pass_and_first_fail():
    planner = ContextLimitPlanner()
    planner.record(16_384, ok=True)
    planner.record(32_768, ok=True)
    planner.record(65_536, ok=False)

    assert planner.max_passing_context == 32_768
    assert planner.first_failing_context == 65_536
    assert planner.verdict() == "needs_refinement"
