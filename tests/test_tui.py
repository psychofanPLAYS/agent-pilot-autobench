import asyncio

from textual.widgets import DataTable, Static

from gguf_limit_bench.evaluation_mode import EvaluationMode
from gguf_limit_bench.modes import RUN_MODES
from gguf_limit_bench.tui import (
    BenchTui,
    candidate_assessment_text,
    candidate_rankings_text,
    decision_runbook_text,
    context_gate_text,
    format_candidate_assessment,
    format_candidate_rankings,
    format_champion_line,
    format_context_gate,
    format_decision_runbook,
    format_operator_verdict,
    format_performance_prediction,
    format_repeatability,
    format_resource_gate,
    format_score_summary,
    format_settings_candidates,
    format_stability_gate,
    format_target_scope,
    operator_verdict_text,
    performance_prediction_text,
    repeatability_text,
    resource_gate_text,
    score_summary_text,
    settings_candidates_text,
    stability_gate_text,
    target_scope_text,
)


def _make_model_dir(tmp_path, names: list[str]):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for index, name in enumerate(names, start=1):
        (model_dir / name).write_bytes(b"1" * index)
    return model_dir


def _path_arg(path) -> str:
    return path.as_posix()


def test_tui_loads_models_and_supports_select_all(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "Qwen3-Small-Q4_K_M.gguf").write_bytes(b"1" * 10)
    (model_dir / "Qwen3-Large-Q4_K_M.gguf").write_bytes(b"1" * 30)

    async def run_tui_check():
        app = BenchTui(root=model_dir)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one("#models", DataTable)
            status = app.query_one("#status", Static)

            assert table.row_count == 2
            assert [model.name for model in app.models] == [
                "Qwen3-Large-Q4_K_M.gguf",
                "Qwen3-Small-Q4_K_M.gguf",
            ]
            assert "2 models found. 0 selected." in str(status.render())
            assert "GB" in [str(column.label) for column in table.columns.values()]

            await pilot.press("a")
            await pilot.pause()

            assert len(app.selection.selected_models()) == 2
            assert "2 selected" in str(status.render())

    asyncio.run(run_tui_check())


def test_history_panel_scales_with_terminal_height(tmp_path):
    model_dir = _make_model_dir(
        tmp_path,
        [
            "Qwen3-Small-Q4_K_M.gguf",
            "Qwen3-Large-Q4_K_M.gguf",
        ],
    )

    async def run_tui_check():
        small = BenchTui(root=model_dir)
        async with small.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            small_history_height = small.query_one("#history_box").region.height

        large = BenchTui(root=model_dir)
        async with large.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            large_history_height = large.query_one("#history_box").region.height

        assert small_history_height >= 3
        assert large_history_height > small_history_height

    asyncio.run(run_tui_check())


def test_history_panel_resize_keys_adjust_and_reset_height(tmp_path):
    model_dir = _make_model_dir(tmp_path, ["Qwen3-Test-Q4_K_M.gguf"])

    async def run_tui_check():
        app = BenchTui(root=model_dir)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            original_height = app.query_one("#history_box").region.height

            await pilot.press("]")
            await pilot.pause()
            grown_height = app.query_one("#history_box").region.height

            await pilot.press("[")
            await pilot.press("0")
            await pilot.pause()
            reset_height = app.query_one("#history_box").region.height

            assert grown_height > original_height
            assert reset_height == original_height

    asyncio.run(run_tui_check())


def test_narrow_tui_shows_full_highlighted_model_details(tmp_path):
    long_name = "Qwen3.6-35B-A3B-Uncensored-Heretic-Native-MTP-Preserved-Q4_K_M.gguf"
    model_dir = _make_model_dir(tmp_path, [long_name])

    async def run_tui_check():
        app = BenchTui(root=model_dir)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            table = app.query_one("#models", DataTable)
            details = app.query_one("#details", Static)

            labels = [str(column.label) for column in table.columns.values()]
            assert labels == ["Sel", "GB", "Model"]
            assert long_name in str(details.render())

    asyncio.run(run_tui_check())


def test_tui_continue_requires_a_selection_then_exits_with_selected_models(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "Qwen3-Test-Q4_K_M.gguf").write_bytes(b"fake")

    async def run_tui_check():
        app = BenchTui(root=model_dir)
        async with app.run_test() as pilot:
            await pilot.pause()
            status = app.query_one("#status", Static)

            await pilot.press("enter")
            await pilot.pause()

            assert app.models_to_run == []
            assert "Select at least one model" in str(status.render())

            await pilot.press("space")
            await pilot.press("enter")
            await pilot.pause()

            assert [model.name for model in app.models_to_run] == ["Qwen3-Test-Q4_K_M.gguf"]

    asyncio.run(run_tui_check())


def test_tui_r_key_also_starts_selected_models(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "Qwen3-Test-Q4_K_M.gguf").write_bytes(b"fake")

    async def run_tui_check():
        app = BenchTui(root=model_dir)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press("space")
            await pilot.press("r")
            await pilot.pause()

            assert [model.name for model in app.models_to_run] == ["Qwen3-Test-Q4_K_M.gguf"]

    asyncio.run(run_tui_check())


def test_tui_can_run_selected_models_inside_the_app(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "Qwen3-Test-Q4_K_M.gguf").write_bytes(b"fake")
    called: list[str] = []

    def fake_run_model(model):
        called.append(model.name)
        return tmp_path / "runs" / model.name

    async def run_tui_check():
        app = BenchTui(root=model_dir, run_model=fake_run_model)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press("space")
            await pilot.press("enter")
            await pilot.pause(0.2)

            assert app.ran_inside_tui is True
            assert called == ["Qwen3-Test-Q4_K_M.gguf"]

    asyncio.run(run_tui_check())


def test_tui_s_key_cycles_sort_modes_and_keeps_selection(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "b-small-Q4_K_M.gguf").write_bytes(b"1" * 10)
    (model_dir / "a-large-Q4_K_M.gguf").write_bytes(b"1" * 30)

    async def run_tui_check():
        app = BenchTui(root=model_dir)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("space")
            selected_before = app.selection.selected_paths()

            await pilot.press("s")
            await pilot.pause()

            assert [model.name for model in app.models] == [
                "a-large-Q4_K_M.gguf",
                "b-small-Q4_K_M.gguf",
            ]
            assert app.selection.selected_paths() == selected_before

    asyncio.run(run_tui_check())


def test_tui_defaults_to_find_best_settings_mode(tmp_path):
    app = BenchTui(root=tmp_path, runs_root=tmp_path)
    assert app.run_mode.id == "best_settings"
    assert app.evaluation_mode is EvaluationMode.BENCHMARK


def test_tui_m_key_cycles_run_mode(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "Qwen3-Test-Q4_K_M.gguf").write_bytes(b"fake")

    async def run_tui_check():
        app = BenchTui(root=model_dir)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.run_mode.id == "best_settings"

            await pilot.press("m")
            await pilot.pause()
            assert app.run_mode.id == "librarian_bench"

            await pilot.press("m")
            await pilot.pause()
            assert app.run_mode.id == "flag_effect"

            # Cycle through to the quick mode; its evaluation is the speed scout.
            for _ in range(len(RUN_MODES)):
                if app.run_mode.id == "quick":
                    break
                await pilot.press("m")
                await pilot.pause()
            assert app.run_mode.id == "quick"
            assert app.evaluation_mode is EvaluationMode.SPEED_SCOUT

    asyncio.run(run_tui_check())


def test_champion_line_formats():
    assert format_champion_line("QwenX", 950.0) == "Top candidate: QwenX (950.00)"
    assert format_champion_line(None, None) == "Top candidate: not decided yet"


def test_format_score_summary_names_measured_benchmark_scores():
    assert format_score_summary(None) == "Benchmark scores: unmeasured"
    line = format_score_summary(
        {
            "score_contract": "agent_bench_score",
            "agent_bench_score": 0.82,
            "general_score": 0.78,
            "agentic_score": 0.86,
            "generation_tps": 42.0,
            "serving_tps": 38.0,
            "context": 131072,
        }
    )
    assert line == (
        "Benchmark scores: contract=agent_bench_score agent=0.8200 "
        "general=0.7800 agentic=0.8600 gen=42.00 tok/s serving=38.00 tok/s "
        "ctx=131072"
    )


def test_score_summary_text_reads_leaderboard_receipts(tmp_path):
    run = tmp_path / "20260706-suite-backed"
    run.mkdir()
    (run / "best-settings.json").write_text(
        (
            '{"model":"G:/models/Winner.gguf","settings":{"context_size":131072},'
            '"result":{"ok":true,"generation_tokens_per_second":42,'
            '"prompt_tokens_per_second":900,"failure":"none",'
            '"agent_bench_score":0.82,"benchmark_suite_ok":true,'
            '"benchmark_suite_general_score":0.78,"benchmark_suite_agentic_score":0.86,'
            '"serving_tokens_per_second":38},"score":0.82}'
        ),
        encoding="utf-8",
    )

    line = score_summary_text(tmp_path)

    assert line.startswith("Benchmark scores: contract=agent_bench_score")
    assert "agent=0.8200" in line


def test_format_candidate_assessment_names_readiness_performance_and_missing_gates():
    assert format_candidate_assessment(None) == "Candidate readiness: unmeasured"
    line = format_candidate_assessment(
        {
            "readiness": "not_recommendable",
            "readiness_score": 0,
            "known_performance": {
                "quality": "unmeasured",
                "speed": "interactive",
                "context_class": "long_agentic",
            },
            "missing_evidence": [{"gate": "model"}, {"gate": "deployment"}, {"gate": "qe"}],
        }
    )
    assert line == (
        "Candidate readiness: not_recommendable (0/100) | "
        "performance quality=unmeasured speed=interactive context=long_agentic | "
        "missing model, deployment, qe"
    )


def test_format_operator_verdict_names_hard_usability_status():
    assert format_operator_verdict(None) == "Operator verdict: unmeasured"
    line = format_operator_verdict(
        {
            "status": "NOT_USABLE_YET",
            "headline": "No deployable recommendation exists.",
            "next_command": "apb benchmark-suite --plan benchmark-suite.plan.json --runs-root _runs",
        }
    )
    assert line == (
        "Operator verdict: NOT_USABLE_YET | No deployable recommendation exists. | "
        "next apb benchmark-suite --plan benchmark-suite.plan.json --runs-root _runs"
    )


def test_format_target_scope_names_no_target_evidence():
    assert format_target_scope(None) == "Target scope: unscoped"
    line = format_target_scope(
        {
            "target_model": "Gemma-4-26B",
            "status": "NO_TARGET_EVIDENCE",
            "matched_receipt_count": 0,
            "ignored_receipt_count": 5,
        }
    )

    assert line == "Target scope: Gemma-4-26B | NO_TARGET_EVIDENCE | matched 0, ignored 5"


def test_format_performance_prediction_names_risk_and_expectation():
    assert format_performance_prediction(None) == "Performance prediction: unmeasured"
    line = format_performance_prediction(
        {
            "status": "LAB_ONLY_SPEED_PROOF",
            "risk": "high",
            "deployment_expectation": "do_not_deploy",
            "expected_user_experience": "Fast but unscored.",
        }
    )
    assert line == (
        "Performance prediction: LAB_ONLY_SPEED_PROOF (high risk) | "
        "expectation=do_not_deploy | Fast but unscored."
    )


def test_format_settings_candidates_ranks_profiles_without_pretending_proof():
    assert format_settings_candidates([]) == "Settings candidates: none"
    line = format_settings_candidates(
        [
            {
                "rank": 1,
                "profile_id": "standard",
                "status": "UNPROVEN",
                "decision": "next_to_test",
                "context_size": 131072,
                "recommendation_score": 32.0,
            }
        ]
    )
    assert (
        line
        == "Settings candidates:\n#1 standard | UNPROVEN | next_to_test | ctx=131072 | score=32.0000"
    )


def test_operator_verdict_text_reads_hard_recommendation_receipts(tmp_path):
    run = tmp_path / "20260706-speed-only"
    run.mkdir()
    (run / "best-settings.json").write_text(
        (
            '{"model":"G:/models/Fast.gguf","settings":{"context_size":262144},'
            '"result":{"ok":true,"generation_tokens_per_second":120,'
            '"prompt_tokens_per_second":900,"failure":"none"},"score":120}'
        ),
        encoding="utf-8",
    )

    line = operator_verdict_text(tmp_path)

    assert line.startswith("Operator verdict: NOT_USABLE_YET")


def test_performance_prediction_text_reads_hard_recommendation_receipts(tmp_path):
    run = tmp_path / "20260706-speed-only"
    run.mkdir()
    (run / "best-settings.json").write_text(
        (
            '{"model":"G:/models/Fast.gguf","settings":{"context_size":262144},'
            '"result":{"ok":true,"generation_tokens_per_second":120,'
            '"prompt_tokens_per_second":900,"failure":"none"},"score":120}'
        ),
        encoding="utf-8",
    )

    line = performance_prediction_text(tmp_path)

    assert line.startswith("Performance prediction: LAB_ONLY_SPEED_PROOF")
    assert "expectation=do_not_deploy" in line


def test_target_scope_text_reads_target_scoped_hard_recommendation_payload(tmp_path):
    run = tmp_path / "old-qwopus"
    run.mkdir()
    (run / "best-settings.json").write_text(
        (
            '{"model":"G:/models/Qwopus.gguf","settings":{"context_size":4096},'
            '"result":{"ok":true,"generation_tokens_per_second":120,'
            '"prompt_tokens_per_second":900,"failure":"none"},"score":120}'
        ),
        encoding="utf-8",
    )

    line = target_scope_text(tmp_path, target_model="Gemma-4-26B")

    assert line == "Target scope: Gemma-4-26B | NO_TARGET_EVIDENCE | matched 0, ignored 1"


def test_tui_target_scoped_helpers_do_not_show_unrelated_candidate(tmp_path):
    run = tmp_path / "old-qwopus"
    run.mkdir()
    (run / "best-settings.json").write_text(
        (
            '{"model":"G:/models/Qwopus.gguf","settings":{"context_size":4096},'
            '"result":{"ok":true,"generation_tokens_per_second":120,'
            '"prompt_tokens_per_second":900,"failure":"none"},"score":120}'
        ),
        encoding="utf-8",
    )

    assert (
        score_summary_text(tmp_path, target_model="Gemma-4-26B") == "Benchmark scores: unmeasured"
    )
    assert (
        candidate_rankings_text(tmp_path, target_model="Gemma-4-26B") == "Candidate rankings: none"
    )
    assert "Gemma-4-26B" in operator_verdict_text(tmp_path, target_model="Gemma-4-26B")


def test_tui_decision_runbook_uses_target_model_path_for_deployment_flags(tmp_path):
    run = tmp_path / "old-qwopus"
    run.mkdir()
    (run / "best-settings.json").write_text(
        (
            '{"model":"G:/models/Qwopus.gguf","settings":{"context_size":4096},'
            '"result":{"ok":true,"generation_tokens_per_second":120,'
            '"prompt_tokens_per_second":900,"failure":"none"},"score":120}'
        ),
        encoding="utf-8",
    )

    line = decision_runbook_text(
        tmp_path,
        target_model="Gemma-4-26B",
        target_model_path="G:/AI/models/Gemma-4-26B-A4B-Q8_0.gguf",
    )

    assert (
        'apb flag-recommendations --model "G:/AI/models/Gemma-4-26B-A4B-Q8_0.gguf" '
        f"--output-dir {_path_arg(tmp_path)}"
    ) in line


def test_settings_candidates_text_reads_hard_recommendation_receipts(tmp_path):
    (tmp_path / "flag-recommendations.json").write_text(
        (
            '{"model":"G:/models/Fast.gguf","model_name":"Fast.gguf",'
            '"profiles":[{"id":"standard","label":"Standard","context_size":131072}]}'
        ),
        encoding="utf-8",
    )
    run = tmp_path / "20260706-speed-only"
    run.mkdir()
    (run / "best-settings.json").write_text(
        (
            '{"model":"G:/models/Fast.gguf","settings":{"context_size":262144},'
            '"result":{"ok":true,"generation_tokens_per_second":120,'
            '"prompt_tokens_per_second":900,"failure":"none"},"score":120}'
        ),
        encoding="utf-8",
    )

    line = settings_candidates_text(tmp_path)

    assert line.startswith("Settings candidates:")
    assert "#1 standard | SYSTEMS_ONLY | needs_agent_score | ctx=131072" in line


def test_candidate_assessment_text_reads_hard_recommendation_receipts(tmp_path):
    run = tmp_path / "20260706-speed-only"
    run.mkdir()
    (run / "best-settings.json").write_text(
        (
            '{"model":"G:/models/Fast.gguf","settings":{"context_size":262144},'
            '"result":{"ok":true,"generation_tokens_per_second":120,'
            '"prompt_tokens_per_second":900,"failure":"none"},"score":120}'
        ),
        encoding="utf-8",
    )

    line = candidate_assessment_text(tmp_path)

    assert "Candidate readiness: not_recommendable (0/100)" in line
    assert "speed=interactive" in line
    assert "missing model, deployment, context, resource, qe" in line


def test_format_candidate_rankings_compacts_top_rows():
    assert format_candidate_rankings([]) == "Candidate rankings: none"
    line = format_candidate_rankings(
        [
            {
                "rank": 1,
                "model": "Beta.gguf",
                "status": "BENCHMARK SUITE",
                "agent_quality_score": 0.82,
                "prediction": {
                    "quality": "strong",
                    "speed": "interactive",
                    "context": "long_agentic",
                },
                "evidence_gaps": [],
            },
            {
                "rank": 2,
                "model": "Alpha.gguf",
                "status": "CONTEXT UNPROVEN",
                "agent_quality_score": None,
                "prediction": {
                    "quality": "unmeasured",
                    "speed": "interactive",
                    "context": "short",
                },
                "evidence_gaps": ["agent_quality", "benchmark_suite", "long_context"],
            },
        ]
    )
    assert line == (
        "Candidate rankings:\n"
        "#1 Beta.gguf | BENCHMARK SUITE | agent=0.8200 | "
        "strong/interactive/long_agentic | gaps=none\n"
        "#2 Alpha.gguf | CONTEXT UNPROVEN | agent=not measured | "
        "unmeasured/interactive/short | gaps=agent_quality, benchmark_suite, long_context"
    )


def test_candidate_rankings_text_reads_hard_recommendation_receipts(tmp_path):
    run = tmp_path / "20260706-speed-only"
    run.mkdir()
    (run / "best-settings.json").write_text(
        (
            '{"model":"G:/models/Fast.gguf","settings":{"context_size":262144},'
            '"result":{"ok":true,"generation_tokens_per_second":120,'
            '"prompt_tokens_per_second":900,"failure":"none"},"score":120}'
        ),
        encoding="utf-8",
    )

    line = candidate_rankings_text(tmp_path)

    assert "Candidate rankings:" in line
    assert "#1 Fast.gguf" in line
    assert "gaps=agent_quality, benchmark_suite, serving" in line


def test_format_decision_runbook_compacts_proof_steps():
    assert format_decision_runbook([]) == "Proof runbook: none"
    line = format_decision_runbook(
        [
            {
                "step": 1,
                "gate": "model",
                "id": "model_plan",
                "status": "pending",
                "proves": "benchmark-suite.plan.json",
            },
            {
                "step": 2,
                "gate": "model",
                "id": "model_score",
                "status": "pending",
                "proves": "_runs/<suite-run>/suite-verdict.json",
            },
        ]
    )
    assert line == (
        "Proof runbook:\n"
        "1. [model/model_plan] pending -> benchmark-suite.plan.json\n"
        "2. [model/model_score] pending -> _runs/<suite-run>/suite-verdict.json"
    )


def test_decision_runbook_text_reads_hard_recommendation_receipts(tmp_path):
    run = tmp_path / "20260706-speed-only"
    run.mkdir()
    (run / "best-settings.json").write_text(
        (
            '{"model":"G:/models/Fast.gguf","settings":{"context_size":262144},'
            '"result":{"ok":true,"generation_tokens_per_second":120,'
            '"prompt_tokens_per_second":900,"failure":"none"},"score":120}'
        ),
        encoding="utf-8",
    )

    line = decision_runbook_text(tmp_path)

    assert "Proof runbook:" in line
    assert "1. [model/model_plan] pending -> benchmark-suite.plan.json" in line


def test_format_repeatability_names_confidence_and_ranges():
    assert format_repeatability(None) == "Repeatability: unmeasured"
    line = format_repeatability(
        {
            "confidence": "repeatable",
            "run_count": 3,
            "score": {"min": 0.8, "max": 0.82},
            "generation_tps": {"min": 40.0, "max": 42.0},
            "cold_ttft_ms": {"min": 410.0, "max": 430.0},
        }
    )
    assert line == (
        "Repeatability: repeatable (3 runs) | "
        "score=0.8000-0.8200 gen=40.0000-42.0000 tok/s ttft=410.0000-430.0000 ms"
    )


def test_repeatability_text_reads_hard_recommendation_receipts(tmp_path):
    run = tmp_path / "20260706-speed-only"
    run.mkdir()
    (run / "best-settings.json").write_text(
        (
            '{"model":"G:/models/Fast.gguf","settings":{"context_size":262144},'
            '"result":{"ok":true,"generation_tokens_per_second":120,'
            '"prompt_tokens_per_second":900,"failure":"none"},"score":120}'
        ),
        encoding="utf-8",
    )

    line = repeatability_text(tmp_path)

    assert line.startswith("Repeatability: single_run (1 run)")


def test_format_stability_gate_names_action_and_required_proof():
    assert format_stability_gate(None) == "Stability gate: unmeasured"
    line = format_stability_gate(
        {
            "action": "RETEST_STABILITY",
            "confidence": "single_run",
            "required": "at least 3 comparable receipts with repeatable measured metrics",
        }
    )
    assert line == (
        "Stability gate: RETEST_STABILITY | confidence=single_run | "
        "required=at least 3 comparable receipts with repeatable measured metrics"
    )


def test_stability_gate_text_reads_hard_recommendation_receipts(tmp_path):
    run = tmp_path / "20260706-speed-only"
    run.mkdir()
    (run / "best-settings.json").write_text(
        (
            '{"model":"G:/models/Fast.gguf","settings":{"context_size":262144},'
            '"result":{"ok":true,"generation_tokens_per_second":120,'
            '"prompt_tokens_per_second":900,"failure":"none"},"score":120}'
        ),
        encoding="utf-8",
    )

    line = stability_gate_text(tmp_path)

    assert line.startswith("Stability gate: WAITING_FOR_PROMOTED_STACK")


def test_format_context_and_resource_gates_name_required_proof():
    assert format_context_gate(None) == "Context gate: unmeasured"
    assert format_resource_gate(None) == "Resource gate: unmeasured"
    assert format_context_gate(
        {
            "action": "RETEST_CONTEXT",
            "required_context": 200000,
            "proven_context": 131072,
            "profile_id": "long_agent",
        }
    ) == ("Context gate: RETEST_CONTEXT | required=200000 | proven=131072 | profile=long_agent")
    assert format_resource_gate(
        {
            "action": "WAITING_FOR_CONTEXT",
            "required": "same-run resource telemetry for the required context profile",
        }
    ) == (
        "Resource gate: WAITING_FOR_CONTEXT | "
        "required=same-run resource telemetry for the required context profile"
    )


def test_context_and_resource_gate_text_read_hard_recommendation_receipts(tmp_path):
    run = tmp_path / "20260706-speed-only"
    run.mkdir()
    (run / "best-settings.json").write_text(
        (
            '{"model":"G:/models/Fast.gguf","settings":{"context_size":262144},'
            '"result":{"ok":true,"generation_tokens_per_second":120,'
            '"prompt_tokens_per_second":900,"failure":"none"},"score":120}'
        ),
        encoding="utf-8",
    )

    assert context_gate_text(tmp_path).startswith("Context gate: WAITING_FOR_DEPLOYMENT")
    assert resource_gate_text(tmp_path).startswith("Resource gate: WAITING_FOR_DEPLOYMENT")


def test_context_gate_text_preserves_required_context(tmp_path):
    (tmp_path / "flag-recommendations.json").write_text(
        (
            '{"model":"G:/models/Fast.gguf","model_name":"Fast.gguf",'
            '"profiles":['
            '{"id":"standard","label":"Standard","context_size":131072},'
            '{"id":"long_agent","label":"Long agent","context_size":200000}'
            "]}"
        ),
        encoding="utf-8",
    )
    run = tmp_path / "20260706-standard-proof"
    run.mkdir()
    (run / "best-settings.json").write_text(
        (
            '{"model":"G:/models/Fast.gguf","settings":{"context_size":131072},'
            '"result":{"ok":true,"generation_tokens_per_second":42,'
            '"prompt_tokens_per_second":900,"failure":"none",'
            '"agent_bench_score":0.82,"benchmark_suite_ok":true,'
            '"benchmark_suite_general_score":0.82,"benchmark_suite_agentic_score":0.82,'
            '"serving_ttft_ms":420,"serving_tokens_per_second":38},'
            '"score":0.82}'
        ),
        encoding="utf-8",
    )

    line = context_gate_text(tmp_path, required_context=200000)

    assert line == (
        "Context gate: RETEST_CONTEXT | required=200000 | proven=131072 | profile=long_agent"
    )


def test_tui_status_helpers_do_not_write_report_artifacts_on_read(tmp_path):
    run = tmp_path / "20260706-speed-only"
    run.mkdir()
    (run / "best-settings.json").write_text(
        (
            '{"model":"G:/models/Fast.gguf","settings":{"context_size":262144},'
            '"result":{"ok":true,"generation_tokens_per_second":120,'
            '"prompt_tokens_per_second":900,"failure":"none"},"score":120}'
        ),
        encoding="utf-8",
    )

    assert score_summary_text(tmp_path).startswith("Benchmark scores:")
    assert operator_verdict_text(tmp_path).startswith("Operator verdict: NOT_USABLE_YET")
    assert performance_prediction_text(tmp_path).startswith("Performance prediction:")
    assert candidate_assessment_text(tmp_path).startswith("Candidate readiness:")
    assert candidate_rankings_text(tmp_path).startswith("Candidate rankings:")
    assert repeatability_text(tmp_path).startswith("Repeatability:")
    assert context_gate_text(tmp_path).startswith("Context gate:")
    assert resource_gate_text(tmp_path).startswith("Resource gate:")
    assert stability_gate_text(tmp_path).startswith("Stability gate:")
    assert decision_runbook_text(tmp_path).startswith("Proof runbook:")

    assert not (tmp_path / "leaderboard.md").exists()
    assert not (tmp_path / "results.html").exists()
    assert not (tmp_path / "hard-recommendations.json").exists()
    assert not (tmp_path / "hard-recommendations.md").exists()


# ---------------------------------------------------------------------------
# Pure formatter unit tests (no Textual App required)
# ---------------------------------------------------------------------------

from gguf_limit_bench.tui import format_scoreboard, format_lifetime_line  # noqa: E402


def test_format_scoreboard_two_packs_one_incomplete():
    result = format_scoreboard(
        [
            {"pack_id": "simple-bench", "correct": 2, "asked": 5, "incomplete": 0},
            {"pack_id": "easy-gotcha", "correct": 4, "asked": 5, "incomplete": 1},
        ]
    )
    assert "simple-bench 2/5" in result
    assert "easy-gotcha 4/5" in result
    assert "(1 incomplete)" in result


def test_format_scoreboard_no_incomplete_suffix_when_zero():
    result = format_scoreboard(
        [
            {"pack_id": "simple-bench", "correct": 2, "asked": 5, "incomplete": 0},
        ]
    )
    assert "incomplete" not in result


def test_format_scoreboard_empty_list():
    assert format_scoreboard([]) == "no packs scored"


def test_format_lifetime_line_contains_expected_parts():
    result = format_lifetime_line("easy-gotcha", {"seen": 50, "correct": 31, "accuracy": 0.62})
    assert "easy-gotcha" in result
    assert "50 seen" in result
    assert "31 correct" in result
    assert "62%" in result


def test_active_run_status_reports_questions(tmp_path):
    from gguf_limit_bench.tui import active_run_status

    run = tmp_path / "20260623-0700-somemodel"
    sub = run / "simplebench-Lmin-stripped"
    sub.mkdir(parents=True)
    (run / "events.jsonl").write_text(
        '{"type":"autoresearch_attempt_started","data":{"settings":{"profile_name":"Lmin-stripped"}}}\n',
        encoding="utf-8",
    )
    (sub / "transcript.jsonl").write_text(
        '{"question_id":1,"correct":true,"predicted_answer":"B"}\n'
        '{"question_id":2,"correct":false,"predicted_answer":"A"}\n',
        encoding="utf-8",
    )
    status = active_run_status(tmp_path)
    assert status is not None
    assert "Lmin-stripped" in status
    assert "asked 2Q" in status and "1 correct" in status


def test_active_run_status_none_when_empty(tmp_path):
    from gguf_limit_bench.tui import active_run_status

    assert active_run_status(tmp_path) is None
