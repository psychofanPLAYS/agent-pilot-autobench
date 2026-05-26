from pathlib import Path

from gguf_limit_bench.autoresearch import AttemptResult, AutoresearchSettings
from gguf_limit_bench.learning import OptunaSettingsLearner, study_name_for_model


def _good_result(settings: AutoresearchSettings, speed: float = 60.0) -> AttemptResult:
    return AttemptResult(
        ok=True,
        generation_tokens_per_second=speed,
        prompt_tokens_per_second=900.0,
        ttft_ms=None,
        context_size=settings.context_size,
        failure="unknown",
        stdout="",
        stderr="",
        returncode=0,
    )


def test_study_name_for_model_is_stable_and_filesystem_safe():
    first = study_name_for_model(Path("G:/AI/models/Qwen3 Test/Qwen3-Test-Q4_K_M.gguf"))
    second = study_name_for_model(Path("G:/AI/models/Qwen3 Test/Qwen3-Test-Q4_K_M.gguf"))

    assert first == second
    assert first.startswith("gguf-")
    assert " " not in first


def test_optuna_learner_persists_trials_between_instances(tmp_path):
    storage_path = tmp_path / "learning" / "optuna.sqlite3"
    model = Path("G:/AI/models/Qwen3-Test-Q4_K_M.gguf")
    learner = OptunaSettingsLearner(storage_path=storage_path, model=model, parallel_max=4)

    suggestion = learner.suggest()
    learner.tell(suggestion, _good_result(suggestion.settings, speed=71.0))

    loaded = OptunaSettingsLearner(storage_path=storage_path, model=model, parallel_max=4)
    best = loaded.best()

    assert storage_path.exists()
    assert best is not None
    assert best["score"] > 0
    assert best["settings"]["kv_unified"] is True


def test_optuna_learner_suggests_baseline_before_exploring(tmp_path):
    learner = OptunaSettingsLearner(
        storage_path=tmp_path / "learning.sqlite3",
        model=Path("G:/AI/models/Qwen3-Test-Q4_K_M.gguf"),
        parallel_max=4,
    )

    suggestion = learner.suggest()

    assert suggestion.settings == AutoresearchSettings()


def test_optuna_learner_records_failed_trials_as_bad_scores(tmp_path):
    learner = OptunaSettingsLearner(
        storage_path=tmp_path / "learning.sqlite3",
        model=Path("G:/AI/models/Qwen3-Test-Q4_K_M.gguf"),
        parallel_max=4,
    )
    suggestion = learner.suggest()
    learner.tell(
        suggestion,
        AttemptResult(
            ok=False,
            generation_tokens_per_second=0.0,
            prompt_tokens_per_second=0.0,
            ttft_ms=None,
            context_size=suggestion.settings.context_size,
            failure="gpu_oom",
            stdout="",
            stderr="CUDA out of memory",
            returncode=1,
        ),
    )

    best = learner.best()

    assert best is not None
    assert best["score"] < 0
