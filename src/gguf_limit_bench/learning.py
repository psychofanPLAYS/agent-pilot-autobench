from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import optuna
from optuna.trial import TrialState

from gguf_limit_bench.autoresearch import AttemptResult, AutoresearchSettings


CONTEXT_CHOICES = [4096, 8192, 16384, 32768, 65536, 131072]
BATCH_CHOICES = [512, 1024, 2048, 4096]
UBATCH_CHOICES = [128, 256, 512, 1024]
GPU_LAYER_CHOICES = [99, 80, 60, 40]
LEARNING_SCORE_VERSION = "score-v4-simplebench"


@dataclass(frozen=True)
class LearningSuggestion:
    trial_id: int
    settings: AutoresearchSettings


class OptunaSettingsLearner:
    def __init__(
        self,
        storage_path: Path,
        model: Path,
        parallel_max: int,
        seed: int = 4090,
        objective: str = "throughput",
    ) -> None:
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.study = optuna.create_study(
            study_name=study_name_for_model(model, objective),
            storage=_sqlite_url(storage_path),
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            load_if_exists=True,
        )
        self.parallel_max = max(1, parallel_max)
        if not self.study.trials:
            self.study.enqueue_trial(AutoresearchSettings().to_dict())

    def suggest(self) -> LearningSuggestion:
        trial = self.study.ask()
        context_size = trial.suggest_categorical("context_size", CONTEXT_CHOICES)
        batch_size = trial.suggest_categorical("batch_size", BATCH_CHOICES)
        ubatch_size = trial.suggest_categorical("ubatch_size", UBATCH_CHOICES)
        gpu_layers = trial.suggest_categorical("gpu_layers", GPU_LAYER_CHOICES)
        parallel = trial.suggest_int("parallel", 1, self.parallel_max)
        flash_attention = trial.suggest_categorical("flash_attention", [True])
        settings = AutoresearchSettings(
            context_size=int(context_size),
            parallel=parallel,
            gpu_layers=int(gpu_layers),
            batch_size=int(batch_size),
            ubatch_size=int(ubatch_size),
            flash_attention=bool(flash_attention),
            kv_unified=True,
        )
        return LearningSuggestion(trial_id=trial.number, settings=settings)

    def enqueue_settings(self, settings: AutoresearchSettings) -> None:
        self.study.enqueue_trial(settings.to_dict(), skip_if_exists=True)

    def tell(self, suggestion: LearningSuggestion, result: AttemptResult) -> None:
        self.study.tell(suggestion.trial_id, result.score())

    def best(self) -> dict | None:
        complete_trials = [
            trial
            for trial in self.study.trials
            if trial.state == TrialState.COMPLETE and trial.value is not None
        ]
        if not complete_trials:
            return None
        trial = self.study.best_trial
        settings = AutoresearchSettings(
            context_size=int(trial.params.get("context_size", 4096)),
            parallel=int(trial.params.get("parallel", 1)),
            gpu_layers=int(trial.params.get("gpu_layers", 99)),
            batch_size=int(trial.params.get("batch_size", 2048)),
            ubatch_size=int(trial.params.get("ubatch_size", 512)),
            flash_attention=bool(trial.params.get("flash_attention", True)),
            kv_unified=True,
        )
        return {
            "score": trial.value,
            "settings": settings.to_dict(),
            "trial_id": trial.number,
            "storage": str(self.storage_path),
        }


def study_name_for_model(model: Path, objective: str = "throughput") -> str:
    stable = hashlib.sha1(str(model).lower().encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", model.stem).strip("-")[:60]
    return f"gguf-{LEARNING_SCORE_VERSION}-{objective}-{slug}-{stable}"


def _sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"
