from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkPack:
    id: str
    version: str
    description: str
    tasks: tuple[str, ...]
    settings_space: dict[str, Any]
    scoring_categories: tuple[str, ...]
    safety_policy: str
    receipt_schema: str = "pack.v1"


BUILTIN_PACK_IDS = {
    "speed",
    "load-smoke",
    "context-limit",
    "kv-falloff",
    "mtp-efficiency",
    "json-discipline",
    "tool-calling",
    "agentic-harness",
    "qa",
    "coding-smoke",
    "bias-probe",
    "hermes-pilot",
    "all",
}


def load_benchmark_packs(plugin_dir: Path | None = None) -> dict[str, BenchmarkPack]:
    packs = _builtin_packs()
    if plugin_dir is not None and plugin_dir.exists():
        for manifest in sorted(plugin_dir.glob("*.json")):
            pack = _pack_from_dict(json.loads(manifest.read_text(encoding="utf-8")))
            packs[pack.id] = pack
    return packs


def _builtin_packs() -> dict[str, BenchmarkPack]:
    common: dict[str, Any] = {
        "version": "2026.05.26",
        "settings_space": {},
        "safety_policy": "local_deterministic",
        "receipt_schema": "pack.v1",
    }
    specs = [
        ("load-smoke", "Load and one tiny speed probe.", ("load",), ("stability",)),
        ("speed", "Raw llama.cpp prompt and generation speed.", ("llama_bench",), ("speed",)),
        (
            "context-limit",
            "Find maximum useful full-GPU context.",
            ("context_ladder",),
            ("context", "stability"),
        ),
        (
            "kv-falloff",
            "Paired KV cache speed and quality falloff.",
            ("paired_kv",),
            ("context", "quality"),
        ),
        (
            "mtp-efficiency",
            "MTP wall-clock gain without quality collapse.",
            ("mtp_matrix",),
            ("speed", "quality"),
        ),
        ("json-discipline", "Strict JSON/schema compliance probes.", ("json_schema",), ("json",)),
        (
            "tool-calling",
            "Hermes-style tool choice and argument tests.",
            ("tool_choice",),
            ("tool", "json"),
        ),
        (
            "agentic-harness",
            "Multi-step agent workflow viability.",
            ("agentic",),
            ("tool", "instruction"),
        ),
        ("qa", "Factual Q&A local fixture pack.", ("qa_fixture",), ("quality",)),
        ("coding-smoke", "Tiny sandbox coding task with tests.", ("coding_fixture",), ("coding",)),
        (
            "bias-probe",
            "Labeled political/bias/refusal probe pack.",
            ("bias_fixture",),
            ("bias", "refusal"),
        ),
        (
            "hermes-pilot",
            "Balanced Hermes pilot deployment score.",
            ("hermes",),
            ("tool", "json", "context", "speed"),
        ),
        (
            "all",
            "Run every built-in pack with campaign caps.",
            tuple(sorted(BUILTIN_PACK_IDS - {"all"})),
            ("balanced",),
        ),
    ]
    return {
        pack_id: BenchmarkPack(
            id=pack_id,
            description=description,
            tasks=tasks,
            scoring_categories=scoring,
            **common,
        )
        for pack_id, description, tasks, scoring in specs
    }


def _pack_from_dict(payload: dict[str, Any]) -> BenchmarkPack:
    required = {
        "id",
        "version",
        "description",
        "tasks",
        "settings_space",
        "scoring_categories",
        "safety_policy",
        "receipt_schema",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Benchmark pack manifest is missing: {', '.join(missing)}")
    return BenchmarkPack(
        id=str(payload["id"]),
        version=str(payload["version"]),
        description=str(payload["description"]),
        tasks=tuple(str(task) for task in payload["tasks"]),
        settings_space=dict(payload["settings_space"]),
        scoring_categories=tuple(str(item) for item in payload["scoring_categories"]),
        safety_policy=str(payload["safety_policy"]),
        receipt_schema=str(payload["receipt_schema"]),
    )
