from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

try:
    from enum import StrEnum
except ImportError:  # Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass


class AnswerType(StrEnum):
    MULTIPLE_CHOICE = "multiple_choice"
    EXACT = "exact"


@dataclass(frozen=True)
class PackQuestion:
    question_id: str
    prompt: str
    answer: str
    answer_source: str
    choices: tuple[str, ...] | None = None
    tags: tuple[str, ...] = ()
    accept: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuestionPack:
    pack_id: str
    title: str
    tier: str
    answer_type: AnswerType
    system_prompt: str
    questions: tuple[PackQuestion, ...]


_DATA_DIR = Path(__file__).resolve().parent / "data"
_PACKS_DIR = _DATA_DIR / "packs"

DEFAULT_PACKS: tuple[str, ...] = ("simple-bench", "easy-gotcha", "easy-mc")

# Procedural long-context packs are generated on demand (RULER-style), not read
# from disk. We surface one id per serious context tier so selection UIs can pick
# them; load_pack() recognises any "ruler-longctx-<tokens>" id.
PROCEDURAL_LONGCTX_TIERS: tuple[int, ...] = (16384, 65536, 131072, 262144)
_PROCEDURAL_LONGCTX_RE = re.compile(r"ruler-longctx-(\d+)")
_PROCEDURAL_LONGCTX_COUNT = 8


def procedural_longctx_pack_ids() -> tuple[str, ...]:
    return tuple(f"ruler-longctx-{tier}" for tier in PROCEDURAL_LONGCTX_TIERS)


def available_packs() -> tuple[str, ...]:
    """Return IDs of all available question packs."""
    dynamic: list[str] = []
    if _PACKS_DIR.exists():
        for path in sorted(_PACKS_DIR.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if "pack_id" in payload:
                    dynamic.append(str(payload["pack_id"]))
            except (OSError, json.JSONDecodeError):
                pass
    known = list(DEFAULT_PACKS)
    for pid in dynamic:
        if pid not in known:
            known.append(pid)
    for pid in procedural_longctx_pack_ids():
        if pid not in known:
            known.append(pid)
    return tuple(known)


def load_pack(pack_id: str) -> QuestionPack:
    """Load a QuestionPack by id.

    Raises KeyError if the pack is not found.
    """
    if pack_id == "simple-bench":
        return _load_simple_bench()

    procedural = _PROCEDURAL_LONGCTX_RE.fullmatch(pack_id)
    if procedural:
        # Local import avoids a circular dependency: procedural_packs imports
        # PackQuestion/QuestionPack from this module.
        from gguf_limit_bench.procedural_packs import build_long_context_pack

        return build_long_context_pack(
            target_tokens=int(procedural.group(1)),
            count=_PROCEDURAL_LONGCTX_COUNT,
            seed=0,
        )

    pack_path = _PACKS_DIR / f"{pack_id}.json"
    if not pack_path.exists():
        raise KeyError(f"Unknown question pack: {pack_id!r}")

    payload = json.loads(pack_path.read_text(encoding="utf-8"))
    if payload.get("pack_id") != pack_id:
        raise KeyError(f"Unknown question pack: {pack_id!r}")

    return _load_json_pack(payload, pack_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_simple_bench() -> QuestionPack:
    data_path = _DATA_DIR / "simple_bench_public.json"
    system_prompt_path = _DATA_DIR / "system_prompt.txt"

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    rows = payload["eval_data"]
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()

    questions: list[PackQuestion] = []
    for row in rows:
        qid = str(row["question_id"])
        questions.append(
            PackQuestion(
                question_id=qid,
                prompt=str(row["prompt"]).strip(),
                answer=str(row["answer"]).strip().upper(),
                answer_source="dataset_label:simple-bench",
                choices=None,
                tags=(),
                accept=(),
            )
        )

    return QuestionPack(
        pack_id="simple-bench",
        title="SimpleBench Public",
        tier="hard",
        answer_type=AnswerType.MULTIPLE_CHOICE,
        system_prompt=system_prompt,
        questions=tuple(questions),
    )


def _derive_answer_source(question_id: str, answer_type: AnswerType) -> str:
    if answer_type is AnswerType.EXACT:
        return "curated_fact"
    # dataset_label:<prefix up to first run of digits>
    match = re.match(r"^(.*?)(\d)", question_id)
    if match:
        prefix = match.group(1).rstrip("-_")
    else:
        prefix = question_id
    return f"dataset_label:{prefix}"


def _load_json_pack(payload: dict[str, Any], pack_path: Path) -> QuestionPack:
    pack_id = str(payload["pack_id"])
    title = str(payload.get("title", pack_id))
    tier = str(payload.get("tier", "unknown"))
    answer_type = AnswerType(str(payload["answer_type"]))

    # Resolve system_prompt: look in packs dir first, then data dir
    system_prompt_ref = payload.get("system_prompt_ref")
    if system_prompt_ref:
        sp_path = pack_path.parent / system_prompt_ref
        if not sp_path.exists():
            sp_path = pack_path.parent.parent / system_prompt_ref
        system_prompt = sp_path.read_text(encoding="utf-8").strip()
    else:
        system_prompt = str(payload.get("system_prompt", ""))

    raw_questions: list[dict[str, Any]] = payload.get("questions", [])
    questions: list[PackQuestion] = []
    for raw in raw_questions:
        qid = str(raw["question_id"])
        choices_raw: list[str] | None = raw.get("choices")
        choices: tuple[str, ...] | None = tuple(str(c) for c in choices_raw) if choices_raw else None
        tags: tuple[str, ...] = tuple(str(t) for t in raw.get("tags", []))
        accept: tuple[str, ...] = tuple(str(a) for a in raw.get("accept", []))
        answer = str(raw["answer"]).strip()
        if answer_type is AnswerType.MULTIPLE_CHOICE:
            answer = answer.upper()

        answer_source = str(raw.get("answer_source", "")) or _derive_answer_source(qid, answer_type)

        # Validate MC questions
        if answer_type is AnswerType.MULTIPLE_CHOICE:
            if not choices:
                raise ValueError(
                    f"Pack {pack_id!r} question {qid!r}: MC question must have non-empty choices"
                )
            if len(answer) != 1 or answer not in "ABCDEF":
                raise ValueError(
                    f"Pack {pack_id!r} question {qid!r}: MC answer must be a single letter A-F"
                )
            letter_index = ord(answer) - ord("A")
            if letter_index >= len(choices):
                raise ValueError(
                    f"Pack {pack_id!r} question {qid!r}: answer {answer!r} out of range for choices"
                )

        questions.append(
            PackQuestion(
                question_id=qid,
                prompt=str(raw["prompt"]).strip(),
                answer=answer,
                answer_source=answer_source,
                choices=choices,
                tags=tags,
                accept=accept,
            )
        )

    return QuestionPack(
        pack_id=pack_id,
        title=title,
        tier=tier,
        answer_type=answer_type,
        system_prompt=system_prompt,
        questions=tuple(questions),
    )


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
