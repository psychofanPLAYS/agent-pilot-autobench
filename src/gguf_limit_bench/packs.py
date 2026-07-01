from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import yaml

# Canonical on-disk question-set formats (self-contained: system prompt + questions +
# answers in one file). YAML is the human-friendly authoring format; JSON still loads.
# See docs/QUESTION-SETS.md.
_PACK_SUFFIXES = (".yaml", ".yml", ".json")


def _read_pack_mapping(path: Path) -> dict[str, Any]:
    """Parse a question-set file (.yaml/.yml/.json) into a mapping."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Question set {path.name!r} must be a mapping at the top level")
    return data


def _pack_id_of(mapping: dict[str, Any]) -> str | None:
    """A set's id — accepts `id` (preferred) or legacy `pack_id`."""
    value = mapping.get("id", mapping.get("pack_id"))
    return str(value) if value is not None else None

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
        for path in sorted(
            p for p in _PACKS_DIR.iterdir() if p.suffix.lower() in _PACK_SUFFIXES
        ):
            try:
                pid = _pack_id_of(_read_pack_mapping(path))
                if pid:
                    dynamic.append(pid)
            except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
                pass
    known = list(DEFAULT_PACKS)
    for pid in dynamic:
        if pid not in known:
            known.append(pid)
    for pid in procedural_longctx_pack_ids():
        if pid not in known:
            known.append(pid)
    # Local import avoids a circular dependency: the librarian package imports
    # PackQuestion/QuestionPack from this module.
    from gguf_limit_bench.librarian.registry import LIBRARIAN_PACK_IDS

    for pid in LIBRARIAN_PACK_IDS:
        if pid not in known:
            known.append(pid)
    return tuple(known)


def load_pack(pack_id: str) -> QuestionPack:
    """Load a QuestionPack by id.

    Raises KeyError if the pack is not found.
    """
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

    # Local import avoids a circular dependency: the librarian package imports
    # PackQuestion/QuestionPack from this module.
    from gguf_limit_bench.librarian.registry import LIBRARIAN_BUILDERS

    if pack_id in LIBRARIAN_BUILDERS:
        return LIBRARIAN_BUILDERS[pack_id](0)

    pack_path = next(
        (p for suffix in _PACK_SUFFIXES if (p := _PACKS_DIR / f"{pack_id}{suffix}").exists()),
        None,
    )
    if pack_path is None:
        raise KeyError(f"Unknown question pack: {pack_id!r}")

    payload = _read_pack_mapping(pack_path)
    if _pack_id_of(payload) != pack_id:
        raise KeyError(f"Unknown question pack: {pack_id!r}")

    return _pack_from_mapping(payload, pack_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


def _pack_from_mapping(payload: dict[str, Any], pack_path: Path) -> QuestionPack:
    """Build a QuestionPack from a self-contained set mapping (YAML or JSON).

    Canonical schema (see docs/QUESTION-SETS.md): `id`, `title`, `tier`,
    `answer_type`, inline `system_prompt`, and `questions` (each `id`, `prompt`,
    `answer`, optional `choices`/`accept`/`tags`). Legacy keys still load:
    `pack_id`, per-question `question_id`, `system_prompt_ref` (external file), and
    `eval_data` in place of `questions`."""
    pack_id = _pack_id_of(payload)
    if not pack_id:
        raise ValueError(f"Question set {pack_path.name!r} is missing an `id`")
    title = str(payload.get("title", pack_id))
    tier = str(payload.get("tier", "unknown"))
    if "answer_type" not in payload:
        raise ValueError(f"Question set {pack_id!r} is missing `answer_type`")
    answer_type = AnswerType(str(payload["answer_type"]))

    # Prefer inline `system_prompt` (the canonical, self-contained form); fall back to
    # a legacy external `system_prompt_ref` for old files.
    system_prompt_ref = payload.get("system_prompt_ref")
    if payload.get("system_prompt"):
        system_prompt = str(payload["system_prompt"]).strip()
    elif system_prompt_ref:
        sp_path = pack_path.parent / system_prompt_ref
        if not sp_path.exists():
            sp_path = pack_path.parent.parent / system_prompt_ref
        system_prompt = sp_path.read_text(encoding="utf-8").strip()
    else:
        raise ValueError(
            f"Question set {pack_id!r} is missing `system_prompt` (put it up top in the file)"
        )

    raw_questions: list[dict[str, Any]] = payload.get("questions") or payload.get("eval_data") or []
    if not raw_questions:
        raise ValueError(f"Question set {pack_id!r} has no `questions`")
    questions: list[PackQuestion] = []
    for raw in raw_questions:
        if "id" not in raw and "question_id" not in raw:
            raise ValueError(f"Question set {pack_id!r}: a question is missing `id`")
        qid = str(raw.get("id", raw.get("question_id")))
        if "answer" not in raw:
            raise ValueError(f"Question set {pack_id!r} question {qid!r}: missing `answer`")
        choices_raw: list[str] | None = raw.get("choices")
        choices: tuple[str, ...] | None = (
            tuple(str(c) for c in choices_raw) if choices_raw else None
        )
        tags: tuple[str, ...] = tuple(str(t) for t in raw.get("tags", []))
        accept: tuple[str, ...] = tuple(str(a) for a in raw.get("accept", []))
        answer = str(raw["answer"]).strip()
        if answer_type is AnswerType.MULTIPLE_CHOICE:
            answer = answer.upper()

        answer_source = str(raw.get("answer_source", "")) or _derive_answer_source(qid, answer_type)

        # Validate MC questions. `choices` is OPTIONAL: some sets (e.g. SimpleBench)
        # embed the options directly in the prompt, so we only range-check when an
        # explicit choices list is provided.
        if answer_type is AnswerType.MULTIPLE_CHOICE:
            if len(answer) != 1 or answer not in "ABCDEF":
                raise ValueError(
                    f"Pack {pack_id!r} question {qid!r}: MC answer must be a single letter A-F"
                )
            if choices and (ord(answer) - ord("A")) >= len(choices):
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
