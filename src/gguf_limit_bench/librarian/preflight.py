"""Fail-fast preflight gates for librarian benchmark cells."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from gguf_limit_bench.answer_scoring import extract_answer
from gguf_limit_bench.autoresearch import AutoresearchSettings
from gguf_limit_bench.discovery import parse_model_name
from gguf_limit_bench.model_identity import IdentityConfidence, resolve_path_identity
from gguf_limit_bench.pack_runner import _chat
from gguf_limit_bench.packs import AnswerType

PREFLIGHT_FAILURE_CLASS = "preflight_fail"
_KNOWN_STRING = "Librarian preflight tokenization check."
_ANSWER_PROMPT = (
    "Preflight format check. Choose the only correct option.\n"
    "A. apple\nB. banana\nC. carrot\n\nFinal Answer:"
)
_THINKING_PROMPT = (
    "Preflight thinking check. Think if your current template enables it, then answer.\n"
    "A. ready\nB. blocked\n\nFinal Answer:"
)


@dataclass(frozen=True)
class PreflightGateReceipt:
    name: str
    status: str
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LibrarianPreflightReceipt:
    ok: bool
    failure_class: str
    model: str
    family: str
    quant: str
    settings: dict[str, Any]
    gates: tuple[PreflightGateReceipt, ...]

    @property
    def failure(self) -> str:
        for gate in self.gates:
            if gate.status == "fail":
                return gate.name
        return "none"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gates"] = [gate.to_dict() for gate in self.gates]
        payload["failure"] = self.failure
        return payload


def run_librarian_preflight(
    *,
    model: Path,
    settings: AutoresearchSettings,
    base_url: str,
    timeout_seconds: int = 600,
) -> LibrarianPreflightReceipt:
    """Run the five librarian preflight gates before any scored questions."""
    info = parse_model_name(model)
    gates = [
        _identity_gate(model),
        _single_bos_gate(base_url, info.family, timeout_seconds),
        _template_load_gate(settings),
        _thinking_sanity_gate(base_url, info.family, settings, timeout_seconds),
        _answer_channel_gate(base_url, timeout_seconds),
    ]
    ok = all(gate.status != "fail" for gate in gates)
    return LibrarianPreflightReceipt(
        ok=ok,
        failure_class="none" if ok else PREFLIGHT_FAILURE_CLASS,
        model=str(model),
        family=info.family,
        quant=info.quant,
        settings=settings.to_dict(),
        gates=tuple(gates),
    )


def write_preflight_receipt(run_dir: Path, receipt: LibrarianPreflightReceipt) -> Path:
    path = run_dir / "preflight.json"
    path.write_text(json.dumps(receipt.to_dict(), ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def _identity_gate(model: Path) -> PreflightGateReceipt:
    identity = resolve_path_identity(model)
    evidence = {
        "repo_id": identity.repo_id,
        "filename": identity.filename,
        "confidence": identity.confidence,
        "source": identity.source,
        "evidence": list(identity.evidence),
    }
    if identity.repo_id and identity.confidence != IdentityConfidence.UNRESOLVED:
        return PreflightGateReceipt("identity", "pass", evidence=evidence)
    return PreflightGateReceipt(
        "identity",
        "fail",
        "GGUF path could not be resolved to a canonical HF repo slug and artifact.",
        evidence,
    )


def _single_bos_gate(
    base_url: str, family: str, timeout_seconds: int
) -> PreflightGateReceipt:
    if family != "gemma":
        return PreflightGateReceipt("single_bos", "skip", "Only applies to Gemma models.")
    plain = _tokenize(base_url, _KNOWN_STRING, add_special=False, timeout_seconds=timeout_seconds)
    special = _tokenize(base_url, _KNOWN_STRING, add_special=True, timeout_seconds=timeout_seconds)
    if plain is None or special is None:
        return PreflightGateReceipt(
            "single_bos",
            "fail",
            "llama-server /tokenize did not return token arrays.",
        )
    added = len(special) - len(plain)
    evidence = {"plain_tokens": len(plain), "special_tokens": len(special), "added": added}
    if added == 1:
        return PreflightGateReceipt("single_bos", "pass", evidence=evidence)
    return PreflightGateReceipt(
        "single_bos",
        "fail",
        f"Expected exactly one BOS/special token; got {added}.",
        evidence,
    )


def _template_load_gate(settings: AutoresearchSettings) -> PreflightGateReceipt:
    args = tuple(settings.extra_server_args)
    evidence: dict[str, Any] = {"extra_server_args": list(args)}
    if "--jinja" not in args:
        return PreflightGateReceipt(
            "template_load",
            "fail",
            "Librarian benchmark requires llama.cpp Jinja template handling (--jinja).",
            evidence,
        )
    template_file = _value_after(args, "--chat-template-file")
    chat_template = _value_after(args, "--chat-template")
    if template_file:
        path = Path(template_file)
        evidence["template_file"] = str(path)
        try:
            evidence["template_sha256"] = sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            return PreflightGateReceipt(
                "template_load",
                "fail",
                f"Template file could not be read: {exc}",
                evidence,
            )
    if chat_template:
        evidence["chat_template"] = chat_template
    return PreflightGateReceipt("template_load", "pass", evidence=evidence)


def _thinking_sanity_gate(
    base_url: str, family: str, settings: AutoresearchSettings, timeout_seconds: int
) -> PreflightGateReceipt:
    if family != "qwen":
        return PreflightGateReceipt("thinking_sanity", "skip", "Only applies to Qwen models.")
    mode = _thinking_mode(settings.extra_server_args)
    if mode is None:
        return PreflightGateReceipt(
            "thinking_sanity",
            "skip",
            "No explicit enable_thinking knob was provided for this cell.",
            {"extra_server_args": list(settings.extra_server_args)},
        )
    text, *_ = _chat(
        base_url=base_url,
        system_prompt="You are a benchmark preflight assistant.",
        user_content=_THINKING_PROMPT,
        max_tokens=256,
        timeout_seconds=timeout_seconds,
    )
    has_think = "<think>" in text.lower()
    evidence = {"thinking_mode": mode, "contains_think_block": has_think}
    if (mode == "on" and has_think) or (mode == "off" and not has_think):
        return PreflightGateReceipt("thinking_sanity", "pass", evidence=evidence)
    expected = "contain" if mode == "on" else "not contain"
    return PreflightGateReceipt(
        "thinking_sanity",
        "fail",
        f"Expected output to {expected} a <think> block for thinking={mode}.",
        evidence,
    )


def _answer_channel_gate(base_url: str, timeout_seconds: int) -> PreflightGateReceipt:
    # Allow enough tokens for a reasoning ("thinking") model to emit its <think>
    # block and still reach the Final Answer line; 64 tokens starved thinking
    # models and produced false answer_channel failures.
    text, *_ = _chat(
        base_url=base_url,
        system_prompt="You are a benchmark preflight assistant. Reply with Final Answer: A.",
        user_content=_ANSWER_PROMPT,
        max_tokens=1024,
        timeout_seconds=timeout_seconds,
    )
    answer = extract_answer(text, AnswerType.MULTIPLE_CHOICE)
    evidence = {"predicted": answer, "response_chars": len(text)}
    if answer in {"A", "B", "C"}:
        return PreflightGateReceipt("answer_channel", "pass", evidence=evidence)
    return PreflightGateReceipt(
        "answer_channel",
        "fail",
        "Warmup answer did not contain a parseable Final Answer / MC letter.",
        evidence,
    )


def _tokenize(
    base_url: str, text: str, *, add_special: bool, timeout_seconds: int
) -> list[Any] | None:
    payload = json.dumps({"content": text, "add_special": add_special}).encode("utf-8")
    request = Request(
        f"{base_url}/tokenize",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError):
        return None
    tokens = data.get("tokens") if isinstance(data, dict) else None
    return tokens if isinstance(tokens, list) else None


def _thinking_mode(args: tuple[str, ...]) -> str | None:
    joined = " ".join(args).lower()
    if "enable_thinking" not in joined:
        return None
    if "enable_thinking\":true" in joined or "enable_thinking:true" in joined:
        return "on"
    if "enable_thinking\":false" in joined or "enable_thinking:false" in joined:
        return "off"
    return None


def _value_after(args: tuple[str, ...], option: str) -> str | None:
    for index, arg in enumerate(args):
        if arg == option and index + 1 < len(args):
            return args[index + 1]
        prefix = f"{option}="
        if arg.startswith(prefix):
            return arg.removeprefix(prefix)
    return None
