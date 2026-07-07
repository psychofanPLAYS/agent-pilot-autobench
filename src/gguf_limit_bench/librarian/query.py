"""Librarian job: ``query`` - query expansion and HyDE intent.

This pack measures whether a local model can choose a useful retrieval-query
payload instead of answering the user's question directly. The correct option
contains a compact lexical vector plus a HyDE-style synthetic document that
describes what an answering memory would look like.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from gguf_limit_bench.librarian._common import (
    LIBRARIAN_SYSTEM_PROMPT,
    AnswerType,
    PackQuestion,
    QuestionPack,
    make_rng,
    shuffle_choices,
)

PACK_ID = "librarian-query"

_INSTRUCTION = (
    "Choose the best retrieval-query expansion payload. The best payload has a "
    "LEX vector with concrete search terms and a HYDE synthetic document that "
    "describes the memory we need to retrieve. Do not answer the user's question. "
    "Reply with the letter of your choice."
)


@dataclass(frozen=True)
class _QueryItem:
    user_query: str
    intent: str
    lex: tuple[str, ...]
    hyde: str
    direct_answer: str
    wrong_intent: str
    keyword_spam: str


_ITEM_BANK: tuple[_QueryItem, ...] = (
    _QueryItem(
        user_query="Why did the Qwen helper stop showing reasoning after the template update?",
        intent="template_reasoning",
        lex=("Qwen", "reasoning_content", "chat template", "enable_thinking", "froggeric"),
        hyde="A troubleshooting note explaining that the Qwen helper needs the active chat template and reasoning_content parsing aligned after a template update.",
        direct_answer="The helper stopped because the template was wrong.",
        wrong_intent="A note about ranking retrieved snippets for a search query.",
        keyword_spam="Qwen Qwen reasoning template helper update tokens llama server answer fix",
    ),
    _QueryItem(
        user_query="Which memories explain why SemLoc should use CLAMSHELL when XTREME is busy?",
        intent="routing_policy",
        lex=("SemLoc", "CLAMSHELL", "XTREME busy", "fallback", "embed rerank"),
        hyde="A routing-policy memory saying SemLoc should prefer the always-on CLAMSHELL retrieval lane when XTREME is busy, gaming, or not worth waking.",
        direct_answer="Use CLAMSHELL when XTREME is busy.",
        wrong_intent="A memory about Windows tray popup hover timing.",
        keyword_spam="SemLoc XTREME CLAMSHELL busy fallback route memory agent use answer",
    ),
    _QueryItem(
        user_query="Find the note about why Gemma 4 replaced Gemma 3 in the active benchmark lane.",
        intent="model_lane_decision",
        lex=("Gemma 4", "Gemma 3", "active comparator", "wiki librarian", "benchmark lane"),
        hyde="A decision note recording that Gemma 4 is the active Google-family comparator and Gemma 3 remains only historical context.",
        direct_answer="Gemma 4 replaced Gemma 3 because the lane was updated.",
        wrong_intent="A note about Python package versions in the QE lab.",
        keyword_spam="Gemma Gemma Gemma benchmark active model old new compare lane answer",
    ),
    _QueryItem(
        user_query="What should I inspect before trusting a llama.cpp flag recommendation?",
        intent="runtime_doctor",
        lex=("llama.cpp", "flag support", "runtime doctor", "props", "slots", "template"),
        hyde="A runtime-doctor checklist saying to inspect local help text, live props, slots, loaded template, and launch profile before trusting flag advice.",
        direct_answer="Inspect props and slots first.",
        wrong_intent="A note about how to write a changelog for a public release.",
        keyword_spam="llama cpp flags props slots help template launch profile trust recommendation",
    ),
    _QueryItem(
        user_query="Where is the rule saying QE models should not answer questions?",
        intent="qe_contract",
        lex=("QE model", "should not answer", "query expansion", "HyDE", "structured lex vec"),
        hyde="A contract note stating that QE models should output query expansions and HyDE retrieval text instead of solving the user's question.",
        direct_answer="QE models should not answer questions.",
        wrong_intent="A note about model VRAM fit and disk usage.",
        keyword_spam="QE answer question not answer query expansion hyde structured vector output",
    ),
    _QueryItem(
        user_query="Find evidence that two-pack librarian samples are not enough for model recommendations.",
        intent="sample_strength",
        lex=(
            "two-pack samples",
            "false tie",
            "librarian benchmark",
            "recommendation-grade",
            "agent_bench_score",
        ),
        hyde="A benchmark finding explaining that small two-pack samples can create false ties and full or broader librarian coverage is needed before recommending a model.",
        direct_answer="Two-pack samples are not enough.",
        wrong_intent="A note about prompt token throughput at long context.",
        keyword_spam="two pack librarian model recommend score sample enough false tie benchmark",
    ),
    _QueryItem(
        user_query="Which memory explains why generated benchmark receipts should not be deleted?",
        intent="receipt_policy",
        lex=("benchmark receipts", "do not delete", "runs", "evidence", "local database"),
        hyde="A project policy memory saying benchmark receipts, model files, and local databases are evidence artifacts and should not be deleted without explicit approval.",
        direct_answer="Because they are evidence.",
        wrong_intent="A note about choosing a CSS color palette for the web UI.",
        keyword_spam="receipts delete benchmark runs evidence files database model local approval",
    ),
    _QueryItem(
        user_query="Show me why the browser UI and TUI should share the same runners.",
        intent="interface_split",
        lex=("browser UI", "TUI", "same runners", "receipts", "no logic fork"),
        hyde="An architecture note saying the website is the human cockpit, the TUI is the hooked operator session, and both must call shared APB runners and receipts.",
        direct_answer="They should share runners to avoid duplicate logic.",
        wrong_intent="A note about long-context KV cache quantization.",
        keyword_spam="browser TUI CLI runners receipt shared logic no fork user agent app",
    ),
    _QueryItem(
        user_query="Find the discussion about 262k context expectations for modern local models.",
        intent="context_expectation",
        lex=("262k context", "modern models", "Gemma 4", "Qwen3.6", "agentic workflows"),
        hyde="A requirements note saying modern 2026 agent workflows expect 128k to 262k context windows, and 32k is not enough for large system prompts.",
        direct_answer="The expected context is 262k.",
        wrong_intent="A note about Easy MC benchmark packs.",
        keyword_spam="context 262k 128k 32k model modern agentic system prompt big window",
    ),
    _QueryItem(
        user_query="Which notes say bad benchmark data must not corrupt the app's learned data?",
        intent="data_hygiene",
        lex=("bad test results", "archive", "clean data", "benchmark data", "do not corrupt"),
        hyde="A data-hygiene note saying failed or bad benchmark experiments should be archived separately so the app can learn without polluting current recommendation data.",
        direct_answer="Bad test results should be archived.",
        wrong_intent="A note about creating a firewall-visible helper process name.",
        keyword_spam="bad data corrupt benchmark archive clean learned results app mistakes",
    ),
    _QueryItem(
        user_query="Find why pilotBENCHY recommendations must be scores, not vibes.",
        intent="product_contract",
        lex=("pilotBENCHY", "scores not vibes", "hard recommendations", "evidence", "receipts"),
        hyde="A product-contract note saying pilotBENCHY must produce scores, performance predictions, hard recommendations, and receipts rather than fit-only proof or vague claims.",
        direct_answer="pilotBENCHY must produce scores and hard recommendations.",
        wrong_intent="A note about shadcn component usage.",
        keyword_spam="pilotBENCHY scores vibes recommendations evidence receipts proof fit app",
    ),
    _QueryItem(
        user_query="Where did we decide Qwen3.6 remains first for the librarian agent lane?",
        intent="qwen_default",
        lex=("Qwen3.6", "librarian lane", "default", "Gemma 4 comparator", "agentic"),
        hyde="A model-lane decision note saying Qwen3.6-first remains the default for wiki-librarian and agentic librarian work while Gemma 4 is the comparator.",
        direct_answer="Qwen3.6 remains first for librarian work.",
        wrong_intent="A note about Semgrep installation state.",
        keyword_spam="Qwen Qwen3.6 default librarian agent lane Gemma comparator first model",
    ),
)


def _payload(*, lex: tuple[str, ...], hyde: str) -> str:
    return f"LEX: {', '.join(lex)}\nHYDE: {hyde}"


def _make_question(rng: random.Random, seed: int, index: int, item: _QueryItem) -> PackQuestion:
    labels = (
        _payload(lex=item.lex, hyde=item.hyde),
        f"ANSWER: {item.direct_answer}\nHYDE: This directly answers the user instead of expanding retrieval intent.",
        _payload(lex=(item.lex[0], item.lex[1], "misc"), hyde=item.wrong_intent),
        _payload(lex=tuple(item.keyword_spam.split()[:6]), hyde=item.keyword_spam),
    )
    choices, answer = shuffle_choices(rng, labels, 0)
    rendered = "\n\n".join(f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices))
    prompt = (
        f"User question:\n{item.user_query}\n\n"
        "Candidate query-expansion payloads:\n"
        f"{rendered}\n\n"
        f"{_INSTRUCTION}"
    )
    return PackQuestion(
        question_id=f"{PACK_ID}-s{seed}-{index}",
        prompt=prompt,
        answer=answer,
        answer_source="librarian:query",
        choices=choices,
        tags=("librarian", "query", "hyde", f"intent={item.intent}"),
        accept=(),
    )


def build(seed: int = 0) -> QuestionPack:
    rng = make_rng(seed)
    bank = list(_ITEM_BANK)
    rng.shuffle(bank)
    count = rng.randint(10, min(16, len(bank)))
    questions = [_make_question(rng, seed, index, item) for index, item in enumerate(bank[:count])]
    return QuestionPack(
        pack_id=PACK_ID,
        title="Librarian: query expansion and HyDE",
        tier="librarian",
        answer_type=AnswerType.MULTIPLE_CHOICE,
        system_prompt=LIBRARIAN_SYSTEM_PROMPT,
        questions=tuple(questions),
    )
