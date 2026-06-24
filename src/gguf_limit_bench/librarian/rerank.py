"""Librarian job: ``rerank`` — retrieval reranking.

When a coding agent asks the memory layer a question, the librarian retrieves a
handful of candidate snippets and must pick the one that actually answers the
query. The hard part is that good distractors *look* relevant: they share the
query's keywords and topic but do not contain the answer. This job measures
whether the local model can tell the answering snippet from the lookalikes.

Each question presents a QUERY and ``k`` candidate snippets (``k`` is 4 or 5)
rendered as MULTIPLE_CHOICE options A..E. Exactly one snippet answers the query;
the rest are keyword-overlap distractors. The option order is shuffled
deterministically with :func:`gguf_limit_bench.librarian._common.make_rng`, and
the gold ``answer`` is the letter at the correct snippet's shuffled position.

Pure and seed-deterministic: :func:`build` called twice with the same seed
returns byte-identical questions. All randomness flows through
:func:`gguf_limit_bench.librarian._common.make_rng`.
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
)

PACK_ID = "librarian-rerank"

_LETTERS = "ABCDEF"

_INSTRUCTION = (
    "Pick the single snippet that actually answers the query. "
    "Reply with the letter of that snippet."
)


@dataclass(frozen=True)
class _RerankItem:
    """One rerank query: its answering snippet plus keyword-overlap distractors."""

    query: str
    correct: str
    distractors: tuple[str, ...]


# Item bank: each query has exactly one answering snippet and 3-4 distractors that
# share keywords/topic with the query but do NOT answer it. Snippets stay short
# and self-contained so the gold is unambiguous.
_ITEM_BANK: tuple[_RerankItem, ...] = (
    _RerankItem(
        query="What port does the metrics server listen on?",
        correct="The metrics server listens on port 9090.",
        distractors=(
            "The metrics server logs to /var/log/metrics.",
            "The web server listens on port 8080.",
            "Metrics are retained for 30 days.",
            "The metrics server runs as a systemd service.",
        ),
    ),
    _RerankItem(
        query="Which database does the project use as its primary store?",
        correct="The project uses PostgreSQL as its primary datastore.",
        distractors=(
            "The project caches query results in Redis.",
            "Database migrations run automatically on deploy.",
            "The project's database backups run nightly.",
            "Database credentials are stored in the vault.",
        ),
    ),
    _RerankItem(
        query="What is the default branch name for this repository?",
        correct="The repository's default branch is named trunk.",
        distractors=(
            "The repository uses protected branches for releases.",
            "Feature branches are deleted after they merge.",
            "The repository requires signed commits.",
            "Branch names should be prefixed with the ticket id.",
        ),
    ),
    _RerankItem(
        query="How long are application logs retained?",
        correct="Application logs are retained for 14 days.",
        distractors=(
            "Application logs are shipped to the central aggregator.",
            "Audit logs are encrypted at rest.",
            "Logs are written in structured JSON format.",
            "Debug logging is disabled in production.",
        ),
    ),
    _RerankItem(
        query="Which Python version does the project target?",
        correct="The project targets Python 3.12.",
        distractors=(
            "The project uses uv to manage Python dependencies.",
            "Python type hints are checked in CI.",
            "The project drops support for older Python runtimes.",
            "Python tests run under pytest.",
        ),
    ),
    _RerankItem(
        query="What is the rate limit for the public API?",
        correct="The public API allows 100 requests per minute.",
        distractors=(
            "The public API requires an API key in the header.",
            "API responses are paginated by default.",
            "The public API is versioned under /v2.",
            "API errors return a structured JSON body.",
        ),
    ),
    _RerankItem(
        query="Where are nightly database backups stored?",
        correct="Nightly database backups are stored in the offsite S3 bucket.",
        distractors=(
            "Database backups are taken every night at midnight.",
            "Backups are verified with a weekly restore test.",
            "The database runs in a single primary configuration.",
            "Backup retention is configured per environment.",
        ),
    ),
    _RerankItem(
        query="What timeout does the HTTP client use?",
        correct="The HTTP client uses a 30 second request timeout.",
        distractors=(
            "The HTTP client retries failed requests three times.",
            "The HTTP client sends a custom user-agent header.",
            "The HTTP client pools connections per host.",
            "HTTP responses are decompressed automatically.",
        ),
    ),
    _RerankItem(
        query="Which cloud region is production deployed in?",
        correct="Production is deployed in the us-east-1 region.",
        distractors=(
            "Production deploys run through a blue-green pipeline.",
            "Staging mirrors the production configuration.",
            "Production secrets are rotated every quarter.",
            "Production traffic is served behind a load balancer.",
        ),
    ),
    _RerankItem(
        query="What is the maximum upload file size?",
        correct="The maximum upload file size is 25 megabytes.",
        distractors=(
            "Uploaded files are scanned for malware on arrival.",
            "Uploads are stored in object storage.",
            "File uploads must use a multipart request.",
            "Upload progress is reported over a websocket.",
        ),
    ),
    _RerankItem(
        query="How many worker processes does the queue run?",
        correct="The queue runs with 8 worker processes.",
        distractors=(
            "The queue retries failed jobs with backoff.",
            "Queue jobs are prioritized by a numeric weight.",
            "The queue is backed by Redis streams.",
            "Dead-letter jobs are moved to a separate queue.",
        ),
    ),
    _RerankItem(
        query="What email address receives on-call alerts?",
        correct="On-call alerts are sent to oncall@example.com.",
        distractors=(
            "On-call rotation changes every Monday.",
            "Alerts are deduplicated within a five minute window.",
            "Critical alerts also page a phone number.",
            "Alert thresholds are defined in the monitoring config.",
        ),
    ),
    _RerankItem(
        query="Which authentication scheme does the gateway require?",
        correct="The gateway requires OAuth 2.0 bearer tokens.",
        distractors=(
            "The gateway terminates TLS for all services.",
            "The gateway rate-limits unauthenticated requests.",
            "Gateway access logs are sampled at ten percent.",
            "The gateway routes by host header.",
        ),
    ),
    _RerankItem(
        query="What is the cache time-to-live for session data?",
        correct="Session data is cached with a 15 minute time-to-live.",
        distractors=(
            "Session data is stored in an encrypted cookie.",
            "The cache evicts the least recently used entries.",
            "Session ids are regenerated on privilege change.",
            "Cached entries are namespaced by tenant.",
        ),
    ),
)


def _make_question(rng: random.Random, seed: int, index: int, item: _RerankItem) -> PackQuestion:
    """Build one shuffled MULTIPLE_CHOICE rerank question from ``item``."""
    # k = 4 or 5: the correct snippet plus 3 or 4 distractors.
    n_distractors = rng.choice((3, 4))
    n_distractors = min(n_distractors, len(item.distractors))
    distractors = rng.sample(item.distractors, n_distractors)

    options = [item.correct, *distractors]
    rng.shuffle(options)

    k = len(options)
    correct_pos = options.index(item.correct)
    answer = _LETTERS[correct_pos]

    rendered = "\n".join(f"{_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    prompt = f"Query: {item.query}\n\nCandidate snippets:\n{rendered}\n\n{_INSTRUCTION}"

    return PackQuestion(
        question_id=f"{PACK_ID}-s{seed}-{index}",
        prompt=prompt,
        answer=answer,
        answer_source="librarian:rerank",
        choices=tuple(options),
        tags=("librarian", "rerank", f"n_choices={k}"),
        accept=(),
    )


def build(seed: int = 0) -> QuestionPack:
    """Build the deterministic ``librarian-rerank`` pack (10..16 MC questions)."""
    rng = make_rng(seed)

    bank = list(_ITEM_BANK)
    rng.shuffle(bank)

    # Choose how many questions to emit (10..16, bounded by the bank size).
    count = rng.randint(10, min(16, len(bank)))
    chosen = bank[:count]

    questions: list[PackQuestion] = [
        _make_question(rng, seed, index, item) for index, item in enumerate(chosen)
    ]

    return QuestionPack(
        pack_id=PACK_ID,
        title="Librarian: rerank retrieved snippets",
        tier="librarian",
        answer_type=AnswerType.MULTIPLE_CHOICE,
        system_prompt=LIBRARIAN_SYSTEM_PROMPT,
        questions=tuple(questions),
    )
