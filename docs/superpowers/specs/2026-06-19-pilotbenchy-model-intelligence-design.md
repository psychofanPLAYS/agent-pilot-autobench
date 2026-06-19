# PilotBENCHY Model Intelligence and Agentic Campaign Design

Date: 2026-06-19

## Purpose

Turn Agent Pilot Autobench (PilotBENCHY) from a collection of useful benchmark
commands into a coherent local-first product that can answer:

> Which downloaded model, model-specific runtime profile, and measured settings
> are best for my agent workload on this machine?

The product will discover downloaded GGUF models, resolve their Hugging Face
provenance, retrieve model cards and structured metadata with explicit confidence,
extract actionable runtime recommendations, and run reproducible campaigns over one,
several, or every discovered model.

No benchmark is part of this implementation phase. The current user-operated LLM
service must remain untouched until a separate, explicit campaign run is requested.

## Product Principles

1. **Evidence before claims.** A model-card statement is a publisher claim until
   PilotBENCHY verifies it against the local file, runtime, or benchmark evidence.
2. **Use upstream tools.** PilotBENCHY orchestrates Hugging Face, llama.cpp, BFCL,
   PawBench, WildClawBench, and other benchmark engines; it does not reimplement them.
3. **Local-first and restartable.** Model files stay where they are. Network metadata
   is cached. Campaigns write append-only receipts and can resume after interruption.
4. **Fair comparisons are labelled.** Same-quant comparisons are distinguished from
   controls with different quantization, context, or runtime profiles.
5. **Quality before speed.** Speed is useful evidence but cannot make a model an
   agentic winner without quality and task-completion scores.
6. **SimpleBench remains mandatory.** Every full autoresearch candidate must answer
   and be scored on the unchanged 10-question public SimpleBench snapshot.

## Scope

### Included

- Filesystem discovery from one or more configured model roots.
- Optional LM Studio inventory discovery through `lms ls --json` when available.
- Hugging Face repository and file identification from local directory structure,
  filenames, LM Studio identifiers, and Hub search.
- Revision-pinned model metadata and README/model-card retrieval.
- Transparent identity, document, and recommendation confidence.
- Model-card recommendation extraction for llama.cpp flags, sampling, context,
  quantization, prompt/chat-template, MTP/speculative decoding, and vision sidecars.
- Validation of extracted flags against the selected local llama.cpp executable.
- A durable local model catalog with human-readable and machine-readable output.
- Campaign selection for one model, explicit model lists, filtered groups, or all
  discovered models.
- Fixed-budget Karpathy-style keep/discard/crash experimentation per model.
- Mandatory 10-question SimpleBench scoring in full autoresearch campaigns.
- External agentic benchmark adapters and tiered product presets.
- Reports that select a winner only from comparable, completed evidence.

### Excluded from this implementation phase

- Running any live benchmark or loading any model.
- Downloading large benchmark datasets, Docker images, or additional GGUF files.
- Replacing llama.cpp, LM Studio, OpenClaw, Hermes Agent, or upstream benchmark code.
- Automatically deleting models, caches, receipts, or failed experiments.
- Uploading private results or local paths to a public service.

## Recommended Architecture

PilotBENCHY owns discovery, provenance, orchestration, receipts, comparison, and
reporting. Upstream projects own their benchmark logic.

```text
Local model roots / LM Studio inventory
                 |
                 v
          Model discovery
                 |
                 v
     Hugging Face identity resolver
                 |
                 v
  revision-pinned card + metadata cache
                 |
                 v
 recommendation parser + confidence engine
                 |
                 v
       local llama.cpp capability check
                 |
                 v
       durable PilotBENCHY catalog
                 |
                 v
 campaign planner (one/list/filter/all)
                 |
                 v
 system + SimpleBench + external agent suites
                 |
                 v
 receipts -> comparisons -> winner with caveats
```

The implementation should use small focused modules rather than extending
`discovery.py` or `autoresearch.py` into large multipurpose files.

## Model Discovery and Identity

### Discovery sources

1. Configured filesystem roots remain authoritative for runnable GGUF paths.
2. LM Studio JSON is an optional additional source of publisher, repository, model
   key, architecture, context, quantization, and tool-use hints.
3. GGUF filename parsing remains a fallback, not the highest-confidence source.

The current LM Studio layout already provides exact identities such as:

```text
G:\AI\models\LM_Studio-gguf\bytkim\Qwen3.6-27B-MTP-pi-reasoning-GGUF\
  Qwen3.6-27B-MTP-pi-reasoning-Q5_K_M.gguf
```

That path deterministically proposes:

```text
repo: bytkim/Qwen3.6-27B-MTP-pi-reasoning-GGUF
file: Qwen3.6-27B-MTP-pi-reasoning-Q5_K_M.gguf
```

### Resolution order

1. Exact LM Studio `indexedModelIdentifier` repo and file.
2. Exact `<owner>/<repo>/<filename>` filesystem layout with the file present in the
   Hub repository.
3. Exact unique filename match plus compatible file size and parsed quantization.
4. Repository-folder match plus compatible filename.
5. Fuzzy Hub search, returned as unresolved candidates for user review.

Ambiguous matches must never be silently promoted to an exact identity.

## Hugging Face Retrieval and Cache

Use the official `huggingface_hub` Python library rather than handwritten Hub API
logic. Network access is optional at runtime.

For each resolved model, store:

- repository ID and exact revision SHA;
- matched GGUF filename;
- retrieval timestamp and source URLs;
- model-card front matter and rendered Markdown source;
- base-model lineage, datasets, license, pipeline tag, library, tags, and file list;
- local file path, size, parsed quant, architecture hints, MTP hints, and mmproj path;
- retrieval or parsing errors without discarding the rest of the catalog entry.

Cache records belong under ignored local state, not source control. A cached revision
remains usable offline. Refreshes create updated evidence rather than rewriting old
campaign receipts.

## Confidence Model

PilotBENCHY must expose three independent confidence dimensions. One blended number
would hide which part is uncertain.

### Identity confidence

| Level | Meaning |
| --- | --- |
| `verified` | Exact repo and exact filename are present at a pinned Hub revision. |
| `strong` | Exact unique filename and compatible local metadata match. |
| `candidate` | Folder/name evidence is plausible but not unique. |
| `unresolved` | No reliable remote identity was found. |

### Document confidence

| Level | Meaning |
| --- | --- |
| `verified` | README and metadata were retrieved from the pinned matched repository revision. |
| `cached` | A previously pinned revision is available but refresh failed or was skipped. |
| `partial` | Structured metadata or README is missing. |
| `unavailable` | No trustworthy document was retrieved. |

### Recommendation confidence

| Level | Meaning |
| --- | --- |
| `locally_validated` | Publisher recommendation and local executable capability agree. |
| `publisher_claim` | Exact model card recommends it, but it has not been tested locally. |
| `upstream_claim` | Official base-model documentation recommends it. |
| `inferred` | PilotBENCHY inferred it from architecture, quant, or general runtime practice. |
| `rejected` | The local executable does not support the documented flag or the setting conflicts with evidence. |

Every recommendation stores its source URL, revision, evidence excerpt location,
parser method, and validation result. Confidence labels are deterministic and tested;
they are not LLM-generated opinions.

## Recommendation Extraction

Extract structured settings from fenced command examples, tables, front matter, and
clearly labelled recommendation sections. Preserve the original text separately from
the normalized setting.

Recognized setting groups include:

- quantization and approximate memory guidance;
- context size and generation budget;
- temperature, top-p, top-k, min-p, and penalties;
- GPU layers, flash attention, KV cache types, unified KV, batch and microbatch;
- parallel slots and continuous batching;
- chat template and thinking/non-thinking mode;
- MTP/speculative decoding flags;
- vision projector and compatible sidecar source.

Unknown flags are retained as claims but are not emitted into launch profiles.
PilotBENCHY checks `llama-server --help` and records the executable version before
promoting a flag to `locally_validated`.

The installed llama.cpp `b9596` proves why this is required: it rejects the existing
`--draft-max` and `--draft-min` flags, while accepting:

```text
--spec-type draft-mtp --spec-draft-n-max 3
```

The stale MTP command generation is a product bug and must be repaired test-first.

## Catalog Product Surface

Add a cohesive `models` command group while preserving existing commands:

```text
apb models scan
apb models enrich
apb models list
apb models show MODEL
apb models recommendations MODEL
apb models export
```

Expected behavior:

- `scan` performs local-only discovery and never contacts the Hub.
- `enrich` retrieves or refreshes Hub evidence; `--offline` uses cached evidence.
- `list` shows concise identity, quant, size, confidence, and evidence freshness.
- `show` explains provenance, lineage, model-card claims, conflicts, and missing data.
- `recommendations` separates publisher claims, locally supported flags, and
  PilotBENCHY inferences.
- `export` writes deterministic JSON and Markdown suitable for review or support.

The TUI may consume the same catalog service later; CLI behavior is the first product
contract and must not depend on an interactive terminal.

## Campaign Selection and All-Model Autoresearch

Autoresearch selection must support:

```text
apb autoresearch --model PATH
apb autoresearch --models PATH1 --models PATH2
apb autoresearch --filter-family qwen --filter-mtp
apb autoresearch --all
```

`--all` means all discovered runnable model weights after deterministic exclusions:

- skip mmproj/projector files;
- deduplicate the same resolved local file;
- skip missing or unreadable files with a recorded reason;
- honor explicit include/exclude filters;
- plan the full queue before starting the first model.

The campaign writes a manifest containing the frozen model queue, catalog revisions,
selected benchmark phases, per-model budget, total estimated work, and resume state.
An interrupted campaign resumes from the manifest without repeating completed models.

The product must support `--dry-run` for one, filtered, or all-model campaigns. Dry
runs perform no server start and show exact planned commands, time budgets, model
order, benchmark phases, and missing prerequisites.

## Benchmark Contract

### Mandatory full-campaign phases

For every autoresearch candidate in a full campaign:

1. System viability and controlled server startup.
2. The unchanged 10-question MIT-licensed SimpleBench public snapshot.
3. Agentic quick suite with deterministic tool-use and structured-output evidence.
4. Optional configured external suites.

The 10 SimpleBench questions must always run in the same order with the same prompt,
answer keys, and scoring. They remain visible per question in transcripts and reports.
No speed result, external score, or model-card claim may substitute for this phase.

A campaign is incomplete when any selected model did not receive all ten questions.
Incomplete campaigns may show provisional rankings but cannot declare a champion.

### Tiered presets

| Preset | Purpose | Required evidence |
| --- | --- | --- |
| `quick` | Fast compatibility scout | load, basic serving, all 10 SimpleBench questions |
| `agentic` | Practical local model comparison | `quick` plus BFCL/Inspect-style deterministic tasks |
| `openclaw` | OpenClaw-targeted acceptance | `agentic` plus configured PinchBench, ClawProBench, PawBench, or WildClaw slice |
| `hermes` | Hermes-targeted acceptance | `agentic` plus configured PawBench or WildClaw Hermes slice |
| `deep` | Decision-grade product campaign | context/runtime ladder, mandatory SimpleBench, agentic suite, selected external acceptance suites |

External harnesses remain optional dependencies. Missing tools must produce actionable
doctor output and failed/not-run evidence, never a false zero or silent skip.

## Pi Thinking versus Non-Thinking Comparison

The initial clean comparison pair is:

- `Qwen3.6-27B-MTP-pi-reasoning-Q5_K_M.gguf`
- `Qwen3.6-27B-MTP-pi-tune-Q5_K_M.gguf`

They have the same base family and quant class, making them suitable for a controlled
head-to-head. The base Qwen3.6-27B currently installed at a different quant is a
reference control and must be labelled as quant-confounded.

Each Pi model receives two profile classes:

1. **Controlled profile:** common context, cache, parallelism, and benchmark contract.
2. **Native profile:** model-card recommended thinking/non-thinking sampling and MTP
   settings after local capability validation.

This separates intrinsic behavior from the advantage of model-specific tuning.
PilotBENCHY must report task success, SimpleBench results, tool validity, failure
recovery, latency, token use, throughput, and timeout behavior. A single composite
score is accompanied by its component scores and never hides a failed mandatory phase.

## Karpathy Autoresearch Provenance

The repository already adapts Andrej Karpathy's fixed-budget loop from pinned upstream
commit `228791fb499afffb54b46200aca536f79142f117`.

The implementation will preserve an attributed, pinned copy of the relevant upstream
`program.md` contract and an upstream notice. It will not vendor the unrelated
`prepare.py` and `train.py` training implementation because PilotBENCHY optimizes
inference profiles and agent evaluation rather than training a language model.

The adapted contract remains:

1. Freeze the harness and budget.
2. Establish a baseline.
3. Change one profile dimension at a time.
4. Measure one declared comparable campaign score.
5. Record every attempt.
6. Keep improvements; discard regressions; label crashes.
7. Preserve raw logs outside the main agent context.

PilotBENCHY uses campaign commits and receipts safely; it must not automatically reset
or discard unrelated user worktree changes.

## Receipts and Reporting

Each catalog refresh and campaign writes immutable receipts containing:

- catalog identity and revision snapshot;
- local runtime version and capability snapshot;
- frozen campaign manifest;
- exact launch argv as JSON data;
- per-question SimpleBench transcript and score;
- per-task external benchmark source/version and score;
- telemetry, timeouts, crashes, OOMs, and skipped prerequisites;
- controlled-profile and native-profile comparisons;
- confidence labels and evidence links;
- provisional or final winner status.

The final product report answers separately:

- best model for quality;
- best model for agentic completion;
- best model for latency/throughput;
- best balanced model under the selected weights;
- confidence and caveats for each conclusion.

## Error Handling and Safety

- Hub unavailable: use pinned cache and label it `cached`.
- Ambiguous identity: show candidates and require explicit selection before applying
  repository-specific settings.
- Missing README: retain structured Hub and local evidence with `partial` documents.
- Unsupported llama.cpp flag: label `rejected`; do not launch with it.
- Existing listener or served LLM: dry-run and code-only operations remain safe;
  live campaign startup requires an explicit later command and a free configured port.
- OOM or crash: preserve receipt, mark the attempt, unload only the benchmark-owned
  process, and continue according to campaign policy.
- User interruption: stop benchmark-owned children cleanly and leave a resumable
  manifest.
- Dirty Git worktree: never reset, delete, or stage unrelated files.

## Testing Strategy

Implementation follows test-driven development.

### Unit tests

- Path and LM Studio identity resolution.
- Exact, strong, candidate, and unresolved confidence outcomes.
- Model-card/front-matter parsing and malformed-card handling.
- Recommendation extraction and provenance.
- llama.cpp help parsing and removed-flag rejection.
- MTP command generation using current native flags.
- One/list/filter/all campaign selection and deterministic queue freezing.
- Mandatory inclusion of exactly ten SimpleBench questions.
- Composite score behavior when a mandatory phase fails.

### Integration tests without model loading

- Recorded Hugging Face API/model-card fixtures.
- Fake llama.cpp help/version executables.
- Fake model roots and LM Studio JSON.
- Campaign `--dry-run` for one and all models.
- Resume manifests with completed, failed, and pending models.
- External adapter doctor checks with tools present and absent.

### Later hardware acceptance tests

Hardware tests are explicitly deferred until the user says the current served LLM may
be stopped or a separate safe lane is available. The first acceptance campaign will
compare the Pi reasoning and non-thinking Q5 models, with the base model as a labelled
control, and will include all ten SimpleBench questions.

## Acceptance Criteria

The implementation is complete only when:

1. All discovered local GGUF weights appear in a catalog without moving the files.
2. Exact LM Studio/Hugging Face layouts resolve to pinned repositories and filenames.
3. Catalog entries expose all three confidence dimensions and evidence provenance.
4. Model-card recommendations are parsed and separated from inference.
5. Local llama.cpp capability validation rejects removed flags and emits valid native
   MTP commands for the installed build.
6. One, list, filter, and all-model campaign planning work deterministically.
7. An all-model dry run plans every eligible model and starts no process.
8. Every full campaign plan includes exactly the unchanged ten SimpleBench questions
   for every model.
9. External OpenClaw/Hermes benchmark adapters have versioned plan support and honest
   prerequisite failures.
10. Karpathy upstream provenance and the adapted fixed-budget contract are present.
11. Existing tests plus new unit/integration tests, formatting, linting, typing,
    compilation, package build, and static checks pass.
12. No model benchmark, service interruption, or large download occurs during this
    code-only implementation phase.
