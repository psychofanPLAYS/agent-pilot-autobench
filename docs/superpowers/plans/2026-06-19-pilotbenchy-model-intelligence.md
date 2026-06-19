# PilotBENCHY Model Intelligence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a product-grade local model catalog, confidence-backed Hugging Face enrichment, current llama.cpp MTP profiles, and resumable one/list/filter/all-model campaign planning that always schedules the unchanged ten SimpleBench questions.

**Architecture:** Keep local discovery, Hub retrieval, recommendation extraction, runtime capability validation, catalog persistence, and campaign planning in focused modules. PilotBENCHY owns orchestration and receipts; existing SimpleBench and external benchmark engines remain authoritative. All verification in this plan is offline or dry-run and must not start a model server or benchmark.

**Tech Stack:** Python 3.11+, Typer, dataclasses, `huggingface_hub`, pytest, Ruff, MyPy, existing PilotBENCHY receipts and SimpleBench assets.

---

## File Structure

Create or change these focused units:

- `src/gguf_limit_bench/model_identity.py`: derive and rank Hugging Face identities from filesystem and LM Studio evidence.
- `src/gguf_limit_bench/hf_catalog.py`: official Hub client adapter, revision-pinned model-card retrieval, offline cache records.
- `src/gguf_limit_bench/model_recommendations.py`: parse model-card settings and attach deterministic confidence/provenance.
- `src/gguf_limit_bench/runtime_capabilities.py`: parse llama.cpp version/help and validate normalized arguments.
- `src/gguf_limit_bench/model_catalog.py`: combine discovery, identity, Hub evidence, recommendations, cache, and exports.
- `src/gguf_limit_bench/campaign.py`: freeze one/list/filter/all model queues and mandatory benchmark phases into resumable manifests.
- `src/gguf_limit_bench/external_suites.py`: versioned external benchmark definitions and prerequisite checks only.
- `src/gguf_limit_bench/flag_ladder.py`: emit supported native MTP flags.
- `src/gguf_limit_bench/cli.py`: register `models` commands and dry-run campaign selection without adding more domain logic.
- `src/gguf_limit_bench/data/external_suites.json`: supported upstream suite metadata.
- `third_party/karpathy-autoresearch/`: pinned upstream program contract and provenance notice.
- Tests mirror each new module and use fixtures instead of live Hub or model processes.

### Task 1: Model identity domain and path resolution

**Files:**
- Create: `src/gguf_limit_bench/model_identity.py`
- Modify: `src/gguf_limit_bench/discovery.py`
- Create: `tests/test_model_identity.py`
- Modify: `tests/test_discovery.py`

- [ ] **Step 1: Write failing identity tests**

```python
from pathlib import Path

from gguf_limit_bench.model_identity import IdentityConfidence, resolve_path_identity


def test_resolve_lm_studio_layout_to_exact_repo_and_file():
    path = Path(
        r"G:\AI\models\LM_Studio-gguf\bytkim\Qwen3.6-27B-MTP-pi-reasoning-GGUF"
        r"\Qwen3.6-27B-MTP-pi-reasoning-Q5_K_M.gguf"
    )
    identity = resolve_path_identity(path)
    assert identity.repo_id == "bytkim/Qwen3.6-27B-MTP-pi-reasoning-GGUF"
    assert identity.filename == path.name
    assert identity.confidence is IdentityConfidence.CANDIDATE
    assert identity.source == "lm_studio_path"


def test_plain_filename_stays_unresolved():
    identity = resolve_path_identity(Path("Qwen3.6-27B-Q5_K_M.gguf"))
    assert identity.repo_id is None
    assert identity.confidence is IdentityConfidence.UNRESOLVED
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `uv run pytest tests/test_model_identity.py -v`

Expected: collection fails because `gguf_limit_bench.model_identity` does not exist.

- [ ] **Step 3: Implement the minimal identity model and resolver**

```python
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class IdentityConfidence(StrEnum):
    VERIFIED = "verified"
    STRONG = "strong"
    CANDIDATE = "candidate"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ModelIdentity:
    repo_id: str | None
    filename: str
    confidence: IdentityConfidence
    source: str
    evidence: tuple[str, ...] = ()


def resolve_path_identity(path: Path) -> ModelIdentity:
    parts = path.parts
    lowered = [part.lower() for part in parts]
    if "lm_studio-gguf" in lowered:
        index = lowered.index("lm_studio-gguf")
        if len(parts) >= index + 4:
            repo_id = f"{parts[index + 1]}/{parts[index + 2]}"
            return ModelIdentity(
                repo_id=repo_id,
                filename=path.name,
                confidence=IdentityConfidence.CANDIDATE,
                source="lm_studio_path",
                evidence=(str(path),),
            )
    return ModelIdentity(None, path.name, IdentityConfidence.UNRESOLVED, "filename")
```

- [ ] **Step 4: Extend `ModelInfo` without breaking existing callers**

Add optional `identity: ModelIdentity | None = None` to `ModelInfo`, and attach
`resolve_path_identity(path)` inside `discover_models`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `uv run pytest tests/test_model_identity.py tests/test_discovery.py -v`

Expected: all tests pass and existing ordering/mmproj behavior is unchanged.

- [ ] **Step 6: Commit the slice**

```bash
git add src/gguf_limit_bench/model_identity.py src/gguf_limit_bench/discovery.py tests/test_model_identity.py tests/test_discovery.py
git commit -m "feat: resolve model identities from local paths"
```

### Task 2: LM Studio JSON inventory as optional evidence

**Files:**
- Modify: `src/gguf_limit_bench/model_identity.py`
- Create: `tests/fixtures/lms-models.json`
- Modify: `tests/test_model_identity.py`

- [ ] **Step 1: Add a failing parser test using a small recorded fixture**

```python
from gguf_limit_bench.model_identity import parse_lm_studio_inventory


def test_parse_lm_studio_inventory_preserves_exact_indexed_identifier(tmp_path):
    payload = (FIXTURES / "lms-models.json").read_text(encoding="utf-8")
    entries = parse_lm_studio_inventory(payload)
    pi = entries["qwen3.6-27b-mtp-pi-reasoning"]
    assert pi.repo_id == "bytkim/Qwen3.6-27B-MTP-pi-reasoning-GGUF"
    assert pi.filename == "Qwen3.6-27B-MTP-pi-reasoning-Q5_K_M.gguf"
    assert pi.max_context_length == 262144
    assert pi.trained_for_tool_use is True
```

The fixture contains only the two Pi entries and no local absolute paths or secrets.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_model_identity.py::test_parse_lm_studio_inventory_preserves_exact_indexed_identifier -v`

Expected: failure because `parse_lm_studio_inventory` is missing.

- [ ] **Step 3: Implement JSON parsing with strict shape checks**

Define `LmStudioModelEvidence` and parse `indexedModelIdentifier` with
`partition("/")`/`split("/", 2)`. Invalid rows are skipped with returned diagnostics;
the function never invokes `lms` itself.

- [ ] **Step 4: Add a local inventory gateway**

```python
def read_lm_studio_inventory(command: Sequence[str] = ("lms", "ls", "--llm", "--json")) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "LM Studio inventory failed")
    return completed.stdout
```

The CLI calls this only when `--lm-studio` is explicitly selected. Tests inject a
fake command; ordinary model scanning remains filesystem-only and cannot wake LM Studio.

- [ ] **Step 5: Verify GREEN and commit**

Run: `uv run pytest tests/test_model_identity.py -v`

```bash
git add src/gguf_limit_bench/model_identity.py tests/test_model_identity.py tests/fixtures/lms-models.json
git commit -m "feat: read optional LM Studio model evidence"
```

### Task 3: Revision-pinned Hugging Face catalog adapter and cache

**Files:**
- Create: `src/gguf_limit_bench/hf_catalog.py`
- Create: `tests/test_hf_catalog.py`
- Create: `tests/fixtures/hf/pi-reasoning-model-info.json`
- Create: `tests/fixtures/hf/pi-reasoning-readme.md`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Write failing gateway and cache tests**

```python
def test_fetch_pins_model_card_to_returned_revision(tmp_path):
    api = FakeHubApi(sha="cc2a865c", siblings=[PI_FILENAME], readme=PI_README)
    record = HubCatalog(api=api, cache_root=tmp_path).fetch(PI_REPO, PI_FILENAME)
    assert record.revision == "cc2a865c"
    assert record.filename_verified is True
    assert record.document_confidence == "verified"
    assert (tmp_path / "bytkim--Qwen3.6-27B-MTP-pi-reasoning-GGUF" / "cc2a865c" / "record.json").exists()


def test_offline_load_uses_pinned_cache(tmp_path):
    cached = write_cached_record(tmp_path)
    record = HubCatalog(api=None, cache_root=tmp_path, offline=True).load(cached.repo_id)
    assert record.revision == cached.revision
    assert record.document_confidence == "cached"
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_hf_catalog.py -v`

Expected: missing module failure.

- [ ] **Step 3: Add the official Hub dependency**

Add `huggingface_hub>=0.34` to project dependencies, then run `uv lock`.

- [ ] **Step 4: Implement a narrow injectable adapter**

`HubCatalog.fetch` calls `HfApi.model_info(repo_id, files_metadata=True)`, loads
`ModelCard` at the returned SHA, verifies the exact filename in siblings, and writes
atomic JSON plus README cache files. Use a temporary sibling followed by `Path.replace`;
never delete prior revisions.

- [ ] **Step 5: Handle partial and cached evidence**

Test missing README, missing filename, network failure with cache, and network failure
without cache. Missing filenames retain candidate identity and cannot become verified.

- [ ] **Step 6: Verify GREEN and commit**

Run: `uv run pytest tests/test_hf_catalog.py -v`

```bash
git add pyproject.toml uv.lock src/gguf_limit_bench/hf_catalog.py tests/test_hf_catalog.py tests/fixtures/hf
git commit -m "feat: cache revision-pinned Hugging Face evidence"
```

### Task 4: Recommendation extraction and independent confidence

**Files:**
- Create: `src/gguf_limit_bench/model_recommendations.py`
- Create: `tests/test_model_recommendations.py`

- [ ] **Step 1: Write failing tests for the two Pi cards**

```python
def test_reasoning_card_extracts_sampling_context_and_mtp_claims():
    recommendations = extract_recommendations(PI_REASONING_README, source=SOURCE)
    values = {item.key: item.value for item in recommendations}
    assert values["temperature"] == 1.0
    assert values["top_p"] == 0.95
    assert values["context_size"] == 131072
    assert values["spec_type"] == "draft-mtp"
    assert values["spec_draft_n_max"] == 3
    assert all(item.confidence == "publisher_claim" for item in recommendations)


def test_nonthinking_card_extracts_direct_sampling_profile():
    values = recommendation_values(extract_recommendations(PI_TUNE_README, source=SOURCE))
    assert values["temperature"] == 0.7
    assert values["top_p"] == 0.8
    assert values["top_k"] == 20
    assert values["presence_penalty"] == 1.5
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_model_recommendations.py -v`

- [ ] **Step 3: Implement normalized recommendation records**

```python
@dataclass(frozen=True)
class Recommendation:
    key: str
    value: str | int | float | bool
    confidence: str
    source_url: str
    revision: str
    evidence: str
    parser: str
    local_validation: str = "not_checked"
```

Parse only recognized arguments from fenced `llama-server` command blocks with
`shlex.split(posix=True)` and explicit argument arities. Parse quant/context tables
separately. Preserve unknown tokens as diagnostics; never emit them into commands.

- [ ] **Step 4: Add conflict tests**

When two blocks recommend different values, retain both source claims and mark the
normalized key `conflicted`; do not silently choose the last block.

- [ ] **Step 5: Verify GREEN and commit**

Run: `uv run pytest tests/test_model_recommendations.py -v`

```bash
git add src/gguf_limit_bench/model_recommendations.py tests/test_model_recommendations.py
git commit -m "feat: extract confidence-backed model recommendations"
```

### Task 5: llama.cpp capability validation and stale MTP regression

**Files:**
- Create: `src/gguf_limit_bench/runtime_capabilities.py`
- Create: `tests/test_runtime_capabilities.py`
- Modify: `src/gguf_limit_bench/autoresearch.py`
- Modify: `src/gguf_limit_bench/flag_ladder.py`
- Modify: `src/gguf_limit_bench/workflows.py`
- Modify: `tests/test_simple_bench.py`
- Modify: `tests/test_workflows.py`

- [ ] **Step 1: Write a failing reproduction for installed-build semantics**

```python
HELP = """
--spec-draft-n-max N number of tokens to draft
--spec-type none,draft-simple,draft-mtp
--draft-max N the argument has been removed
"""


def test_removed_alias_is_rejected_and_native_mtp_is_supported():
    capabilities = parse_llama_help("version: 9596 (18ef86ece)", HELP)
    assert capabilities.supports("--spec-type")
    assert capabilities.supports("--spec-draft-n-max")
    assert not capabilities.supports("--draft-max")


def test_mtp_ladder_emits_current_native_arguments():
    ladder = build_core_flag_ladder(enable_mtp=True)
    command = build_server_args(ladder[-1])
    assert command[-4:] == ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
    assert "--draft-max" not in command
```

- [ ] **Step 2: Verify RED for the correct reason**

Run: `uv run pytest tests/test_runtime_capabilities.py tests/test_simple_bench.py -v`

Expected: current command contains removed `--draft-max`.

- [ ] **Step 3: Implement capability parsing**

Store version, commit, supported flags, removed flags, and raw-help SHA256. Exact flags
must be parsed from the option column, not substring searched.

- [ ] **Step 4: Replace generic draft fields with explicit spec fields**

Change `AutoresearchSettings` to `spec_type`, `spec_draft_n_max`,
`spec_draft_n_min`, and `spec_draft_p_min`. Generate one conservative native MTP
profile (`draft-mtp`, three tokens) matching current upstream/model-card guidance.

- [ ] **Step 5: Update workflow and docs-facing plan output**

Both server and CLI command builders emit current native arguments. If a selected
binary lacks them, dry-run marks the profile unsupported instead of launching it.

- [ ] **Step 6: Verify GREEN and commit**

Run: `uv run pytest tests/test_runtime_capabilities.py tests/test_simple_bench.py tests/test_workflows.py -v`

```bash
git add src/gguf_limit_bench/runtime_capabilities.py src/gguf_limit_bench/autoresearch.py src/gguf_limit_bench/flag_ladder.py src/gguf_limit_bench/workflows.py tests/test_runtime_capabilities.py tests/test_simple_bench.py tests/test_workflows.py
git commit -m "fix: use supported native llama.cpp MTP flags"
```

### Task 6: Durable model catalog service and exports

**Files:**
- Create: `src/gguf_limit_bench/model_catalog.py`
- Create: `tests/test_model_catalog.py`

- [ ] **Step 1: Write failing catalog tests**

```python
def test_catalog_keeps_three_confidence_dimensions(tmp_path):
    catalog = ModelCatalog(cache_root=tmp_path, hub=fake_hub()).build([model_info()])
    entry = catalog.entries[0]
    assert entry.identity_confidence == "verified"
    assert entry.document_confidence == "verified"
    assert entry.recommendations[0].confidence == "publisher_claim"


def test_catalog_exports_stable_json_and_markdown(tmp_path):
    paths = write_catalog(catalog_fixture(), tmp_path)
    assert json.loads(paths.json.read_text())["schema_version"] == 1
    assert "Identity confidence" in paths.markdown.read_text()
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_model_catalog.py -v`

- [ ] **Step 3: Implement catalog composition**

Build immutable `CatalogEntry` objects from `ModelInfo`, `ModelIdentity`, `HubRecord`,
recommendations, and runtime validation. One failed enrichment cannot abort other
entries. Sort by normalized repo ID then local path for stable exports.

- [ ] **Step 4: Implement atomic local exports**

Write `catalog.json` and `catalog.md` beneath `_db/catalog/`. Include schema version,
generated timestamp, local roots, errors, confidence, revision, evidence links, and
recommendations. Do not copy full model cards into Markdown output.

- [ ] **Step 5: Verify GREEN and commit**

Run: `uv run pytest tests/test_model_catalog.py -v`

```bash
git add src/gguf_limit_bench/model_catalog.py tests/test_model_catalog.py
git commit -m "feat: build durable local model catalog"
```

### Task 7: Product CLI for model intelligence

**Files:**
- Modify: `src/gguf_limit_bench/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_package_metadata.py`

- [ ] **Step 1: Write failing CLI contract tests**

```python
def test_models_scan_is_local_only(monkeypatch, tmp_path):
    monkeypatch.setattr("gguf_limit_bench.cli.HubCatalog", forbidden_hub)
    result = runner.invoke(app, ["models", "scan", "--model-root", str(tmp_path), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["network_used"] is False


def test_models_enrich_offline_uses_cache(tmp_path):
    result = runner.invoke(app, ["models", "enrich", "--offline", "--cache-root", str(tmp_path)])
    assert result.exit_code == 0
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_cli.py -k 'models_' -v`

- [ ] **Step 3: Register a Typer sub-application**

```python
models_app = typer.Typer(help="Discover and explain local model evidence.")
app.add_typer(models_app, name="models")
```

Add `scan`, `enrich`, `list`, `show`, `recommendations`, and `export` commands. Each
delegates to `model_catalog.py`; CLI code only validates options and renders output.

- [ ] **Step 4: Add deterministic JSON output tests**

Cover ambiguous identities, offline cache, missing model roots, and unsupported flags.
Ensure errors go to stderr and JSON stdout remains parseable.

- [ ] **Step 5: Verify GREEN and commit**

Run: `uv run pytest tests/test_cli.py tests/test_package_metadata.py -v`

```bash
git add src/gguf_limit_bench/cli.py tests/test_cli.py tests/test_package_metadata.py
git commit -m "feat: add model intelligence CLI"
```

### Task 8: Frozen resumable campaign manifests and all-model selection

**Files:**
- Create: `src/gguf_limit_bench/campaign.py`
- Create: `tests/test_campaign.py`
- Modify: `src/gguf_limit_bench/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing selection tests**

```python
def test_all_selects_every_weight_once_and_skips_mmproj():
    plan = plan_campaign(catalog_with_three_weights_and_mmproj(), CampaignSelection(all=True))
    assert [item.path.name for item in plan.models] == ["a.gguf", "b.gguf", "c.gguf"]
    assert plan.skipped[0].reason == "vision_projector"


def test_filters_are_applied_before_manifest_is_frozen():
    plan = plan_campaign(catalog_fixture(), CampaignSelection(all=True, family="qwen", mtp=True))
    assert all(item.family == "qwen" and item.has_mtp for item in plan.models)


def test_resume_preserves_original_queue_order(tmp_path):
    manifest = write_manifest(plan_fixture(), tmp_path)
    mark_completed(manifest, "model-a")
    resumed = load_manifest(manifest)
    assert [item.status for item in resumed.models] == ["completed", "pending"]
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_campaign.py -v`

- [ ] **Step 3: Implement selection and manifest types**

`CampaignSelection` accepts exactly one of explicit model paths, filters, or `all`.
`CampaignManifest` stores schema version, catalog revision digest, frozen ordered model
queue, preset, phases, budgets, runtime capability digest, and status per model.

- [ ] **Step 4: Add `autoresearch --models`, filters, and `--all` dry-run**

Keep the existing singular `--model`. Reject conflicting selection flags. An all-model
campaign defaults to a safe dry-run; `--all --execute` is the explicit live form David
can choose later. Dry-run writes the manifest and exact planned commands but starts no
subprocess. This implementation adds the live path but this session must not invoke it.

- [ ] **Step 5: Prove no process start in tests**

Patch every subprocess gateway to raise if invoked and run an all-model dry-run. Assert
the manifest contains all eligible models and `processes_started == 0`.

- [ ] **Step 6: Verify GREEN and commit**

Run: `uv run pytest tests/test_campaign.py tests/test_cli.py -k 'campaign or autoresearch_all or dry_run' -v`

```bash
git add src/gguf_limit_bench/campaign.py src/gguf_limit_bench/cli.py tests/test_campaign.py tests/test_cli.py
git commit -m "feat: plan resumable all-model campaigns"
```

### Task 9: Make all ten SimpleBench questions mandatory in campaign plans

**Files:**
- Modify: `src/gguf_limit_bench/campaign.py`
- Modify: `src/gguf_limit_bench/simple_bench.py`
- Modify: `tests/test_campaign.py`
- Modify: `tests/test_simple_bench.py`

- [ ] **Step 1: Write the failing invariant tests**

```python
def test_every_full_campaign_model_has_exactly_ten_simplebench_questions():
    plan = plan_campaign(catalog_fixture(3), CampaignSelection(all=True), preset="deep")
    assert len(plan.models) == 3
    for model in plan.models:
        phase = next(item for item in model.phases if item.id == "simplebench-public-10")
        assert phase.required is True
        assert len(phase.question_ids) == 10


def test_campaign_cannot_declare_champion_when_simplebench_is_incomplete():
    result = campaign_result(simplebench_answered=9, simplebench_expected=10)
    assert result.champion is None
    assert result.status == "partial"
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_campaign.py -k simplebench -v`

- [ ] **Step 3: Add a named immutable SimpleBench phase descriptor**

Load the shipped dataset through `load_simple_bench_questions()`, require exactly ten
unique IDs, and store those IDs plus the shipped data checksum in every `quick`,
`agentic`, `openclaw`, `hermes`, and `deep` model plan.

- [ ] **Step 4: Gate champion eligibility**

Campaign aggregation requires ten scored results for each selected model. Missing,
duplicate, failed, or reordered result sets make the campaign partial and expose only
a provisional ranking.

- [ ] **Step 5: Verify GREEN and commit**

Run: `uv run pytest tests/test_campaign.py tests/test_simple_bench.py -v`

```bash
git add src/gguf_limit_bench/campaign.py src/gguf_limit_bench/simple_bench.py tests/test_campaign.py tests/test_simple_bench.py
git commit -m "feat: require ten SimpleBench questions per model"
```

### Task 10: Versioned external OpenClaw and Hermes suite adapters

**Files:**
- Create: `src/gguf_limit_bench/external_suites.py`
- Create: `src/gguf_limit_bench/data/external_suites.json`
- Create: `tests/test_external_suites.py`
- Modify: `src/gguf_limit_bench/campaign.py`

- [ ] **Step 1: Write failing registry tests**

```python
def test_registry_contains_supported_upstream_suites():
    suites = load_external_suites()
    assert {item.id for item in suites} >= {
        "bfcl", "pawbench", "wildclawbench", "pinchbench", "clawprobench"
    }
    assert all(item.repository.startswith("https://github.com/") for item in suites)
    assert all(item.license for item in suites)


def test_missing_external_tool_is_not_scored_as_zero():
    result = check_suite_prerequisites(suite("pawbench"), which=lambda _: None)
    assert result.status == "not_installed"
    assert result.score is None
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_external_suites.py -v`

- [ ] **Step 3: Add source-controlled suite metadata**

Record repository, license, intended harnesses, default smoke slice, heavy storage
warning, required executables, and result parser contract. Do not vendor suite tasks or
download images.

- [ ] **Step 4: Implement prerequisite and plan adapters**

Return versioned command plans only. PawBench and WildClawBench plans support OpenClaw
and Hermes. BFCL remains the lightweight function-calling suite. PinchBench and
ClawProBench are OpenClaw-only. Missing prerequisites remain `not_installed`.

- [ ] **Step 5: Attach configured suites to campaign presets**

`agentic` includes BFCL/Inspect when installed; `openclaw` and `hermes` add only suites
explicitly selected or configured. `deep` freezes selected versions in the manifest.

- [ ] **Step 6: Verify GREEN and commit**

Run: `uv run pytest tests/test_external_suites.py tests/test_campaign.py -v`

```bash
git add src/gguf_limit_bench/external_suites.py src/gguf_limit_bench/data/external_suites.json src/gguf_limit_bench/campaign.py tests/test_external_suites.py tests/test_campaign.py
git commit -m "feat: plan versioned agent benchmark adapters"
```

### Task 11: Preserve the pinned Karpathy operating contract

**Files:**
- Create: `third_party/karpathy-autoresearch/program.md`
- Create: `third_party/karpathy-autoresearch/UPSTREAM.md`
- Modify: `docs/AUTORESEARCH-PROGRAM.md`
- Create: `tests/test_upstream_provenance.py`

- [ ] **Step 1: Write a failing provenance test**

```python
def test_karpathy_program_is_pinned_and_attributed():
    upstream = Path("third_party/karpathy-autoresearch/UPSTREAM.md").read_text()
    program = Path("third_party/karpathy-autoresearch/program.md").read_text()
    assert "228791fb499afffb54b46200aca536f79142f117" in upstream
    assert "https://github.com/karpathy/autoresearch" in upstream
    assert "MIT" in upstream
    assert "The experiment loop" in program
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_upstream_provenance.py -v`

- [ ] **Step 3: Copy the exact pinned upstream `program.md`**

Use the already pulled upstream checkout at
`G:\AI\_codex_projects\_upstream\karpathy-autoresearch`, verify commit
`228791fb499afffb54b46200aca536f79142f117`, and copy only `program.md`.

- [ ] **Step 4: Write the provenance notice**

Record upstream URL, commit, file SHA256, retrieval date, README-declared MIT license,
what was adapted, and why unrelated training files were not vendored.

- [ ] **Step 5: Verify GREEN and commit**

Run: `uv run pytest tests/test_upstream_provenance.py -v`

```bash
git add third_party/karpathy-autoresearch docs/AUTORESEARCH-PROGRAM.md tests/test_upstream_provenance.py
git commit -m "docs: preserve pinned Karpathy autoresearch contract"
```

### Task 12: Product documentation and operator-safe defaults

**Files:**
- Modify: `README.md`
- Modify: `docs/COMMAND-BOARD.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `pilotBENCHY_toDO.md`
- Modify: `CHANGELOG.md`
- Create: `tests/test_model_intelligence_docs.py`

- [ ] **Step 1: Write failing documentation contract tests**

```python
def test_readme_documents_catalog_and_all_model_dry_run():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "apb models scan" in readme
    assert "apb models enrich" in readme
    assert "apb autoresearch --all --dry-run" in readme
    assert "10-question SimpleBench" in readme
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_model_intelligence_docs.py -v`

- [ ] **Step 3: Document the product workflow**

Explain scan -> enrich -> inspect recommendations -> dry-run campaign -> later live
campaign. Include confidence meanings, offline behavior, external-suite prerequisites,
all-model resume semantics, and the promise that scan/enrich/dry-run start no model.

- [ ] **Step 4: Correct obsolete MTP documentation**

Replace all claims about `--draft-max 8/16/32` with runtime-capability validation and
current `draft-mtp` examples. Clearly label model-card recommendations versus measured
local winners.

- [ ] **Step 5: Verify GREEN and commit**

Run: `uv run pytest tests/test_model_intelligence_docs.py tests/test_public_identity.py -v`

```bash
git add README.md docs/COMMAND-BOARD.md docs/ARCHITECTURE.md pilotBENCHY_toDO.md CHANGELOG.md tests/test_model_intelligence_docs.py
git commit -m "docs: explain model intelligence campaigns"
```

### Task 13: Code-only product verification

**Files:**
- Modify only files required by failures discovered in this task.

- [ ] **Step 1: Confirm no benchmark-owned process was started**

Run read-only process inspection for `llama-server.exe` and record that existing user
services were neither stopped nor changed. Do not interpret an existing process as a
test process.

- [ ] **Step 2: Run the complete unit suite**

Run: `uv run pytest -q`

Expected: all tests pass; no test loads a GGUF or contacts live model endpoints.

- [ ] **Step 3: Run formatting, linting, and typing**

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
```

Expected: zero errors.

- [ ] **Step 4: Run packaging verification**

```bash
uv run python -m compileall -q src tests
uv build
uv run agent-autobench --help
uv run agent-autobench models --help
uv run agent-autobench autoresearch --all --dry-run --model-root tests/fixtures/models
```

Expected: build succeeds; commands exit zero; dry-run reports zero processes started
and ten SimpleBench question IDs per eligible model.

- [ ] **Step 5: Run static security checks**

Run `semgrep --config auto src tests` when available. If absent, record the absence and
use Ruff, MyPy, compileall, tests, and `git diff --check` as the documented fallback.

- [ ] **Step 6: Audit the approved specification**

Create a requirement-to-evidence checklist from all twelve acceptance criteria in
`docs/superpowers/specs/2026-06-19-pilotbenchy-model-intelligence-design.md`. Mark
hardware acceptance explicitly deferred and do not claim a best model yet.

- [ ] **Step 7: Keep verification fixes inside their owning task**

If verification exposes a defect, return to the task that owns that behavior, add a
failing regression test, implement the minimal fix, rerun that task's verification,
and amend or add a scoped commit there. Do not create a vague mixed verification commit.
