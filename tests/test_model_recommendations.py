from pathlib import Path

from gguf_limit_bench.model_recommendations import (
    RecommendationSource,
    extract_recommendations,
    recommendation_values,
)


FIXTURES = Path(__file__).parent / "fixtures" / "hf"
SOURCE = RecommendationSource(
    url="https://huggingface.co/bytkim/model/blob/cc2a/README.md",
    revision="cc2a",
)


def test_reasoning_card_extracts_sampling_context_and_mtp_claims():
    readme = (FIXTURES / "pi-reasoning-readme.md").read_text(encoding="utf-8")

    recommendations = extract_recommendations(readme, source=SOURCE)
    values = recommendation_values(recommendations)

    assert values["temperature"] == 1.0
    assert values["top_p"] == 0.95
    assert values["context_size"] == 131072
    assert values["spec_type"] == "draft-mtp"
    assert values["spec_draft_n_max"] == 3
    assert all(item.confidence == "publisher_claim" for item in recommendations)
    assert all(item.source_url == SOURCE.url for item in recommendations)


def test_nonthinking_card_extracts_direct_sampling_profile():
    readme = (FIXTURES / "pi-tune-readme.md").read_text(encoding="utf-8")

    values = recommendation_values(extract_recommendations(readme, source=SOURCE))

    assert values["temperature"] == 0.7
    assert values["top_p"] == 0.8
    assert values["top_k"] == 20
    assert values["presence_penalty"] == 1.5
    assert values["cache_type_k"] == "q8_0"


def test_conflicting_command_blocks_are_exposed_not_silently_overwritten():
    readme = """
```bash
llama-server -m model.gguf --temp 0.7
```
```bash
llama-server -m model.gguf --temp 1.0
```
"""

    recommendations = extract_recommendations(readme, source=SOURCE)

    temperatures = [item for item in recommendations if item.key == "temperature"]
    assert [item.value for item in temperatures] == [0.7, 1.0]
    assert all(item.conflicted for item in temperatures)
    assert "temperature" not in recommendation_values(recommendations)


def test_unknown_flags_are_not_emitted_as_recommendations():
    readme = """
```bash
llama-server -m model.gguf --mystery-mode turbo --temp 0.7
```
"""

    recommendations = extract_recommendations(readme, source=SOURCE)

    assert recommendation_values(recommendations) == {"temperature": 0.7}
