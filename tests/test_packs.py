import json

from gguf_limit_bench.packs import BUILTIN_PACK_IDS, BenchmarkPack, load_benchmark_packs


def test_builtin_benchmark_packs_cover_autoresearch_plan():
    packs = load_benchmark_packs()

    assert BUILTIN_PACK_IDS <= set(packs)
    assert packs["hermes-pilot"].safety_policy == "local_deterministic"
    assert "tool" in packs["tool-calling"].scoring_categories


def test_user_pack_manifest_is_loaded_and_versioned(tmp_path):
    plugin_dir = tmp_path / "plugins" / "benchmarks"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "private-pack.json").write_text(
        json.dumps(
            {
                "id": "private-pack",
                "version": "2026.05.26",
                "description": "Private local prompt pack.",
                "tasks": ["local_fixture"],
                "settings_space": {"context": [16384]},
                "scoring_categories": ["quality"],
                "safety_policy": "local_fixture_only",
                "receipt_schema": "pack.v1",
            }
        ),
        encoding="utf-8",
    )

    packs = load_benchmark_packs(plugin_dir)

    assert isinstance(packs["private-pack"], BenchmarkPack)
    assert packs["private-pack"].version == "2026.05.26"
