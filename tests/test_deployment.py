import json

from gguf_limit_bench.deployment import export_champion_profile


def test_export_champion_profile_writes_yaml_and_powershell(tmp_path):
    champion = {
        "model_name": "Winner.gguf",
        "model_path": "G:/AI/models/Winner.gguf",
        "settings": {
            "context_size": 135936,
            "parallel": 1,
            "gpu_layers": 99,
            "batch_size": 2048,
            "ubatch_size": 512,
            "flash_attention": True,
        },
        "score": 100.0,
    }
    champion_path = tmp_path / "champion.json"
    champion_path.write_text(json.dumps(champion), encoding="utf-8")

    outputs = export_champion_profile(
        champion_path=champion_path,
        output_dir=tmp_path / "champions",
        llama_server="G:/AI/llama.cpp/llama-server.exe",
        lane="hermes_pilot",
    )

    assert outputs.yaml_path.exists()
    assert outputs.powershell_path.exists()
    assert "--ctx-size 135936" in outputs.powershell_path.read_text(encoding="utf-8")
    assert "--host 127.0.0.1" in outputs.powershell_path.read_text(encoding="utf-8")
    assert "Hermes provider URL" in outputs.note_path.read_text(encoding="utf-8")
