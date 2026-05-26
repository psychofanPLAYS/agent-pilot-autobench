from io import BytesIO
from pathlib import Path

from gguf_limit_bench.autoresearch import AutoresearchSettings
from gguf_limit_bench.server_probe import (
    DEFAULT_AGENT_TTFT_QUESTIONS,
    build_llama_server_command,
    iter_llama_completion_stream_events,
    measure_llama_completion_stream_ttft,
    serving_prompts_for_context,
)


def test_build_llama_server_command_uses_real_llama_server_flags():
    command = build_llama_server_command(
        llama_server=Path("llama-server.exe"),
        model=Path("model.gguf"),
        settings=AutoresearchSettings(context_size=8192, parallel=2, flash_attention=True),
        host="127.0.0.1",
        port=8088,
    )

    assert command[:3] == ["llama-server.exe", "--model", "model.gguf"]
    assert "--host" in command
    assert "127.0.0.1" in command
    assert "--port" in command
    assert "8088" in command
    assert "--ctx-size" in command
    assert "8192" in command
    assert "--parallel" in command
    assert "2" in command
    assert "--metrics" in command
    assert "--no-webui" in command


def test_iter_llama_completion_stream_events_parses_sse_json_and_stops_at_done():
    stream = BytesIO(
        b"event: message\n"
        b'data: {"content":"hello","tokens":[123],"stop":false}\n\n'
        b'data: {"content":" world","tokens":[456],"stop":false}\n\n'
        b"data: [DONE]\n"
        b'data: {"content":"ignored","tokens":[789]}\n'
    )

    events = list(iter_llama_completion_stream_events(stream))

    assert [event["content"] for event in events] == ["hello", " world"]
    assert [event["tokens"] for event in events] == [[123], [456]]


def test_measure_llama_completion_stream_ttft_uses_cache_prompt_and_timing_fields(monkeypatch):
    captured: dict = {}

    class FakeResponse(BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def fake_urlopen(request, timeout):
        captured["payload"] = request.data.decode("utf-8")
        return FakeResponse(
            b'data: {"content":"hello","tokens":[123],"stop":false}\n\n'
            b'data: {"content":"","stop":true,"tokens_cached":7,"tokens_evaluated":2,'
            b'"timings":{"predicted_per_second":42.5}}\n\n'
        )

    monkeypatch.setattr("gguf_limit_bench.server_probe.urlopen", fake_urlopen)

    result = measure_llama_completion_stream_ttft(
        base_url="http://127.0.0.1:8080",
        prompt="hello",
        max_tokens=1,
        cache_prompt=True,
        timeout_seconds=30,
    )

    assert '"cache_prompt": true' in captured["payload"]
    assert result.ok is True
    assert result.tokens_per_second == 42.5
    assert result.ttft_samples_ms
    assert result.tokens_cached_samples == [7]
    assert result.tokens_evaluated_samples == [2]
    assert result.question_results[0]["question_id"] == "single_prompt"
    assert result.question_results[0]["question_index"] == 1


def test_high_context_uses_stable_five_question_suite():
    prompts = serving_prompts_for_context(
        context_size=32_768,
        fallback_prompt=DEFAULT_AGENT_TTFT_QUESTIONS[0].prompt,
        samples=0,
    )

    assert [prompt.id for prompt in prompts] == [
        prompt.id for prompt in DEFAULT_AGENT_TTFT_QUESTIONS
    ]
    assert len(prompts) == 5


def test_context_tiers_progressively_add_ordered_questions():
    assert [
        prompt.id
        for prompt in serving_prompts_for_context(
            context_size=4_096,
            fallback_prompt=DEFAULT_AGENT_TTFT_QUESTIONS[0].prompt,
            samples=0,
        )
    ] == ["latency_definition"]
    assert [
        prompt.id
        for prompt in serving_prompts_for_context(
            context_size=8_192,
            fallback_prompt=DEFAULT_AGENT_TTFT_QUESTIONS[0].prompt,
            samples=0,
        )
    ] == ["latency_definition", "tool_plan"]
    assert [
        prompt.id
        for prompt in serving_prompts_for_context(
            context_size=16_384,
            fallback_prompt=DEFAULT_AGENT_TTFT_QUESTIONS[0].prompt,
            samples=0,
        )
    ] == ["latency_definition", "tool_plan", "json_receipt"]


def test_samples_can_raise_deterministic_question_count():
    prompts = serving_prompts_for_context(
        context_size=8_192,
        fallback_prompt=DEFAULT_AGENT_TTFT_QUESTIONS[0].prompt,
        samples=4,
    )

    assert [prompt.id for prompt in prompts] == [
        "latency_definition",
        "tool_plan",
        "json_receipt",
        "failure_triage",
    ]


def test_custom_prompt_is_repeated_only_when_explicitly_supplied():
    prompts = serving_prompts_for_context(
        context_size=4_096,
        fallback_prompt="fallback",
        samples=3,
    )

    assert [prompt.id for prompt in prompts] == ["custom_prompt"] * 3
    assert [prompt.prompt for prompt in prompts] == ["fallback"] * 3
