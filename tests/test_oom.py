from gguf_limit_bench.oom import is_oom_failure, oom_failure_label


def test_detects_cuda_out_of_memory():
    stderr = (
        "ggml_backend_cuda_buffer_type_alloc_buffer: failed to allocate 8192.00 MiB on device 0: "
        "cudaMalloc failed: out of memory\n"
        "llama_kv_cache_init: failed to allocate buffer for kv cache\n"
    )
    assert is_oom_failure(stderr) is True


def test_detects_plain_out_of_memory():
    assert is_oom_failure("CUDA error: out of memory") is True


def test_detects_failed_to_allocate_on_device():
    assert is_oom_failure("failed to allocate 24000.00 MiB on device 0") is True


def test_timeout_text_is_not_oom():
    assert is_oom_failure("llama-server did not become ready: connection refused") is False
    assert is_oom_failure("llama-server exited early with code 1") is False


def test_empty_stderr_is_not_oom():
    assert is_oom_failure("") is False
    assert is_oom_failure(None or "") is False


def test_pinned_host_memory_warning_alone_is_not_fatal():
    # llama.cpp falls back to pageable memory; this warning is not a hard OOM.
    warning = "ggml_cuda_host_malloc: failed to allocate 512.00 MiB of pinned memory"
    assert is_oom_failure(warning) is False


def test_oom_label_notes_the_context():
    assert "256k" in oom_failure_label(262_144)
