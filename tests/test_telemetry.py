from gguf_limit_bench.telemetry import classify_failure


def test_classify_failure_recognizes_cuda_and_memory_failures():
    assert classify_failure("CUDA error: out of memory") == "gpu_oom"
    assert classify_failure("failed to allocate memory for kv cache") == "memory_allocation"
    assert classify_failure("Segmentation fault core dumped") == "crash"
    assert classify_failure("process timed out after 300 seconds") == "timeout"
    assert classify_failure("main: error: failed to load model 'bad.gguf'") == "model_load"
    assert classify_failure("process exited with code 0") == "unknown"
