from gguf_limit_bench.telemetry import TelemetrySnapshot, classify_failure


def test_classify_failure_recognizes_cuda_and_memory_failures():
    assert classify_failure("CUDA error: out of memory") == "gpu_oom"
    assert classify_failure("failed to allocate memory for kv cache") == "memory_allocation"
    assert classify_failure("Segmentation fault core dumped") == "crash"
    assert classify_failure("process timed out after 300 seconds") == "timeout"
    assert classify_failure("main: error: failed to load model 'bad.gguf'") == "model_load"
    assert classify_failure("process exited with code 0") == "unknown"


def test_telemetry_snapshot_exposes_resource_debugging_fields():
    snapshot = TelemetrySnapshot(
        ram_available_mb=1024,
        ram_used_percent=50.0,
        cpu_used_percent=12.5,
        swap_used_percent=0.0,
        disk_read_mb=1.0,
        disk_write_mb=2.0,
        gpu_used_mb=1000,
        gpu_total_mb=24000,
        gpu_util_percent=90,
        gpu_power_watts=350.0,
    )

    payload = snapshot.to_dict()

    assert payload["cpu_used_percent"] == 12.5
    assert payload["swap_used_percent"] == 0.0
    assert payload["gpu_power_watts"] == 350.0
