from gguf_limit_bench.telemetry import PeakEnergySampler, TelemetrySnapshot, classify_failure


def _energy_snap(used, power):
    return TelemetrySnapshot(
        ram_available_mb=0,
        ram_used_percent=0.0,
        gpu_used_mb=used,
        gpu_total_mb=24000,
        gpu_power_watts=power,
    )


def test_peak_vram_tracks_max():
    times = iter([0.0, 1.0, 2.0])
    snaps = iter([_energy_snap(1000, 100.0), _energy_snap(3000, 100.0), _energy_snap(2000, 100.0)])
    sampler = PeakEnergySampler(sampler=lambda: next(snaps), clock=lambda: next(times))
    sampler.sample()
    sampler.sample()
    sampler.sample()
    assert sampler.peak_vram_mb == 3000


def test_energy_integrates_power_over_time():
    times = iter([0.0, 1.0, 2.0])
    snaps = iter([_energy_snap(1000, 100.0), _energy_snap(1000, 100.0), _energy_snap(1000, 100.0)])
    sampler = PeakEnergySampler(sampler=lambda: next(snaps), clock=lambda: next(times))
    sampler.sample()
    sampler.sample()
    sampler.sample()
    assert round(sampler.energy_joules, 3) == 200.0
    assert round(sampler.duration_s, 3) == 2.0


def test_energy_none_when_power_unavailable():
    times = iter([0.0, 1.0])
    snaps = iter([_energy_snap(1000, None), _energy_snap(1000, None)])
    sampler = PeakEnergySampler(sampler=lambda: next(snaps), clock=lambda: next(times))
    sampler.sample()
    sampler.sample()
    assert sampler.energy_joules is None


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
