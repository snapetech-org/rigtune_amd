from __future__ import annotations

from pathlib import Path
from threading import Event
from time import sleep

from rigtune.amd import has_junction_sensor, normalize_pci_bus_id, read_sysfs_metrics
from rigtune.backends.base import LlmtraceHooks
from rigtune.calibrate.measure import Telemetry, _Sample, _summarize


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="ascii")


def _fake_gpu(tmp_path: Path, *, junction: str | None = "97000") -> tuple[Path, Path]:
    drm = tmp_path / "sys" / "class" / "drm"
    device = tmp_path / "sys" / "devices" / "pci0000:15" / "0000:15:00.0"
    (drm / "card0").mkdir(parents=True)
    (drm / "card0" / "device").symlink_to(device)
    _write(device / "gpu_busy_percent", "99\n")
    _write(device / "mem_info_vram_used", str(10 * 1024**3))
    hwmon = device / "hwmon" / "hwmon5"
    _write(hwmon / "power1_average", "231000000\n")
    _write(hwmon / "freq1_label", "sclk\n")
    _write(hwmon / "freq1_input", "2400000000\n")
    _write(hwmon / "temp1_label", "edge\n")
    _write(hwmon / "temp1_input", "81000\n")
    if junction is not None:
        _write(hwmon / "temp2_label", "junction\n")
        _write(hwmon / "temp2_input", f"{junction}\n")
    return drm, device


def test_normalize_pci_bus_id_accepts_nvml_domain_width():
    assert normalize_pci_bus_id("00000000:15:00.0") == "0000:15:00.0"
    assert normalize_pci_bus_id("0000:15:00.0") == "0000:15:00.0"


def test_sysfs_telemetry_maps_power_clock_memory_usage_and_junction(tmp_path):
    drm, _ = _fake_gpu(tmp_path)
    sample = read_sysfs_metrics("00000000:15:00.0", drm)
    assert sample.memory_used_mb == 10240
    assert sample.utilization_pct == 99
    assert sample.power_w == 231
    assert sample.clock_mhz == 2400
    assert sample.junction_temp_c == 97
    assert has_junction_sensor("0000:15:00.0", drm)


def test_sysfs_does_not_treat_edge_temperature_as_junction(tmp_path):
    drm, _ = _fake_gpu(tmp_path, junction=None)
    sample = read_sysfs_metrics("0000:15:00.0", drm)
    assert sample.junction_temp_c is None
    assert not has_junction_sensor("0000:15:00.0", drm)


def test_unknown_pci_device_returns_unavailable_readings(tmp_path):
    sample = read_sysfs_metrics("0000:ff:00.0", tmp_path / "missing")
    assert sample.memory_used_mb is None
    assert sample.power_w is None
    assert sample.junction_temp_c is None


def test_telemetry_stops_on_junction_guard_and_records_peak(tmp_path, monkeypatch):
    drm, _ = _fake_gpu(tmp_path, junction="97000")
    import rigtune.calibrate.measure as measure
    monkeypatch.setattr(measure, "read_sysfs_metrics", lambda bdf: read_sysfs_metrics(bdf, drm))
    hooks = LlmtraceHooks(gpu_vendor="amd", gpu_bdf="0000:15:00.0")
    stopped = Event()
    tel = Telemetry(hooks, interval_ms=5, junction_temp_limit_c=96,
                    on_thermal_limit=lambda _temp: stopped.set())
    tel.start()
    assert stopped.wait(1)
    tel.stop()
    assert tel.summary.source == "amd_sysfs"
    assert tel.summary.junction_temp_min_c == 97
    assert tel.summary.junction_temp_peak_c == 97
    assert tel.summary.thermal_limit_hit


def test_telemetry_summary_preserves_junction_temperature_range():
    summary = _summarize([_Sample(0, junction_temp_c=91, power_w=100),
                          _Sample(1, junction_temp_c=97, power_w=200)], "amd_sysfs")
    assert summary.junction_temp_min_c == 91
    assert summary.junction_temp_peak_c == 97
    assert summary.energy_j == 150  # trapezoid integration over timestamped board-power samples


def test_telemetry_records_sensor_but_does_not_call_edge_a_junction(tmp_path, monkeypatch):
    drm, _ = _fake_gpu(tmp_path, junction=None)
    import rigtune.calibrate.measure as measure
    monkeypatch.setattr(measure, "read_sysfs_metrics", lambda bdf: read_sysfs_metrics(bdf, drm))
    tel = Telemetry(LlmtraceHooks(gpu_vendor="amd", gpu_bdf="0000:15:00.0"), interval_ms=5,
                    junction_temp_limit_c=96)
    tel.start()
    sleep(0.02)
    tel.stop()
    assert tel.summary.source == "amd_sysfs"
    assert tel.summary.junction_temp_peak_c is None
    assert not tel.summary.thermal_limit_hit


def test_junction_guard_requires_supported_opt_in_and_saves_limit_in_profile_key(monkeypatch, hw_rx7900xt):
    import rigtune.amd as amd
    import rigtune.power as power
    from rigtune.pipeline import SearchOptions, validate_junction_guard, validate_power_bounds
    from rigtune.power import Capabilities

    gpu = hw_rx7900xt.gpu.model_copy(update={"pci_bus_id": "0000:15:00.0", "hip_uuid": "GPU-test",
                                            "architecture": "gfx1100"})
    hw = hw_rx7900xt.model_copy(update={"gpus": [gpu]})
    monkeypatch.setattr(amd, "has_junction_sensor", lambda _bdf: True)

    class Capable:
        def capabilities(self):
            return Capabilities(gpu_index=1, power_limit_min_w=120, power_limit_max_w=290,
                                power_limit_default_w=231, power_limit_current_w=231, can_cap=True,
                                junction_slowdown_c=110)

    monkeypatch.setattr(power, "controller_for", lambda *args, **kwargs: Capable())
    validate_junction_guard(hw, "cap", 97)
    assert SearchOptions(max_junction_temp_c=97).key()["junction"] == "97C"
    assert SearchOptions(max_junction_temp_c=96.5).key()["junction"] == "96.5C"
    assert SearchOptions(power_min_w=150, power_max_w=200).key() == {"power_min": "150W", "power_max": "200W"}
    validate_power_bounds(hw, "cap", 150, 200)
    validate_junction_guard(hw, "cap", 97, power_min_w=150, power_max_w=200)

    import pytest
    with pytest.raises(ValueError, match="requires --power cap"):
        validate_junction_guard(hw, "off", 97)
    with pytest.raises(ValueError, match="below this GPU's hotspot"):
        validate_junction_guard(hw, "cap", 110)
    with pytest.raises(ValueError, match="--skip-calibration"):
        validate_junction_guard(hw, "cap", 97, skip_calibration=True)
    with pytest.raises(ValueError, match="require --power cap"):
        validate_power_bounds(hw, "off", power_min_w=150)
    with pytest.raises(ValueError, match="less than or equal"):
        validate_power_bounds(hw, "cap", power_min_w=201, power_max_w=200)
    with pytest.raises(ValueError, match="supported minimum"):
        validate_power_bounds(hw, "cap", power_min_w=100)


def test_rocm_torch_engines_require_matching_runtime_and_pin_the_selected_amd_gpu(monkeypatch, hw_rx7900xt,
                                                                                  prepared_vllm):
    from rigtune.backends import get_backend
    from rigtune.models import ArchInfo, ModelSpec

    gpu = hw_rx7900xt.gpu.model_copy(update={"pci_bus_id": "0000:15:00.0", "hip_uuid": "GPU-test",
                                            "architecture": "gfx1100"})
    backends = {name: item.model_copy(update={"available": True, "runtime": "rocm"})
                for name, item in hw_rx7900xt.backends.items()}
    hw = hw_rx7900xt.model_copy(update={"gpus": [gpu], "backends": backends})
    import rigtune.backends.vllm as vllm_module
    import rigtune.backends.sglang as sglang_module
    monkeypatch.setattr(vllm_module, "load_arch", lambda _spec: ArchInfo(
        architecture="LlamaForCausalLM", num_layers=2, hidden_size=64, num_attention_heads=2,
        num_kv_heads=2, head_dim=32))
    monkeypatch.setattr(sglang_module, "load_arch", lambda _spec: ArchInfo(
        architecture="LlamaForCausalLM", num_layers=2, hidden_size=64, num_attention_heads=2,
        num_kv_heads=2, head_dim=32))

    for name in ("vllm", "sglang"):
        backend = get_backend(name)
        assert backend.supports(hw, ModelSpec(hf_id="org/model"))
        config = backend.default_config(hw, prepared_vllm)
        spec = backend.launch_spec(config, prepared_vllm, 8000)
        assert spec.env["ROCR_VISIBLE_DEVICES"] == "GPU-test"
        assert spec.env["HIP_VISIBLE_DEVICES"] == "0"
        assert not any("rigtune_gpu" in arg for arg in spec.args)

    cuda_only = hw.model_copy(update={"backends": {
        **backends, "vllm": backends["vllm"].model_copy(update={"runtime": "cuda"})}})
    assert not get_backend("vllm").supports(cuda_only, ModelSpec(hf_id="org/model"))
