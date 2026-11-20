from __future__ import annotations

from rigtune.hardware import hardware_hash, probe
from rigtune.hfconfig import arch_from_config, dtype_bytes
from tests.conftest import LLAMA_3B_CONFIG, make_hw


def test_hardware_hash_is_stable_and_ignores_free_memory():
    a = make_hw("a100")
    b = make_hw("a100")
    b.gpus[0].vram_free_bytes //= 2
    b.cpu.ram_free_bytes //= 3
    assert hardware_hash(a) == hardware_hash(b)
    assert len(hardware_hash(a)) == 16


def test_hardware_hash_differs_across_machines():
    hashes = {hardware_hash(make_hw(k)) for k in ("a100", "rtx4090", "gtx1080", "cpu")}
    assert len(hashes) == 4


def test_probe_runs_on_this_machine():
    hw = probe()
    assert hw.cpu.logical_cores >= 1
    assert set(hw.backends) == {
        "vllm", "sglang", "llamacpp-cuda", "llamacpp-hip", "llamacpp-vulkan", "llamacpp-cpu", "vllm-cpu"
    }


def test_arch_from_llama_config():
    arch = arch_from_config(LLAMA_3B_CONFIG)
    assert arch.architecture == "LlamaForCausalLM"
    assert arch.num_layers == 28 and arch.num_kv_heads == 8 and arch.head_dim == 128
    # Llama-3.2-3B has 3.21B parameters; the estimate should land within 3%.
    assert abs(arch.num_params - 3.21e9) / 3.21e9 < 0.03
    # KV bytes/token in bf16: 2 * 28 * 8 * 128 * 2 = 114688
    assert arch.kv_bytes_per_token(2) == 114_688


def test_arch_nested_text_config():
    cfg = {"architectures": ["Gemma3ForConditionalGeneration"], "text_config": dict(LLAMA_3B_CONFIG)}
    arch = arch_from_config(cfg)
    assert arch.num_layers == 28


def test_dtype_bytes():
    assert dtype_bytes("bfloat16") == 2 and dtype_bytes("fp8") == 1 and dtype_bytes("float32") == 4


def test_cuda_visible_devices_is_honoured(monkeypatch):
    """NVML enumerates physical devices regardless of CUDA_VISIBLE_DEVICES; the backends do not."""
    from rigtune.hardware import visible_gpus
    from rigtune.models import GPUInfo

    gpus = [GPUInfo(vendor="nvidia", name="A", index=0, uuid="GPU-aaa"),
            GPUInfo(vendor="nvidia", name="B", index=1, uuid="GPU-bbb")]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert visible_gpus(gpus) == gpus
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert visible_gpus(gpus) == []          # CPU-only as far as any backend is concerned
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert visible_gpus(gpus) == []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert [g.name for g in visible_gpus(gpus)] == ["B"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaa")
    assert [g.name for g in visible_gpus(gpus)] == ["A"]


def test_rocm_smi_json_maps_amd_uuid_memory_and_runtime_index():
    from rigtune.hardware import _parse_rocm_smi_json

    raw = '{"card1":{"Unique ID":"0x6bdce6ea1d388c5c","VRAM Total Memory (B)":"21458059264",' \
          '"VRAM Total Used Memory (B)":"58236928","Card Series":"AMD Radeon RX 7900 XT",' \
          '"Card Vendor":"Advanced Micro Devices, Inc. [AMD/ATI]","Node ID":"2"}}'
    gpus = _parse_rocm_smi_json(raw)
    assert len(gpus) == 1
    assert gpus[0].name == "AMD Radeon RX 7900 XT" and gpus[0].index == 1
    assert gpus[0].uuid == "GPU-6bdce6ea1d388c5c"
    assert gpus[0].vram_free_bytes == 21_399_822_336


def test_amd_smi_inventory_uses_hip_order_and_stable_ids():
    from rigtune.hardware import _parse_amd_smi_inventory

    listing = [{"gpu": 0, "bdf": "0000:03:00.0", "uuid": "23007550-0000-1000-8088-c382a826700f",
                "node_id": 1, "hip_id": 0, "hip_uuid": "GPU-2388c382a826700f"}]
    static = {"gpu_data": [{"gpu": 0, "asic": {"market_name": "AMD Radeon RX 7900 XT",
                                                    "target_graphics_version": "gfx1100"},
                             "vram": {"size": {"value": 20480, "unit": "MB"}}}]}
    metrics = {"gpu_data": [{"gpu": 0, "mem_usage": {"total_vram": {"value": 20480, "unit": "MB"},
                                                       "free_vram": {"value": 19000, "unit": "MB"}}}]}

    (gpu,) = _parse_amd_smi_inventory(listing, static, metrics)
    assert (gpu.name, gpu.index, gpu.architecture) == ("AMD Radeon RX 7900 XT", 0, "gfx1100")
    assert gpu.pci_bus_id == "0000:03:00.0"
    assert gpu.uuid == "23007550-0000-1000-8088-c382a826700f"
    assert gpu.hip_uuid == "GPU-2388c382a826700f"
    assert gpu.vram_total_bytes == 20 * 1024**3
    assert gpu.vram_free_bytes == 19000 * 1024**2


def test_amd_smi_inventory_keeps_missing_memory_unavailable():
    from rigtune.hardware import _parse_amd_smi_inventory

    listing = [{"gpu": 0, "bdf": "0000:03:00.0", "hip_id": 0}]
    static = {"gpu_data": [{"gpu": 0, "asic": {"market_name": "AMD Radeon RX 7900 XT"}}]}
    (gpu,) = _parse_amd_smi_inventory(listing, static)
    assert gpu.vram_total_bytes == 0
    assert gpu.vram_free_bytes is None


def test_amd_smi_inventory_skips_malformed_and_duplicate_rows():
    from rigtune.hardware import _parse_amd_smi_inventory

    assert _parse_amd_smi_inventory("not json", {}, {}) == []
    listing = [{"gpu": 0, "bdf": "0000:03:00.0", "hip_id": 0},
               {"gpu": 0, "bdf": "0000:03:00.0", "hip_id": 0}]
    static = {"gpu_data": [{"gpu": 0, "asic": {"market_name": "AMD Radeon RX 7900 XT"}}]}
    assert len(_parse_amd_smi_inventory(listing, static)) == 1


def test_llama_server_device_match_refuses_duplicate_names(monkeypatch):
    import rigtune.hardware as hardware

    monkeypatch.setattr(hardware, "_llama_server_info", lambda _binary: (None, {"vulkan"}, [
        ("vulkan", 0, "AMD Radeon RX 7900 XT"), ("vulkan", 1, "AMD Radeon RX 7900 XT")]))
    assert hardware.llama_server_device_index("vulkan", "AMD Radeon RX 7900 XT", "/llama") is None


def test_llama_server_device_match_prefers_unique_pci_identity(monkeypatch):
    import rigtune.hardware as hardware

    monkeypatch.setattr(hardware, "_llama_server_info", lambda _binary: (None, {"vulkan"}, [
        ("vulkan", 0, "AMD Radeon RX 7900 XT [0000:03:00.0]"),
        ("vulkan", 1, "AMD Radeon RX 7900 XT [0000:04:00.0]")]))
    assert hardware.llama_server_device_index("vulkan", "AMD Radeon RX 7900 XT", "/llama",
                                              gpu_identity="0000:04:00.0") == 1


def test_rocm_visibility_masks_allow_uuid_and_hip_relative_index(monkeypatch):
    from rigtune.hardware import visible_gpus
    from rigtune.models import GPUInfo

    gpus = [GPUInfo(vendor="amd", name="RX 9070 XT", index=0, uuid="GPU-9070"),
            GPUInfo(vendor="amd", name="RX 7900 XT", index=1, uuid="GPU-7900")]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-7900")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    assert [gpu.name for gpu in visible_gpus(gpus)] == ["RX 7900 XT"]


def test_rocr_visibility_accepts_amd_smi_hip_uuid(monkeypatch):
    from rigtune.hardware import visible_gpus
    from rigtune.models import GPUInfo

    gpu = GPUInfo(vendor="amd", name="RX 7900 XT", index=1, hip_uuid="GPU-hip-7900")
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-hip-7900")
    assert visible_gpus([gpu]) == [gpu]


def test_backend_reasons_explain_a_masked_gpu(monkeypatch):
    """With no visible GPU an installed vLLM must say why, not report an empty reason."""
    from rigtune.hardware import probe_backends

    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name in ("vllm", "sglang") else None)
    monkeypatch.setattr("rigtune.hardware._torch_is_cuda_build", lambda: True)
    out = probe_backends([])  # no visible GPUs
    assert not out["vllm"].available and "NVIDIA" in out["vllm"].reason
    assert not out["sglang"].available and "NVIDIA" in out["sglang"].reason


def test_backend_probe_distinguishes_rocm_torch_from_cuda_and_cpu(monkeypatch):
    from rigtune.hardware import probe_backends
    from rigtune.models import GPUInfo

    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name in ("torch", "vllm", "sglang")
                        else None)
    monkeypatch.setattr("rigtune.hardware._pkg_version", lambda name: "1.0" if name in ("vllm", "sglang") else None)
    monkeypatch.setattr("rigtune.hardware._sglang_version", lambda _python: "1.0")
    monkeypatch.setattr("rigtune.hardware._torch_runtime", lambda _python=None: "rocm")
    amd = GPUInfo(vendor="amd", name="AMD Radeon RX 7900 XT", index=1, architecture="gfx1100")
    found = probe_backends([amd])
    assert found["vllm"].available and found["vllm"].runtime == "rocm"
    assert found["sglang"].available and found["sglang"].runtime == "rocm"
    assert found["vllm"].gpu_architecture == "gfx1100"
    assert found["vllm"].validation_level == "detected_only"

    unknown_arch = amd.model_copy(update={"architecture": "gfx9999"})
    found = probe_backends([unknown_arch])
    assert not found["vllm"].available and "unsupported or unknown AMD GPU architecture" in found["vllm"].reason
    assert not found["sglang"].available and "unsupported or unknown AMD GPU architecture" in found["sglang"].reason

    monkeypatch.setattr("rigtune.hardware._torch_runtime", lambda _python=None: "cpu")
    found = probe_backends([amd])
    assert found["vllm-cpu"].available
    assert not found["vllm"].available and "CPU build" in found["vllm"].reason
    assert not found["sglang"].available and "CPU build" in found["sglang"].reason

    monkeypatch.setattr("rigtune.hardware._torch_runtime", lambda _python=None: "cuda")
    found = probe_backends([amd])
    assert not found["vllm"].available and "not a ROCm build" in found["vllm"].reason
    assert not found["sglang"].available and "not a ROCm build" in found["sglang"].reason


def test_nvidia_hardware_hash_remains_compatible_with_existing_profiles(hw_a100):
    from rigtune.hardware import hardware_hash

    assert hardware_hash(hw_a100) == "23b55d8b4f1b7839"


def test_amd_architecture_is_included_in_hardware_hash(hw_rx7900xt):
    from rigtune.hardware import hardware_hash

    with_arch = hw_rx7900xt.model_copy(deep=True)
    without_arch = hw_rx7900xt.model_copy(deep=True)
    with_arch.gpu.architecture = "gfx1100"
    without_arch.gpu.architecture = "gfx1101"
    assert hardware_hash(with_arch) != hardware_hash(without_arch)
