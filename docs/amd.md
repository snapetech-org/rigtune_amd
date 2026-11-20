# AMD GPU discovery and device selection

Rigtune prefers AMD SMI for GPU inventory. It reads the AMD SMI GPU UUID, PCI bus address, HIP runtime index, product name, GPU architecture, and VRAM. When AMD SMI is unavailable, it falls back to ROCm SMI. Missing VRAM readings remain unknown; Rigtune does not infer them from the card model.

Check the detected device and the llama.cpp builds available on this host:

```bash
amd-smi list --enumeration
amd-smi static --asic --vram
amd-smi metric --mem-usage
rigtune probe --json
```

Install ROCm or the AMD GPU driver stack that provides the runtime used by the selected llama.cpp build. Point Rigtune at a specific binary when multiple builds are installed:

```bash
LLAMA_SERVER_HIP=/path/to/hip/llama-server rigtune probe
LLAMA_SERVER_VULKAN=/path/to/vulkan/llama-server rigtune probe
```

For HIP, Rigtune pins the selected GPU using its HIP UUID when AMD SMI provides one, then exposes it to the process as device zero. For Vulkan, Rigtune reads `llama-server --list-devices`, resolves the selected card to one runtime device, and uses `--device VulkanN` when the server supports that selector. Older builds use `GGML_VK_VISIBLE_DEVICES` only when the user has not already set it. If duplicate device names cannot be disambiguated by a stable identifier in the runtime listing, Rigtune refuses that launch instead of choosing the first matching card.

The tool respects `ROCR_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` while probing HIP devices, and Vulkan's listing honors `GGML_VK_VISIBLE_DEVICES` when it is set. Vulkan devices are resolved through the Vulkan build's own device listing because Vulkan ordering can differ from HIP ordering. Use `rigtune probe --json` to inspect the selected GPU's PCI bus ID, UUIDs, architecture, runtime index, and available memory before calibration.

Discovery means the runtime can see a device; it does not establish that a model can run or that a backend has been hardware-validated. See [benchmarks.md](benchmarks.md) for the measured hardware matrix.

## ROCm vLLM and SGLang

Rigtune checks the PyTorch runtime in each engine environment before offering vLLM or SGLang on AMD. The detected engine must use a ROCm build and the selected GPU architecture must be recognized. Probe and compare metadata label this `detected_only`; a profile says `hardware_trial_completed` only after a successful trial on the selected GPU. For these ROCm paths, Rigtune currently offers BF16 and FP16 weights; it does not reuse CUDA-only compute-capability rules to offer FP8, AWQ, GPTQ, or CUDA KV-cache options.

For vLLM, use its [AMD ROCm installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) and a wheel or image that supports the GPU and ROCm release on the host. For SGLang, use its [AMD GPU guide](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/amd_gpu.mdx); when SGLang lives in a separate environment, set `SGLANG_PYTHON` to that environment's Python. The detection check only verifies the installed runtime and architecture. It does not certify a particular model, quantization method, or performance on Radeon hardware.

The default ROCm engines are pinned to the GPU selected by discovery using its HIP UUID when available, with the HIP process seeing the selected card as device zero. `rigtune probe --json` reports the chosen GPU and which engine runtimes were detected.

## AMD power caps and junction guard

AMD telemetry reads the selected card through the Linux `amdgpu` hwmon/sysfs interface. Missing readings stay unavailable; Rigtune does not substitute zero or the edge temperature for an absent junction reading. When board power is readable, energy is integrated from timestamped power samples with the trapezoid rule. The AMD driver documents power in microwatts and temperature in millidegrees Celsius in its [hwmon interface](https://docs.kernel.org/gpu/amdgpu/thermal.html). Trial rows show the sampled junction-temperature range.

`rigtune power status` reads the selected AMD GPU's PPT0 cap range from AMD SMI. AMD cap writes require root on the host. The controller selects the physical card by PCI bus address, records its exact original cap before writing, checks each write by reading the cap back, and retains the recovery record if restoration cannot be verified. Power modification remains off by default. After an interrupted run, restore it with:

```bash
sudo rigtune power reset
```

Use `--power cap` to include lower PPT0 caps in a calibration. AMD clock locking is not currently supported, so `--power both` can only try caps. `--power-min-w` and `--power-max-w` restrict the cap candidates to an inclusive watt range reported as supported by AMD SMI. The original cap is the baseline and generated candidates never raise it. Bounds, device support, and root access are checked before model download or a cap write. If a card reports its current limit at the minimum supported limit, there is no lower cap to try.

For example, when the selected GPU supports the requested range:

```bash
sudo rigtune recalibrate Qwen/Qwen2.5-3B-Instruct --power cap \
  --power-min-w 150 --power-max-w 220 --max-junction-temp-c 97
```

The optional `--max-junction-temp-c` guard applies to calibration trials when used with `--power cap` or `--power both`:

```bash
sudo rigtune recalibrate Qwen/Qwen2.5-3B-Instruct --power cap --max-junction-temp-c 97
```

Rigtune requires an AMD junction sensor, an available lower PPT0 cap, root permission, and a target below the GPU's reported hotspot slowdown threshold. It monitors warm-up and measured requests at 100 ms intervals. When a sample reaches the target, Rigtune stops the engine, marks that trial, and skips the remaining lower-cap candidates. This is best-effort sampling, not a live serving limiter or a replacement for the GPU's own protection. The winning cap is applied while serving and restored when the server exits; use `rigtune power reset` if a hard kill leaves a recovery record. Calibration rows show the observed junction-temperature minimum and peak when readable, with no edge-temperature substitution.
