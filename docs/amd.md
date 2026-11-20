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
