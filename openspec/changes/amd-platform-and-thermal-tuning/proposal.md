# Proposal

## Why

The fork can discover AMD cards and compare llama.cpp HIP/Vulkan, but its device inventory and measurement controls still assume NVIDIA in important places. That leaves AMD runs harder to trust, prevents power and thermal objectives on the RX 7900 XT, and hides other ROCm-capable engines from the search.

## What Changes

- Make AMD and Vulkan device discovery and per-device launch selection deterministic, including support for AMD SMI installations and clear backend diagnostics.
- Add vendor-neutral GPU power, energy, clock, memory, and temperature telemetry, plus opt-in AMD power-cap trials guarded by temperature limits and reliable state restoration.
- Discover ROCm-capable vLLM and SGLang installations and admit them only when the installed runtime, GPU, and model are supported.
- Correct installation and Docker guidance so AMD users install this fork and can choose a compatible native or ROCm container path.
- Keep CPU-only unit tests on public GitHub-hosted runners; record real hardware validation separately and never claim results for untested AMD cards or engines.

## Capabilities

### New Capabilities

- `amd-gpu-tuning`: AMD device discovery, exact device selection, telemetry, guarded power-cap calibration, and restoration behavior.
- `rocm-engine-selection`: Capability-aware discovery and selection of ROCm-backed inference engines.

### Modified Capabilities

None.

## Impact

Affected areas include `rigtune/hardware.py`, llama.cpp and engine backend selection, `rigtune/power.py`, calibration and comparison telemetry, CLI options, packaging metadata, Docker configuration, tests, and AMD usage/benchmark documentation. Runtime dependencies remain optional by backend; GPU controls require explicit opt-in and privileged access only when a user requests power-cap changes.
