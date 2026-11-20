# Tasks

## 1. AMD Inventory and Exact Device Selection

- [x] 1.1 Add AMD SMI JSON discovery with ROCm SMI fallback and stable UUID/PCI identity; verify fixtures cover available, missing, malformed, and duplicate-device output.
- [x] 1.2 Carry stable identity and runtime index through HIP/Vulkan selection and fail closed on ambiguous mappings; verify two same-name devices and missing runtime devices in backend tests.
- [x] 1.3 Document AMD discovery prerequisites and per-device selection; verify documented commands match the CLI help and llama.cpp device listing.

## 2. AMD Telemetry and Thermal Guard

- [x] 2.1 Add vendor-neutral timestamped GPU readings and AMD power/energy/temperature/clock/memory provider; verify partial sensor support is represented as unavailable, never zero.
- [x] 2.2 Connect AMD readings to benchmark summaries, compare-state checks, and provenance; verify AMD measurements identify the selected GPU and preserve temperature range.
- [x] 2.3 Add an opt-in junction-temperature guard that stops unsafe candidates without changing power settings; verify a threshold crossing stops the candidate and records the reason.
- [x] 2.4 Document telemetry availability and best-effort thermal guard behavior; verify examples distinguish sensor-only operation from power modification.

## 3. Reversible AMD Power-Cap Tuning

- [x] 3.1 Implement AMD power-cap query/set/verify using the existing restore-record lifecycle; verify write failure, interruption cleanup, and successful recovery with a mocked AMD SMI adapter.
- [x] 3.2 Add explicit CLI/config options for AMD cap bounds and the thermal ceiling; verify defaults are read-only and invalid/unsupported bounds fail before a write.
- [x] 3.3 Document privilege requirements and recovery procedure; verify the recovery example restores and verifies the saved cap.

## 4. ROCm Engine Selection

- [x] 4.1 Add per-environment ROCm runtime and architecture probes for vLLM; verify CPU-only, CUDA-only, ROCm-supported, and unknown architectures report distinct availability reasons.
- [x] 4.2 Add guarded SGLang ROCm detection only for architectures/runtime combinations the installed engine reports as supported; verify unsupported or unvalidated RX 7900 XT combinations are not advertised as hardware-validated.
- [x] 4.3 Include runtime, architecture, and validation level in engine availability/results and update engine guidance; verify result metadata includes these fields without performance claims from detection alone.

## 5. Install and Container Guidance

- [x] 5.1 Correct package/install instructions to install this fork's distribution; verify the package name and console scripts against `pyproject.toml`.
- [x] 5.2 Add a separate ROCm container path while retaining NVIDIA Docker behavior; verify the Dockerfile uses a ROCm base/runtime and documented host prerequisites.
- [x] 5.3 Keep public hosted CI focused on hardware-independent checks and update AMD usage docs; verify workflow runner labels and run the selected checks once.

## 6. Final Integration

- [x] 6.1 Run final focused and full CPU-only validation, inspect all changes and repo status, and record RX 7900 XT checks that could not run without changing host GPU settings.
- [x] 6.2 Commit and push focused milestones to the existing feature branch; verify remote head matches the final local commit and PR #1 remains open.
