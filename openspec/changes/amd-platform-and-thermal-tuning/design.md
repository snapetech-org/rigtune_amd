# Design

## Context

See `proposal.md` and the capability requirements in `specs/`. The current fork already has HIP and Vulkan llama.cpp launch paths, but inventory, telemetry, power objectives, and the vLLM/SGLang gates still contain NVIDIA-only assumptions. An upstream update is available, but this published feature branch is based on the fork's current `main`; its history will be preserved and unrelated upstream commits will not be folded into the open PR as part of this change.

## Goals / Non-Goals

**Goals:**

- Make device identity and runtime ordering explicit across discovery, backend selection, measurements, and reports.
- Reuse the existing power-state persistence and restoration lifecycle for AMD limits.
- Keep AMD power modification opt-in and keep sensor-only runs read-only.
- Make ROCm engine availability truthful at the installed runtime and device capability level.
- Keep ordinary CI hardware-independent and document the boundary between detection and RX 7900 XT validation.

**Non-Goals:**

- Do not set fan curves, undervolt, alter a persistent GPU profile, or change the host's current GPU settings.
- Do not claim SGLang or other ROCm engines are RX 7900 XT validated without a successful hardware run.
- Do not change upstream or fork default branches, rewrite published history, or open an upstream PR.

## Decisions

1. **Use stable identity in addition to runtime index.** Extend the GPU inventory with PCI bus identity where available. AMD SMI is the preferred AMD management interface; ROCm SMI remains a fallback for older installations. Preserve runtime index separately because HIP/Vulkan visibility can remap it. Device matching uses UUID or PCI identity where supported; name-only matches must be unique or fail with an actionable diagnostic. Vulkan selection is resolved from llama.cpp's runtime device listing and pinned with its supported device selector/visibility mechanism.

   **Alternative:** match only model names and host indexes. Rejected because duplicate cards and container-visible index remapping can benchmark the wrong GPU.

2. **Keep telemetry provider-neutral at the Rigtune boundary.** Introduce a small sampler/provider interface that emits timestamped readings tied to a stable GPU identity. Keep NVML as its current provider and add an AMD SMI/ROCm SMI provider. Map supported fields into the existing measurement/provenance structures; unsupported sensors are explicit unavailable values. If an optional CLI or library is missing, discovery reports that limitation without fabricating zeros.

   **Alternative:** add a mandatory telemetry dependency. Rejected because llama.cpp-only AMD installs should remain usable without ROCm management tooling.

3. **Implement AMD power caps as an opt-in controller using existing restore records.** Query the original and supported cap range before writes, persist the restore record before changing state, and verify every write. Reuse the controller cleanup/reset path for normal exit, exceptions, and supported signals. A sampled junction-temperature guard terminates an unsafe candidate and enters the same restoration path. Recovery clears its record only after the original value is read back. Temperature guard alone is read-only and does not claim to be a hardware-enforced cap.

   **Alternative:** continuously change the cap as a feedback loop or manipulate fan/voltage controls. Rejected because the current task needs reversible power-limit trials and dependable thermal stopping, not host profile management or unstable closed-loop controls.

4. **Probe each inference engine in its own executable/environment.** For vLLM and SGLang, inspect the engine's Python environment and runtime build, then check AMD architecture support before marking it usable. Separate detection from hardware validation in availability and benchmark metadata. Unknown or contradictory runtime/device information stays unavailable with a reason. Avoid a shared CUDA-derived predicate for HIP builds.

   **Alternative:** infer ROCm support from package presence or installed ROCm libraries. Rejected because CPU-only and CUDA-only installations can coexist with ROCm libraries.

5. **Keep packaging paths explicit.** Correct install instructions to name this fork's distribution. Keep the existing NVIDIA Dockerfile intact and provide a separate ROCm/container recipe with runtime prerequisites, rather than trying to make one image satisfy incompatible CUDA and ROCm base images. Public-hosted jobs run CPU-only tests/lint/type checks; real GPU checks remain a separately recorded hardware validation.

   **Alternative:** replace the existing Docker image with a ROCm image. Rejected because it would break the established NVIDIA path.

## Risks / Trade-offs

- [AMD SMI output and flags vary across releases] → parse documented JSON where possible, keep ROCm SMI fallback, and cover representative outputs with fixtures.
- [Power cap writes can fail or the process can be killed hard] → persist before writes, read back after writes, restore on all supported exits, expose recovery, and document that SIGKILL/power loss requires recovery on the next invocation.
- [Temperature sampling cadence can miss short spikes] → treat the guard as a best-effort benchmark safety stop, report sampling interval and observed peak, and retain the card's own thermal protections.
- [Engine/model support changes between ROCm releases] → report exact runtime/architecture and do not convert detection into a performance or validation claim.
- [Upstream changed shared files after this fork branch point] → preserve published history, implement against the PR base, and avoid importing unrelated upstream benchmark artifacts or workflow changes.

## Migration Plan

No migration is needed for existing configs. New power control remains disabled unless explicitly requested. Existing NVIDIA behavior and image paths remain available. After changes, run focused CPU-only tests, lint, and type checks once per milestone; run the full repository validation on the final head. Perform RX 7900 XT discovery/benchmark checks only when the device and relevant runtime are available, and record unavailable checks rather than inventing results. Rollback is a normal revert of the focused commits; no persistent host setting is changed by this repository work.
