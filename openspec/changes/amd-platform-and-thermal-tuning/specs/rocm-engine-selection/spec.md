# Spec Delta

## Purpose

Lets Rigtune compare ROCm inference engines only when their installed runtime, GPU architecture, and model support are actually usable on the current host.

## ADDED Requirements

### Requirement: ROCm engine availability is runtime-specific
The system SHALL probe each installed engine in the Python environment that will run it and distinguish CUDA, ROCm, CPU, and unsupported runtime builds.

#### Scenario: vLLM is installed in a ROCm environment
- **WHEN** the selected AMD GPU architecture and model are reported as supported by the installed vLLM runtime
- **THEN** vLLM is offered as a calibration candidate with its ROCm runtime and version recorded

#### Scenario: An engine installation is CPU-only or CUDA-only
- **WHEN** the engine package exists but its runtime cannot use the selected AMD GPU
- **THEN** the engine is marked unavailable with a diagnostic instead of being selected for calibration

### Requirement: ROCm engine support is gated by device and model compatibility
The system SHALL check the selected AMD GPU architecture and model compatibility before offering a ROCm engine.

#### Scenario: The model is not supported by the installed engine
- **WHEN** the engine registry or runtime probe excludes the model architecture
- **THEN** the engine is not offered for that model and the reason is included in the probe output

#### Scenario: Engine compatibility cannot be established
- **WHEN** runtime or architecture information is missing or contradictory
- **THEN** the system reports the uncertainty and does not claim the engine is validated on that device

### Requirement: AMD engine results identify their validation level
The system SHALL distinguish software capability detection from successful hardware execution and measured benchmark evidence.

#### Scenario: A ROCm engine has not completed a hardware trial
- **WHEN** a runtime probe succeeds but no calibration trial has completed on the selected GPU
- **THEN** reports label the engine as detected but not hardware-validated and publish no performance claim

#### Scenario: An AMD engine calibration completes
- **WHEN** an AMD engine finishes calibration and confirmation
- **THEN** its result records the GPU architecture, runtime version, engine version, model, workload, and validation outcome
