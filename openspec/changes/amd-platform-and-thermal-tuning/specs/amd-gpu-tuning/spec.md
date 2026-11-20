# Spec Delta

## Purpose

Provides reliable AMD GPU selection and measured power and thermal tuning so AMD calibration results describe the device and operating limits users actually selected.

## ADDED Requirements

### Requirement: AMD GPU inventory identifies stable devices
The system SHALL discover supported AMD GPUs through available AMD management interfaces and preserve a stable device identifier alongside runtime ordering.

#### Scenario: AMD SMI is installed without ROCm SMI
- **WHEN** AMD SMI returns a supported AMD GPU and ROCm SMI is unavailable
- **THEN** the probe reports the GPU with its model, memory, UUID or bus identifier, and runtime device index

#### Scenario: An AMD management interface is unavailable
- **WHEN** no supported AMD management interface can report an AMD GPU
- **THEN** the probe explains that AMD hardware discovery is unavailable and does not invent memory, UUID, or telemetry values

### Requirement: GPU backend launches are pinned to the selected device
The system SHALL restrict every HIP or Vulkan trial to the exact selected visible device and SHALL refuse ambiguous or unmatched device mappings.

#### Scenario: Two devices have the same display name
- **WHEN** a Vulkan device name matches more than one runtime device
- **THEN** the trial resolves selection using a stable identifier or reports an ambiguity before scoring a result

#### Scenario: Runtime device inventory does not contain the selection
- **WHEN** the selected GPU is absent from the llama.cpp runtime device list
- **THEN** the trial fails with a diagnostic naming the requested GPU and runtime backend

### Requirement: AMD trials record available GPU telemetry
The system SHALL record available power, energy, temperature, clock, and memory readings for the selected AMD GPU and SHALL represent unsupported or missing readings as unavailable.

#### Scenario: A junction temperature sensor is available
- **WHEN** an AMD trial starts and completes
- **THEN** its result records the sampled junction-temperature range and the selected GPU identity

#### Scenario: A telemetry field is unsupported
- **WHEN** the driver or device does not expose a telemetry field
- **THEN** the result marks that field unavailable rather than treating it as zero or omitting the limitation

### Requirement: AMD power-cap tuning is explicit and reversible
The system SHALL change AMD power limits only when the user opts into power tuning, SHALL respect queried device capability bounds and the requested temperature ceiling, and SHALL restore the original power limit after completion, failure, or supported interruption recovery.

#### Scenario: Power tuning is not requested
- **WHEN** calibration runs without the power-control option
- **THEN** the system reads available telemetry without changing the GPU power limit or profile

#### Scenario: A candidate reaches the requested temperature ceiling
- **WHEN** sampled junction temperature reaches the configured ceiling during a candidate
- **THEN** the candidate is stopped and recorded as thermally unsafe, and the original GPU power limit is restored

#### Scenario: Power-control permission or capability is unavailable
- **WHEN** a requested power-cap operation cannot be authorized or supported
- **THEN** calibration stops before changing GPU state and reports the exact capability or permission failure

#### Scenario: A prior process ended before restoring power
- **WHEN** a user invokes the recovery command and a valid restore record exists
- **THEN** the original power limit is restored for the recorded device and the restore record is cleared only after verification
