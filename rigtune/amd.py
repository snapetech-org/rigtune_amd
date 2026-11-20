"""Read AMD GPU telemetry through the Linux amdgpu sysfs interface.

The kernel exports sensor values directly, so sampling does not need to start
amd-smi for every measurement. AMD SMI remains the management interface for
inventory and explicit power-cap control.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AmdGpuReading:
    memory_used_mb: Optional[float] = None
    utilization_pct: Optional[float] = None
    power_w: Optional[float] = None
    clock_mhz: Optional[float] = None
    junction_temp_c: Optional[float] = None


def normalize_pci_bus_id(value: str) -> str:
    """Normalize sysfs/AMD-SMI BDFs and NVML's eight-digit domain form."""
    parts = value.strip().lower().split(":")
    if len(parts) != 3:
        return value.strip().lower()
    try:
        domain = f"{int(parts[0], 16):04x}"
    except ValueError:
        return value.strip().lower()
    return f"{domain}:{parts[1]}:{parts[2]}"


def device_path(pci_bus_id: str, drm_root: Path = Path("/sys/class/drm")) -> Optional[Path]:
    """Find the amdgpu device by PCI BDF; card numbers are not stable across boots."""
    wanted = normalize_pci_bus_id(pci_bus_id)
    for candidate in drm_root.glob("card[0-9]*/device"):
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if normalize_pci_bus_id(resolved.name) == wanted:
            return candidate
    return None


def _read_number(path: Path) -> Optional[float]:
    try:
        return float(path.read_text(encoding="ascii").strip())
    except (OSError, UnicodeError, ValueError):
        return None


def _hwmon_files(device: Path):
    for hwmon in (device / "hwmon").glob("hwmon*"):
        try:
            labels = list(hwmon.glob("*_label"))
        except OSError:
            continue
        yield hwmon, labels


def read_sysfs_metrics(pci_bus_id: str, drm_root: Path = Path("/sys/class/drm")) -> AmdGpuReading:
    """Read best-effort AMD GPU readings; unavailable fields stay ``None``."""
    device = device_path(pci_bus_id, drm_root)
    if device is None:
        return AmdGpuReading()

    used = _read_number(device / "mem_info_vram_used")
    utilization = _read_number(device / "gpu_busy_percent")
    memory_mb = used / 2**20 if used is not None and used >= 0 else None
    if utilization is not None and not 0 <= utilization <= 100:
        utilization = None

    power_w = None
    clock_mhz = None
    junction_c = None
    for hwmon, labels in _hwmon_files(device):
        if power_w is None:
            for name in ("power1_average", "power1_input"):
                raw = _read_number(hwmon / name)
                if raw is not None and raw >= 0:
                    power_w = raw / 1_000_000.0
                    break
        for label_file in labels:
            try:
                label = label_file.read_text(encoding="ascii").strip().casefold()
            except (OSError, UnicodeError):
                continue
            input_file = label_file.with_name(label_file.name.removesuffix("_label") + "_input")
            raw = _read_number(input_file)
            if raw is None:
                continue
            if label in ("junction", "hotspot", "hot spot", "hot-spot") and junction_c is None:
                junction_c = raw / 1000.0
            if label in ("sclk", "gfx", "gfxclk") and clock_mhz is None and raw >= 0:
                clock_mhz = raw / 1_000_000.0 if raw > 100_000 else raw

    if junction_c is not None and not -40 <= junction_c <= 200:
        junction_c = None
    return AmdGpuReading(memory_mb, utilization, power_w, clock_mhz, junction_c)


def has_junction_sensor(pci_bus_id: str, drm_root: Path = Path("/sys/class/drm")) -> bool:
    """Whether the selected AMD GPU exposes a readable hotspot/junction sensor."""
    return read_sysfs_metrics(pci_bus_id, drm_root).junction_temp_c is not None
