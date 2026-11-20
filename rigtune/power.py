"""GPU power and clock control for energy tuning.

LLM decode is memory-bandwidth bound: past a certain core clock the SMs wait on memory, so more
frequency buys almost no throughput while power keeps rising steeply. There is therefore a power
cap (or clock) below peak where joules per token is lowest. NVIDIA caps and clock locks are applied
through NVML while the backend is already running. AMD PPT0 caps use AMD SMI; AMD clock locking is unsupported:

  * NVIDIA power cap - nvmlDeviceSetPowerManagementLimit: the board power limit, in watts. Portable
                    across consumer and datacentre cards; the driver picks the clock under the cap.
  * NVIDIA locked clocks - nvmlDeviceSetGpuLockedClocks: pins the SM clock range, in MHz. Direct control
                    of frequency; support varies by card and driver.
  * AMD power cap - AMD SMI's PPT0 setting, selected and restored by PCI bus address.

Both are machine-wide and need root. Every change is recorded in a restore file before it is made
(~/.rigtune/power-restore.json), undone in a finally block, on interpreter exit and on SIGTERM,
and `rigtune power reset` restores it after a hard kill.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

from rigtune import paths
from rigtune.models import Config

logger = logging.getLogger(__name__)

MODES = ("off", "cap", "clock", "both")
# Fractions of the default power limit / max SM clock to try below 100%.
FRACTIONS = (0.85, 0.70, 0.55)


class PowerControlUnavailable(RuntimeError):
    """The requested power or clock change is not permitted or not supported on this machine."""


@dataclass(frozen=True)
class PowerSetting:
    power_limit_w: Optional[int] = None
    sm_clock_mhz: Optional[int] = None

    @property
    def is_default(self) -> bool:
        return self.power_limit_w is None and self.sm_clock_mhz is None

    def label(self) -> str:
        if self.is_default:
            return "default"
        parts = []
        if self.power_limit_w is not None:
            parts.append(f"cap {self.power_limit_w} W")
        if self.sm_clock_mhz is not None:
            parts.append(f"clock <= {self.sm_clock_mhz} MHz")
        return ", ".join(parts)


def setting_of(cfg: Config) -> PowerSetting:
    return PowerSetting(power_limit_w=cfg.power_limit_w, sm_clock_mhz=cfg.sm_clock_mhz)


def with_power(cfg: Config, setting: PowerSetting) -> Config:
    return cfg.model_copy(update={"power_limit_w": setting.power_limit_w, "sm_clock_mhz": setting.sm_clock_mhz})


@dataclass
class Capabilities:
    gpu_index: int
    power_limit_min_w: Optional[int] = None
    power_limit_max_w: Optional[int] = None
    power_limit_default_w: Optional[int] = None
    power_limit_current_w: Optional[int] = None
    sm_clocks_mhz: List[int] = field(default_factory=list)  # supported graphics clocks, descending
    sm_clock_max_mhz: Optional[int] = None
    junction_slowdown_c: Optional[int] = None
    can_cap: bool = False
    can_lock: bool = False
    reasons: Dict[str, str] = field(default_factory=dict)


class PowerController(Protocol):
    @property
    def applied(self) -> Optional[PowerSetting]: ...

    def capabilities(self) -> Capabilities: ...

    def apply(self, setting: PowerSetting) -> None: ...

    def restore(self) -> None: ...


# --------------------------------------------------------------------------- restore file


def restore_path() -> Path:
    return paths.home() / "power-restore.json"


def _write_restore(state: Dict[str, Any]) -> None:
    p = restore_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, staged = tempfile.mkstemp(dir=p.parent, prefix=".power-restore-", suffix=".tmp")
    tmp = Path(staged)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def _clear_restore() -> None:
    try:
        restore_path().unlink()
    except FileNotFoundError:
        pass


def pending_restore() -> Optional[Dict[str, Any]]:
    p = restore_path()
    if not p.exists():
        return None
    try:
        value = json.loads(p.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


# --------------------------------------------------------------------------- NVML implementation


def _load_nvml() -> Any:
    try:
        import pynvml  # type: ignore
    except ImportError as exc:
        raise PowerControlUnavailable("pynvml is not installed (pip install nvidia-ml-py)") from exc
    return pynvml


class NvmlPowerController:
    """Power cap and clock lock for one GPU through NVML. `nvml` is injectable for tests."""

    def __init__(self, gpu_index: int = 0, nvml: Any = None):
        self.gpu_index = gpu_index
        self._nvml = nvml
        self._handle: Any = None
        self._original_limit_mw: Optional[int] = None
        self._clocks_locked = False
        self._applied: Optional[PowerSetting] = None
        self._hooks_installed = False
        self._lock = threading.Lock()

    # ---- plumbing

    @property
    def applied(self) -> Optional[PowerSetting]:
        return self._applied

    def _lib(self) -> Any:
        if self._nvml is None:
            self._nvml = _load_nvml()
        if self._handle is None:
            try:
                self._nvml.nvmlInit()
                self._handle = self._nvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            except Exception as exc:
                raise PowerControlUnavailable(f"NVML unavailable: {exc}") from exc
        return self._nvml

    def _call(self, name: str, *args: Any) -> Any:
        nv = self._lib()
        try:
            return getattr(nv, name)(self._handle, *args)
        except Exception as exc:
            raise PowerControlUnavailable(_explain(name, exc)) from exc

    def _try(self, name: str, *args: Any) -> Any:
        try:
            return self._call(name, *args)
        except PowerControlUnavailable:
            return None

    # ---- query

    def capabilities(self) -> Capabilities:
        caps = Capabilities(gpu_index=self.gpu_index)
        nv = self._lib()
        cons = self._try("nvmlDeviceGetPowerManagementLimitConstraints")
        if cons:
            caps.power_limit_min_w, caps.power_limit_max_w = int(cons[0]) // 1000, int(cons[1]) // 1000
        dflt = self._try("nvmlDeviceGetPowerManagementDefaultLimit")
        if dflt:
            caps.power_limit_default_w = int(dflt) // 1000
        cur = self._try("nvmlDeviceGetPowerManagementLimit")
        if cur:
            caps.power_limit_current_w = int(cur) // 1000
        mems = self._try("nvmlDeviceGetSupportedMemoryClocks") or []
        if mems:
            clocks = self._try("nvmlDeviceGetSupportedGraphicsClocks", max(mems)) or []
            caps.sm_clocks_mhz = sorted({int(c) for c in clocks}, reverse=True)
        mx = self._try("nvmlDeviceGetMaxClockInfo", getattr(nv, "NVML_CLOCK_SM", 1))
        caps.sm_clock_max_mhz = int(mx) if mx else (caps.sm_clocks_mhz[0] if caps.sm_clocks_mhz else None)

        # Permission probes: both writes below leave the device exactly as it was.
        if cur:
            try:
                self._call("nvmlDeviceSetPowerManagementLimit", int(cur))
                caps.can_cap = True
            except PowerControlUnavailable as exc:
                caps.reasons["cap"] = str(exc)
        else:
            caps.reasons["cap"] = "power limit not reported by the driver"
        if caps.sm_clocks_mhz:
            try:
                self._call("nvmlDeviceSetGpuLockedClocks", caps.sm_clocks_mhz[-1], caps.sm_clocks_mhz[0])
                self._call("nvmlDeviceResetGpuLockedClocks")
                caps.can_lock = True
            except PowerControlUnavailable as exc:
                caps.reasons["clock"] = str(exc)
        else:
            caps.reasons["clock"] = "supported clocks not reported by the driver"
        return caps

    # ---- control

    def apply(self, setting: PowerSetting) -> None:
        with self._lock:
            if setting.is_default:
                self._restore_locked()
                return
            self._remember_original()
            self._install_hooks()
            if setting.power_limit_w is not None:
                self._call("nvmlDeviceSetPowerManagementLimit", int(setting.power_limit_w) * 1000)
            elif self._original_limit_mw is not None and self._applied and self._applied.power_limit_w is not None:
                self._call("nvmlDeviceSetPowerManagementLimit", self._original_limit_mw)
            if setting.sm_clock_mhz is not None:
                lo = self._min_clock()
                self._call("nvmlDeviceSetGpuLockedClocks", min(lo, setting.sm_clock_mhz), int(setting.sm_clock_mhz))
                self._clocks_locked = True
            elif self._clocks_locked:
                self._call("nvmlDeviceResetGpuLockedClocks")
                self._clocks_locked = False
            self._applied = setting
            logger.info("GPU %d power setting: %s", self.gpu_index, setting.label())

    def restore(self) -> None:
        with self._lock:
            self._restore_locked()

    def _restore_locked(self) -> None:
        if self._applied is None and not self._clocks_locked and pending_restore() is None:
            return
        errors = []
        if self._original_limit_mw is not None:
            try:
                self._call("nvmlDeviceSetPowerManagementLimit", self._original_limit_mw)
            except PowerControlUnavailable as exc:
                errors.append(str(exc))
        if self._clocks_locked or (self._applied and self._applied.sm_clock_mhz is not None):
            try:
                self._call("nvmlDeviceResetGpuLockedClocks")
            except PowerControlUnavailable as exc:
                errors.append(str(exc))
        self._clocks_locked = False
        self._applied = None
        if errors:
            logger.error("could not fully restore GPU %d power state: %s (run `rigtune power reset`)",
                         self.gpu_index, "; ".join(errors))
            return
        _clear_restore()
        logger.info("GPU %d power state restored", self.gpu_index)

    def _remember_original(self) -> None:
        if self._original_limit_mw is not None:
            return
        existing = pending_restore()
        if restore_path().exists() and existing is None:
            raise PowerControlUnavailable("power restore record is unreadable; refusing to overwrite it")
        if existing and existing.get("gpu_index") == self.gpu_index and existing.get("power_limit_mw"):
            # A previous run died without restoring; the file holds the true original, not the cap.
            self._original_limit_mw = int(existing["power_limit_mw"])
            return
        cur = self._call("nvmlDeviceGetPowerManagementLimit")
        self._original_limit_mw = int(cur)
        _write_restore({"gpu_index": self.gpu_index, "power_limit_mw": self._original_limit_mw})

    def _min_clock(self) -> int:
        mems = self._try("nvmlDeviceGetSupportedMemoryClocks") or []
        clocks = self._try("nvmlDeviceGetSupportedGraphicsClocks", max(mems)) if mems else None
        return min(int(c) for c in clocks) if clocks else 0

    def _install_hooks(self) -> None:
        if self._hooks_installed:
            return
        self._hooks_installed = True
        atexit.register(self.restore)
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            previous = signal.getsignal(signal.SIGTERM)

            def _on_term(signum: int, frame: Any) -> None:
                self.restore()
                if callable(previous) and previous not in (signal.SIG_DFL, signal.SIG_IGN):
                    previous(signum, frame)
                raise SystemExit(128 + signum)

            signal.signal(signal.SIGTERM, _on_term)
        except (ValueError, OSError, AttributeError):  # not the main thread, or no SIGTERM on this platform
            pass


class AmdSmiPowerController:
    """AMD PPT0 cap control through AMD SMI, selected by stable PCI BDF.

    AMD SMI does not expose a read-only permission probe. We therefore report
    control as available only for root with a complete cap range and never
    issue a no-op setter just to test permission.
    """

    def __init__(self, gpu_index: int, pci_bus_id: str, hip_uuid: Optional[str] = None,
                 junction_temp_limit_c: Optional[float] = None, binary: Optional[str] = None,
                 run: Any = subprocess.run, is_root: Optional[bool] = None):
        self.gpu_index = gpu_index
        self.pci_bus_id = _normalize_bdf(pci_bus_id)
        self.hip_uuid = hip_uuid
        self.junction_temp_limit_c = junction_temp_limit_c
        self.binary = binary
        self._run = run
        self._is_root = is_root if is_root is not None else bool(getattr(os, "geteuid", lambda: 1)() == 0)
        self._original_limit_w: Optional[int] = None
        self._applied: Optional[PowerSetting] = None
        self._hooks_installed = False
        self._lock = threading.Lock()

    @property
    def applied(self) -> Optional[PowerSetting]:
        return self._applied

    def _binary(self) -> str:
        if self.binary:
            return self.binary
        from rigtune.hardware import _amd_smi_binary

        found = _amd_smi_binary()
        if not found:
            raise PowerControlUnavailable("amd-smi is not installed or not executable")
        self.binary = found
        return found

    def _call(self, *args: str) -> str:
        try:
            result = self._run([self._binary(), *args], capture_output=True, text=True, timeout=20, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            raise PowerControlUnavailable(f"amd-smi {' '.join(args)} failed: {exc}") from exc
        if result.returncode != 0:
            detail = (getattr(result, "stderr", "") or getattr(result, "stdout", "") or "").strip()
            if "permission" in detail.casefold() or "sudo" in detail.casefold():
                raise PowerControlUnavailable("amd-smi requires root privileges for power-cap control")
            raise PowerControlUnavailable(f"amd-smi {' '.join(args)} failed: {detail or result.returncode}")
        return getattr(result, "stdout", "") or ""

    def _static(self) -> Dict[str, Any]:
        raw = self._call("static", "-g", self.pci_bus_id, "-l", "--json")
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise PowerControlUnavailable(f"amd-smi returned invalid static JSON: {exc}") from exc
        rows = parsed.get("gpu_data") if isinstance(parsed, dict) else None
        if not isinstance(rows, list):
            raise PowerControlUnavailable("amd-smi static output has no gpu_data list")
        if len(rows) != 1 or not isinstance(rows[0], dict):
            raise PowerControlUnavailable(
                f"amd-smi returned an ambiguous response for GPU {self.pci_bus_id}; refusing to use it"
            )
        return rows[0]

    @staticmethod
    def _watts(value: Any) -> Optional[int]:
        if isinstance(value, dict):
            value = value.get("value")
        try:
            number = float(str(value).replace(",", "").strip())
            return int(number) if number > 0 else None
        except (TypeError, ValueError):
            return None

    def _limit_values(self) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        row = self._static()
        limits = row.get("limit") if isinstance(row.get("limit"), dict) else {}
        ppt0 = limits.get("ppt0") if isinstance(limits.get("ppt0"), dict) else {}
        return (self._watts(ppt0.get("min_power_limit")),
                self._watts(ppt0.get("max_power_limit")),
                self._watts(ppt0.get("socket_power_limit")),
                self._watts(limits.get("slowdown_hotspot_temperature")))

    def capabilities(self) -> Capabilities:
        lo, hi, current, slowdown = self._limit_values()
        caps = Capabilities(gpu_index=self.gpu_index, power_limit_min_w=lo, power_limit_max_w=hi,
                            power_limit_default_w=current, power_limit_current_w=current,
                            junction_slowdown_c=slowdown)
        caps.can_lock = False
        caps.reasons["clock"] = "AMD clock locking is not supported by rigtune"
        if lo is None or hi is None or current is None or lo > hi or not lo <= current <= hi:
            caps.reasons["cap"] = "amd-smi did not report a valid PPT0 range and current limit"
        elif not self._is_root:
            caps.reasons["cap"] = "AMD power-cap control requires root on the host"
        else:
            caps.can_cap = True
        return caps

    def apply(self, setting: PowerSetting) -> None:
        with self._lock:
            if setting.sm_clock_mhz is not None:
                raise PowerControlUnavailable("AMD clock locking is not supported by rigtune")
            if setting.is_default:
                self._restore_locked()
                return
            if setting.power_limit_w is None:
                raise PowerControlUnavailable("AMD SMI supports only a PPT0 power cap")
            caps = self.capabilities()
            if not caps.can_cap:
                raise PowerControlUnavailable(caps.reasons.get("cap", "AMD power capping is unavailable"))
            lo, hi = caps.power_limit_min_w, caps.power_limit_max_w
            if lo is None or hi is None or not lo <= setting.power_limit_w <= hi:
                raise PowerControlUnavailable(f"requested cap {setting.power_limit_w} W is outside the AMD PPT0 "
                                               f"range {lo}-{hi} W")
            self._remember_original(caps.power_limit_current_w)
            self._install_hooks()
            self._set_limit(setting.power_limit_w)
            self._applied = setting
            logger.info("GPU %d (%s) power setting: %s", self.gpu_index, self.pci_bus_id, setting.label())

    def restore(self) -> None:
        with self._lock:
            self._restore_locked()

    def _restore_locked(self) -> None:
        state = pending_restore()
        if restore_path().exists() and state is None:
            raise PowerControlUnavailable("power restore record is unreadable; it was preserved")
        if state is not None and (state.get("vendor") != "amd" or state.get("pci_bus_id") != self.pci_bus_id):
            raise PowerControlUnavailable("power restore record belongs to another GPU; it was preserved")
        saved = self._original_limit_w
        if saved is None and state:
            saved = self._watts(state.get("power_limit_w"))
        if saved is None:
            if self._applied is None:
                return
            raise PowerControlUnavailable("original AMD power cap is unavailable; recovery record was preserved")
        try:
            self._set_limit(saved)
        except PowerControlUnavailable as exc:
            logger.error("could not restore AMD GPU %s power cap: %s (run `rigtune power reset`)",
                         self.pci_bus_id, exc)
            raise
        self._original_limit_w = None
        self._applied = None
        _clear_restore()
        logger.info("GPU %d (%s) power cap restored to %d W", self.gpu_index, self.pci_bus_id, saved)

    def _set_limit(self, watts: int) -> None:
        self._call("set", "-g", self.pci_bus_id, "-o", str(int(watts)), "ppt0", "--json")
        _, _, actual, _ = self._limit_values()
        if actual != int(watts):
            raise PowerControlUnavailable(f"AMD SMI did not apply {watts} W cap (read back {actual} W)")

    def _remember_original(self, current: Optional[int]) -> None:
        if self._original_limit_w is not None:
            return
        existing = pending_restore()
        if restore_path().exists() and existing is None:
            raise PowerControlUnavailable("power restore record is unreadable; refusing to overwrite it")
        if existing:
            if (existing.get("vendor") == "amd" and existing.get("pci_bus_id") == self.pci_bus_id
                    and self._watts(existing.get("power_limit_w")) is not None):
                self._original_limit_w = self._watts(existing.get("power_limit_w"))
                return
            raise PowerControlUnavailable("another GPU has an unresolved power restore record; run `rigtune power reset` "
                                           "before changing a different GPU")
        if current is None:
            raise PowerControlUnavailable("amd-smi did not report the current PPT0 power limit")
        self._original_limit_w = current
        _write_restore({"vendor": "amd", "gpu_index": self.gpu_index, "pci_bus_id": self.pci_bus_id,
                        "hip_uuid": self.hip_uuid, "power_limit_w": current})

    def _install_hooks(self) -> None:
        if self._hooks_installed:
            return
        self._hooks_installed = True
        atexit.register(self.restore)
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            previous = signal.getsignal(signal.SIGTERM)

            def _on_term(signum: int, frame: Any) -> None:
                self.restore()
                if callable(previous) and previous not in (signal.SIG_DFL, signal.SIG_IGN):
                    previous(signum, frame)
                raise SystemExit(128 + signum)

            signal.signal(signal.SIGTERM, _on_term)
        except (ValueError, OSError, AttributeError):
            pass


def _normalize_bdf(value: str) -> str:
    from rigtune.amd import normalize_pci_bus_id

    normalized = normalize_pci_bus_id(value)
    if len(normalized.split(":")) != 3 or "." not in normalized:
        raise PowerControlUnavailable(f"invalid AMD PCI bus address: {value!r}")
    return normalized


def _explain(call: str, exc: Exception) -> str:
    name = type(exc).__name__
    if "NoPermission" in name or "permission" in str(exc).lower():
        return f"{call}: insufficient permissions (power and clock control need root on the host)"
    if "NotSupported" in name or "not supported" in str(exc).lower():
        return f"{call}: not supported on this GPU/driver"
    return f"{call}: {exc}"


def controller_for(gpu_index: int, vendor: str = "nvidia", pci_bus_id: Optional[str] = None,
                   hip_uuid: Optional[str] = None, junction_temp_limit_c: Optional[float] = None) \
        -> PowerController:
    """Create a controller for a selected physical GPU; tests monkeypatch this factory."""
    if vendor == "amd":
        if not pci_bus_id:
            raise PowerControlUnavailable("AMD power control requires a stable PCI bus address")
        return AmdSmiPowerController(gpu_index, pci_bus_id, hip_uuid=hip_uuid,
                                     junction_temp_limit_c=junction_temp_limit_c)
    return NvmlPowerController(gpu_index)


def reset_from_file(gpu_index: Optional[int] = None, nvml: Any = None,
                    amd_smi_run: Any = subprocess.run) -> Optional[Dict[str, Any]]:
    """Undo a recorded change, preserving the recovery file unless hardware readback verifies it."""
    state = pending_restore()
    if restore_path().exists() and state is None:
        raise PowerControlUnavailable("power restore record is unreadable; it was preserved")
    if state and state.get("vendor") == "amd":
        if gpu_index is not None and int(gpu_index) != int(state.get("gpu_index", -1)):
            raise PowerControlUnavailable(f"recorded change belongs to GPU {state.get('gpu_index')}; refusing to "
                                           f"reset GPU {gpu_index}")
        bdf = state.get("pci_bus_id")
        watts = AmdSmiPowerController._watts(state.get("power_limit_w"))
        if not bdf or watts is None:
            raise PowerControlUnavailable("AMD power recovery record is incomplete; it was preserved")
        ctl = AmdSmiPowerController(int(state.get("gpu_index", 0)), str(bdf),
                                    hip_uuid=state.get("hip_uuid"), run=amd_smi_run)
        ctl._set_limit(watts)
        _clear_restore()
        return state
    idx = gpu_index if gpu_index is not None else (state or {}).get("gpu_index", 0)
    ctl = NvmlPowerController(int(idx), nvml=nvml)
    if state and state.get("power_limit_mw"):
        ctl._call("nvmlDeviceSetPowerManagementLimit", int(state["power_limit_mw"]))
    ctl._try("nvmlDeviceResetGpuLockedClocks")
    _clear_restore()
    return state


# --------------------------------------------------------------------------- search points


def _snap(target: float, supported: Sequence[int]) -> int:
    return min(supported, key=lambda c: (abs(c - target), c))


def candidate_points(caps: Capabilities, mode: str, fractions: Sequence[float] = FRACTIONS,
                    power_min_w: Optional[int] = None, power_max_w: Optional[int] = None) -> List[PowerSetting]:
    """Settings the power stage should try; the unchanged default is always first.

    Explicit watt bounds constrain the swept cap range. The original cap remains the baseline and is never raised
    by the generated candidates, even when the requested upper bound is higher.
    """
    if mode not in MODES:
        raise ValueError(f"power mode must be one of {', '.join(MODES)}")
    points: List[PowerSetting] = [PowerSetting()]
    if mode in ("cap", "both") and caps.can_cap and caps.power_limit_default_w:
        dflt = caps.power_limit_default_w
        lo = caps.power_limit_min_w or 1
        hi = caps.power_limit_max_w or dflt
        if power_min_w is None and power_max_w is None:
            for f in fractions:
                w = int(round(min(hi, max(lo, dflt * f))))
                if w < dflt:
                    points.append(PowerSetting(power_limit_w=w))
        else:
            if power_min_w is not None and power_min_w < lo:
                raise ValueError(f"--power-min-w {power_min_w} is below the supported minimum {lo} W")
            if power_max_w is not None and power_max_w > hi:
                raise ValueError(f"--power-max-w {power_max_w} is above the supported maximum {hi} W")
            lower = power_min_w if power_min_w is not None else lo
            upper = min(power_max_w if power_max_w is not None else dflt, dflt)
            if lower > upper:
                raise ValueError(f"power cap bounds {lower}-{upper} W contain no cap below the current {dflt} W")
            targets = [upper]
            targets.extend(int(round(lower + (upper - lower) * f)) for f in fractions)
            targets.append(lower)
            points.extend(PowerSetting(power_limit_w=w) for w in targets if w < dflt)
    if mode in ("clock", "both") and caps.can_lock and caps.sm_clocks_mhz:
        top = caps.sm_clock_max_mhz or caps.sm_clocks_mhz[0]
        for f in fractions:
            mhz = _snap(top * f, caps.sm_clocks_mhz)
            if mhz < top:
                points.append(PowerSetting(sm_clock_mhz=mhz))
    seen, out = set(), []
    for p in points:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out
