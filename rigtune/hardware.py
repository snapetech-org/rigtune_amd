"""Step 1: probe. Produce a HardwareDescriptor and a stable hardware hash."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import psutil

from rigtune.models import BackendAvailability, CPUInfo, GPUInfo, HardwareDescriptor

logger = logging.getLogger(__name__)

BACKEND_NAMES = (
    "vllm", "sglang", "llamacpp-cuda", "llamacpp-hip", "llamacpp-vulkan", "llamacpp-cpu", "vllm-cpu"
)
ROCM_GPU_ARCHES = frozenset({"gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1101", "gfx1150", "gfx1151",
                             "gfx1200", "gfx1201"})


# --------------------------------------------------------------------------- GPU


def _probe_gpus_pynvml() -> List[GPUInfo]:
    try:
        import pynvml  # type: ignore
    except ImportError:
        return []
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # NVML present but no driver
        logger.debug("nvmlInit failed: %s", exc)
        return []
    gpus: List[GPUInfo] = []
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
        try:
            cuda_raw = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            cuda_version: Optional[str] = f"{cuda_raw // 1000}.{(cuda_raw % 1000) // 10}"
        except Exception:
            cuda_version = None
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
                cc: Optional[Tuple[int, int]] = (int(major), int(minor))
            except Exception:
                cc = None
            try:
                uuid = pynvml.nvmlDeviceGetUUID(h)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode()
            except Exception:
                uuid = None
            try:
                pci = pynvml.nvmlDeviceGetPciInfo(h)
                pci_bus_id = getattr(pci, "busId", None)
                if isinstance(pci_bus_id, bytes):
                    pci_bus_id = pci_bus_id.decode()
            except Exception:
                pci_bus_id = None
            gpus.append(
                GPUInfo(
                    vendor="nvidia",
                    name=str(name),
                    index=i,
                    compute_capability=cc,
                    vram_total_bytes=int(mem.total),
                    vram_free_bytes=int(mem.free),
                    driver_version=str(driver),
                    cuda_version=cuda_version,
                    uuid=uuid,
                    pci_bus_id=str(pci_bus_id) if pci_bus_id else None,
                )
            )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return gpus


def nvlink_between(indices: List[int]) -> Optional[bool]:
    """Whether every GPU in `indices` has an active NVLink; None when NVML cannot tell.

    Tensor parallelism over PCIe alone can hang in NCCL's peer-to-peer path inside containers
    (measured on a pair of A40s), so a tensor-parallel launch needs to know its interconnect.
    """
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        for i in indices:
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            active = False
            for link in range(18):  # NVML_NVLINK_MAX_LINKS
                try:
                    if pynvml.nvmlDeviceGetNvLinkState(h, link) == pynvml.NVML_FEATURE_ENABLED:
                        active = True
                        break
                except Exception:
                    break  # past this GPU's last link, or no NVLink at all
            if not active:
                return False
        return True
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _probe_gpus_torch() -> List[GPUInfo]:
    if importlib.util.find_spec("torch") is None:
        return []
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return []
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            gpus.append(
                GPUInfo(
                    vendor="nvidia" if not getattr(torch.version, "hip", None) else "amd",
                    name=props.name,
                    index=i,
                    compute_capability=(props.major, props.minor),
                    vram_total_bytes=int(total),
                    vram_free_bytes=int(free),
                    cuda_version=getattr(torch.version, "cuda", None),
                    architecture=(str(getattr(props, "gcnArchName", "")).split(":", 1)[0] or None),
                )
            )
        return gpus
    except Exception as exc:
        logger.debug("torch GPU probe failed: %s", exc)
        return []


def _probe_gpus_nvidia_smi() -> List[GPUInfo]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [
                smi,
                "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx, name, total, free, driver, cc = parts[:6]
        try:
            ccm = tuple(int(x) for x in cc.split("."))
            cct: Optional[Tuple[int, int]] = (ccm[0], ccm[1])
        except Exception:
            cct = None
        gpus.append(
            GPUInfo(
                vendor="nvidia",
                name=name,
                index=int(idx),
                compute_capability=cct,
                vram_total_bytes=int(float(total)) * 1024 * 1024,
                vram_free_bytes=int(float(free)) * 1024 * 1024,
                driver_version=driver,
            )
        )
    return gpus


def _rocm_smi_binary() -> Optional[str]:
    for candidate in (shutil.which("rocm-smi"), "/opt/rocm/bin/rocm-smi"):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _amd_smi_binary() -> Optional[str]:
    for candidate in (shutil.which("amd-smi"), "/opt/rocm/bin/amd-smi"):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _amd_smi_json(binary: str, *args: str) -> Optional[object]:
    try:
        out = subprocess.run([binary, *args, "--json"], capture_output=True, text=True, timeout=20, check=True)
        raw = out.stdout.strip()
        return json.loads(raw) if raw else None
    except Exception as exc:
        logger.debug("amd-smi %s failed: %s", " ".join(args), exc)
        return None


def _amd_smi_gpu_data(raw: object) -> Dict[int, dict]:
    if not isinstance(raw, dict) or not isinstance(raw.get("gpu_data"), list):
        return {}
    out = {}
    for item in raw["gpu_data"]:
        if isinstance(item, dict):
            try:
                out[int(item["gpu"])] = item
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _amd_smi_int(value: object, *, multiplier: int = 1) -> Optional[int]:
    if isinstance(value, dict):
        value = value.get("value")
    try:
        if value is None or str(value).strip().casefold() in ("", "n/a", "na", "none"):
            return None
        return int(float(str(value).replace(",", "")) * multiplier)
    except (TypeError, ValueError):
        return None


def _parse_amd_smi_inventory(list_raw: object, static_raw: object, metric_raw: object = None) -> List[GPUInfo]:
    """Join ``amd-smi list -e --json`` with ``amd-smi static -a -v --json``.

    The AMD-SMI ID is not necessarily the HIP runtime ID. Keep the former only as a final
    fallback and use ``hip_id`` for launch ordering; the PCI BDF and UUID are stable identities.
    """
    if not isinstance(list_raw, list):
        return []
    static = _amd_smi_gpu_data(static_raw)
    metrics = _amd_smi_gpu_data(metric_raw)
    gpus: List[GPUInfo] = []
    seen: set[str] = set()
    for item in list_raw:
        if not isinstance(item, dict):
            continue
        try:
            smi_index = int(item["gpu"])
        except (KeyError, TypeError, ValueError):
            continue
        details = static.get(smi_index, {})
        asic = details.get("asic") if isinstance(details.get("asic"), dict) else {}
        vram = details.get("vram") if isinstance(details.get("vram"), dict) else {}
        name = str(asic.get("market_name") or "").strip()
        if not name:
            continue
        memory = metrics.get(smi_index, {}).get("mem_usage", {})
        total_mb = _amd_smi_int(memory.get("total_visible_vram") or memory.get("total_vram"))
        free_mb = _amd_smi_int(memory.get("free_visible_vram") or memory.get("free_vram"))
        raw_size = vram.get("size") if isinstance(vram, dict) else None
        size_mb = total_mb if total_mb is not None else _amd_smi_int(raw_size)
        hip_id = _amd_smi_int(item.get("hip_id"))
        node_id = _amd_smi_int(item.get("node_id"))
        index = hip_id if hip_id is not None else max(0, (node_id or smi_index + 1) - 1)
        total = size_mb * 1024 * 1024 if size_mb is not None else 0
        free = free_mb * 1024 * 1024 if free_mb is not None else None
        hip_uuid = str(item.get("hip_uuid") or "").strip()
        uuid = str(item.get("uuid") or "").strip() or None
        physical_id = str(item.get("bdf") or uuid or hip_uuid or smi_index).casefold()
        if physical_id in seen:
            continue
        seen.add(physical_id)
        gpus.append(GPUInfo(
            vendor="amd",
            name=name,
            index=index,
            vram_total_bytes=total,
            vram_free_bytes=free,
            uuid=uuid,
            hip_uuid=hip_uuid or None,
            pci_bus_id=str(item.get("bdf") or "").strip() or None,
            architecture=str(asic.get("target_graphics_version") or "").strip() or None,
        ))
    return sorted(gpus, key=lambda gpu: (gpu.index, gpu.name.casefold()))


def _probe_gpus_amd_smi() -> List[GPUInfo]:
    smi = _amd_smi_binary()
    if not smi:
        return []
    listing = _amd_smi_json(smi, "list", "--enumeration")
    static = _amd_smi_json(smi, "static", "--asic", "--vram")
    metrics = _amd_smi_json(smi, "metric", "--mem-usage")
    return _parse_amd_smi_inventory(listing, static, metrics)


def _merge_amd_inventory(primary: List[GPUInfo], fallback: List[GPUInfo]) -> List[GPUInfo]:
    """Keep AMD-SMI identities while filling memory readings from ROCm-SMI where available."""
    if not primary:
        return fallback
    by_index = {gpu.index: gpu for gpu in fallback}
    merged = []
    for gpu in primary:
        other = by_index.pop(gpu.index, None)
        if other is not None:
            gpu = gpu.model_copy(update={
                "vram_total_bytes": gpu.vram_total_bytes or other.vram_total_bytes,
                "vram_free_bytes": gpu.vram_free_bytes if gpu.vram_free_bytes is not None else other.vram_free_bytes,
                "uuid": gpu.uuid or other.uuid,
                "hip_uuid": gpu.hip_uuid or other.hip_uuid,
                "pci_bus_id": gpu.pci_bus_id or other.pci_bus_id,
            })
        merged.append(gpu)
    merged.extend(by_index.values())
    return sorted(merged, key=lambda gpu: (gpu.index, gpu.name.casefold()))


def _parse_rocm_smi_json(raw: str) -> List[GPUInfo]:
    """Parse `rocm-smi --showproductname --showmeminfo vram --showuniqueid --json`.

    Node ID follows the ROCm runtime's device ordering (card1 can be ROCm0), unlike the
    DRM card number. Prefer the hardware UUID for pinning because it survives either order.
    """
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    gpus: List[GPUInfo] = []
    for key, item in data.items():
        match = re.fullmatch(r"card(\d+)", str(key), re.I)
        if not match or not isinstance(item, dict):
            continue
        vendor = str(item.get("Card Vendor", ""))
        if "amd" not in vendor.lower() and "ati" not in vendor.lower():
            continue
        name = str(item.get("Card Series") or item.get("Card Model") or "AMD GPU").strip()
        try:
            total = int(item.get("VRAM Total Memory (B)", 0))
            used = int(item.get("VRAM Total Used Memory (B)", 0))
        except (TypeError, ValueError):
            total, used = 0, 0
        try:
            index = max(0, int(item.get("Node ID", int(match.group(1)) + 1)) - 1)
        except (TypeError, ValueError):
            index = int(match.group(1))
        raw_uuid = str(item.get("Unique ID", "")).strip().lower()
        raw_uuid = re.sub(r"^0x", "", raw_uuid)
        uuid = f"GPU-{raw_uuid}" if raw_uuid and raw_uuid.strip("0") else None
        gpus.append(GPUInfo(
            vendor="amd", name=name, index=index, vram_total_bytes=total,
            vram_free_bytes=max(0, total - used), uuid=uuid,
            hip_uuid=uuid,
            pci_bus_id=str(item.get("PCI Bus") or item.get("PCI Bus ID") or "").strip() or None,
        ))
    return sorted(gpus, key=lambda gpu: (gpu.index, gpu.name.casefold()))


def _probe_gpus_rocm_smi() -> List[GPUInfo]:
    smi = _rocm_smi_binary()
    if not smi:
        return []
    commands = (
        [smi, "--showproductname", "--showmeminfo", "vram", "--showuniqueid", "--json"],
        [smi, "--json"],
    )
    for cmd in commands:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except Exception as exc:
            logger.debug("rocm-smi probe failed: %s", exc)
            continue
        gpus = _parse_rocm_smi_json(out.stdout)
        if gpus:
            return gpus
    return []


def _visible_subset(gpus: List[GPUInfo], env: str, *, indices_are_relative: bool = False) -> List[GPUInfo]:
    wanted = [token.strip() for token in env.split(",") if token.strip()]
    if not wanted or wanted == ["-1"]:
        return []
    if indices_are_relative:
        selected = set()
        for token in wanted:
            if token.isdigit() and int(token) < len(gpus):
                selected.add(int(token))
        return [gpu for i, gpu in enumerate(gpus) if i in selected]
    out = []
    for gpu in gpus:
        for token in wanted:
            if token.isdigit() and str(gpu.index) == token:
                out.append(gpu)
                break
            if gpu.uuid and (gpu.uuid.casefold() == token.casefold()
                             or gpu.uuid.casefold().removeprefix("gpu-").startswith(token.casefold().removeprefix("gpu-"))):
                out.append(gpu)
                break
            if gpu.hip_uuid and (gpu.hip_uuid.casefold() == token.casefold()
                                 or gpu.hip_uuid.casefold().removeprefix("gpu-").startswith(
                                     token.casefold().removeprefix("gpu-"))):
                out.append(gpu)
                break
    return out


def visible_gpus(gpus: List[GPUInfo]) -> List[GPUInfo]:
    """Apply the vendor visibility masks, which inventory tools may otherwise ignore.

    ROCm and CUDA use different masks. When both ROCm masks are set, HIP_VISIBLE_DEVICES
    applies to the list already filtered by ROCR_VISIBLE_DEVICES.
    """
    nvidia = [gpu for gpu in gpus if gpu.vendor == "nvidia"]
    amd = [gpu for gpu in gpus if gpu.vendor == "amd"]
    other = [gpu for gpu in gpus if gpu.vendor not in ("nvidia", "amd")]
    cuda_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_mask is not None:
        nvidia = _visible_subset(nvidia, cuda_mask)
    rocr_mask = os.environ.get("ROCR_VISIBLE_DEVICES")
    hip_mask = os.environ.get("HIP_VISIBLE_DEVICES")
    if rocr_mask is not None:
        amd = _visible_subset(amd, rocr_mask)
        if hip_mask is not None:
            amd = _visible_subset(amd, hip_mask, indices_are_relative=True)
    elif hip_mask is not None:
        amd = _visible_subset(amd, hip_mask)
    return sorted(nvidia + amd + other, key=lambda gpu: (gpu.index, gpu.name.casefold()))


def probe_gpus() -> List[GPUInfo]:
    # Prefer AMD SMI's PCI BDF, UUID, and HIP ordering; ROCm SMI remains the compatibility fallback.
    # NVML and AMD SMI describe different vendors, so combine them on mixed hosts.
    amd = _probe_gpus_amd_smi()
    if not amd or any(gpu.vram_free_bytes is None for gpu in amd):
        amd = _merge_amd_inventory(amd, _probe_gpus_rocm_smi())
    gpus = _probe_gpus_pynvml() + amd
    if not gpus:
        gpus = _probe_gpus_torch()
    if not gpus:
        gpus = _probe_gpus_nvidia_smi()
    return visible_gpus(gpus)


# --------------------------------------------------------------------------- CPU


def _cpu_flags() -> Tuple[str, set]:
    """Return (model name, flag set) from /proc/cpuinfo, py-cpuinfo, or nothing."""
    name = platform.processor() or "unknown"
    flags: set = set()
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
            if m:
                name = m.group(1).strip()
            m = re.search(r"^flags\s*:\s*(.+)$", text, re.M)
            if m:
                flags = set(m.group(1).split())
            return name, flags
        except OSError:
            pass
    try:  # optional dependency, helps on macOS/Windows dev boxes
        import cpuinfo  # type: ignore

        info = cpuinfo.get_cpu_info()
        name = info.get("brand_raw", name)
        flags = set(info.get("flags", []))
    except Exception:
        pass
    return name, flags


def _cpu_quota(root: str = "/sys/fs/cgroup") -> Optional[float]:
    """CPUs this process may use under a cgroup quota (containers), or None when unlimited or unknown."""
    try:  # cgroup v2: "<quota|max> <period>"
        with open(os.path.join(root, "cpu.max"), encoding="utf-8") as fh:
            quota, period = fh.read().split()[:2]
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        with open(os.path.join(root, "cpu", "cpu.cfs_quota_us"), encoding="utf-8") as fh:
            quota_us = int(fh.read())
        with open(os.path.join(root, "cpu", "cpu.cfs_period_us"), encoding="utf-8") as fh:
            period_us = int(fh.read())
        return None if quota_us <= 0 or period_us <= 0 else quota_us / period_us
    except (OSError, ValueError):
        return None


_CGROUP_UNLIMITED = 1 << 60  # cgroup v1 writes "no limit" as a page-rounded 2**63 - 1


def _memory_limit(root: str = "/sys/fs/cgroup") -> Optional[int]:
    """Bytes this process's cgroup may still use under its memory limit (containers), or None when unlimited
    or unknown.

    Usage leaves out the file cache, active and inactive, which the kernel reclaims before it enforces the
    limit: a container that has just downloaded a model's weights holds them as page cache, and counting that
    would report it as full and refuse plans that fit. psutil's available memory, which this is compared with,
    counts the host's file cache as available too."""
    v1 = os.path.join(root, "memory")
    for limit_file, usage_file, stat_file, file_keys in (
        (os.path.join(root, "memory.max"), os.path.join(root, "memory.current"),
         os.path.join(root, "memory.stat"), ("active_file", "inactive_file")),  # cgroup v2
        (os.path.join(v1, "memory.limit_in_bytes"), os.path.join(v1, "memory.usage_in_bytes"),
         os.path.join(v1, "memory.stat"), ("total_active_file", "total_inactive_file")),  # cgroup v1
    ):
        try:
            with open(limit_file, encoding="utf-8") as fh:
                raw = fh.read().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            limit = int(raw)
        except ValueError:
            return None
        if limit <= 0 or limit >= _CGROUP_UNLIMITED:
            return None
        usage = 0
        try:
            with open(usage_file, encoding="utf-8") as fh:
                usage = int(fh.read().strip())
        except (OSError, ValueError):
            pass
        cache = 0
        try:
            with open(stat_file, encoding="utf-8") as fh:
                for line in fh:
                    key, _, value = line.partition(" ")
                    if key in file_keys:
                        cache += int(value)
        except (OSError, ValueError):
            pass
        return max(0, limit - max(0, usage - cache))
    return None


def _usable_cpus() -> Optional[int]:
    """Whole CPUs this process may run on: its affinity mask, capped by a cgroup quota."""
    limits: List[int] = []
    if hasattr(os, "sched_getaffinity"):
        try:
            limits.append(len(os.sched_getaffinity(0)))
        except OSError:
            pass
    quota = _cpu_quota()
    if quota is not None:
        limits.append(max(1, int(quota)))
    return min(limits) if limits else None


def probe_cpu() -> CPUInfo:
    name, flags = _cpu_flags()
    vm = psutil.virtual_memory()
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or 1
    # A container sees the host's cores but may use only a quota of them: a RunPod pod showed 96 CPUs
    # under a 7.65-CPU quota, and llama.cpp given 48 threads ran throttled at 27 tok/s on a 0.5B model.
    usable = _usable_cpus()
    if usable is not None:
        physical, logical = min(physical, usable), min(logical, usable)
    # Likewise memory: psutil reads /proc/meminfo, which a container shares with its host, so a pod reports
    # the host's free RAM whatever its own limit, and the host-RAM check for llama.cpp's offloaded layers
    # could pass a config the limit then kills. Free RAM is capped by what the cgroup still allows. Total RAM
    # stays the host's: it is part of the hardware hash, and changing it would orphan every profile already
    # cached in a container.
    free = int(vm.available)
    room = _memory_limit()
    if room is not None:
        free = min(free, room)
    return CPUInfo(
        model_name=name,
        physical_cores=physical,
        logical_cores=logical,
        avx2="avx2" in flags,
        avx512=any(f.startswith("avx512") for f in flags),
        ram_total_bytes=int(vm.total),
        ram_free_bytes=free,
        arch=platform.machine() or "unknown",
    )


# --------------------------------------------------------------------------- backends


def _pkg_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def llama_server_binary(backend: Optional[str] = None) -> Optional[str]:
    """Path to llama-server, preferring a backend-specific override when supplied.

    `LLAMA_SERVER` remains the common override. `LLAMA_SERVER_HIP`,
    `LLAMA_SERVER_VULKAN`, and `LLAMA_SERVER_CUDA` allow separate builds to be
    calibrated by the same installation.
    """
    specific = {"hip": "LLAMA_SERVER_HIP", "vulkan": "LLAMA_SERVER_VULKAN", "cuda": "LLAMA_SERVER_CUDA"}
    env_names = ([specific[backend]] if backend in specific else [])
    env_names += ["LLAMA_SERVER"]
    if backend is None:
        env_names += [specific[k] for k in ("hip", "vulkan", "cuda")]
    for name in dict.fromkeys(env_names):
        path = os.environ.get(name)
        if path and os.path.isfile(path):
            return path
    for cand in ("llama-server", "llama-server.exe"):
        p = shutil.which(cand)
        if p:
            return p
    return None


def _llama_server_info(binary: str) -> Tuple[Optional[str], set[str], List[Tuple[str, int, str]]]:
    """Return the version, accelerator backends, and listed devices for a llama-server binary."""
    version: Optional[str] = None
    text = ""
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=15)
        version_text = (out.stdout or "") + (out.stderr or "")
        m = re.search(r"version:\s*(\S+)", version_text)
        if m:
            version = m.group(1)
        text += "\n" + version_text
    except Exception as exc:
        logger.debug("llama-server --version failed: %s", exc)
    try:
        out = subprocess.run([binary, "--list-devices"], capture_output=True, text=True, timeout=15)
        text += "\n" + (out.stdout or "") + "\n" + (out.stderr or "")
    except Exception as exc:
        logger.debug("llama-server --list-devices failed: %s", exc)

    devices: List[Tuple[str, int, str]] = []
    capabilities = set()
    prefix_map = {"cuda": "cuda", "rocm": "hip", "hip": "hip", "vulkan": "vulkan"}
    for line in text.splitlines():
        match = re.match(r"\s*(CUDA|ROCm|HIP|Vulkan)(\d+)\s*:\s*(.*)$", line, re.I)
        if not match:
            continue
        prefix, index, name = match.groups()
        runtime = prefix_map[prefix.casefold()]
        capabilities.add(runtime)
        name = re.sub(r"\s+\(\d+\s+MiB,.*\)\s*$", "", name).strip()
        devices.append((runtime, int(index), name))
    # Some builds print the backend in --version even when no device is visible.
    for prefix, runtime in prefix_map.items():
        if re.search(rf"\b{prefix}(?:\s+backend|\s+build)?\b", text, re.I):
            capabilities.add(runtime)
    return version, capabilities, devices


def llama_server_device_index(backend: str, gpu_name: str, binary: Optional[str] = None,
                             gpu_identity: Optional[str] = None) -> Optional[int]:
    """Find one unambiguous device in llama.cpp's list, preferring a stable identity over its name."""
    path = binary or llama_server_binary(backend)
    if not path:
        return None
    _, _, devices = _llama_server_info(path)
    runtime_devices = [(index, name) for runtime, index, name in devices if runtime == backend and name]
    if gpu_identity:
        identity = re.sub(r"[^a-z0-9]", "", gpu_identity.casefold())
        matches = [index for index, name in runtime_devices
                   if identity and identity in re.sub(r"[^a-z0-9]", "", name.casefold())]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return None
    needle = re.sub(r"[^a-z0-9]", "", gpu_name.casefold())
    candidates = [(index, re.sub(r"[^a-z0-9]", "", name.casefold())) for index, name in runtime_devices]
    exact = [index for index, listed in candidates if listed == needle and needle]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return None
    partial = [index for index, listed in candidates if needle and (needle in listed or listed in needle)]
    return partial[0] if len(partial) == 1 else None


def sglang_python() -> Optional[str]:
    """The Python that runs SGLang: $SGLANG_PYTHON when set, else this interpreter if sglang imports here.

    SGLang and vLLM pin shared dependencies (torch, FlashInfer, transformers) differently from release
    to release, so they are safest in separate environments; pointing $SGLANG_PYTHON at SGLang's lets
    one calibration choose between them."""
    env = os.environ.get("SGLANG_PYTHON")
    if env:
        return env if os.path.exists(env) else None
    return sys.executable if importlib.util.find_spec("sglang") is not None else None


# What each engine's environment has of the libraries its releases pin (name shown -> distribution name). vLLM
# 0.33.0 pins torch 2.14.0 and transformers >=5.16.1,<5.21.0, SGLang 0.5.23 torch 2.14.1 and transformers 5.19.0
# (vLLM requirements/cuda.txt and common.txt, SGLang python/pyproject.toml), so the two need separate
# environments, and each one's libraries are read there. arctic-inference holds vLLM's suffix decoding
# (speculative.suffix_refusal), so a profile records which one it was calibrated with; a library that is not
# installed is left out, as before. These are recorded and shown, not part of a profile's cache key (cache.invalid_reason
# compares the hardware and the engine's version), so a changed install takes `rigtune recalibrate`.
ENGINE_LIBS = {"torch": "torch", "transformers": "transformers", "flashinfer": "flashinfer-python",
               "tokenizers": "tokenizers", "arctic-inference": "arctic_inference"}
_VERSIONS_CODE = ("import importlib.metadata as m, json, sys\n"
                  "def v(n):\n"
                  "    try:\n"
                  "        return m.version(n)\n"
                  "    except Exception:\n"
                  "        return None\n"
                  "print(json.dumps({n: v(n) for n in sys.argv[1:]}))\n")
_SGLANG_LIBS: Dict[str, Dict[str, str]] = {}  # SGLang's python -> its libraries, read with its version


def _libs_here() -> Dict[str, str]:
    """ENGINE_LIBS as installed beside this interpreter."""
    found = {name: _pkg_version(dist) for name, dist in ENGINE_LIBS.items()}
    return {name: v for name, v in found.items() if v}


def _sglang_version(python: str) -> Optional[str]:
    """SGLang's version in `python`'s environment. Another interpreter is asked once for SGLang's version and its
    libraries together (kept for _sglang_libs)."""
    if python == sys.executable:
        return _pkg_version("sglang")
    try:
        out = subprocess.run([python, "-c", _VERSIONS_CODE, "sglang", *ENGINE_LIBS.values()],
                             capture_output=True, text=True, timeout=60)
        found = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else {}
    except Exception as exc:
        logger.debug("%s could not report an sglang version: %s", python, exc)
        return None
    _SGLANG_LIBS[python] = {name: found[dist] for name, dist in ENGINE_LIBS.items() if found.get(dist)}
    return found.get("sglang") or None


def _sglang_libs(python: str) -> Dict[str, str]:
    return _libs_here() if python == sys.executable else dict(_SGLANG_LIBS.get(python, {}))


def _torch_is_cuda_build() -> Optional[bool]:
    if importlib.util.find_spec("torch") is None:
        return None
    try:
        import torch  # type: ignore

        return bool(getattr(torch.version, "cuda", None))
    except Exception:
        return None


_TORCH_RUNTIME_CODE = (
    "import json\n"
    "try:\n"
    " import torch\n"
    " runtime = 'rocm' if getattr(torch.version, 'hip', None) else ('cuda' if getattr(torch.version, 'cuda', None) else 'cpu')\n"
    "except Exception:\n"
    " runtime = None\n"
    "print(json.dumps({'runtime': runtime}))\n"
)


def _torch_runtime(python: Optional[str] = None) -> Optional[str]:
    """Return the accelerator runtime compiled into the engine's PyTorch, never from device visibility."""
    if python and python != sys.executable:
        try:
            out = subprocess.run([python, "-c", _TORCH_RUNTIME_CODE], capture_output=True, text=True, timeout=30)
            if out.returncode == 0:
                row = json.loads(out.stdout.strip().splitlines()[-1])
                return row.get("runtime") if row.get("runtime") in ("cuda", "rocm", "cpu") else None
        except Exception as exc:
            logger.debug("could not identify torch runtime in %s: %s", python, exc)
        return None
    cuda = _torch_is_cuda_build()
    if cuda is True:
        return "cuda"
    if importlib.util.find_spec("torch") is None:
        return None
    try:
        import torch  # type: ignore

        if getattr(torch.version, "hip", None):
            return "rocm"
    except Exception:
        return None
    return "cpu" if cuda is False else None


def probe_backends(gpus: List[GPUInfo]) -> Dict[str, BackendAvailability]:
    out: Dict[str, BackendAvailability] = {}
    has_nvidia = any(g.vendor == "nvidia" for g in gpus)
    has_amd = any(g.vendor == "amd" for g in gpus)
    amd_gpu = next((gpu for gpu in gpus if gpu.vendor == "amd"), None)
    amd_arch = amd_gpu.architecture if amd_gpu else None
    amd_arch_supported = amd_arch in ROCM_GPU_ARCHES

    vllm_ver = _pkg_version("vllm")
    vllm_importable = importlib.util.find_spec("vllm") is not None
    torch_cuda = _torch_is_cuda_build()
    vllm_runtime = _torch_runtime()
    if not vllm_importable:
        vllm_reason = "vllm not importable"
    elif has_amd and vllm_runtime == "cpu":
        vllm_reason = "vLLM's PyTorch is a CPU build; an AMD GPU requires a ROCm build"
    elif has_amd and not has_nvidia and vllm_runtime != "rocm":
        vllm_reason = "vLLM's PyTorch is not a ROCm build for the visible AMD GPU"
    elif has_amd and vllm_runtime == "rocm" and not amd_arch_supported:
        vllm_reason = f"unsupported or unknown AMD GPU architecture for ROCm vLLM: {amd_arch or 'unknown'}"
    elif not has_nvidia and not (has_amd and vllm_runtime == "rocm"):
        vllm_reason = "no usable NVIDIA GPU (check CUDA_VISIBLE_DEVICES)"
    elif has_nvidia and vllm_runtime == "rocm" and not has_amd:
        vllm_reason = "vLLM's PyTorch is a ROCm build but no AMD GPU is visible"
    elif torch_cuda is False and vllm_runtime != "rocm":
        vllm_reason = "torch is a CPU build"
    else:
        vllm_reason = None
    vllm_libs = _libs_here() if vllm_ver else {}
    out["vllm"] = BackendAvailability(
        name="vllm",
        available=bool(vllm_importable and ((has_nvidia and vllm_runtime in ("cuda", None))
                                             or (has_amd and vllm_runtime == "rocm" and amd_arch_supported))),
        version=vllm_ver,
        reason=vllm_reason,
        runtime=vllm_runtime,
        gpu_architecture=amd_arch,
        validation_level=("detected_only" if has_amd and vllm_runtime == "rocm" else None),
        libs=vllm_libs,
    )
    # vLLM-CPU: the same package built against a CPU-only torch.
    out["vllm-cpu"] = BackendAvailability(
        name="vllm-cpu",
        available=bool(vllm_importable and vllm_runtime == "cpu"),
        version=vllm_ver,
        libs=dict(vllm_libs),
        reason=(
            "vllm not importable" if not vllm_importable
            else ("torch is an accelerator build; vLLM-CPU needs the CPU wheel" if vllm_runtime != "cpu" else None)
        ),
        runtime=vllm_runtime,
    )

    sgl_python = sglang_python()
    sgl_ver = _sglang_version(sgl_python) if sgl_python else None
    sgl_found = sgl_python is not None and (sgl_python == sys.executable or sgl_ver is not None)
    sgl_runtime = _torch_runtime(sgl_python) if has_amd and sgl_found else None
    sgl_rocm_ok = has_amd and sgl_runtime == "rocm" and amd_arch_supported
    sgl_nvidia_ok = has_nvidia and sgl_runtime in (None, "cuda")
    if not sgl_found:
        sgl_reason = "sglang not found: install it here, or set $SGLANG_PYTHON to the python of an environment that has it"
    elif sgl_runtime == "cpu":
        sgl_reason = "SGLang's PyTorch is a CPU build"
    elif has_amd and sgl_runtime == "rocm" and not amd_arch_supported:
        sgl_reason = f"unsupported or unknown AMD GPU architecture for ROCm SGLang: {amd_arch or 'unknown'}"
    elif has_amd and not has_nvidia and sgl_runtime != "rocm":
        sgl_reason = "SGLang's PyTorch is not a ROCm build for the visible AMD GPU"
    elif has_nvidia and sgl_runtime == "rocm" and not sgl_rocm_ok:
        sgl_reason = "SGLang's PyTorch is a ROCm build but no supported AMD GPU is visible"
    elif not sgl_nvidia_ok and not sgl_rocm_ok:
        sgl_reason = "no usable NVIDIA GPU (check CUDA_VISIBLE_DEVICES)"
    else:
        sgl_reason = None
    out["sglang"] = BackendAvailability(
        name="sglang",
        available=bool(sgl_found and (sgl_nvidia_ok or sgl_rocm_ok)),
        version=sgl_ver,
        runtime=sgl_runtime,
        gpu_architecture=amd_arch,
        validation_level=("detected_only" if has_amd and sgl_runtime == "rocm" else None),
        libs=_sglang_libs(sgl_python) if sgl_python and sgl_ver else {},
        reason=sgl_reason,
    )

    llama_variants = (
        ("llamacpp-cuda", "cuda", "nvidia", "CUDA"),
        ("llamacpp-hip", "hip", "amd", "HIP/ROCm"),
        ("llamacpp-vulkan", "vulkan", None, "Vulkan"),
    )
    for name, runtime, vendor, label in llama_variants:
        binary = llama_server_binary(runtime)
        if not binary:
            out[name] = BackendAvailability(
                name=name, available=False, reason=f"llama-server {label} build not found (set LLAMA_SERVER_{runtime.upper()})"
            )
            continue
        ver, capabilities, devices = _llama_server_info(binary)
        has_gpu = any(gpu.vendor == vendor for gpu in gpus) if vendor else bool(gpus)
        if runtime in ("hip", "vulkan") and devices:
            indices = [llama_server_device_index(runtime, gpu.name, binary,
                                                 gpu_identity=gpu.pci_bus_id or gpu.uuid)
                       for gpu in gpus]
            device_matches = any(index is not None for index in indices)
        else:
            device_matches = has_gpu
        if runtime not in capabilities:
            reason = f"llama-server built without {label}"
        elif not has_gpu:
            reason = ("no usable NVIDIA GPU (check CUDA_VISIBLE_DEVICES)" if vendor == "nvidia"
                      else "no usable AMD GPU (check ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES)" if vendor == "amd"
                      else "no usable GPU is visible")
        elif not device_matches:
            reason = f"llama-server does not list a visible GPU for {label}"
        else:
            reason = None
        out[name] = BackendAvailability(
            name=name, available=reason is None, version=ver, reason=reason,
            gpu_architecture=(amd_arch if has_amd and runtime in ("hip", "vulkan") else None),
            validation_level=("detected_only" if has_amd and runtime in ("hip", "vulkan") and reason is None else None),
        )

    binary = llama_server_binary()
    if binary:
        ver, _, _ = _llama_server_info(binary)
        out["llamacpp-cpu"] = BackendAvailability(name="llamacpp-cpu", available=True, version=ver)
    else:
        out["llamacpp-cpu"] = BackendAvailability(
            name="llamacpp-cpu", available=False, reason="llama-server not on PATH (set $LLAMA_SERVER)"
        )
    return out


# --------------------------------------------------------------------------- entry


def probe() -> HardwareDescriptor:
    gpus = probe_gpus()
    cpu = probe_cpu()
    return HardwareDescriptor(
        os=f"{platform.system()} {platform.release()}",
        python=sys.version.split()[0],
        gpus=gpus,
        cpu=cpu,
        backends=probe_backends(gpus),
    )


def hardware_hash(hw: HardwareDescriptor) -> str:
    """Stable identity of the machine for cache keys.

    Uses the *shape* of the hardware (GPU model, CC, total VRAM, CPU model,
    cores, SIMD flags, total RAM), not volatile fields like free memory.
    """
    parts = []
    for g in hw.gpus:
        # Keep the pre-AMD NVIDIA shape unchanged so adding optional device metadata does not invalidate
        # existing hardware-keyed profiles. AMD needs its ROCm target in the key because two cards with the
        # same marketing name can expose different supported engine and kernel sets.
        architecture = f":{g.architecture or ''}" if g.vendor == "amd" else ""
        parts.append(f"gpu:{g.vendor}:{g.name}:{g.cc[0]}.{g.cc[1]}{architecture}:"
                     f"{g.vram_total_bytes // (256 * 1024 * 1024)}")
    c = hw.cpu
    parts.append(f"cpu:{c.model_name}:{c.physical_cores}:{c.logical_cores}:{int(c.avx2)}{int(c.avx512)}")
    parts.append(f"ram:{c.ram_total_bytes // (1024 * 1024 * 1024)}")
    parts.append(f"arch:{c.arch}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def llmtrace_version() -> Optional[str]:
    return _pkg_version("llmtrace")
