"""Energy tuning: power cap and clock lock. No GPU needed; NVML is faked."""

from __future__ import annotations

import json
import subprocess
from typing import List, Optional

import pytest
from typer.testing import CliRunner

from rigtune import power as P
from rigtune.calibrate.objectives import Constraints, pick
from rigtune.calibrate.search import StagedSearch, SubprocessTrialRunner
from rigtune.calibrate.workload import Workload
from rigtune.models import ArchInfo, Config, ModelSpec, PreparedModel, TrialMetrics, TrialResult
from rigtune.power import Capabilities, PowerControlUnavailable, PowerSetting, candidate_points, with_power
from tests.test_objectives_and_search import FakeRunner, _feasible_a100

DEFAULT_W, MAX_MHZ = 350, 1695


# --------------------------------------------------------------------------- fakes


class FakeNVML:
    """Module-shaped stand-in for pynvml with the calls NvmlPowerController makes."""

    NVML_CLOCK_SM = 1

    class NVMLError(Exception):
        pass

    class NVMLError_NoPermission(NVMLError):
        pass

    def __init__(self, permit_cap: bool = True, permit_lock: bool = True):
        self.permit_cap, self.permit_lock = permit_cap, permit_lock
        self.limit_mw = DEFAULT_W * 1000
        self.locked: Optional[tuple] = None
        self.calls: List[tuple] = []

    def nvmlInit(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, i):
        return f"gpu{i}"

    def nvmlDeviceGetPowerManagementLimitConstraints(self, h):
        return [100_000, 400_000]

    def nvmlDeviceGetPowerManagementDefaultLimit(self, h):
        return DEFAULT_W * 1000

    def nvmlDeviceGetPowerManagementLimit(self, h):
        return self.limit_mw

    def nvmlDeviceSetPowerManagementLimit(self, h, mw):
        if not self.permit_cap:
            raise self.NVMLError_NoPermission("Insufficient Permissions")
        self.calls.append(("set_limit", mw))
        self.limit_mw = mw

    def nvmlDeviceGetSupportedMemoryClocks(self, h):
        return [9751, 405]

    def nvmlDeviceGetSupportedGraphicsClocks(self, h, mem):
        return list(range(MAX_MHZ, 209, -15))

    def nvmlDeviceGetMaxClockInfo(self, h, kind):
        return MAX_MHZ

    def nvmlDeviceSetGpuLockedClocks(self, h, lo, hi):
        if not self.permit_lock:
            raise self.NVMLError_NoPermission("Insufficient Permissions")
        self.calls.append(("lock", lo, hi))
        self.locked = (lo, hi)

    def nvmlDeviceResetGpuLockedClocks(self, h):
        if not self.permit_lock:
            raise self.NVMLError_NoPermission("Insufficient Permissions")
        self.calls.append(("reset_clocks",))
        self.locked = None


class FakeController:
    """In-memory PowerController for search / runner / supervisor tests."""

    def __init__(self, fail_on: Optional[PowerSetting] = None):
        self.applied: Optional[PowerSetting] = None
        self.history: List[str] = []
        self.fail_on = fail_on

    def capabilities(self) -> Capabilities:
        return _caps()

    def apply(self, setting: PowerSetting) -> None:
        if self.fail_on is not None and setting == self.fail_on:
            raise PowerControlUnavailable("insufficient permissions")
        self.applied = None if setting.is_default else setting
        self.history.append(setting.label())

    def restore(self) -> None:
        self.applied = None
        self.history.append("restore")


class FakeAmdSmi:
    def __init__(self, current: int = 231, *, apply: bool = True):
        self.current = current
        self.apply_changes = apply
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        args = cmd[1:]
        if args[0] == "static":
            row = {"gpu": 1, "limit": {
                "ppt0": {"min_power_limit": {"value": 120}, "max_power_limit": {"value": 290},
                         "socket_power_limit": {"value": self.current}},
                "slowdown_hotspot_temperature": {"value": 110},
            }}
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"gpu_data": [row]}), "")
        if args[0] == "set" and self.apply_changes:
            self.current = int(args[args.index("-o") + 1])
        return subprocess.CompletedProcess(cmd, 0, "{}", "")


def _caps(can_cap: bool = True, can_lock: bool = True) -> Capabilities:
    return Capabilities(gpu_index=0, power_limit_min_w=100, power_limit_max_w=400, power_limit_default_w=DEFAULT_W,
                        power_limit_current_w=DEFAULT_W, sm_clocks_mhz=list(range(MAX_MHZ, 209, -15)),
                        sm_clock_max_mhz=MAX_MHZ, can_cap=can_cap, can_lock=can_lock)


def _fraction(cfg: Config) -> float:
    """How far below full power a config runs, as a fraction of the default cap or max clock."""
    if cfg.power_limit_w is not None:
        return cfg.power_limit_w / DEFAULT_W
    if cfg.sm_clock_mhz is not None:
        return cfg.sm_clock_mhz / MAX_MHZ
    return 1.0


class PowerFakeRunner(FakeRunner):
    """Memory-bound decode: throughput is flat down to 70% of full power, power falls with f^2.

    So energy per token keeps falling as the cap drops, and throughput only starts to fall once
    compute becomes the bottleneck below the knee.
    """

    KNEE = 0.70

    def run(self, cfg: Config, stage: str) -> TrialResult:
        base = super().run(cfg.model_copy(update={"power_limit_w": None, "sm_clock_mhz": None}), stage)
        if not base.ok:
            return TrialResult(config=cfg, stage=stage, metrics=base.metrics, launched=base.launched, error=base.error)
        f = _fraction(cfg)
        tok = base.metrics.tok_s * min(1.0, f / self.KNEE)
        watts = 300.0 * f * f
        m = base.metrics.model_copy(update={"tok_s": tok, "power_w": watts, "joules_per_token": watts / tok,
                                            "sm_clock_mhz": MAX_MHZ * f})
        return TrialResult(config=cfg, stage=stage, metrics=m)


# --------------------------------------------------------------------------- points


def test_candidate_points_per_mode():
    caps = _caps()
    assert candidate_points(caps, "off") == [PowerSetting()]
    cap = candidate_points(caps, "cap")
    assert cap[0].is_default and [p.power_limit_w for p in cap[1:]] == [298, 245, 193]  # 350 W x .85/.70/.55, rounded
    clk = candidate_points(caps, "clock")
    assert [p.sm_clock_mhz for p in clk[1:]] == [1440, 1185, 930]  # snapped to supported 15 MHz steps
    assert all(c in caps.sm_clocks_mhz for c in (1440, 1185, 930))
    both = candidate_points(caps, "both")
    assert len(both) == 7 and both[0].is_default  # default + 3 caps + 3 clocks, not the cross product
    with pytest.raises(ValueError):
        candidate_points(caps, "turbo")


def test_candidate_points_respect_limits_and_permissions():
    caps = _caps()
    caps.power_limit_min_w = 250  # the card refuses to go below 250 W
    assert [p.power_limit_w for p in candidate_points(caps, "cap")[1:]] == [298, 250]  # clamped and de-duplicated
    assert candidate_points(_caps(can_cap=False), "cap") == [PowerSetting()]
    assert len(candidate_points(_caps(can_cap=False), "both")) == 4  # clocks still offered


def test_candidate_points_apply_explicit_watt_bounds_without_raising_the_baseline():
    caps = _caps()
    bounded = candidate_points(caps, "cap", power_min_w=200, power_max_w=300)
    assert bounded[0] == PowerSetting()
    assert [p.power_limit_w for p in bounded[1:]] == [300, 285, 270, 255, 200]
    assert candidate_points(caps, "cap", power_min_w=350, power_max_w=400) == [PowerSetting()]
    with pytest.raises(ValueError, match="supported minimum"):
        candidate_points(caps, "cap", power_min_w=50)


# --------------------------------------------------------------------------- NVML controller


def test_nvml_controller_caps_locks_and_restores(tmp_home):
    nv = FakeNVML()
    ctl = P.NvmlPowerController(0, nvml=nv)
    caps = ctl.capabilities()
    assert caps.can_cap and caps.can_lock and caps.power_limit_default_w == 350
    assert caps.power_limit_min_w == 100 and caps.power_limit_max_w == 400 and caps.sm_clock_max_mhz == MAX_MHZ
    assert nv.limit_mw == 350_000 and nv.locked is None  # permission probes left the device unchanged
    nv.calls.clear()

    ctl.apply(PowerSetting(power_limit_w=245))
    assert nv.limit_mw == 245_000 and ("set_limit", 245_000) in nv.calls
    restore_file = P.restore_path()
    assert json.loads(restore_file.read_text())["power_limit_mw"] == 350_000  # the original, written first

    ctl.apply(PowerSetting(sm_clock_mhz=1185))  # switch knobs: cap lifted, clocks locked
    assert nv.limit_mw == 350_000 and nv.locked == (210, 1185)

    ctl.restore()
    assert nv.limit_mw == 350_000 and nv.locked is None and ctl.applied is None
    assert not restore_file.exists()


def test_nvml_controller_reports_missing_permission(tmp_home):
    nv = FakeNVML(permit_cap=False, permit_lock=False)
    ctl = P.NvmlPowerController(0, nvml=nv)
    caps = ctl.capabilities()
    assert not caps.can_cap and not caps.can_lock
    assert "root" in caps.reasons["cap"] and "root" in caps.reasons["clock"]
    with pytest.raises(PowerControlUnavailable, match="root"):
        ctl.apply(PowerSetting(power_limit_w=245))


def test_amd_smi_cap_is_selected_by_bdf_verified_and_restored(tmp_home):
    smi = FakeAmdSmi()
    ctl = P.AmdSmiPowerController(1, "0000:15:00.0", hip_uuid="GPU-test", binary="amd-smi", run=smi,
                                  is_root=True)
    caps = ctl.capabilities()
    assert caps.can_cap and not caps.can_lock
    assert (caps.power_limit_min_w, caps.power_limit_max_w, caps.power_limit_current_w) == (120, 290, 231)
    assert caps.junction_slowdown_c == 110
    assert all(call[1] == "static" for call in smi.calls)  # capability detection never writes a setting

    ctl.apply(PowerSetting(power_limit_w=180))
    state = json.loads(P.restore_path().read_text())
    assert state == {"vendor": "amd", "gpu_index": 1, "pci_bus_id": "0000:15:00.0", "hip_uuid": "GPU-test",
                     "power_limit_w": 231}
    assert smi.current == 180
    assert ["amd-smi", "set", "-g", "0000:15:00.0", "-o", "180", "ppt0", "--json"] in smi.calls

    ctl.restore()
    assert smi.current == 231 and P.pending_restore() is None


def test_amd_smi_permission_and_invalid_range_are_read_only(tmp_home):
    smi = FakeAmdSmi()
    ctl = P.AmdSmiPowerController(1, "0000:15:00.0", binary="amd-smi", run=smi, is_root=False)
    caps = ctl.capabilities()
    assert not caps.can_cap and "root" in caps.reasons["cap"]
    assert all(call[1] == "static" for call in smi.calls)
    with pytest.raises(PowerControlUnavailable, match="root"):
        ctl.apply(PowerSetting(power_limit_w=180))

    root_ctl = P.AmdSmiPowerController(1, "0000:15:00.0", binary="amd-smi", run=smi, is_root=True)
    with pytest.raises(PowerControlUnavailable, match="range"):
        root_ctl.apply(PowerSetting(power_limit_w=100))


def test_amd_smi_rejects_ambiguous_bdf_selection_response(tmp_home):
    class AmbiguousAmdSmi(FakeAmdSmi):
        def __call__(self, cmd, **kwargs):
            result = super().__call__(cmd, **kwargs)
            if cmd[1] == "static":
                rows = json.loads(result.stdout)["gpu_data"]
                rows.append(dict(rows[0]))
                result.stdout = json.dumps({"gpu_data": rows})
            return result

    ctl = P.AmdSmiPowerController(1, "0000:15:00.0", binary="amd-smi", run=AmbiguousAmdSmi(), is_root=True)
    with pytest.raises(PowerControlUnavailable, match="ambiguous response"):
        ctl.capabilities()


def test_amd_smi_failed_readback_keeps_restore_record(tmp_home):
    smi = FakeAmdSmi(apply=False)
    ctl = P.AmdSmiPowerController(1, "0000:15:00.0", binary="amd-smi", run=smi, is_root=True)
    with pytest.raises(PowerControlUnavailable, match="did not apply"):
        ctl.apply(PowerSetting(power_limit_w=180))
    assert P.pending_restore()["power_limit_w"] == 231
    ctl.restore()  # the setter was a no-op; readback confirms the original cap is already restored
    assert P.pending_restore() is None


def test_amd_failed_restore_preserves_recovery_record(tmp_home):
    state = {"vendor": "amd", "gpu_index": 1, "pci_bus_id": "0000:15:00.0", "power_limit_w": 231}
    P._write_restore(state)
    smi = FakeAmdSmi(current=180, apply=False)
    ctl = P.AmdSmiPowerController(1, "0000:15:00.0", binary="amd-smi", run=smi, is_root=True)
    with pytest.raises(PowerControlUnavailable, match="did not apply"):
        ctl.restore()
    assert P.pending_restore() == state


@pytest.mark.parametrize("state", [
    {"vendor": "amd", "gpu_index": 0, "pci_bus_id": "0000:16:00.0", "power_limit_w": 231},
    {"vendor": "nvidia", "gpu_index": 0, "power_limit_mw": 231000},
])
def test_amd_restore_never_clears_another_gpu_recovery_record(tmp_home, state):
    P._write_restore(state)
    ctl = P.AmdSmiPowerController(1, "0000:15:00.0", binary="amd-smi", run=FakeAmdSmi(), is_root=True)
    with pytest.raises(PowerControlUnavailable, match="another GPU"):
        ctl.restore()
    assert P.pending_restore() == state


def test_reset_from_file_restores_amd_original_and_checks_gpu(tmp_home):
    state = {"vendor": "amd", "gpu_index": 1, "pci_bus_id": "0000:15:00.0", "power_limit_w": 231}
    P._write_restore(state)
    smi = FakeAmdSmi(current=180)
    with pytest.raises(PowerControlUnavailable, match="belongs to GPU 1"):
        P.reset_from_file(gpu_index=0, amd_smi_run=smi)
    assert P.pending_restore() == state
    assert P.reset_from_file(amd_smi_run=smi) == state
    assert smi.current == 231 and P.pending_restore() is None


def test_reset_from_file_undoes_a_crashed_run(tmp_home):
    nv = FakeNVML()
    P.NvmlPowerController(0, nvml=nv).apply(PowerSetting(power_limit_w=192))
    assert nv.limit_mw == 192_000 and P.pending_restore() is not None
    # The process dies here without restoring. A later `rigtune power reset`:
    state = P.reset_from_file(nvml=nv)
    assert state["power_limit_mw"] == 350_000 and nv.limit_mw == 350_000
    assert P.pending_restore() is None


def test_controller_recovers_the_true_original_after_a_crash(tmp_home):
    nv = FakeNVML()
    P.NvmlPowerController(0, nvml=nv).apply(PowerSetting(power_limit_w=192))  # crashed, left at 192 W
    ctl = P.NvmlPowerController(0, nvml=nv)
    ctl.apply(PowerSetting(power_limit_w=245))
    ctl.restore()
    assert nv.limit_mw == 350_000  # not 192: the restore file, not the current limit, is the original


# --------------------------------------------------------------------------- objective


def _power_result(f: float, tok: float, watts: float, key_ctx: int = 4096) -> TrialResult:
    cfg = Config(backend="vllm", quant="fp8", ctx=key_ctx, batch=64, gpu_memory_utilization=0.9,
                 power_limit_w=None if f == 1.0 else int(round(DEFAULT_W * f)))
    m = TrialMetrics(tok_s=tok, ttft_ms=40, power_w=watts, joules_per_token=watts / tok, requests=48,
                     output_tokens=6000)
    return TrialResult(config=cfg, stage="t", metrics=m)


def test_objective_takes_free_energy_savings_only():
    results = [_power_result(1.0, 1000, 300), _power_result(0.85, 999, 217),
               _power_result(0.70, 995, 147), _power_result(0.55, 786, 91)]
    w, _ = pick(results, "balanced", Constraints())
    assert w.config.power_limit_w == 245  # 0.5% slower is noise; 51% less energy per token
    w, _ = pick(results, "balanced", Constraints(power_max_loss=0.25))
    assert w.config.power_limit_w == 193  # a looser budget buys the deeper cap
    w, _ = pick(results, "throughput", Constraints(power_max_loss=0.0, noise_tolerance=0.0))
    assert w.config.power_limit_w is None  # no tolerance: any throughput loss disqualifies a variant
    w, _ = pick(results, "efficiency", Constraints())
    assert w.config.power_limit_w == 193  # min J/token above the 50% throughput floor


def test_an_exact_throughput_tie_goes_to_the_lower_energy_setting():
    tie = [_power_result(1.0, 1000, 300), _power_result(0.85, 1000, 217)]
    w, _ = pick(tie, "throughput", Constraints(power_max_loss=0.0, noise_tolerance=0.0))
    assert w.config.power_limit_w == 298


def test_power_variants_never_beat_a_genuinely_faster_config():
    faster_other = _power_result(1.0, 1200, 300, key_ctx=8192)
    capped = _power_result(0.70, 1000, 147)
    w, _ = pick([faster_other, capped], "balanced", Constraints(power_max_loss=0.25))
    assert w is faster_other  # the energy tie-break only applies to variants of the leader


# --------------------------------------------------------------------------- search stage 4


def test_staged_search_power_stage_picks_the_knee(hw_a100, prepared_vllm):
    feasible = _feasible_a100(hw_a100, prepared_vllm)
    search = StagedSearch(objective="balanced", runner=PowerFakeRunner(), constraints=Constraints(ttft_ceiling_ms=300),
                          power_points=candidate_points(_caps(), "both"))
    winner, notes = search.run(feasible)
    power_rows = [r for r in search.results if r.stage == "power"]
    assert len(power_rows) == 6  # 3 caps + 3 clocks; the runner has no sweep, so no same-launch baseline
    assert _fraction(winner.config) == pytest.approx(0.70, abs=0.01)  # deepest setting at no throughput cost
    base = next(r for r in search.results if r.config.key() == winner.config.base_key())
    assert winner.metrics.tok_s >= 0.98 * base.metrics.tok_s
    assert winner.metrics.joules_per_token < 0.55 * base.metrics.joules_per_token
    assert any(n.startswith("power setting chosen") and "J/token" in n for n in notes)


def test_staged_search_efficiency_goes_below_the_knee(hw_a100, prepared_vllm):
    feasible = _feasible_a100(hw_a100, prepared_vllm)
    search = StagedSearch(objective="efficiency", runner=PowerFakeRunner(), power_points=candidate_points(_caps(), "cap"))
    winner, _ = search.run(feasible)
    assert winner.config.power_limit_w == 193  # trades 21% throughput for the lowest J/token


def test_staged_search_without_power_points_is_unchanged(hw_a100, prepared_vllm):
    feasible = _feasible_a100(hw_a100, prepared_vllm)
    search = StagedSearch(objective="balanced", runner=PowerFakeRunner(), constraints=Constraints(ttft_ceiling_ms=300))
    winner, _ = search.run(feasible)
    assert not any(r.stage == "power" for r in search.results)
    assert winner.config.power_limit_w is None and winner.config.sm_clock_mhz is None


# --------------------------------------------------------------------------- real subprocess runner + supervisor


def _fake_model() -> PreparedModel:
    arch = ArchInfo(num_layers=2, hidden_size=64, num_attention_heads=2, num_kv_heads=2, head_dim=32, num_params=1000)
    return PreparedModel(spec=ModelSpec(hf_id="x/y"), backend="fake", arch=arch, weights_bytes={"none": 1000})


class CountingBackend:
    def __init__(self):
        from tests.test_supervisor import FakeBackend

        self.inner = FakeBackend()
        self.launches = 0
        self.health_path = self.inner.health_path

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def launch(self, *a, **kw):
        self.launches += 1
        return self.inner.launch(*a, **kw)


def test_subprocess_sweep_uses_one_launch_and_always_restores(tmp_path):
    from tests.conftest import make_hw

    be = CountingBackend()
    ctl = FakeController(fail_on=PowerSetting(sm_clock_mhz=930))
    wl = Workload(n_prompts=2, prefill_tokens=8, decode_tokens=4, concurrencies=(1,))
    runner = SubprocessTrialRunner(backends={"fake": be}, models={"fake": _fake_model()}, hw=make_hw("cpu"),
                                   workload=wl, log_dir=tmp_path, startup_timeout=15, power=ctl, power_settle_s=0)
    base = Config(backend="fake", quant="none", ctx=128, batch=1)
    points = [PowerSetting(), PowerSetting(power_limit_w=245), PowerSetting(sm_clock_mhz=930)]
    seen = []
    results = runner.sweep(base, points, "power", progress=lambda st, c, r: seen.append((c.key(), r is not None)))
    assert be.launches == 1
    assert [r.config.power_limit_w for r in results] == [None, 245, None]
    assert results[0].ok and results[1].ok
    assert not results[2].ok and "power control unavailable" in results[2].error  # one refusal, the rest measured
    assert ctl.history[-1] == "restore" and ctl.applied is None
    assert len(seen) == 6


def test_junction_guard_marks_failure_and_skips_remaining_power_candidates(tmp_path, monkeypatch):
    from tests.conftest import make_hw

    class GuardController(FakeController):
        junction_temp_limit_c = 96.0

    be = CountingBackend()
    ctl = GuardController()
    wl = Workload(n_prompts=2, prefill_tokens=8, decode_tokens=4, concurrencies=(1,))
    runner = SubprocessTrialRunner(backends={"fake": be}, models={"fake": _fake_model()}, hw=make_hw("cpu"),
                                   workload=wl, log_dir=tmp_path, startup_timeout=15, power=ctl, power_settle_s=0)
    calls = 0

    def guarded_trial(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return TrialMetrics(tok_s=5, output_tokens=4, requests=1)
        return TrialMetrics(junction_temp_min_c=96.0, junction_temp_peak_c=97.0,
                            thermal_limit_hit=True, thermal_limit_c=96.0, telemetry_source="amd_sysfs")

    monkeypatch.setattr("rigtune.calibrate.search.run_trial", guarded_trial)
    points = [PowerSetting(), PowerSetting(power_limit_w=245), PowerSetting(power_limit_w=193)]
    results = runner.sweep(Config(backend="fake", quant="none", ctx=128, batch=1), points, "power")
    assert calls == 2
    assert results[1].failure_kind == "thermal_limit"
    assert "junction temperature 97 C reached" in results[1].error
    assert results[2].failure_kind == "thermal_limit" and not results[2].launched
    assert "not measured because the junction temperature guard stopped" in results[2].error
    assert ctl.applied is None and ctl.history[-1] == "restore"


def test_runner_run_applies_and_restores_a_power_config(tmp_path):
    from tests.conftest import make_hw
    from tests.test_supervisor import FakeBackend

    ctl = FakeController()
    wl = Workload(n_prompts=2, prefill_tokens=8, decode_tokens=4, concurrencies=(1,))
    runner = SubprocessTrialRunner(backends={"fake": FakeBackend()}, models={"fake": _fake_model()},
                                   hw=make_hw("cpu"), workload=wl, startup_timeout=15, power=ctl, power_settle_s=0)
    res = runner.run(Config(backend="fake", quant="none", ctx=128, batch=1, power_limit_w=245), "power")
    assert res.ok and ctl.history == ["cap 245 W", "restore"]
    no_ctl = SubprocessTrialRunner(backends={"fake": FakeBackend()}, models={"fake": _fake_model()},
                                   hw=make_hw("cpu"), workload=wl, startup_timeout=15)
    res = no_ctl.run(Config(backend="fake", quant="none", ctx=128, batch=1, power_limit_w=245), "power")
    assert not res.ok and "no power controller" in res.error


def test_supervisor_serves_at_the_calibrated_setting(tmp_path):
    from rigtune.serve.supervisor import Supervisor
    from tests.test_supervisor import FakeBackend

    ctl = FakeController()
    cfg = Config(backend="fake", quant="none", ctx=128, batch=1, sm_clock_mhz=1185)
    sup = Supervisor(FakeBackend(), cfg, None, log_path=tmp_path / "s.log", startup_timeout=15, power=ctl)
    sup.start()
    try:
        assert ctl.applied == PowerSetting(sm_clock_mhz=1185)
        assert sup.status()["power"] == {"applied": "clock <= 1185 MHz", "error": None}
    finally:
        sup.stop()
    assert ctl.applied is None and ctl.history[-1] == "restore"

    refusing = FakeController(fail_on=PowerSetting(sm_clock_mhz=1185))
    sup = Supervisor(FakeBackend(), cfg, None, log_path=tmp_path / "t.log", startup_timeout=15, power=refusing)
    sup.start()
    try:
        assert sup.healthy() and "insufficient" in sup.status()["power"]["error"]  # serves uncapped, says why
    finally:
        sup.stop()


# --------------------------------------------------------------------------- config, cache, CLI


def test_config_keys_distinguish_power_but_share_a_base():
    c = Config(backend="vllm", quant="fp8", ctx=4096, batch=64, gpu_memory_utilization=0.9)
    capped = with_power(c, PowerSetting(power_limit_w=245))
    locked = with_power(c, PowerSetting(sm_clock_mhz=1185))
    assert capped.key().endswith("/pl245") and locked.key().endswith("/clk1185")
    assert c.key() == c.base_key() == capped.base_key() == locked.base_key()
    assert len({c.key(), capped.key(), locked.key()}) == 3


def test_profile_cache_is_keyed_by_power_mode(tmp_home, hw_a100, spec, prepared_vllm):
    from rigtune import cache
    from tests.test_cache_and_pipeline import _profile

    p = _profile(hw_a100, spec, prepared_vllm)
    cache.save(p)
    path = cache.save(p.model_copy(update={"power_mode": "both"}))
    assert path.name == "balanced-power-both.json"
    assert cache.load(hw_a100, spec, "balanced").power_mode == "off"
    assert cache.load(hw_a100, spec, "balanced", power="both").power_mode == "both"
    assert cache.load(hw_a100, spec, "balanced", power="cap") is None


def test_setup_power_explains_what_it_skips(hw_a100, hw_cpu):
    from rigtune.pipeline import setup_power

    ctl, pts, notes = setup_power(hw_cpu, "both")
    assert ctl is None and pts == [] and "no GPU" in notes[0]

    class NoLock(FakeController):
        def capabilities(self):
            c = _caps(can_lock=False)
            c.reasons["clock"] = "insufficient permissions (power and clock control need root on the host)"
            return c

    ctl, pts, notes = setup_power(hw_a100, "both", NoLock())
    assert ctl is not None and len(pts) == 4  # default + 3 caps; clocks refused
    assert any("clock locking not permitted" in n and "root" in n for n in notes)
    assert any(n.startswith("power stage tried 3 settings") for n in notes)


def test_cli_power_status_and_option_validation(monkeypatch, hw_a100):
    import rigtune.hardware as hwmod
    from rigtune.cli import app

    monkeypatch.setattr(hwmod, "probe", lambda: hw_a100)
    monkeypatch.setattr(P, "controller_for", lambda i: FakeController())
    r = CliRunner().invoke(app, ["power", "status"])
    assert r.exit_code == 0, r.output
    assert "permitted" in r.output and "350 W" in r.output and "--power both would try" in r.output
    r = CliRunner().invoke(app, ["bench", "x/y", "--power", "turbo"])
    assert r.exit_code != 0
