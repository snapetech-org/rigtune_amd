"""llama.cpp backend: CUDA, AMD HIP/ROCm, Vulkan, and CPU variants."""

from __future__ import annotations

import functools
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rigtune import speculative
from rigtune.backends.base import BaseBackend, LaunchSpec, LlmtraceHooks, ctx_grid, render_extra
from rigtune.gguf import (
    GGUF_QUANTS,
    GGUFCandidate,
    convert_and_quantize,
    download_gguf,
    estimate_gguf_bytes,
    search_hub_gguf,
)
from rigtune.hardware import llama_server_binary, llama_server_device_index
from rigtune.hfconfig import arch_from_config, load_arch, load_gguf_config
from rigtune.memory import HybridKV, MemoryModel
from rigtune.models import ArchInfo, Config, HardwareDescriptor, MiB, ModelSpec, PreparedModel


@functools.lru_cache(maxsize=8)
def server_help(binary: str) -> str:
    """`llama-server --help`, to follow flag renames across llama.cpp versions. Empty if it cannot run."""
    try:
        out = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=60)
        return out.stdout + out.stderr
    except Exception:
        return ""


def _modern_spec(binary: str) -> bool:
    return "--spec-type" in server_help(binary)

logger = logging.getLogger(__name__)


class LlamaCppBackend(BaseBackend):
    """Shared implementation; CUDA, HIP, Vulkan, and CPU variants differ in offload and device selection."""

    name = "llamacpp"
    accelerator: Optional[str] = None
    gpu_vendor: Optional[str] = None
    gpu_backend = False
    runtime_workspace_bytes = 768 * MiB  # compute buffers (scale with n_batch) + accelerator context

    def supports(self, hw: HardwareDescriptor, model: ModelSpec) -> bool:
        binary = llama_server_binary(self.accelerator)
        if not binary:
            return False
        if not self.gpu_backend:
            return True
        return any(gpu.vendor == self.gpu_vendor for gpu in hw.gpus) if self.gpu_vendor else bool(hw.gpus)

    def _device_extra(self, hw: HardwareDescriptor) -> Dict[str, object]:
        if self.accelerator not in ("hip", "vulkan") or hw.gpu is None:
            return {}
        extra: Dict[str, object] = {
            "rigtune_gpu_index": hw.gpu.index,
            "rigtune_gpu_name": hw.gpu.name,
        }
        if hw.gpu.uuid:
            extra["rigtune_gpu_uuid"] = hw.gpu.uuid
        return extra

    # ---- prepare: resolve (no download); materialize: download/convert only what the planner kept

    def prepare(self, model: ModelSpec, hw: HardwareDescriptor, quants: Optional[List[str]] = None) -> PreparedModel:
        local_gguf = os.path.isfile(model.hf_id) and model.hf_id.lower().endswith(".gguf")
        gguf_config = load_gguf_config(model.hf_id) if local_gguf else None
        arch = arch_from_config(gguf_config) if gguf_config is not None else load_arch(model)
        quants = list(quants or GGUF_QUANTS)
        prepared = PreparedModel(spec=model, backend=self.name, arch=arch)
        if local_gguf:
            # Local GGUF file given directly.
            q = _quant_from_filename(model.hf_id) or "Q4_K_M"
            prepared.gguf_paths[q] = model.hf_id
            prepared.weights_bytes[q] = os.path.getsize(model.hf_id)
            if gguf_config and gguf_config.get("_gguf_mtp_tensor_count") and arch.mtp_layers:
                prepared.spec_repos["mtp"] = model.hf_id
            return prepared
        try:
            found: Dict[str, GGUFCandidate] = search_hub_gguf(model, quants)
        except Exception as exc:
            logger.warning("GGUF hub search failed (%s); will fall back to conversion", exc)
            found = {}
        params = arch.num_params or 0
        for q in quants:
            cand = found.get(q)
            if cand is not None:
                prepared.weights_bytes[q] = cand.size_bytes or estimate_gguf_bytes(params, q)
                prepared.gguf_paths[q] = f"hf://{cand.repo_id}/{cand.filename}"
            else:
                prepared.weights_bytes[q] = estimate_gguf_bytes(params, q)
                prepared.gguf_paths[q] = f"convert://{q}"
        # A small same-family draft model for speculative decoding, when the Hub has a GGUF of it.
        draft = speculative.draft_for(model.hf_id)
        if draft:
            try:
                d = search_hub_gguf(ModelSpec(hf_id=draft), ["Q8_0"]).get("Q8_0")
                if d is not None:
                    prepared.draft_paths[draft] = f"hf://{d.repo_id}/{d.filename}"
            except Exception as exc:
                logger.debug("no draft GGUF for %s: %s", draft, exc)
        return prepared

    def materialize(self, model: PreparedModel, quants: List[str]) -> PreparedModel:
        need_convert: List[str] = []
        for q in quants:
            ref = model.gguf_paths.get(q)
            if ref is None:
                continue
            if ref.startswith("hf://"):
                repo, _, fname = ref[len("hf://"):].rpartition("/")
                path = download_gguf(GGUFCandidate(repo_id=repo, filename=fname, quant=q))
                model.gguf_paths[q] = str(path)
                model.weights_bytes[q] = path.stat().st_size
            elif ref.startswith("convert://"):
                need_convert.append(q)
        if need_convert:
            produced = convert_and_quantize(model.spec, need_convert)
            for q, path in produced.items():
                model.gguf_paths[q] = str(path)
                model.weights_bytes[q] = Path(path).stat().st_size
            for q in need_convert:
                if q not in produced:
                    model.gguf_paths.pop(q, None)
                    model.weights_bytes.pop(q, None)
        for draft, ref in list(model.draft_paths.items()):
            if ref.startswith("hf://"):
                repo, _, fname = ref[len("hf://"):].rpartition("/")
                try:
                    model.draft_paths[draft] = str(download_gguf(GGUFCandidate(repo_id=repo, filename=fname,
                                                                               quant="Q8_0")))
                except Exception as exc:
                    logger.warning("draft model %s unavailable: %s", draft, exc)
                    model.draft_paths.pop(draft, None)
        return model

    # ---- memory: attention KV spans per-slot ctx x n_parallel; recurrent state is per parallel sequence

    def memory_model(self, hw: HardwareDescriptor) -> MemoryModel:
        hybrid_fn = llamacpp_hybrid_kv if self.gpu_backend else None
        return self.calibrated_memory(hw, lambda cfg: cfg.ctx * cfg.batch, "gpu" if self.gpu_backend else "cpu",
                                      hybrid_fn=hybrid_fn)

    # ---- configs

    def supported_quants(self, hw: HardwareDescriptor) -> List[str]:
        return list(GGUF_QUANTS)

    def kv_dtypes(self, hw: HardwareDescriptor) -> List[str]:
        return ["q8_0", "q4_0"]

    def batch_ladder(self) -> Tuple[int, ...]:
        return (1, 4, 8, 16)

    def prefix_variants(self, cfg: Config) -> List[Config]:
        # Reuse cached prompt chunks by KV shifting, and share one KV buffer across slots so a common
        # prefix computed by one slot serves every slot.
        options = [{"cache_reuse": 256}, {"kv_unified": True}]
        return [cfg.model_copy(update={"extra": {**cfg.extra, **o}}) for o in options
                if any(cfg.extra.get(k) != v for k, v in o.items())]

    def spec_variants(self, cfg: Config, model: PreparedModel, hw: Optional[HardwareDescriptor] = None,
                      all_methods: bool = False, min_batch: Optional[int] = None) -> List[Config]:
        specs: List[str] = []
        binary = llama_server_binary(self.accelerator)
        if binary and _modern_spec(binary):  # built-in n-gram lookup: no second model needed
            specs.append(speculative.ngram(speculative.NGRAM_TOKENS_LLAMACPP))
        if (all_methods and binary and "draft-mtp" in server_help(binary)
                and speculative.MTP_KEY in model.spec_repos):
            specs.extend(speculative.mtp(k) for k in (*speculative.MTP_TOKENS, 2))
        draft = speculative.draft_for(model.spec.hf_id)
        path = model.draft_paths.get(draft or "")
        if draft and path and not path.startswith(("hf://", "convert://")):
            specs.append(speculative.draft(draft, speculative.DRAFT_TOKENS_LLAMACPP))
        return [cfg.model_copy(update={"spec_decode": s}) for s in specs if s != cfg.spec_decode]

    def _quants(self, model: PreparedModel) -> List[str]:
        return [q for q in GGUF_QUANTS if q in model.weights_bytes] or list(model.weights_bytes)

    def candidate_configs(self, hw: HardwareDescriptor, model: PreparedModel, min_ctx: int = 0) -> List[Config]:
        layers = model.arch.num_layers
        ctxs = ctx_grid(model.arch.max_position_embeddings, min_ctx)
        if self.gpu_backend:
            ngls = sorted({layers + 1, (3 * layers) // 4, layers // 2}, reverse=True)  # +1 = output layer too
            n_batches = [512]
        else:
            ngls = [0]
            n_batches = [512, 2048]
        out: List[Config] = []
        for quant in self._quants(model):
            for ngl in ngls:
                for ctx in ctxs:
                    for np_ in (1, 4, 8):
                        for nb in n_batches:
                            out.append(
                                Config(
                                    backend=self.name, quant=quant, ctx=ctx, batch=np_, n_gpu_layers=ngl,
                                    n_batch=nb, extra=self._device_extra(hw),
                                )
                            )
        return out

    def default_config(self, hw: HardwareDescriptor, model: PreparedModel, min_ctx: int = 0) -> Config:
        quants = self._quants(model)
        return Config(
            backend=self.name,
            quant="Q4_K_M" if "Q4_K_M" in quants else quants[0],
            ctx=min(max(4096, min_ctx), model.arch.max_position_embeddings),
            batch=4,
            n_gpu_layers=(model.arch.num_layers + 1) if self.gpu_backend else 0,
            n_batch=512,
            extra=self._device_extra(hw),
        )

    # ---- launch

    def _spec_args(self, spec: str, model: PreparedModel, binary: str) -> List[str]:
        """Speculative-decoding flags. llama.cpp (2026) moved to `--spec-type` and renamed --draft-max;
        older builds only know the draft-model flags."""
        kind, draft, k = speculative.parse(spec)
        modern = _modern_spec(binary)
        if kind == "mtp":
            if "draft-mtp" not in server_help(binary):
                raise RuntimeError("this llama-server has no built-in MTP speculative decoding")
            return ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(k)]
        if kind == "ngram":
            if not modern:
                raise RuntimeError("this llama-server has no built-in n-gram speculative decoding")
            return ["--spec-type", "ngram-mod", "--spec-ngram-mod-n-max", str(k)]
        if kind != "draft":
            raise RuntimeError(f"llama.cpp has no {kind} speculative decoding")
        draft_path = model.draft_paths.get(draft or "")
        if not draft_path or draft_path.startswith(("hf://", "convert://")):
            raise RuntimeError(f"draft model for {spec} not materialized")
        out = ["-md", draft_path]
        out += ["--spec-type", "draft-simple", "--spec-draft-n-max", str(k)] if modern else ["--draft-max", str(k)]
        if self.gpu_backend:
            out += ["-ngld", "999"]
        return out

    def launch_spec(self, cfg: Config, model: PreparedModel, port: int) -> LaunchSpec:
        binary = llama_server_binary(self.accelerator)
        if binary is None:
            raise RuntimeError("llama-server not found on PATH (set $LLAMA_SERVER)")
        path = model.gguf_paths.get(cfg.quant)
        if not path or path.startswith(("hf://", "convert://")):
            raise RuntimeError(f"GGUF for {cfg.quant} not materialized: {path}")
        n_parallel = max(1, cfg.batch)
        args = [
            binary,
            "-m", path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", str(cfg.ctx * n_parallel),  # -c is total context, split across slots
            "-np", str(n_parallel),
            "-b", str(max(cfg.n_batch or 512, cfg.prefill_budget or 0)),  # logical batch must cover -ub
            "-ngl", str(cfg.n_gpu_layers if cfg.n_gpu_layers is not None else (999 if self.gpu_backend else 0)),
            "--alias", model.spec.hf_id,
        ]
        template = Path(path).parent / "chat_template.jinja"
        if template.is_file():
            args += ["--chat-template-file", str(template), "--jinja"]
        reasoning = os.environ.get("RIGTUNE_LLAMA_REASONING")
        if reasoning in ("on", "off", "auto"):
            args += ["--reasoning", reasoning]
        if cfg.prefill_budget is not None:
            args += ["-ub", str(cfg.prefill_budget)]
        threads = cfg.extra.get("threads")
        if threads:
            args += ["-t", str(threads)]
        if self.gpu_backend or cfg.kv_dtype != "auto":
            args += ["-fa", "on"]  # a quantized V cache needs flash attention, on CPU too
        if cfg.kv_dtype != "auto":
            args += ["-ctk", cfg.kv_dtype, "-ctv", cfg.kv_dtype]
        if cfg.spec_decode:
            args += self._spec_args(cfg.spec_decode, model, binary)
        args += render_extra(cfg.extra, skip=("threads", "rigtune_gpu_index", "rigtune_gpu_name", "rigtune_gpu_uuid"))
        env = {}
        if self.accelerator == "hip":
            selected = cfg.extra.get("rigtune_gpu_uuid") or cfg.extra.get("rigtune_gpu_index")
            if selected is not None:
                env["ROCR_VISIBLE_DEVICES"] = str(selected)
                env["HIP_VISIBLE_DEVICES"] = "0"
        elif self.accelerator == "vulkan":
            gpu_name = str(cfg.extra.get("rigtune_gpu_name") or "")
            index = llama_server_device_index("vulkan", gpu_name, binary)
            if index is None:
                raise RuntimeError(f"llama-server Vulkan build does not list selected GPU {gpu_name!r}")
            env["GGML_VK_VISIBLE_DEVICES"] = str(index)
        elif not self.gpu_backend:
            env.update({
                "CUDA_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
                "GGML_VK_VISIBLE_DEVICES": "",
            })
        return LaunchSpec(args=args, env=env)

    def replica_launch_spec(self, cfg: Config, model: PreparedModel, port: int, gpu_index: int) -> LaunchSpec:
        spec = self.launch_spec(cfg, model, port)
        if self.accelerator == "hip":
            spec.env = {**spec.env, "ROCR_VISIBLE_DEVICES": str(gpu_index), "HIP_VISIBLE_DEVICES": "0"}
        elif self.accelerator == "vulkan":
            spec.env = {**spec.env, "GGML_VK_VISIBLE_DEVICES": str(gpu_index)}
        elif self.accelerator == "cuda":
            spec.env = {**spec.env, "CUDA_VISIBLE_DEVICES": str(gpu_index)}
        return spec

    # Physical prompt-processing batch (-ub) to try around the engine default of 512.
    UBATCHES = (256, 1024, 2048)

    def prefill_variants(self, cfg: Config) -> List[Config]:
        return [cfg.model_copy(update={"prefill_budget": ub, "n_batch": max(cfg.n_batch or 512, ub)})
                for ub in self.UBATCHES if ub != cfg.prefill_budget]

    def workload_hooks(self, hw: HardwareDescriptor, model: PreparedModel) -> LlmtraceHooks:
        return LlmtraceHooks(
            health_path="/health",
            model_name=model.spec.hf_id,
            tokenizer_id=model.spec.hf_id,
            gpu_ids=[hw.gpu.index] if (self.accelerator == "cuda" and hw.gpu) else [],
            process_memory=not self.gpu_backend,
        )


class LlamaCppCudaBackend(LlamaCppBackend):
    name = "llamacpp-cuda"
    accelerator = "cuda"
    gpu_vendor = "nvidia"
    gpu_backend = True


class LlamaCppHipBackend(LlamaCppBackend):
    name = "llamacpp-hip"
    accelerator = "hip"
    gpu_vendor = "amd"
    gpu_backend = True


class LlamaCppVulkanBackend(LlamaCppBackend):
    name = "llamacpp-vulkan"
    accelerator = "vulkan"
    gpu_backend = True


def llamacpp_hybrid_kv(cfg: Config, arch: ArchInfo) -> HybridKV:
    """llama.cpp hybrid cache: attention KV per token and F32 recurrent state per parallel sequence.

    The rules follow llama-memory-hybrid.cpp and llama-memory-recurrent.cpp. The built-in MTP context
    adds its dense attention layer when enabled; its recurrent snapshot count is zero.
    """
    kind = speculative.parse(cfg.spec_decode)[0] if cfg.spec_decode else None
    full_layers = (arch.full_attention_layers or 0) + (
        arch.mtp_layers if kind in speculative.SELF_HEAD else 0
    )
    return HybridKV(
        state_slots=max(1, cfg.batch),
        state_bytes=arch.linear_state_bytes_llamacpp,
        full_layers=full_layers or None,
    )


class LlamaCppCpuBackend(LlamaCppBackend):
    name = "llamacpp-cpu"
    accelerator = None
    gpu_backend = False
    runtime_workspace_bytes = 512 * MiB

    def candidate_configs(self, hw: HardwareDescriptor, model: PreparedModel, min_ctx: int = 0) -> List[Config]:
        cfgs = super().candidate_configs(hw, model, min_ctx)
        for c in cfgs:
            c.extra["threads"] = hw.cpu.physical_cores
        return cfgs

    def default_config(self, hw: HardwareDescriptor, model: PreparedModel, min_ctx: int = 0) -> Config:
        c = super().default_config(hw, model, min_ctx)
        c.extra["threads"] = hw.cpu.physical_cores
        return c


def _quant_from_filename(path: str) -> Optional[str]:
    base = os.path.basename(path).upper()
    for q in GGUF_QUANTS + ("F16", "BF16", "F32"):
        if q in base:
            return q
    return None
