import struct

from rigtune.backends.llamacpp import LlamaCppHipBackend
from rigtune.hfconfig import arch_from_config, load_gguf_config
from rigtune.models import Config, ModelSpec


def _gguf_string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def _string_entry(key, value):
    return _gguf_string(key) + struct.pack("<I", 8) + _gguf_string(value)


def _uint_entry(key, value):
    return _gguf_string(key) + struct.pack("<II", 4, value)


def _array_entry(key, count):
    return _gguf_string(key) + struct.pack("<IIQ", 9, 4, count) + bytes(count * 4)


def _qwen35_gguf(path):
    entries = [
        _string_entry("general.architecture", "qwen35"),
        _uint_entry("qwen35.block_count", 65),
        _uint_entry("qwen35.context_length", 262144),
        _uint_entry("qwen35.embedding_length", 5120),
        _uint_entry("qwen35.feed_forward_length", 17408),
        _uint_entry("qwen35.attention.head_count", 24),
        _uint_entry("qwen35.attention.head_count_kv", 4),
        _uint_entry("qwen35.attention.key_length", 256),
        _uint_entry("qwen35.full_attention_interval", 4),
        _uint_entry("qwen35.ssm.conv_kernel", 4),
        _uint_entry("qwen35.ssm.state_size", 128),
        _uint_entry("qwen35.ssm.group_count", 16),
        _uint_entry("qwen35.ssm.inner_size", 6144),
        _uint_entry("qwen35.nextn_predict_layers", 1),
        _array_entry("tokenizer.ggml.tokens", 256),
    ]
    tensor = (_gguf_string("blk.64.nextn.eh_proj.weight") + struct.pack("<I", 1)
              + struct.pack("<QIQ", 1, 0, 0))
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, len(entries)) + b"".join(entries) + tensor)


def test_gguf_header_provides_qwen35_architecture_and_mtp(tmp_path):
    model_path = tmp_path / "qwen35-Q4_K_M.gguf"
    _qwen35_gguf(model_path)

    config = load_gguf_config(model_path)
    arch = arch_from_config(config)

    assert arch.architecture == "Qwen3_5ForCausalLM"
    assert arch.num_layers == 65 and arch.max_position_embeddings == 262144
    assert arch.vocab_size == 256 and arch.mtp_layers == 1
    assert arch.full_attention_layers == 16 and arch.linear_layers == 48
    assert arch.linear_state_bytes > 0
    assert arch.linear_state_bytes_llamacpp == 48 * (
        3 * (6144 + 2 * 16 * 128) + 128 * 6144
    ) * 4
    assert config["_gguf_mtp_tensor_count"] == 1


def test_local_gguf_prepares_with_embedded_mtp_and_local_template(tmp_path, monkeypatch, hw_rx7900xt):
    from rigtune.backends import llamacpp as module

    model_path = tmp_path / "qwen35-Q4_K_M.gguf"
    template = tmp_path / "chat_template.jinja"
    template.write_text("{{ messages }}", encoding="utf-8")
    _qwen35_gguf(model_path)
    backend = LlamaCppHipBackend()
    prepared = backend.prepare(ModelSpec(hf_id=str(model_path)), hw_rx7900xt)
    assert prepared.spec_repos["mtp"] == str(model_path)
    assert prepared.gguf_paths["Q4_K_M"] == str(model_path)

    monkeypatch.setattr(module, "llama_server_binary", lambda backend=None: "/opt/llama-server")
    monkeypatch.setattr(module, "server_help", lambda binary: "--spec-type draft-mtp")
    variants = backend.spec_variants(
        backend.default_config(hw_rx7900xt, prepared), prepared, hw_rx7900xt, all_methods=True,
    )
    assert {variant.spec_decode for variant in variants} >= {"mtp:1", "mtp:2", "mtp:3"}
    cfg = Config(backend="llamacpp-hip", quant="Q4_K_M", ctx=4096, batch=1, n_gpu_layers=66,
                 extra=backend._device_extra(hw_rx7900xt))
    monkeypatch.setenv("RIGTUNE_LLAMA_REASONING", "off")
    launch = backend.launch_spec(cfg, prepared, 18000)
    assert launch.env["ROCR_VISIBLE_DEVICES"] == "GPU-6bdce6ea1d388c5c"
    assert launch.env["HIP_VISIBLE_DEVICES"] == "0"
    assert str(template) in launch.args
    assert launch.args[launch.args.index("--reasoning") + 1] == "off"
    assert backend._spec_args("mtp:2", prepared, "/opt/llama-server") == [
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
    ]


def test_llamacpp_hybrid_cache_models_parallel_state_and_mtp_kv():
    from rigtune.backends.llamacpp import llamacpp_hybrid_kv

    config = {
        "architectures": ["Qwen3_5ForCausalLM"], "num_hidden_layers": 65, "hidden_size": 5120,
        "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
        "full_attention_interval": 4, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "linear_num_key_heads": 16, "linear_num_value_heads": 48,
        "linear_inner_size": 6144, "num_nextn_predict_layers": 1, "_gguf_block_count_includes_mtp": True,
    }
    arch = arch_from_config(config)
    cfg = Config(backend="llamacpp-hip", quant="Q4_K_M", ctx=65536, batch=2, n_gpu_layers=66,
                 spec_decode="mtp:2")
    hybrid = llamacpp_hybrid_kv(cfg, arch)
    assert hybrid.state_slots == 2
    assert hybrid.state_bytes == arch.linear_state_bytes_llamacpp
    assert hybrid.full_layers == 17
