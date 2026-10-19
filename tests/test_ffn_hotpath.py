"""AR004 FFN hot-path tests (FR4.1 fingerprint O(1))."""

import time

import torch

from shadowinfer.ffn_optimizer.packed_weight import PackedFFNWeight


def _mixed_precision_map(channels: int) -> dict:
    mapping = {}
    for ch in range(channels):
        if ch % 4 == 0:
            mapping[ch] = "fp16"
        elif ch % 4 == 1:
            mapping[ch] = "int8"
        else:
            mapping[ch] = "int4"
    return mapping


def _packed(channels: int = 64, dim: int = 256) -> PackedFFNWeight:
    torch.manual_seed(0)
    weight = torch.randn(channels, dim)
    return PackedFFNWeight.pack(weight, _mixed_precision_map(channels), group_size=32)


class _CountingPacked(PackedFFNWeight):
    unpack_calls = 0

    def unpack(self):
        type(self).unpack_calls += 1
        return super().unpack()


def test_ut1_dequant_cached_single_unpack():
    _CountingPacked.unpack_calls = 0
    torch.manual_seed(0)
    weight = torch.randn(64, 256)
    packed = _CountingPacked.pack(weight, _mixed_precision_map(64), group_size=32)

    first = packed.dequantize_for_matmul()
    for _ in range(99):
        packed.dequantize_for_matmul()

    assert _CountingPacked.unpack_calls == 1
    assert torch.equal(first, packed.unpack())


def test_ut1_fingerprint_is_o1():
    packed = _packed(channels=1024, dim=256)

    start = time.perf_counter()
    for _ in range(100):
        packed._precision_plan_fingerprint()
    elapsed_us = (time.perf_counter() - start) * 1e6 / 100

    assert packed._precision_plan_fingerprint() == tuple(
        (ch, packed._data[ch]["precision"]) for ch in sorted(packed._data)
    )
    assert elapsed_us <= 10.0


def test_ut2_repack_invalidates_cache():
    packed = _packed()
    packed.dequantize_for_matmul()
    assert packed._dequant_cache is not None

    weight = torch.randn(64, 256)
    mapping = {ch: "fp16" for ch in range(64)}
    packed._do_pack(weight, mapping, group_size=32)

    assert packed._dequant_cache is None
    assert packed._dequant_cache_fingerprint is None
    assert packed._precision_plan_fingerprint() == tuple((ch, "fp16") for ch in range(64))
    assert torch.allclose(packed.dequantize_for_matmul(), weight.half().float(), atol=1e-3)


def test_ut13b_empty_unpack_all_zeros():
    empty = PackedFFNWeight((8, 16))
    assert torch.count_nonzero(empty.unpack()) == 0


def _agent(**overrides):
    from shadowinfer.ffn_optimizer.ffn_optimizer_agent import FFNOptimizerAgent

    cfg = {
        "hidden_dim": 32,
        "intermediate_dim": 64,
        "channel_pruning_enabled": False,
        "channel_pruning_ratio": 0.0,
        "use_packed_weights": True,
    }
    cfg.update(overrides)
    return FFNOptimizerAgent(cfg)


def _step(agent, x, up, down, layer_id=0, sensitivity=0.9, mode="full"):
    from shadowinfer.core.structs import StepConfig

    return agent.on_step(
        StepConfig(
            step_id=0,
            total_steps=1,
            noise_level=0.5,
            shadowkv_mode="balanced",
            ffn_mode=mode,
        ),
        {
            "ffn_input_current": x,
            "weights": {"up": up, "down": down},
            "layer_id": layer_id,
            "qdrift_signal": {"sensitivity_score": sensitivity, "ffn_mode": mode},
        },
    )


def test_ut6_layer_isolation_full_path():
    torch.manual_seed(0)
    x = torch.randn(1, 8, 32)
    up1, down1 = torch.randn(64, 32) * 0.05, torch.randn(32, 64) * 0.05
    up2, down2 = torch.randn(64, 32) * 0.05, torch.randn(32, 64) * 0.05
    agent = _agent()

    out1 = _step(agent, x, up1, down1, layer_id=0)["output"]
    out2 = _step(agent, x, up2, down2, layer_id=1)["output"]
    out1_again = _step(agent, x, up1, down1, layer_id=0)["output"]

    def direct(up, down):
        return torch.nn.functional.linear(
            torch.nn.functional.gelu(torch.nn.functional.linear(x, up)), down
        )

    assert torch.allclose(out1, direct(up1, down1), atol=1e-6)
    assert torch.allclose(out2, direct(up2, down2), atol=1e-6)
    assert torch.equal(out1, out1_again)


def test_ut6b_lifecycle_and_overflow_warning():
    agent = _agent(num_layers=2)
    x = torch.randn(1, 4, 32)
    up, down = torch.randn(64, 32) * 0.05, torch.randn(32, 64) * 0.05
    _step(agent, x, up, down, layer_id=0)
    assert 0 in agent._layer_states

    for _ in range(3):
        _step(agent, x, up, down, layer_id=99)
    overflow_keys = [k for k in agent._warned if k == (99, "layer_overflow")]
    assert len(overflow_keys) == 1

    try:
        _step(agent, x, up, down, layer_id=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative layer_id must raise ValueError")

    agent.on_init({"num_layers": 2})
    assert agent._layer_states == {}
    assert agent._warned == set()


def test_ut16_static_pruning_uses_per_layer_importance():
    torch.manual_seed(0)
    x = torch.randn(1, 8, 32)
    up, down = torch.randn(64, 32) * 0.05, torch.randn(32, 64) * 0.05
    agent = _agent(channel_pruning_enabled=True, channel_pruning_ratio=0.5)

    imp1 = torch.zeros(64)
    imp1[:64] = torch.linspace(1.0, 0.0, 64)
    imp2 = torch.zeros(64)
    imp2[:64] = torch.linspace(0.0, 1.0, 64)

    agent._layer_state(0).channel_importance = imp1
    agent._layer_state(1).channel_importance = imp2

    _step(agent, x, up, down, layer_id=0, sensitivity=0.5, mode="mixed")
    mask0 = agent._layer_state(0).channel_mask.clone()
    _step(agent, x, up, down, layer_id=1, sensitivity=0.5, mode="mixed")
    mask1 = agent._layer_state(1).channel_mask.clone()

    k = 32
    assert set(torch.topk(imp1, k).indices.tolist()) == set(mask0.nonzero().flatten().tolist())
    assert set(torch.topk(imp2, k).indices.tolist()) == set(mask1.nonzero().flatten().tolist())
    assert agent._layer_state(1).channel_importance is imp2


def test_ut8_mixed_mask_golden():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 32)
    up, down = torch.randn(64, 32) * 0.05, torch.randn(32, 64) * 0.05
    agent = _agent()
    mask = torch.zeros(64, dtype=torch.bool)
    mask[torch.randperm(64)[:48]] = True

    packed_up = PackedFFNWeight.pack(up, {i: "fp16" for i in range(64)})
    packed_down = PackedFFNWeight.pack(down, {i: "fp16" for i in range(32)}, pack_dim=1)
    mixed = agent._mixed_compute(x, packed_up, packed_down, mask)
    dense_up = packed_up.unpack()
    dense_down = packed_down.unpack()
    expected = agent._full_compute(x, dense_up, dense_down, mask)
    assert mixed.shape == (2, 4, 32)
    assert torch.allclose(mixed, expected, atol=1e-6)


def test_ut9_all_false_mask_zero_with_single_warning():
    x = torch.randn(1, 4, 32)
    up, down = torch.randn(64, 32) * 0.05, torch.randn(32, 64) * 0.05
    agent = _agent()
    mask = torch.zeros(64, dtype=torch.bool)

    out = agent._mixed_compute(x, up, down, mask, warning_key=(0, "all_false_mask"))
    out2 = agent._mixed_compute(x, up, down, mask, warning_key=(0, "all_false_mask"))
    assert torch.count_nonzero(out) == 0 and out.shape == (1, 4, 32)
    assert torch.count_nonzero(out2) == 0
    assert sum(1 for k in agent._warned if k == (0, "all_false_mask")) == 1


def test_ut21_full_all_false_mask():
    x = torch.randn(1, 4, 32)
    up, down = torch.randn(64, 32) * 0.05, torch.randn(32, 64) * 0.05
    agent = _agent()
    mask = torch.zeros(64, dtype=torch.bool)

    out = agent._full_compute(x, up, down, mask, warning_key=(2, "all_false_mask"))
    agent._full_compute(x, up, down, mask, warning_key=(2, "all_false_mask"))
    assert torch.count_nonzero(out) == 0
    assert sum(1 for k in agent._warned if k == (2, "all_false_mask")) == 1


def test_ut13a_invalid_mask_dim():
    agent = _agent()
    x = torch.randn(1, 4, 32)
    up = torch.randn(64, 32)
    try:
        agent._linear_forward(x, up, mask_dim=2)
    except ValueError:
        pass
    else:
        raise AssertionError("mask_dim=2 must raise ValueError")


def _mixed_map(n=64):
    mapping = {}
    for i in range(n):
        mapping[i] = "fp16" if i % 5 == 0 else ("int8" if i % 5 in (1, 2) else "int4")
    return mapping


def test_ut4_unpack_golden_vec_vs_reference():
    torch.manual_seed(0)
    for pack_dim, shape in ((0, (64, 48)), (1, (32, 64))):
        w = torch.randn(*shape)
        packed = PackedFFNWeight.pack(
            w, _mixed_map(shape[pack_dim]), pack_dim=pack_dim, group_size=16
        )
        assert torch.allclose(packed.unpack(), packed._unpack_reference(), atol=1e-6), pack_dim


def test_ut4_int4_non_divisible_group():
    torch.manual_seed(0)
    w = torch.randn(16, 30)
    packed = PackedFFNWeight.pack(w, {i: "int4" for i in range(16)}, group_size=16)
    assert packed._data[0]["group_size"] == 30
    assert torch.allclose(packed.unpack(), packed._unpack_reference(), atol=1e-6)


def test_ut10_quantize_golden_vec_vs_reference():
    from shadowinfer.ffn_optimizer.ffn_optimizer_agent import FFNOptimizerAgent

    torch.manual_seed(0)
    up = torch.randn(64, 32)
    down = torch.randn(32, 64)
    imp = torch.linspace(0.0, 1.0, 64)
    agent = FFNOptimizerAgent(
        {"use_packed_weights": False, "hidden_dim": 32, "intermediate_dim": 64}
    )

    q_up, q_down, scales = agent.quantize_mixed_precision(up, down, imp)
    r_up, r_down, r_scales = agent._quantize_mixed_precision_reference(up, down, imp)

    assert torch.allclose(q_up, r_up, atol=1e-6)
    assert torch.allclose(q_down, r_down, atol=1e-6)
    assert scales == r_scales


def test_ut10_zero_channel_preserved():
    from shadowinfer.ffn_optimizer.ffn_optimizer_agent import FFNOptimizerAgent

    up = torch.randn(8, 16)
    down = torch.randn(16, 8)
    up[3] = 0.0
    down[:, 3] = 0.0
    imp = torch.tensor([0.1] * 8)
    agent = FFNOptimizerAgent(
        {"use_packed_weights": False, "hidden_dim": 16, "intermediate_dim": 8}
    )
    q_up, q_down, scales = agent.quantize_mixed_precision(up, down, imp)
    assert torch.count_nonzero(q_up[3]) == 0
    assert torch.count_nonzero(q_down[:, 3]) == 0
    assert scales[3] == ("int4", (0.0, 0.0))


def test_ut3_quantization_info_cached_and_complete():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 32)
    up, down = torch.randn(64, 32) * 0.05, torch.randn(32, 64) * 0.05
    agent = _agent()
    outs = [_step(agent, x, up, down, sensitivity=0.5, mode="mixed") for _ in range(3)]
    infos = [o["quantization"] for o in outs]
    assert infos[0] == infos[1] == infos[2]
    info = infos[0]
    assert info["num_fp16_channels"] + info["num_int8_channels"] + info["num_int4_channels"] == 64
    assert "compression_ratio" in info


def test_ut3b_stats_reads_zero_iteration_after_warmup():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 32)
    up, down = torch.randn(64, 32) * 0.05, torch.randn(32, 64) * 0.05
    agent = _agent()
    _step(agent, x, up, down, sensitivity=0.5, mode="mixed")
    state = agent._layer_state(0)

    class CountingDict(dict):
        iterations = 0

        def items(self):
            type(self).iterations += 1
            return super().items()

    counting = CountingDict(state.precision_scales)
    state.precision_scales = counting
    out = _step(agent, x, up, down, sensitivity=0.5, mode="mixed")
    assert out["quantization"]["num_fp16_channels"] >= 1
    assert CountingDict.iterations == 0


def test_ut6c_sparse_and_incremental_layer_isolation():
    torch.manual_seed(0)
    x_prev = torch.randn(1, 8, 32)
    x_cur = x_prev + 0.001
    up1, down1 = torch.randn(64, 32) * 0.05, torch.randn(32, 64) * 0.05
    up2, down2 = torch.randn(64, 32) * 0.05, torch.randn(32, 64) * 0.05
    agent = _agent()

    def call(x, layer, up, down, mode):
        from shadowinfer.core.structs import StepConfig

        out_prev = agent._full_compute(x, up, down)
        for sens in (0.2, 0.5):
            ret = agent.on_step(
                StepConfig(
                    step_id=0,
                    total_steps=1,
                    noise_level=0.5,
                    shadowkv_mode="balanced",
                    ffn_mode=mode,
                ),
                {
                    "ffn_input_current": x,
                    "ffn_input_previous": x_prev,
                    "ffn_output_previous": out_prev,
                    "weights": {"up": up, "down": down},
                    "layer_id": layer,
                    "qdrift_signal": {"sensitivity_score": sens, "ffn_mode": mode},
                },
            )
            if ret["compute_path"] == mode:
                return

    call(x_cur, 0, up1, down1, "sparse")
    call(x_cur, 1, up2, down2, "sparse")
    call(x_cur, 0, up1, down1, "incremental")
    call(x_cur, 1, up2, down2, "incremental")
    assert torch.equal(agent._layer_state(0).weight_up, up1)
    assert torch.equal(agent._layer_state(1).weight_up, up2)


def test_ut9_mask_exception_resets_state():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 32)
    up, down = torch.randn(64, 32) * 0.05, torch.randn(32, 64) * 0.05
    agent = _agent(channel_pruning_enabled=True, channel_pruning_ratio=0.5)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    agent.compute_channel_mask = boom
    out = _step(agent, x, up, down, sensitivity=0.5, mode="mixed")
    state = agent._layer_state(0)
    assert state.channel_mask is None
    assert state.active_channels == 0
    assert out["output"].shape == (1, 4, 32)


def test_ut14_threshold_boundary_matches_reference():
    from shadowinfer.ffn_optimizer.ffn_optimizer_agent import FFNOptimizerAgent

    torch.manual_seed(0)
    up = torch.randn(8, 16)
    down = torch.randn(16, 8)
    for value in (0.7, 0.3, 0.6999999, 0.3000001, 0.0, 1.0):
        imp = torch.full((8,), value, dtype=torch.float32)
        agent = FFNOptimizerAgent(
            {"use_packed_weights": False, "hidden_dim": 16, "intermediate_dim": 8}
        )
        q_up, q_down, scales = agent.quantize_mixed_precision(up, down, imp)
        r_up, r_down, r_scales = agent._quantize_mixed_precision_reference(up, down, imp)
        assert torch.allclose(q_up, r_up, atol=1e-6), value
        assert torch.allclose(q_down, r_down, atol=1e-6), value
        assert scales == r_scales, value


def test_ut18a_num_layers_none_and_bounds():
    agent = _agent()
    agent.on_init({"num_layers": None})
    assert agent._num_layers == 0
    agent.on_init({"num_layers": 0})
    assert agent._num_layers == 0


def test_ut18b_memory_accounting_once_per_layer():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 32)
    up, down = torch.randn(64, 32) * 0.05, torch.randn(32, 64) * 0.05
    agent = _agent(num_layers=4)
    _step(agent, x, up, down, layer_id=0, sensitivity=0.5, mode="mixed")
    first = agent._total_memory_bytes_saved
    _step(agent, x, up, down, layer_id=0, sensitivity=0.5, mode="mixed")
    assert agent._total_memory_bytes_saved == first
    _step(agent, x, up, down, layer_id=1, sensitivity=0.5, mode="mixed")
    assert agent._total_memory_bytes_saved == first * 2
    agent.on_init({"num_layers": 4})
    assert agent._total_memory_bytes_saved == 0.0


def test_ut20_sensitivity_threshold_paths():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 32)
    up, down = torch.randn(64, 32) * 0.05, torch.randn(32, 64) * 0.05
    agent = _agent()
    out_hi = _step(agent, x, up, down, sensitivity=0.7, mode="mixed")
    assert out_hi["compute_path"] == "full"
    assert out_hi["quantization"] == {}
    agent2 = _agent()
    out_lo = _step(agent2, x, up, down, sensitivity=0.69, mode="mixed")
    assert out_lo["compute_path"] == "mixed_full"
    assert out_lo["quantization"] != {}


def test_ut5_ut11_vectorization_speedup():
    torch.manual_seed(0)
    for channels, dim in ((256, 128), (1024, 256)):
        w = torch.randn(channels, dim)
        imp = torch.linspace(0.0, 1.0, channels)
        from shadowinfer.ffn_optimizer.ffn_optimizer_agent import FFNOptimizerAgent as _Agent

        agent = _Agent(
            {"use_packed_weights": False, "hidden_dim": dim, "intermediate_dim": channels}
        )

        def median_time(fn, n=3):
            samples = []
            for _ in range(n):
                start = time.perf_counter()
                fn()
                samples.append(time.perf_counter() - start)
            return sorted(samples)[len(samples) // 2]

        vec_q = median_time(lambda: agent.quantize_mixed_precision(w, w.T, imp))
        ref_q = median_time(lambda: agent._quantize_mixed_precision_reference(w, w.T, imp))
        assert vec_q <= ref_q * 0.5

        pmap = {i: ("fp16" if i % 3 == 0 else "int8") for i in range(channels)}
        packed = PackedFFNWeight.pack(w, pmap, group_size=16)
        vec_u = median_time(packed.unpack)
        ref_u = median_time(packed._unpack_reference)
        assert vec_u <= ref_u * 0.5


def test_ut13b_invalid_method_and_non_2d():
    agent = _agent()
    x = torch.randn(1, 4, 32)
    up = torch.randn(64, 32)
    mask = agent.compute_channel_mask(x, up, method="dynamic_activation")
    assert mask.shape == (64,)

    agent.channel_pruning_ratio = 0.5
    agent.channel_pruning_method = "static_importance"
    agent.channel_importance = torch.rand(64)
    fallback = agent.compute_channel_mask(x, up, method="unknown_method")
    assert fallback.shape == (64,)
    agent.channel_importance = None
    try:
        agent.compute_channel_mask(x, up, method="unknown_method")
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing importance must raise RuntimeError")

    wide = torch.randn(64, 32)
    agent._precompute_weights(wide, wide.T, layer_id=2)
    assert agent._layer_state(2).weight_up is wide


def test_ut15_reuse_returns_previous():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 32)
    prev = torch.randn(1, 4, 32)
    up, down = torch.randn(64, 32) * 0.05, torch.randn(32, 64) * 0.05
    agent = _agent()
    from shadowinfer.core.structs import StepConfig

    out = agent.on_step(
        StepConfig(
            step_id=0, total_steps=1, noise_level=0.5, shadowkv_mode="balanced", ffn_mode="sparse"
        ),
        {
            "ffn_input_current": x,
            "ffn_input_previous": x,
            "ffn_output_previous": prev,
            "weights": {"up": up, "down": down},
            "layer_id": 0,
            "qdrift_signal": {"sensitivity_score": 0.1, "ffn_mode": "sparse"},
        },
    )
    assert out["compute_path"] == "reuse"
    assert torch.equal(out["output"], prev)


def test_ut19_missing_previous_falls_back_to_full():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 32)
    up, down = torch.randn(64, 32) * 0.05, torch.randn(32, 64) * 0.05
    agent = _agent()
    from shadowinfer.core.structs import StepConfig

    original = agent.select_compute_path
    for mode in ("reuse", "incremental", "sparse"):
        agent.select_compute_path = lambda *a, m=mode, **k: m
        out = agent.on_step(
            StepConfig(
                step_id=0,
                total_steps=1,
                noise_level=0.5,
                shadowkv_mode="balanced",
                ffn_mode="sparse",
            ),
            {
                "ffn_input_current": x,
                "weights": {"up": up, "down": down},
                "layer_id": 0,
                "qdrift_signal": {"sensitivity_score": 0.4, "ffn_mode": "sparse"},
            },
        )
        assert out["compute_path"] == "full", mode
    agent.select_compute_path = original


def test_ut17_version_strings_consistent():
    import tomllib
    from pathlib import Path

    import shadowinfer

    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == shadowinfer.__version__ == "3.11.0"


def _assert_pack_data_equal(vec, ref, label=""):
    assert vec._data.keys() == ref._data.keys(), label
    for ch, ref_entry in ref._data.items():
        entry = vec._data[ch]
        assert entry.keys() == ref_entry.keys(), (label, ch)
        for key, expected in ref_entry.items():
            value = entry[key]
            if isinstance(expected, torch.Tensor):
                assert torch.equal(value, expected), (label, ch, key)
                assert value.dtype == expected.dtype, (label, ch, key)
                assert value.shape == expected.shape, (label, ch, key)
            else:
                assert value == expected, (label, ch, key)
    assert vec.memory_bytes() == ref.memory_bytes(), label
    assert vec._precision_plan_fingerprint() == ref._precision_plan_fingerprint(), label


def test_ut61_pack_golden_mixed_precision():
    torch.manual_seed(0)
    for pack_dim, shape, group_size in (
        (0, (64, 48), 16),
        (1, (32, 64), 16),
        (0, (33, 30), 16),
        (0, (48, 32), 64),
    ):
        weight = torch.randn(*shape)
        pmap = _mixed_map(shape[pack_dim])
        vec = PackedFFNWeight.pack(weight, pmap, pack_dim=pack_dim, group_size=group_size)
        ref = PackedFFNWeight(shape, pack_dim=pack_dim)
        ref._do_pack_reference(weight, pmap, group_size=group_size)
        _assert_pack_data_equal(vec, ref, f"dim{pack_dim}-{shape}")


def test_ut62_pack_golden_odd_int4_length():
    torch.manual_seed(0)
    weight = torch.randn(16, 15)
    pmap = {i: "int4" for i in range(16)}
    vec = PackedFFNWeight.pack(weight, pmap, group_size=16)
    ref = PackedFFNWeight(weight.shape)
    ref._do_pack_reference(weight, pmap, group_size=16)
    _assert_pack_data_equal(vec, ref, "odd-int4")
    assert vec._data[0]["group_size"] == 15


def test_ut63_pack_empty_weight():
    weight = torch.zeros(0, 32)
    vec = PackedFFNWeight.pack(weight, {}, group_size=16)
    assert vec._data == {}
    assert vec.memory_bytes() == 0
    assert vec._precision_plan_fingerprint() == ()


def test_ut64_pack_then_unpack_matches_reference():
    torch.manual_seed(0)
    weight = torch.randn(40, 24)
    pmap = _mixed_map(40)
    vec = PackedFFNWeight.pack(weight, pmap, group_size=16)
    assert torch.allclose(vec.unpack(), vec._unpack_reference(), atol=1e-6)


def test_ut65_pack_vectorized_speedup():
    torch.manual_seed(0)
    weight = torch.randn(1024, 256)
    pmap = _mixed_map(1024)

    def timed(fn, n=3):
        samples = []
        for _ in range(n):
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
        return sorted(samples)[len(samples) // 2]

    vec = timed(lambda: PackedFFNWeight.pack(weight, pmap, group_size=64))
    ref = timed(
        lambda: PackedFFNWeight(weight.shape)._do_pack_reference(weight, pmap, group_size=64)
    )
    assert vec <= ref * 0.5


def test_ut61b_pack_golden_fp32_and_unknown_precision():
    torch.manual_seed(0)
    weight = torch.randn(12, 20)
    pmap = {ch: ("fp32" if ch % 2 == 0 else "bogus") for ch in range(12)}
    vec = PackedFFNWeight.pack(weight, pmap, group_size=16)
    ref = PackedFFNWeight(weight.shape)
    ref._do_pack_reference(weight, pmap, group_size=16)
    _assert_pack_data_equal(vec, ref, "fp32-unknown")
    assert vec._data[0]["precision"] == "fp32"
    assert vec._data[1]["precision"] == "fp16"
    try:
        PackedFFNWeight.pack(torch.randn(4, 4, 4), {}, group_size=16)
    except ValueError:
        pass
    else:
        raise AssertionError("3D weight must raise ValueError")
