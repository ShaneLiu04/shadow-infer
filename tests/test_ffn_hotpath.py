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
