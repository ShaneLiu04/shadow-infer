"""AR006 baseline / A-B benchmark for FFN packed-weight pack (CPU, fast)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shadowinfer.ffn_optimizer.packed_weight import PackedFFNWeight  # noqa: E402

torch.manual_seed(42)


def median(values):
    return sorted(values)[len(values) // 2]


def precision_map(channels):
    return {
        ch: ("fp16" if ch % 4 == 0 else ("int8" if ch % 4 == 1 else "int4"))
        for ch in range(channels)
    }


def ab(channels, dim):
    weight = torch.randn(channels, dim)
    pmap = precision_map(channels)

    vec, ref = [], []
    for _ in range(3):
        start = time.perf_counter()
        PackedFFNWeight.pack(weight, pmap, group_size=64)
        vec.append(time.perf_counter() - start)
        start = time.perf_counter()
        obj = PackedFFNWeight(weight.shape)
        obj._do_pack_reference(weight, pmap, group_size=64)
        ref.append(time.perf_counter() - start)

    print(
        f"pack {channels}x{dim}: vec_ms={median(vec) * 1e3:.2f} "
        f"ref_ms={median(ref) * 1e3:.2f} ratio={median(vec) / median(ref):.3f}"
    )


def golden_check():
    torch.manual_seed(0)
    for pack_dim, shape in ((0, (33, 30)), (1, (24, 35))):
        weight = torch.randn(*shape)
        pmap = precision_map(shape[pack_dim])
        vec = PackedFFNWeight.pack(weight, pmap, pack_dim=pack_dim, group_size=16)
        ref = PackedFFNWeight(shape, pack_dim=pack_dim)
        ref._do_pack_reference(weight, pmap, group_size=16)
        assert vec._data.keys() == ref._data.keys()
        for ch in ref._data:
            for key in ref._data[ch]:
                value = vec._data[ch][key]
                expected = ref._data[ch][key]
                if isinstance(expected, torch.Tensor):
                    assert torch.equal(value, expected), (pack_dim, ch, key)
                else:
                    assert value == expected, (pack_dim, ch, key)
        assert vec.memory_bytes() == ref.memory_bytes()
        assert vec._precision_plan_fingerprint() == ref._precision_plan_fingerprint()
    print("golden: _data bytes/shapes/scales/fingerprint/memory identical")


ab(256, 256)
ab(1024, 256)
golden_check()
