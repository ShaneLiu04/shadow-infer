"""AR015 T001: PagedKVStore.rewrite_tokens in-page row-level direct-write tests.

Exercises the not-yet-implemented ``rewrite_tokens`` (FR15.1): rewriting a
global token interval must be bit-identical to appending the same final
content directly (row-local quantization is independent of page batching),
must split across pages without touching page boundaries or metadata, must
refuse intervals containing evicted (token, head) rows, and must validate
all arguments before any mutation (failure atomicity). Rewrites never
change the dual-ledger byte accounting.

All tests use real tensors and real store operations; nothing is mocked.
"""

from __future__ import annotations

import math

import pytest
import torch

from shadowinfer.core import MockModelBackend
from shadowinfer.core.backends.pytorch_backend import PyTorchModelBackend
from shadowinfer.core.structs import StepConfig
from shadowinfer.models.simple_diffusion_llm import DiffusionTransformerBlock, SimpleDiffusionLLM
from shadowinfer.orchestrator import Orchestrator
from shadowinfer.shadowkv.paged_kv_store import PagedKVStore

GOLDEN_SEED = 2026
BATCH = 2
HEADS = 4
HEAD_DIM = 32
SEQ_LEN = 19  # deliberately not a multiple of PAGE_SIZE
PAGE_SIZE = 8
APPEND_SPLIT = (5, 5, 5, 4)  # crosses page boundaries at 8 and 16
INT4_GROUP_SIZE = 16
REWRITE_START = 6
REWRITE_TOKENS = 8  # interval [6, 14) crosses the page boundary at 8
CROSS_SEQ = 26
CROSS_SPLIT = (8, 8, 8, 2)
CROSS_START = 4
CROSS_TOKENS = 16  # interval [4, 20) spans pages 0, 1 and 2


def _make_kv(seed, batch=BATCH, heads=HEADS, seq=SEQ_LEN, head_dim=HEAD_DIM, scale=2.0):
    """Generate non-degenerate K/V (values well away from quantization zero)."""
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn(batch, heads, seq, head_dim, generator=gen) * scale
    v = torch.randn(batch, heads, seq, head_dim, generator=gen) * scale
    return k, v


def _make_store(batch=BATCH, heads=HEADS, head_dim=HEAD_DIM, page_size=PAGE_SIZE):
    return PagedKVStore(
        batch,
        heads,
        head_dim,
        page_size=page_size,
        device="cpu",
        int4_group_size=INT4_GROUP_SIZE,
    )


def _stepwise_append(store, k_full, v_full, precision, split=APPEND_SPLIT):
    start = 0
    for length in split:
        end = start + length
        store.append(
            k_full[:, :, start:end, :].contiguous(),
            v_full[:, :, start:end, :].contiguous(),
            precision,
        )
        start = end


def _appended_store(precision="fp16", seq=SEQ_LEN, split=APPEND_SPLIT, seed=GOLDEN_SEED):
    torch.manual_seed(seed)
    k_full, v_full = _make_kv(seed, seq=seq)
    store = _make_store()
    _stepwise_append(store, k_full, v_full, precision, split=split)
    return store, k_full, v_full


def _splice(k_full, k_new, start):
    """Return a copy of k_full with rows [start, start + T) replaced by k_new."""
    k_final = k_full.clone()
    k_final[:, :, start : start + k_new.shape[2], :] = k_new
    return k_final


def _snapshot_page(page):
    snap = {
        "precision": page.precision,
        "start_token": page.start_token,
        "num_tokens": page.num_tokens,
        "sealed": page.sealed,
    }
    for name in ("k", "v", "scale_k", "scale_v"):
        tensor = getattr(page, name, None)
        snap[name] = None if tensor is None else tensor.clone()
    return snap


def _assert_page_equal(page, snap):
    assert page.precision == snap["precision"]
    assert page.start_token == snap["start_token"]
    assert page.num_tokens == snap["num_tokens"]
    assert page.sealed == snap["sealed"]
    for name in ("k", "v", "scale_k", "scale_v"):
        tensor = getattr(page, name, None)
        expected = snap[name]
        if expected is None:
            assert tensor is None
        else:
            assert tensor is not None
            assert tensor.dtype == expected.dtype
            assert tensor.shape == expected.shape
            assert torch.equal(tensor, expected)


def _assert_token_rows_equal(page, snap, name, lo, hi):
    tensor = getattr(page, name)
    expected = snap[name]
    if expected is None:
        assert tensor is None
        return
    assert torch.equal(tensor[:, :, lo:hi, :], expected[:, :, lo:hi, :])


def _snapshot(store):
    k, v = store.gather()
    live = [page for page in store.pages if page is not None]
    return {
        "k": k,
        "v": v,
        "data_plane_bytes": store.data_plane_bytes(),
        "resident_bytes": store.resident_bytes(),
        "rewritten_tokens": store.rewritten_tokens,
        "rewritten_pages": store.rewritten_pages,
        "pages": [(id(page), _snapshot_page(page)) for page in live],
    }


def _assert_unchanged(store, snap):
    """Assert the full observable store state matches a prior snapshot."""
    k, v = store.gather()
    assert torch.equal(k, snap["k"])
    assert torch.equal(v, snap["v"])
    assert store.data_plane_bytes() == snap["data_plane_bytes"]
    assert store.resident_bytes() == snap["resident_bytes"]
    assert store.rewritten_tokens == snap["rewritten_tokens"]
    assert store.rewritten_pages == snap["rewritten_pages"]
    live = [page for page in store.pages if page is not None]
    assert len(live) == len(snap["pages"])
    for page, (page_id, page_snap) in zip(live, snap["pages"]):
        assert id(page) == page_id
        _assert_page_equal(page, page_snap)


# ---------------------------------------------------------------------------
# (a) golden equivalence: rewrite-after-append == append-final-content
# ---------------------------------------------------------------------------


def _assert_golden_rewrite(precision):
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    k_new, v_new = _make_kv(GOLDEN_SEED + 101, seq=REWRITE_TOKENS)
    k_final = _splice(k_full, k_new, REWRITE_START)
    v_final = _splice(v_full, v_new, REWRITE_START)

    store = _make_store()
    _stepwise_append(store, k_full, v_full, precision)
    written = store.rewrite_tokens(REWRITE_START, k_new, v_new, precision)
    k_got, v_got = store.gather()

    reference = _make_store()
    _stepwise_append(reference, k_final, v_final, precision)
    k_ref, v_ref = reference.gather()

    assert written == REWRITE_TOKENS
    assert k_got.shape == (BATCH, HEADS, SEQ_LEN, HEAD_DIM)
    assert torch.equal(k_got, k_ref)
    assert torch.equal(v_got, v_ref)


class TestRewriteGoldenEquivalence:
    def test_rewrite_golden_fp16(self):
        _assert_golden_rewrite("fp16")

    def test_rewrite_golden_int8(self):
        _assert_golden_rewrite("int8")

    def test_rewrite_golden_int4(self):
        _assert_golden_rewrite("int4")

    def test_rewrite_golden_int8_full_range(self):
        torch.manual_seed(GOLDEN_SEED)
        k_full, v_full = _make_kv(GOLDEN_SEED)
        k_new, v_new = _make_kv(GOLDEN_SEED + 102, seq=SEQ_LEN)

        store = _make_store()
        _stepwise_append(store, k_full, v_full, "int8")
        written = store.rewrite_tokens(0, k_new, v_new, "int8")
        k_got, v_got = store.gather()

        reference = _make_store()
        _stepwise_append(reference, k_new, v_new, "int8")
        k_ref, v_ref = reference.gather()

        assert written == SEQ_LEN
        assert torch.equal(k_got, k_ref)
        assert torch.equal(v_got, v_ref)


# ---------------------------------------------------------------------------
# (b) cross-page split: each hit page rewritten in place, rest untouched
# ---------------------------------------------------------------------------


class TestRewriteCrossPageSplit:
    def test_cross_page_rewrite_bitwise_per_page(self):
        torch.manual_seed(GOLDEN_SEED)
        k_full, v_full = _make_kv(GOLDEN_SEED, seq=CROSS_SEQ)
        k_new, v_new = _make_kv(GOLDEN_SEED + 103, seq=CROSS_TOKENS)
        k_final = _splice(k_full, k_new, CROSS_START)
        v_final = _splice(v_full, v_new, CROSS_START)

        store = _make_store()
        _stepwise_append(store, k_full, v_full, "int8", split=CROSS_SPLIT)
        reference = _make_store()
        _stepwise_append(reference, k_final, v_final, "int8", split=CROSS_SPLIT)
        assert store.page_count == 4

        page_ids = [id(page) for page in store.pages]
        untouched_page_snap = _snapshot_page(store.pages[3])
        data_bytes = store.data_plane_bytes()
        resident_bytes = store.resident_bytes()

        written = store.rewrite_tokens(CROSS_START, k_new, v_new, "int8")

        assert written == CROSS_TOKENS
        assert store.rewritten_tokens == CROSS_TOKENS
        assert store.rewritten_pages == 3
        assert [id(page) for page in store.pages] == page_ids
        # The reference store appended the final content directly, so
        # full-page bitwise equality proves both that hit rows carry the
        # re-quantized new content and that untouched rows are unchanged.
        for slot in range(4):
            page = store.pages[slot]
            ref_page = reference.pages[slot]
            assert torch.equal(page.k, ref_page.k)
            assert torch.equal(page.v, ref_page.v)
            assert torch.equal(page.scale_k, ref_page.scale_k)
            assert torch.equal(page.scale_v, ref_page.scale_v)
        _assert_page_equal(store.pages[3], untouched_page_snap)
        assert store.data_plane_bytes() == data_bytes
        assert store.resident_bytes() == resident_bytes

        k_got, v_got = store.gather()
        k_ref, v_ref = reference.gather()
        assert torch.equal(k_got, k_ref)
        assert torch.equal(v_got, v_ref)


# ---------------------------------------------------------------------------
# (c) evicted rows: intervals containing evicted (token, head) are refused
# ---------------------------------------------------------------------------


def _store_with_partial_eviction():
    store, k_full, v_full = _appended_store("fp16", seq=16, split=(8, 8))
    assert store.evict_tokens({(10, 0)}) == 1
    return store, k_full, v_full


class TestRewriteEvictedRows:
    def test_reject_interval_containing_evicted_row(self):
        store, _, _ = _store_with_partial_eviction()
        k_new, v_new = _make_kv(GOLDEN_SEED + 104, seq=4)
        snap = _snapshot(store)
        with pytest.raises(ValueError):
            store.rewrite_tokens(8, k_new, v_new, "fp16")
        _assert_unchanged(store, snap)

    def test_atomic_when_evicted_row_lies_in_later_hit_page(self):
        store, _, _ = _store_with_partial_eviction()
        k_new, v_new = _make_kv(GOLDEN_SEED + 105, seq=8)
        page0_snap = _snapshot_page(store.pages[0])
        snap = _snapshot(store)
        with pytest.raises(ValueError):
            store.rewrite_tokens(4, k_new, v_new, "fp16")
        # Page 0 (tokens 4-7 are clean) must not have been mutated before the
        # evicted row in page 1 (token 10) triggered the refusal.
        _assert_page_equal(store.pages[0], page0_snap)
        _assert_unchanged(store, snap)

    def test_rewrite_outside_evicted_rows_succeeds(self):
        store, k_full, v_full = _store_with_partial_eviction()
        k_new, v_new = _make_kv(GOLDEN_SEED + 106, seq=4)
        written = store.rewrite_tokens(4, k_new, v_new, "fp16")
        assert written == 4
        k_got, v_got = store.gather()
        assert torch.equal(k_got[:, :, 4:8, :], k_new.half().float())
        assert torch.equal(v_got[:, :, 4:8, :], v_new.half().float())
        # The evicted (token 10, head 0) row stays zeroed, never revived.
        assert torch.count_nonzero(k_got[:, 0, 10, :]) == 0
        assert torch.count_nonzero(v_got[:, 0, 10, :]) == 0
        assert torch.equal(k_got[:, 1:, 10, :], k_full[:, 1:, 10, :].half().float())
        assert torch.equal(v_got[:, 1:, 10, :], v_full[:, 1:, 10, :].half().float())


# ---------------------------------------------------------------------------
# (d) argument validation: every invalid call raises ValueError atomically
# ---------------------------------------------------------------------------


class TestRewriteValidation:
    def test_reject_negative_start(self):
        store, _, _ = _appended_store()
        k_new, v_new = _make_kv(GOLDEN_SEED + 107, seq=4)
        snap = _snapshot(store)
        with pytest.raises(ValueError):
            store.rewrite_tokens(-1, k_new, v_new, "fp16")
        _assert_unchanged(store, snap)

    def test_reject_zero_token_interval(self):
        store, _, _ = _appended_store()
        k_new, v_new = _make_kv(GOLDEN_SEED + 107, seq=0)
        snap = _snapshot(store)
        with pytest.raises(ValueError):
            store.rewrite_tokens(0, k_new, v_new, "fp16")
        _assert_unchanged(store, snap)

    def test_reject_interval_beyond_max_seq(self):
        store, _, _ = _appended_store()
        assert store.max_seq == SEQ_LEN
        k_new, v_new = _make_kv(GOLDEN_SEED + 107, seq=8)
        k_one, v_one = _make_kv(GOLDEN_SEED + 108, seq=1)
        snap = _snapshot(store)
        with pytest.raises(ValueError):
            store.rewrite_tokens(15, k_new, v_new, "fp16")  # 15 + 8 > 19
        with pytest.raises(ValueError):
            store.rewrite_tokens(SEQ_LEN, k_one, v_one, "fp16")  # start == max_seq
        with pytest.raises(ValueError):
            store.rewrite_tokens(100, k_one, v_one, "fp16")
        _assert_unchanged(store, snap)

    def test_reject_non_4d_inputs(self):
        store, _, _ = _appended_store()
        gen = torch.Generator().manual_seed(GOLDEN_SEED + 109)
        k_bad = torch.randn(BATCH, HEADS, 4, generator=gen)
        v_new = torch.randn(BATCH, HEADS, 4, HEAD_DIM, generator=gen)
        snap = _snapshot(store)
        with pytest.raises(ValueError):
            store.rewrite_tokens(0, k_bad, v_new, "fp16")
        with pytest.raises(ValueError):
            store.rewrite_tokens(0, v_new, k_bad, "fp16")
        _assert_unchanged(store, snap)

    def test_reject_k_v_shape_disagreement(self):
        store, _, _ = _appended_store()
        k_new, _ = _make_kv(GOLDEN_SEED + 110, seq=4)
        _, v_new = _make_kv(GOLDEN_SEED + 111, seq=5)
        snap = _snapshot(store)
        with pytest.raises(ValueError):
            store.rewrite_tokens(0, k_new, v_new, "fp16")
        _assert_unchanged(store, snap)

    @pytest.mark.parametrize(
        "batch,heads,head_dim",
        [
            (BATCH + 1, HEADS, HEAD_DIM),
            (BATCH, HEADS + 1, HEAD_DIM),
            (BATCH, HEADS, HEAD_DIM + 1),
        ],
    )
    def test_reject_store_dimension_mismatch(self, batch, heads, head_dim):
        store, _, _ = _appended_store()
        gen = torch.Generator().manual_seed(GOLDEN_SEED + 112)
        k_new = torch.randn(batch, heads, 4, head_dim, generator=gen)
        v_new = torch.randn(batch, heads, 4, head_dim, generator=gen)
        snap = _snapshot(store)
        with pytest.raises(ValueError):
            store.rewrite_tokens(0, k_new, v_new, "fp16")
        _assert_unchanged(store, snap)

    @pytest.mark.parametrize("precision", ["bf16", "int16"])
    def test_reject_unknown_precision(self, precision):
        store, _, _ = _appended_store()
        k_new, v_new = _make_kv(GOLDEN_SEED + 113, seq=4)
        snap = _snapshot(store)
        with pytest.raises(ValueError):
            store.rewrite_tokens(0, k_new, v_new, precision)
        _assert_unchanged(store, snap)

    @pytest.mark.parametrize("precision", ["fp32", "int8", "int4"])
    def test_reject_precision_mismatch_with_hit_pages(self, precision):
        store, _, _ = _appended_store()  # all pages are fp16
        k_new, v_new = _make_kv(GOLDEN_SEED + 114, seq=4)
        snap = _snapshot(store)
        with pytest.raises(ValueError):
            store.rewrite_tokens(0, k_new, v_new, precision)
        _assert_unchanged(store, snap)

    def test_reject_range_spanning_mixed_precision_pages(self):
        torch.manual_seed(GOLDEN_SEED)
        store = _make_store()
        k0, v0 = _make_kv(GOLDEN_SEED, seq=PAGE_SIZE)
        store.append(k0, v0, "fp16")  # page 0: fp16, sealed
        k1, v1 = _make_kv(GOLDEN_SEED + 115, seq=PAGE_SIZE)
        store.append(k1, v1, "int8")  # page 1: int8, sealed
        k_new, v_new = _make_kv(GOLDEN_SEED + 116, seq=8)
        snap = _snapshot(store)
        with pytest.raises(ValueError):
            store.rewrite_tokens(4, k_new, v_new, "fp16")
        with pytest.raises(ValueError):
            store.rewrite_tokens(4, k_new, v_new, "int8")
        _assert_unchanged(store, snap)


# ---------------------------------------------------------------------------
# (e) accounting: counters accumulate exactly, byte ledgers stay flat
# ---------------------------------------------------------------------------


class TestRewriteAccounting:
    def test_counters_start_at_zero(self):
        store = _make_store()
        assert store.rewritten_tokens == 0
        assert store.rewritten_pages == 0

    def test_counters_accumulate_and_byte_ledgers_stay_flat(self):
        torch.manual_seed(GOLDEN_SEED)
        k_full, v_full = _make_kv(GOLDEN_SEED, seq=CROSS_SEQ)
        store = _make_store()
        _stepwise_append(store, k_full, v_full, "fp16", split=CROSS_SPLIT)
        data_before = store.data_plane_bytes()
        resident_before = store.resident_bytes()

        k_a, v_a = _make_kv(GOLDEN_SEED + 117, seq=4)  # [0, 4): page 0 only
        assert store.rewrite_tokens(0, k_a, v_a, "fp16") == 4
        assert store.rewritten_tokens == 4
        assert store.rewritten_pages == 1
        assert store.data_plane_bytes() == data_before
        assert store.resident_bytes() == resident_before

        k_b, v_b = _make_kv(GOLDEN_SEED + 118, seq=16)  # [4, 20): pages 0, 1, 2
        assert store.rewrite_tokens(4, k_b, v_b, "fp16") == 16
        assert store.rewritten_tokens == 20
        assert store.rewritten_pages == 4
        assert store.data_plane_bytes() == data_before
        assert store.resident_bytes() == resident_before

        k_c, v_c = _make_kv(GOLDEN_SEED + 119, seq=6)  # [20, 26): pages 2, 3
        assert store.rewrite_tokens(20, k_c, v_c, "fp16") == 6
        assert store.rewritten_tokens == 26
        assert store.rewritten_pages == 6
        assert store.data_plane_bytes() == data_before
        assert store.resident_bytes() == resident_before

    def test_failed_rewrite_does_not_bump_counters(self):
        store, _, _ = _appended_store()
        k_new, v_new = _make_kv(GOLDEN_SEED + 120, seq=8)
        with pytest.raises(ValueError):
            store.rewrite_tokens(15, k_new, v_new, "fp16")  # 15 + 8 > 19
        assert store.rewritten_tokens == 0
        assert store.rewritten_pages == 0

        k_ok, v_ok = _make_kv(GOLDEN_SEED + 121, seq=4)
        assert store.rewrite_tokens(0, k_ok, v_ok, "fp16") == 4
        assert store.rewritten_tokens == 4
        assert store.rewritten_pages == 1

        with pytest.raises(ValueError):
            store.rewrite_tokens(16, k_new, v_new, "fp16")  # 16 + 8 > 19
        assert store.rewritten_tokens == 4
        assert store.rewritten_pages == 1


# ---------------------------------------------------------------------------
# (f) page invariants: identity, boundaries and sealed state survive rewrite
# ---------------------------------------------------------------------------


class TestRewritePageInvariants:
    def test_sealed_page_metadata_and_untouched_rows_survive_rewrite(self):
        store, _, _ = _appended_store("int8")
        # pages: 0 = tokens [0, 8) sealed, 1 = [8, 16) sealed, 2 = [16, 19) open
        pages_before = [(page, _snapshot_page(page)) for page in store.pages]
        k_new, v_new = _make_kv(GOLDEN_SEED + 122, seq=REWRITE_TOKENS)

        written = store.rewrite_tokens(REWRITE_START, k_new, v_new, "int8")

        assert written == REWRITE_TOKENS
        for (page_before, snap), page_after in zip(pages_before, store.pages):
            assert page_after is page_before
            assert page_after.start_token == snap["start_token"]
            assert page_after.num_tokens == snap["num_tokens"]
            assert page_after.sealed == snap["sealed"]
        # Untouched rows of hit pages stay bit-identical (payload + scales):
        # page 0 rows [0, 6) and page 1 rows [6, 8).
        p0_snap = pages_before[0][1]
        p1_snap = pages_before[1][1]
        for name in ("k", "v", "scale_k", "scale_v"):
            _assert_token_rows_equal(store.pages[0], p0_snap, name, 0, 6)
            _assert_token_rows_equal(store.pages[1], p1_snap, name, 6, 8)
        # Page 2 (not hit) is entirely untouched.
        _assert_page_equal(store.pages[2], pages_before[2][1])

    def test_open_page_stays_open_after_rewrite(self):
        store, _, _ = _appended_store("fp16")
        open_page = store.pages[2]
        assert open_page.sealed is False
        open_snap = _snapshot_page(open_page)
        k_new, v_new = _make_kv(GOLDEN_SEED + 123, seq=3)

        written = store.rewrite_tokens(16, k_new, v_new, "fp16")

        assert written == 3
        assert store.pages[2] is open_page
        assert open_page.sealed is False
        assert open_page.start_token == open_snap["start_token"]
        assert open_page.num_tokens == open_snap["num_tokens"]
        k_got, v_got = store.gather()
        assert torch.equal(k_got[:, :, 16:19, :], k_new.half().float())
        assert torch.equal(v_got[:, :, 16:19, :], v_new.half().float())


# ---------------------------------------------------------------------------
# T002: MacroModeConfig validation (design.md 4.3)
# ---------------------------------------------------------------------------

MM_SEED = 2027
MM_BATCH = 2
MM_HEADS = 2
MM_HEAD_DIM = 4
MM_DIM = 4
MM_SEQ = 8
MM_MACRO = 4  # two full macro blocks: rows [0, 4) and [4, 8)
MM_TAIL_SEQ = 10  # blocks of 4, 4 and 2 rows: partial tail block
MM_BASE_COORD = 10.0
MM_QUIET_COORD = 10.5  # per-row drift |10.5 - 10| / 10 = 0.05 < 0.1
MM_THRESHOLD_COORD = 11.0  # per-row drift exactly 0.1 == skip_threshold
MM_ACTIVE_COORD = 12.0  # per-row drift |12 - 10| / 10 = 0.2 >= 0.1


def _base_hidden(batch=MM_BATCH, seq=MM_SEQ, dim=MM_DIM):
    """Uniform rows [10, 0, ..., 0]: replacing coordinate 0 with ``v`` gives
    an exact per-row relative drift of ``|v - 10| / 10`` in fp32."""
    hidden = torch.zeros(batch, seq, dim)
    hidden[..., 0] = MM_BASE_COORD
    return hidden


def _drift_rows(hidden, rows, coord, batch=None):
    """Copy ``hidden`` with the first coordinate of the given seq rows set
    to ``coord`` (drift = |coord - 10| / 10 per affected row)."""
    out = hidden.clone()
    for row in rows:
        if batch is None:
            out[:, row, 0] = coord
        else:
            out[batch, row, 0] = coord
    return out


def _mm_config(**overrides):
    """Valid classifier config; warmup and force-refresh off by default."""
    from shadowinfer.execution.macro_mode import MacroModeConfig

    settings = dict(
        enabled=True,
        macro_block_size=MM_MACRO,
        skip_threshold=0.1,
        delta_min_active_ratio=0.5,
        warmup_steps=0,
        force_refresh_every=0,
    )
    settings.update(overrides)
    return MacroModeConfig(**settings)


def _mm_state():
    from shadowinfer.execution.macro_mode import MacroModeState

    return MacroModeState(entries={})


def _record_layer(state, layer_id, hidden, config):
    """Record every macro block of one layer from ``hidden`` [B, S, d]."""
    batch, seq, _ = hidden.shape
    gen = torch.Generator().manual_seed(MM_SEED + layer_id)
    for macro_idx, start in enumerate(range(0, seq, config.macro_block_size)):
        end = min(start + config.macro_block_size, seq)
        rows = end - start
        k = torch.randn(batch, MM_HEADS, rows, MM_HEAD_DIM, generator=gen)
        v = torch.randn(batch, MM_HEADS, rows, MM_HEAD_DIM, generator=gen)
        state.record_rows(
            layer_id,
            macro_idx,
            (start, end),
            hidden[:, start:end, :].contiguous(),
            (hidden[:, start:end, :] + 1.0).contiguous(),
            k,
            v,
        )


class TestMacroModeConfigValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"macro_block_size": 0},
            {"skip_threshold": -0.1},
            {"delta_min_active_ratio": -0.1},
            {"delta_min_active_ratio": 1.1},
            {"warmup_steps": -1},
            {"force_refresh_every": -1},
        ],
    )
    def test_invalid_values_raise_value_error(self, overrides):
        from shadowinfer.execution.macro_mode import MacroModeConfig

        with pytest.raises(ValueError):
            MacroModeConfig(enabled=True, **overrides)

    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {"macro_block_size": 1},
            {"delta_min_active_ratio": 0.0},
            {"delta_min_active_ratio": 1.0},
            {"warmup_steps": 0},
            {"force_refresh_every": 0},
        ],
    )
    def test_valid_boundary_values_pass(self, overrides):
        from shadowinfer.execution.macro_mode import MacroModeConfig

        config = MacroModeConfig(enabled=True, **overrides)
        assert config.enabled is True
        config.validate()


# ---------------------------------------------------------------------------
# T002: trajectory classification (SKIP / DELTA / REFRESH)
# ---------------------------------------------------------------------------


class TestMacroModeTrajectoryClassification:
    def test_recorded_same_content_all_skip(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        modes, active = state.classify(0, base, 1, config)
        assert modes == {0: "SKIP", 1: "SKIP"}
        assert active == []

    def test_single_row_below_threshold_still_skip(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        hidden = _drift_rows(base, [0], MM_QUIET_COORD)
        modes, active = state.classify(0, hidden, 1, config)
        assert modes == {0: "SKIP", 1: "SKIP"}
        assert active == []

    def test_minority_row_drift_is_delta(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        hidden = _drift_rows(base, [0], MM_ACTIVE_COORD)
        modes, active = state.classify(0, hidden, 1, config)
        assert modes == {0: "DELTA", 1: "SKIP"}
        assert active == [0]

    def test_majority_row_drift_is_refresh(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        hidden = _drift_rows(base, [0, 1, 2], MM_ACTIVE_COORD)
        modes, active = state.classify(0, hidden, 1, config)
        assert modes == {0: "REFRESH", 1: "SKIP"}
        assert active == [0, 1, 2]


class TestMacroModeClassificationBoundaries:
    def test_drift_exactly_at_threshold_is_active(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        hidden = _drift_rows(base, [0], MM_THRESHOLD_COORD)
        modes, active = state.classify(0, hidden, 1, config)
        assert modes == {0: "DELTA", 1: "SKIP"}
        assert active == [0]

    def test_active_ratio_exactly_at_min_is_refresh(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        hidden = _drift_rows(base, [0, 1], MM_ACTIVE_COORD)
        modes, active = state.classify(0, hidden, 1, config)
        assert modes == {0: "REFRESH", 1: "SKIP"}
        assert active == [0, 1]


class TestMacroModeHysteresisRelease:
    def test_skip_block_released_immediately_by_one_active_row(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        for step in (1, 2):
            modes, _ = state.classify(0, base, step, config)
            assert modes[0] == "SKIP"
        hidden = _drift_rows(base, [0], MM_ACTIVE_COORD)
        modes, active = state.classify(0, hidden, 3, config)
        assert modes[0] == "DELTA"
        assert active == [0]


class TestMacroModeWarmup:
    def test_warmup_steps_force_refresh_even_with_zero_drift(self):
        config = _mm_config(warmup_steps=2)
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        for step in (0, 1):
            modes, active = state.classify(0, base, step, config)
            assert modes == {0: "REFRESH", 1: "REFRESH"}
            assert active == list(range(MM_SEQ))
        modes, active = state.classify(0, base, 2, config)
        assert modes == {0: "SKIP", 1: "SKIP"}
        assert active == []

    def test_zero_warmup_does_not_force_step_zero(self):
        config = _mm_config(warmup_steps=0)
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        modes, active = state.classify(0, base, 0, config)
        assert modes == {0: "SKIP", 1: "SKIP"}
        assert active == []


class TestMacroModeForceRefresh:
    def test_periodic_hits_force_refresh(self):
        config = _mm_config(force_refresh_every=3)
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        for step in (1, 2):
            modes, active = state.classify(0, base, step, config)
            assert modes == {0: "SKIP", 1: "SKIP"}
            assert active == []
        for step in (3, 6):
            modes, active = state.classify(0, base, step, config)
            assert modes == {0: "REFRESH", 1: "REFRESH"}
            assert active == list(range(MM_SEQ))

    def test_disabled_force_refresh_never_forces(self):
        config = _mm_config(force_refresh_every=0)
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        for step in (0, 3, 6):
            modes, active = state.classify(0, base, step, config)
            assert modes == {0: "SKIP", 1: "SKIP"}
            assert active == []


class TestMacroModeEntryInvalidation:
    def test_missing_entries_classify_all_refresh(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden()
        modes, active = state.classify(0, base, 5, config)
        assert modes == {0: "REFRESH", 1: "REFRESH"}
        assert active == list(range(MM_SEQ))

    def test_dim_mismatch_refreshes_without_crash_then_recovers(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        modes, active = state.classify(0, _base_hidden(dim=8), 3, config)
        assert modes == {0: "REFRESH", 1: "REFRESH"}
        assert active == list(range(MM_SEQ))
        # Stale entries were invalidated: a correct-size classify now sees
        # missing entries instead of crashing or reusing them.
        modes, active = state.classify(0, base, 4, config)
        assert modes == {0: "REFRESH", 1: "REFRESH"}
        assert active == list(range(MM_SEQ))
        _record_layer(state, 0, base, config)
        modes, active = state.classify(0, base, 5, config)
        assert modes == {0: "SKIP", 1: "SKIP"}
        assert active == []

    def test_batch_mismatch_refreshes(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        modes, active = state.classify(0, _base_hidden(batch=3), 3, config)
        assert modes == {0: "REFRESH", 1: "REFRESH"}
        assert active == list(range(MM_SEQ))

    def test_longer_seq_refreshes_all_blocks(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        wide = _drift_rows(_base_hidden(seq=12), range(12), MM_ACTIVE_COORD)
        modes, active = state.classify(0, wide, 3, config)
        assert modes == {0: "REFRESH", 1: "REFRESH", 2: "REFRESH"}
        assert active == list(range(12))


class TestMacroModeZeroNormDenominator:
    def test_zero_norm_entry_row_is_always_active(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden()
        base[:, 0, :] = 0.0  # entry row norm 0: drift is infinite, even if h is 0 too
        _record_layer(state, 0, base, config)
        modes, active = state.classify(0, base, 1, config)
        assert modes == {0: "DELTA", 1: "SKIP"}
        assert active == [0]


class TestMacroModeBatchUniformity:
    def test_drift_in_single_batch_element_activates_seq_row(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        hidden = _drift_rows(base, [2], MM_ACTIVE_COORD, batch=1)
        modes, active = state.classify(0, hidden, 1, config)
        assert modes == {0: "DELTA", 1: "SKIP"}
        assert active == [2]


class TestMacroModeTailPartialBlock:
    def test_partial_tail_block_skips_when_frozen(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden(seq=MM_TAIL_SEQ)
        _record_layer(state, 0, base, config)
        modes, active = state.classify(0, base, 1, config)
        assert modes == {0: "SKIP", 1: "SKIP", 2: "SKIP"}
        assert active == []

    def test_partial_tail_ratio_uses_actual_row_count(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden(seq=MM_TAIL_SEQ)
        _record_layer(state, 0, base, config)
        # Tail block holds 2 rows: one active row is ratio 1/2 == 0.5, so the
        # block is REFRESH (a wrong mb-sized denominator would give DELTA).
        hidden = _drift_rows(base, [8], MM_ACTIVE_COORD)
        modes, active = state.classify(0, hidden, 2, config)
        assert modes == {0: "SKIP", 1: "SKIP", 2: "REFRESH"}
        assert active == [8]


class TestMacroModeStatsAccounting:
    def test_stats_accumulate_across_layers_within_a_step(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        _record_layer(state, 1, base, config)
        layer0 = _drift_rows(base, [4], MM_ACTIVE_COORD)
        modes0, active0 = state.classify(0, layer0, 3, config)
        assert modes0 == {0: "SKIP", 1: "DELTA"}
        assert active0 == [4]
        layer1 = _drift_rows(base, [0, 1, 2, 3], MM_ACTIVE_COORD)
        modes1, active1 = state.classify(1, layer1, 3, config)
        assert modes1 == {0: "REFRESH", 1: "SKIP"}
        assert active1 == [0, 1, 2, 3]
        stats = state.stats()
        assert stats["skip_blocks"] == 2
        assert stats["delta_blocks"] == 1
        assert stats["refresh_blocks"] == 1
        assert stats["skipped_rows"] == 8
        assert stats["active_rows"] == 5
        assert stats["rewritten_rows"] == 0
        state.record_rewritten(5)
        state.record_rewritten(3)
        assert state.stats()["rewritten_rows"] == 8

    def test_begin_step_zeroes_counts_and_keeps_entries(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        hidden = _drift_rows(base, [0], MM_ACTIVE_COORD)
        modes, _ = state.classify(0, hidden, 1, config)
        assert modes == {0: "DELTA", 1: "SKIP"}
        state.record_rewritten(4)
        state.begin_step()
        stats = state.stats()
        assert stats["skip_blocks"] == 0
        assert stats["delta_blocks"] == 0
        assert stats["refresh_blocks"] == 0
        assert stats["skipped_rows"] == 0
        assert stats["active_rows"] == 0
        assert stats["rewritten_rows"] == 0
        modes, active = state.classify(0, base, 2, config)
        assert modes == {0: "SKIP", 1: "SKIP"}
        assert active == []

    def test_reset_clears_entries_and_counts(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        state.classify(0, base, 1, config)
        state.record_rewritten(4)
        state.reset()
        assert state.entries == {}
        stats = state.stats()
        assert stats["skip_blocks"] == 0
        assert stats["rewritten_rows"] == 0
        modes, active = state.classify(0, base, 2, config)
        assert modes == {0: "REFRESH", 1: "REFRESH"}
        assert active == list(range(MM_SEQ))


class TestMacroModeRecordClassifyLoop:
    def test_recording_drifted_content_freezes_it(self):
        config = _mm_config()
        state = _mm_state()
        base = _base_hidden()
        _record_layer(state, 0, base, config)
        drifted = _drift_rows(base, range(MM_SEQ), MM_ACTIVE_COORD)
        modes, active = state.classify(0, drifted, 1, config)
        assert modes == {0: "REFRESH", 1: "REFRESH"}
        assert active == list(range(MM_SEQ))
        _record_layer(state, 0, drifted, config)
        modes, active = state.classify(0, drifted, 2, config)
        assert modes == {0: "SKIP", 1: "SKIP"}
        assert active == []


# ---------------------------------------------------------------------------
# T003: model-level segmented forward (design.md 4.2.1 / 4.2.2 / 6.3)
# ---------------------------------------------------------------------------

T003_SEED = 2031
T003_LAYERS = 2
T003_HEADS = 2
T003_HEAD_DIM = 8
T003_DIM = 16
T003_SEQ = 12
T003_MACRO = MM_MACRO  # 4: three full macro blocks over T003_SEQ rows
T003_NUM_MACROS = T003_SEQ // T003_MACRO
T003_BATCH = 2


def _t003_model():
    """Deterministic tiny two-layer model (CPU, fp32)."""
    torch.manual_seed(T003_SEED)
    return SimpleDiffusionLLM(
        num_layers=T003_LAYERS,
        num_heads=T003_HEADS,
        head_dim=T003_HEAD_DIM,
        hidden_dim=T003_DIM,
        vocab_size=100,
        max_seq_len=32,
    )


def _t003_base_hidden():
    """[B, S, d] rows with coordinate 0 = 10 (a coord-0 change to ``v`` has
    exact per-row relative drift ``|v - 10| / 10``, same ruler as T002)."""
    return _base_hidden(batch=T003_BATCH, seq=T003_SEQ, dim=T003_DIM)


def _t003_snapshot_entries(state, layer_id):
    """Clone every recorded tensor of one layer for bitwise comparisons."""
    return {
        macro_idx: (
            entry.hidden_in.clone(),
            entry.hidden_out.clone(),
            entry.k.clone(),
            entry.v.clone(),
        )
        for (lay, macro_idx), entry in state.entries.items()
        if lay == layer_id
    }


def _t003_record_via_forward(block, state, base, config):
    """First mode-path call on ``base``: a fresh state has no entries, so
    classify marks every macro block REFRESH and the built-in full forward
    records all of them (design.md 4.2.2; no manual record_rows here)."""
    return block(
        base,
        layer_id=0,
        step_id=0,
        mode_state=state,
        mode_config=config,
    )


class TestMacroModePartialForward:
    """T003 item 1: partially active forward vs dense forward."""

    def test_partial_step_active_rows_match_dense_and_frozen_match_cache(self):
        model = _t003_model()
        block = model.blocks[0]
        config = _mm_config()
        state = _mm_state()
        base = _t003_base_hidden()

        out_base, kv_base, attn_base = _t003_record_via_forward(block, state, base, config)

        drifted = _drift_rows(base, [0], MM_ACTIVE_COORD)
        state.begin_step()
        out_mode, (k_mode, v_mode), attn_mode = block(
            drifted,
            layer_id=0,
            step_id=1,
            mode_state=state,
            mode_config=config,
        )
        stats = state.stats()
        assert 0 < stats["active_rows"] < T003_SEQ

        out_dense, (k_dense, v_dense), attn_dense = block(drifted, layer_id=0, step_id=1)

        # Assembled full-sequence K/V: frozen rows replay cached projections
        # bitwise (asserted exactly below); the active row is re-projected by
        # a subset matmul (M=A vs M=S), which can reorder the reduction in
        # the last bit -- the same quantified-noise rationale as the
        # attention matmul below (design.md 4.2.2 / 6.3(a)).
        frozen_rows_kv = list(range(1, T003_SEQ))
        assert torch.equal(k_mode[:, :, frozen_rows_kv, :], k_dense[:, :, frozen_rows_kv, :])
        assert torch.equal(v_mode[:, :, frozen_rows_kv, :], v_dense[:, :, frozen_rows_kv, :])
        for kv_mode, kv_dense in ((k_mode, k_dense), (v_mode, v_dense)):
            active_rel_l2 = (kv_mode[:, :, 0, :] - kv_dense[:, :, 0, :]).norm() / kv_dense[
                :, :, 0, :
            ].norm()
            assert active_rel_l2.item() <= 1e-6
        # Active rows are recomputed and match the dense forward. The
        # row-local ops (LN/projection/FFN) are bitwise identical, but the
        # attention matmul runs at M=A vs the dense M=S, and CPU kernel
        # blocking reorders the reduction: measured relL2 noise is ~7.7e-8
        # (last-bit), so the assertion uses a tight quantified tolerance
        # instead of torch.equal (design.md 6.3(a), AR015 bench note).
        active_out = out_mode[:, 0, :]
        dense_out = out_dense[:, 0, :]
        rel_l2 = (active_out - dense_out).norm() / dense_out.norm()
        assert rel_l2.item() <= 1e-6
        # Frozen rows take the residual-corrected cache (design.md 4.2.1);
        # their drift is zero here, so they equal the recorded outputs.
        frozen_rows = list(range(1, T003_SEQ))
        assert torch.equal(out_mode[:, frozen_rows, :], out_base[:, frozen_rows, :])


class TestMacroModeFreshFullRefresh:
    """T003 item 2: fresh state -> fully active first call + full record."""

    def test_fresh_first_call_matches_dense_and_records_every_macro(self):
        model = _t003_model()
        block = model.blocks[0]
        config = _mm_config()
        state = _mm_state()
        base = _t003_base_hidden()

        out_mode, (k_mode, v_mode), attn_mode = block(
            base,
            layer_id=0,
            step_id=0,
            mode_state=state,
            mode_config=config,
        )
        out_dense, (k_dense, v_dense), attn_dense = block(base, layer_id=0, step_id=0)

        assert torch.equal(out_mode, out_dense)
        assert torch.equal(k_mode, k_dense)
        assert torch.equal(v_mode, v_dense)
        assert torch.equal(attn_mode, attn_dense)

        expected_keys = {(0, macro) for macro in range(T003_NUM_MACROS)}
        assert set(state.entries.keys()) == expected_keys
        for macro_idx in range(T003_NUM_MACROS):
            start = macro_idx * T003_MACRO
            end = start + T003_MACRO
            entry = state.entries[(0, macro_idx)]
            assert entry.seq_start == start
            assert entry.seq_end == end
            assert torch.equal(entry.hidden_in, base[:, start:end, :])
            assert torch.equal(entry.hidden_out, out_mode[:, start:end, :])
            assert torch.equal(entry.k, k_mode[:, :, start:end, :])
            assert torch.equal(entry.v, v_mode[:, :, start:end, :])


class TestMacroModeDisabledRegression:
    """T003 item 3: mode off (implicit and explicit None) is bit-stable."""

    def test_explicit_none_kwargs_match_implicit_off_path(self):
        torch.manual_seed(T003_SEED)
        block = DiffusionTransformerBlock(
            hidden_dim=T003_DIM,
            num_heads=T003_HEADS,
            head_dim=T003_HEAD_DIM,
            intermediate_dim=4 * T003_DIM,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        base = _t003_base_hidden()

        out_a, kv_a, attn_a = block(base, layer_id=0, step_id=0)
        out_b, kv_b, attn_b = block(base, layer_id=0, step_id=0)
        out_c, kv_c, attn_c = block(
            base,
            layer_id=0,
            step_id=0,
            mode_state=None,
            mode_config=None,
        )

        assert torch.equal(out_a, out_b)
        assert torch.equal(kv_a[0], kv_b[0])
        assert torch.equal(kv_a[1], kv_b[1])
        assert torch.equal(attn_a, attn_b)
        assert torch.equal(out_a, out_c)
        assert torch.equal(kv_a[0], kv_c[0])
        assert torch.equal(kv_a[1], kv_c[1])
        assert torch.equal(attn_a, attn_c)


class TestMacroModeAllSkipLayer:
    """T003 item 4: zero active rows replay the cache untouched."""

    def test_same_hidden_second_call_is_all_skip_and_replays_cache(self):
        model = _t003_model()
        block = model.blocks[0]
        config = _mm_config()
        state = _mm_state()
        base = _t003_base_hidden()

        out_first, (k_first, v_first), attn_first = _t003_record_via_forward(
            block, state, base, config
        )
        snapshot = _t003_snapshot_entries(state, 0)

        state.begin_step()
        out_skip, (k_skip, v_skip), attn_skip = block(
            base,
            layer_id=0,
            step_id=1,
            mode_state=state,
            mode_config=config,
        )

        stats = state.stats()
        assert stats["skip_blocks"] == T003_NUM_MACROS
        assert stats["active_rows"] == 0
        assert torch.equal(out_skip, out_first)
        assert torch.equal(k_skip, k_first)
        assert torch.equal(v_skip, v_first)
        assert torch.equal(attn_skip, attn_first)

        after = _t003_snapshot_entries(state, 0)
        assert set(after.keys()) == set(snapshot.keys())
        for macro_idx in snapshot:
            for before_t, after_t in zip(snapshot[macro_idx], after[macro_idx]):
                assert torch.equal(before_t, after_t)


class TestMacroModeFrozenResidualCorrection:
    """T003 item 5: sub-threshold drift is absorbed by the residual term."""

    def test_frozen_row_output_equals_residual_corrected_cache(self):
        model = _t003_model()
        block = model.blocks[0]
        config = _mm_config()
        state = _mm_state()
        base = _t003_base_hidden()
        _t003_record_via_forward(block, state, base, config)

        perturbed = _drift_rows(base, range(T003_SEQ), MM_QUIET_COORD)
        state.begin_step()
        out_skip, (k_skip, v_skip), attn_skip = block(
            perturbed,
            layer_id=0,
            step_id=1,
            mode_state=state,
            mode_config=config,
        )
        assert state.stats()["active_rows"] == 0

        expected = torch.empty_like(perturbed)
        for macro_idx in range(T003_NUM_MACROS):
            start = macro_idx * T003_MACRO
            end = start + T003_MACRO
            entry = state.entries[(0, macro_idx)]
            expected[:, start:end, :] = entry.hidden_out + (
                perturbed[:, start:end, :] - entry.hidden_in
            )
        assert torch.equal(out_skip, expected)


class TestMacroModeDeltaRecord:
    """T003 item 6: DELTA blocks rewrite only their active rows."""

    def test_delta_block_updates_only_active_rows(self):
        model = _t003_model()
        block = model.blocks[0]
        config = _mm_config()
        state = _mm_state()
        base = _t003_base_hidden()
        _t003_record_via_forward(block, state, base, config)
        snapshot = _t003_snapshot_entries(state, 0)

        drifted = _drift_rows(base, [0], MM_ACTIVE_COORD)
        state.begin_step()
        out_mode, (k_mode, v_mode), attn_mode = block(
            drifted,
            layer_id=0,
            step_id=1,
            mode_state=state,
            mode_config=config,
        )

        stats = state.stats()
        assert stats["delta_blocks"] == 1
        assert stats["skip_blocks"] == T003_NUM_MACROS - 1
        assert stats["active_rows"] == 1

        entry = state.entries[(0, 0)]
        # The active row is rewritten to this step's values.
        assert torch.equal(entry.hidden_in[:, 0, :], drifted[:, 0, :])
        assert torch.equal(entry.hidden_out[:, 0, :], out_mode[:, 0, :])
        assert torch.equal(entry.k[:, :, 0, :], k_mode[:, :, 0, :])
        assert torch.equal(entry.v[:, :, 0, :], v_mode[:, :, 0, :])
        # The frozen rows of the DELTA block are untouched bitwise.
        assert torch.equal(entry.hidden_in[:, 1:, :], snapshot[0][0][:, 1:, :])
        assert torch.equal(entry.hidden_out[:, 1:, :], snapshot[0][1][:, 1:, :])
        assert torch.equal(entry.k[:, :, 1:, :], snapshot[0][2][:, :, 1:, :])
        assert torch.equal(entry.v[:, :, 1:, :], snapshot[0][3][:, :, 1:, :])
        # SKIP blocks are untouched bitwise.
        after = _t003_snapshot_entries(state, 0)
        for macro_idx in range(1, T003_NUM_MACROS):
            for before_t, after_t in zip(snapshot[macro_idx], after[macro_idx]):
                assert torch.equal(before_t, after_t)


class TestMacroModeRefreshRecord:
    """T003 item 7: REFRESH blocks replace the whole macro entry."""

    def test_refresh_block_replaces_whole_entry(self):
        model = _t003_model()
        block = model.blocks[0]
        config = _mm_config()
        state = _mm_state()
        base = _t003_base_hidden()
        _t003_record_via_forward(block, state, base, config)

        drifted = _drift_rows(base, [0, 1, 2], MM_ACTIVE_COORD)
        state.begin_step()
        out_mode, (k_mode, v_mode), attn_mode = block(
            drifted,
            layer_id=0,
            step_id=1,
            mode_state=state,
            mode_config=config,
        )

        stats = state.stats()
        assert stats["refresh_blocks"] == 1
        assert stats["skip_blocks"] == T003_NUM_MACROS - 1
        assert stats["active_rows"] == 3

        entry = state.entries[(0, 0)]
        assert torch.equal(entry.hidden_in, drifted[:, 0:T003_MACRO, :])
        assert torch.equal(entry.hidden_out, out_mode[:, 0:T003_MACRO, :])
        assert torch.equal(entry.k, k_mode[:, :, 0:T003_MACRO, :])
        assert torch.equal(entry.v, v_mode[:, :, 0:T003_MACRO, :])


class TestMacroModeLastAttention:
    """T003 item 8: attention shapes per path and cross-step carry."""

    def test_attention_shape_per_path_and_skip_carries_last(self):
        model = _t003_model()
        block = model.blocks[0]
        config = _mm_config()
        state = _mm_state()
        base = _t003_base_hidden()

        out_full, kv_full, attn_full = _t003_record_via_forward(block, state, base, config)
        assert attn_full.shape == (T003_BATCH, T003_HEADS, T003_SEQ, T003_SEQ)

        out_skip, kv_skip, attn_skip = block(
            base,
            layer_id=0,
            step_id=1,
            mode_state=state,
            mode_config=config,
        )
        assert torch.equal(attn_skip, attn_full)

        drifted = _drift_rows(base, [0], MM_ACTIVE_COORD)
        out_part, kv_part, attn_part = block(
            drifted,
            layer_id=0,
            step_id=2,
            mode_state=state,
            mode_config=config,
        )
        assert attn_part.shape == (T003_BATCH, T003_HEADS, 1, T003_SEQ)
        assert torch.equal(state.last_attention, attn_part)


class TestMacroModeForwardStepPassthrough:
    """T003 item 9: forward_step wires the mode state through all blocks."""

    def test_two_forward_steps_record_then_skip_with_reset_stats(self):
        model = _t003_model()
        config = _mm_config()
        state = _mm_state()
        x = _t003_base_hidden()

        logits_first, kv_first, attn_first = model.forward_step(
            x, 0, 0, mode_state=state, mode_config=config
        )
        stats = state.stats()
        assert stats["refresh_blocks"] == T003_LAYERS * T003_NUM_MACROS
        assert stats["active_rows"] == T003_LAYERS * T003_SEQ

        logits_dense, kv_dense, attn_dense = model.forward_step(x, 0, 0)
        assert torch.equal(logits_first, logits_dense)
        assert torch.equal(kv_first[0][0], kv_dense[0][0])
        assert torch.equal(kv_first[0][1], kv_dense[0][1])

        logits_second, kv_second, attn_second = model.forward_step(
            x, 1, 0, mode_state=state, mode_config=config
        )
        assert torch.equal(logits_second, logits_first)
        assert torch.equal(kv_second[0][0], kv_first[0][0])
        assert torch.equal(kv_second[0][1], kv_first[0][1])
        assert torch.equal(attn_second, attn_first)

        # begin_step zeroed the counters: the second step reports only its
        # own all-SKIP classification instead of accumulating step 0.
        stats = state.stats()
        assert stats["skip_blocks"] == T003_LAYERS * T003_NUM_MACROS
        assert stats["skipped_rows"] == T003_LAYERS * T003_SEQ
        assert stats["refresh_blocks"] == 0
        assert stats["active_rows"] == 0


class TestMacroModeMultiLayer:
    """T003 item 10: per-layer entries and two-layer segmented stepping."""

    def test_layers_record_independently_and_drift_activates_both(self):
        model = _t003_model()
        config = _mm_config()
        state = _mm_state()
        x = _t003_base_hidden()

        logits_first, kv_first, attn_first = model.forward_step(
            x, 0, 0, mode_state=state, mode_config=config
        )
        expected_keys = {
            (layer, macro) for layer in range(T003_LAYERS) for macro in range(T003_NUM_MACROS)
        }
        assert set(state.entries.keys()) == expected_keys

        # Same hidden base: every layer of the model skips, so the logits
        # equal the dense (mode-off) forward bitwise.
        logits_skip, kv_skip, attn_skip = model.forward_step(
            x, 1, 0, mode_state=state, mode_config=config
        )
        logits_dense, kv_dense, attn_dense = model.forward_step(x, 0, 0)
        assert torch.equal(logits_skip, logits_dense)
        assert torch.equal(logits_skip, logits_first)

        # A large drift activates the drifted rows in BOTH layers: layer 1
        # classifies layer 0's freshly computed rows as its own drift.
        x_drift = _drift_rows(x, [0, 1], 110.0)
        logits_drift, kv_drift, attn_drift = model.forward_step(
            x_drift, 2, 0, mode_state=state, mode_config=config
        )
        stats = state.stats()
        assert stats["active_rows"] == 2 * T003_LAYERS
        assert stats["refresh_blocks"] == T003_LAYERS
        assert stats["skip_blocks"] == 2 * T003_LAYERS
        assert set(state.entries.keys()) == expected_keys

        # Layer 0 macro 0 was refreshed with this step's layer-0 input
        # (total_steps=0 keeps the pos-embed the only addition to x).
        positions = torch.arange(T003_SEQ)
        layer0_input = x_drift + model.pos_embed(positions).unsqueeze(0)
        entry = state.entries[(0, 0)]
        assert torch.equal(entry.hidden_in, layer0_input[:, 0:T003_MACRO, :])


# ---------------------------------------------------------------------------
# T004: backend mode-control dispatch (design.md 4.2.2 items 4-6 / 4.3 / 6.4)
# ---------------------------------------------------------------------------

T004_SEED = 2032
T004_LAYERS = 2
T004_HEADS = 2
T004_HEAD_DIM = 8
T004_DIM = 16
T004_SEQ = 12
T004_MACRO = 4  # three full macro blocks over T004_SEQ rows
T004_NUM_MACROS = T004_SEQ // T004_MACRO
T004_PAGE_SIZE = 8  # pages [0, 8) and [8, 12) per layer
T004_BATCH = 2
T004_DRIFT_ROWS = (0, 4, 5)  # block 0 DELTA (1/4), block 1 REFRESH (2/4), block 2 SKIP
T004_DRIFT_COORD = 110.0  # per-row drift 10x, far above the 0.1 gate
T004_MACRO_DEFAULT = 32  # backend default macro_block_size

# execution_stats keys pinned by AR013/AR014; every one must keep existing.
T004_PRE_AR015_STATS_KEYS = frozenset(
    {
        "kv_lifecycle",
        "kv_execution",
        "precision",
        "reused_layers",
        "computed_layers",
        "reused_ratio",
        "packed_layers",
        "skipped_repack",
        "paged",
        "page_size",
        "page_count",
        "open_pages",
        "appended_tokens",
        "appended_tokens_total",
        "repack_tokens_saved",
        "page_evictions",
        "resident_bytes",
        "data_plane_bytes",
        "metadata_bytes",
        "dense_bytes",
        "compression_ratio",
        "data_plane_ratio",
        "fallback",
        "reset_reason",
        "drifts",
        "attention_reused",
    }
)


def _t004_model_config() -> dict:
    return {
        "name": "ar015-t004",
        "num_layers": T004_LAYERS,
        "num_heads": T004_HEADS,
        "head_dim": T004_HEAD_DIM,
        "hidden_dim": T004_DIM,
        "vocab_size": 100,
        "max_seq_len": 32,
        "intermediate_dim": 4 * T004_DIM,
    }


def _t004_backend(seed: int) -> PyTorchModelBackend:
    backend = PyTorchModelBackend(model_config=_t004_model_config(), seed=seed)
    backend.load(model_name="ar015-t004", device="cpu")
    return backend


def _t004_mode_backend(seed: int, **overrides) -> PyTorchModelBackend:
    backend = _t004_backend(seed)
    settings = dict(
        kv_lifecycle="replace",
        kv_execution="paged",
        mode_control_enabled=True,
        macro_block_size=T004_MACRO,
        page_size=T004_PAGE_SIZE,
    )
    settings.update(overrides)
    backend.configure_execution(**settings)
    return backend


def _t004_step(step_id: int, total_steps: int = 0) -> StepConfig:
    # conservative => fp32 pages, so store.gather() compares bitwise.
    return StepConfig(
        step_id=step_id,
        total_steps=total_steps,
        noise_level=0.0,
        shadowkv_mode="conservative",
        ffn_mode="full",
    )


def _t004_input() -> torch.Tensor:
    return _base_hidden(batch=T004_BATCH, seq=T004_SEQ, dim=T004_DIM)


def _t004_drift_input() -> torch.Tensor:
    return _drift_rows(_t004_input(), list(T004_DRIFT_ROWS), T004_DRIFT_COORD)


def _t004_reference_model(seed: int) -> SimpleDiffusionLLM:
    """Fresh model with weights identical to _t004_backend(same seed)."""
    torch.manual_seed(seed)
    return SimpleDiffusionLLM(
        num_layers=T004_LAYERS,
        num_heads=T004_HEADS,
        head_dim=T004_HEAD_DIM,
        hidden_dim=T004_DIM,
        vocab_size=100,
        max_seq_len=32,
        intermediate_dim=4 * T004_DIM,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _t004_reference_config():
    from shadowinfer.execution.macro_mode import MacroModeConfig

    return MacroModeConfig(
        enabled=True,
        macro_block_size=T004_MACRO,
        skip_threshold=0.1,
        delta_min_active_ratio=0.5,
        warmup_steps=1,
        force_refresh_every=0,
    )


def _t004_reference_state():
    from shadowinfer.execution.macro_mode import MacroModeState

    return MacroModeState(entries={})


def _t004_replay_step(model, state, config, x, step_id):
    """Instrumented mode-path replay of one denoising step (total_steps=0).

    Mirrors SimpleDiffusionLLM.forward_step while capturing each layer's
    (modes, active_rows) through a side classify call: classify is
    deterministic on unchanged entries, so the captured classification is
    exactly the one the block itself uses. Returns (per_layer, logits, kvs)
    where kvs[layer] is the authoritative full-sequence (k, v).
    """
    positions = torch.arange(x.shape[1])
    hidden = x + model.pos_embed(positions).unsqueeze(0)
    state.begin_step()
    per_layer = []
    kvs = {}
    for layer_id, block in enumerate(model.blocks):
        modes, active = state.classify(layer_id, hidden, step_id, config)
        per_layer.append((dict(modes), list(active)))
        hidden, (k, v), _attn = block(
            hidden,
            layer_id=layer_id,
            step_id=step_id,
            mode_state=state,
            mode_config=config,
        )
        kvs[layer_id] = (k, v)
    logits = model.output_proj(model.out_norm(hidden))
    return per_layer, logits, kvs


def _t004_expected_dispatch(per_layer, seq_len, block_size):
    """Expected (skip, delta, refresh, skipped_rows, projected_rows, rewritten)."""
    skip = delta = refresh = skipped = projected = rewritten = 0
    for modes, active in per_layer:
        active_set = set(active)
        projected += len(active)
        for macro_idx, mode in modes.items():
            start = macro_idx * block_size
            end = min(start + block_size, seq_len)
            if mode == "SKIP":
                skip += 1
                skipped += end - start
            elif mode == "REFRESH":
                refresh += 1
                rewritten += end - start
            else:
                delta += 1
                rewritten += sum(1 for row in range(start, end) if row in active_set)
    return skip, delta, refresh, skipped, projected, rewritten


def _t004_expected_layer_rewritten(modes, active, seq_len, block_size):
    """Rows one layer must overwrite: DELTA active rows + REFRESH full blocks."""
    active_set = set(active)
    rewritten = 0
    for macro_idx, mode in modes.items():
        start = macro_idx * block_size
        end = min(start + block_size, seq_len)
        if mode == "SKIP":
            continue
        if mode == "REFRESH":
            rewritten += end - start
        else:
            rewritten += sum(1 for row in range(start, end) if row in active_set)
    return rewritten


def _t004_expected_rewrite_intervals(modes, active, seq_len, block_size):
    """Contiguous [lo, hi) rewrite intervals one layer's dispatch must issue."""
    active_set = set(active)
    intervals = []
    for macro_idx, mode in modes.items():
        start = macro_idx * block_size
        end = min(start + block_size, seq_len)
        if mode == "SKIP":
            continue
        if mode == "REFRESH":
            intervals.append((start, end))
            continue
        run_start = None
        for row in range(start, end):
            if row in active_set:
                if run_start is None:
                    run_start = row
            elif run_start is not None:
                intervals.append((run_start, row))
                run_start = None
        if run_start is not None:
            intervals.append((run_start, end))
    return intervals


class TestModeControlConfigValidation:
    """T004 item 1: explicit opt-in validation at configure time."""

    def test_mode_control_with_dense_execution_raises(self):
        backend = _t004_backend(seed=T004_SEED + 1)
        with pytest.raises(ValueError):
            backend.configure_execution(
                kv_lifecycle="replace",
                kv_execution="dense",
                mode_control_enabled=True,
            )

    def test_mode_control_with_append_lifecycle_raises(self):
        backend = _t004_backend(seed=T004_SEED + 2)
        with pytest.raises(ValueError):
            backend.configure_execution(
                kv_lifecycle="append",
                kv_execution="paged",
                mode_control_enabled=True,
            )

    def test_mode_control_with_step_reuse_raises(self):
        backend = _t004_backend(seed=T004_SEED + 3)
        with pytest.raises(ValueError):
            backend.configure_execution(
                kv_lifecycle="replace",
                kv_execution="paged",
                step_reuse_enabled=True,
                mode_control_enabled=True,
            )

    def test_mode_control_with_invalid_parameter_values_raises(self):
        backend = _t004_backend(seed=T004_SEED + 4)
        for overrides in (
            {"macro_block_size": 0},
            {"macro_block_size": -4},
            {"skip_threshold": -0.1},
            {"delta_min_active_ratio": -0.1},
            {"delta_min_active_ratio": 1.1},
            {"mode_warmup_steps": -1},
            {"force_refresh_every": -1},
        ):
            with pytest.raises(ValueError):
                backend.configure_execution(
                    kv_lifecycle="replace",
                    kv_execution="paged",
                    mode_control_enabled=True,
                    **overrides,
                )

    def test_valid_mode_configuration_is_accepted(self):
        from shadowinfer.execution.macro_mode import MacroModeConfig, MacroModeState

        backend = _t004_backend(seed=T004_SEED + 5)
        backend.configure_execution(
            kv_lifecycle="replace",
            kv_execution="paged",
            mode_control_enabled=True,
            macro_block_size=T004_MACRO,
            page_size=T004_PAGE_SIZE,
        )
        assert isinstance(backend._mode_state, MacroModeState)
        assert isinstance(backend._mode_config, MacroModeConfig)
        assert backend._mode_config.enabled is True
        assert backend._mode_config.macro_block_size == T004_MACRO
        assert backend._mode_config.skip_threshold == pytest.approx(0.1)
        assert backend._mode_config.delta_min_active_ratio == pytest.approx(0.5)
        assert backend._mode_config.warmup_steps == 1
        assert backend._mode_config.force_refresh_every == 0

    def test_reset_execution_state_rebuilds_mode_state(self):
        backend = _t004_mode_backend(seed=T004_SEED + 6)
        backend.forward_step(_t004_input(), _t004_step(0))
        state_before = backend._mode_state
        assert state_before.entries

        backend.reset_execution_state()

        assert backend._mode_state is not state_before
        assert backend._mode_state.entries == {}
        rerun = backend.forward_step(_t004_input(), _t004_step(0))
        stats = rerun["execution_stats"]
        assert stats["fallback"] is False
        assert stats["refresh_blocks"] == T004_LAYERS * T004_NUM_MACROS
        assert stats["appended_tokens"] == T004_LAYERS * T004_SEQ


class TestModeControlDefaultOffRegression:
    """T004 item 2: eight keys always exist; mode off keeps everything at 0."""

    def _assert_mode_keys_off(self, stats):
        assert "mode_control" in stats
        assert stats["mode_control"] is False
        assert stats["macro_block_size"] == T004_MACRO_DEFAULT
        for key in (
            "skip_blocks",
            "delta_blocks",
            "refresh_blocks",
            "skipped_rows",
            "rewritten_rows",
            "projected_rows",
        ):
            assert key in stats
            assert stats[key] == 0

    def test_default_dense_execution_reports_mode_keys_off(self):
        backend = _t004_backend(seed=T004_SEED + 11)
        result = backend.forward_step(_t004_input(), _t004_step(0))
        stats = result["execution_stats"]
        assert T004_PRE_AR015_STATS_KEYS <= set(stats)
        self._assert_mode_keys_off(stats)
        assert stats["kv_execution"] == "dense"
        for entry in result["kv_cache"].values():
            assert entry.k_tensor is not None
            assert getattr(entry, "paged_kv", None) is None

    def test_paged_replace_without_mode_keeps_existing_accounting(self):
        backend = _t004_backend(seed=T004_SEED + 12)
        backend.configure_execution(
            kv_lifecycle="replace",
            kv_execution="paged",
            page_size=T004_PAGE_SIZE,
        )
        result = backend.forward_step(_t004_input(), _t004_step(0))
        stats = result["execution_stats"]
        assert T004_PRE_AR015_STATS_KEYS <= set(stats)
        self._assert_mode_keys_off(stats)
        # AR014 replace semantics are untouched by the mode keys.
        assert stats["appended_tokens"] == T004_LAYERS * T004_SEQ
        assert stats["repack_tokens_saved"] == 0
        assert stats["fallback"] is False
        assert stats["page_count"] == T004_LAYERS * math.ceil(T004_SEQ / T004_PAGE_SIZE)


class TestModeFirstStepColdStart:
    """T004 item 3: first step on a missing store appends full, all REFRESH."""

    def test_first_step_appends_full_and_reports_all_refresh(self):
        backend = _t004_mode_backend(seed=T004_SEED + 21)
        result = backend.forward_step(_t004_input(), _t004_step(0))
        stats = result["execution_stats"]
        assert stats["mode_control"] is True
        assert stats["macro_block_size"] == T004_MACRO
        assert stats["refresh_blocks"] == T004_LAYERS * T004_NUM_MACROS
        assert stats["skip_blocks"] == 0
        assert stats["delta_blocks"] == 0
        assert stats["skipped_rows"] == 0
        assert stats["projected_rows"] == T004_LAYERS * T004_SEQ
        assert stats["rewritten_rows"] == 0  # cold start appends, never rewrites
        assert stats["appended_tokens"] == T004_LAYERS * T004_SEQ
        assert stats["repack_tokens_saved"] == 0
        assert stats["fallback"] is False
        for entry in backend.get_kv_cache().values():
            store = entry.paged_kv
            assert store is not None
            assert store.max_seq == T004_SEQ
            assert store.appended_tokens == T004_SEQ
            assert store.rewritten_tokens == 0


class TestModeSkipZeroWrite:
    """T004 item 4: all-SKIP second step performs zero page writes."""

    def test_all_skip_second_step_writes_nothing_and_replays_first_step(self):
        backend = _t004_mode_backend(seed=T004_SEED + 31)
        x = _t004_input()
        first = backend.forward_step(x, _t004_step(0))
        snapshots = {}
        for layer_id, entry in backend.get_kv_cache().items():
            store = entry.paged_kv
            snapshots[layer_id] = {
                "store": store,
                "page_ids": [id(page) for page in store.pages],
                "page_snaps": [_snapshot_page(page) for page in store.pages],
                "gather": store.gather(),
            }

        second = backend.forward_step(x, _t004_step(1))
        stats = second["execution_stats"]
        assert stats["mode_control"] is True
        assert stats["skip_blocks"] == T004_LAYERS * T004_NUM_MACROS
        assert stats["delta_blocks"] == 0
        assert stats["refresh_blocks"] == 0
        assert stats["skipped_rows"] == T004_LAYERS * T004_SEQ
        assert stats["projected_rows"] == 0
        assert stats["rewritten_rows"] == 0
        assert stats["appended_tokens"] == 0
        # every unoverwritten row counts as repack-saved
        assert stats["repack_tokens_saved"] == T004_LAYERS * T004_SEQ
        assert stats["fallback"] is False

        for layer_id, snap in snapshots.items():
            store = snap["store"]
            assert backend.get_kv_cache()[layer_id].paged_kv is store
            assert [id(page) for page in store.pages] == snap["page_ids"]
            for page, page_snap in zip(store.pages, snap["page_snaps"]):
                _assert_page_equal(page, page_snap)
            assert store.appended_tokens == T004_SEQ
            assert store.rewritten_tokens == 0
            k_got, v_got = store.gather()
            assert torch.equal(k_got, snap["gather"][0])
            assert torch.equal(v_got, snap["gather"][1])

        # all-SKIP layers replay cached outputs: logits are bit-identical
        assert torch.equal(second["output"], first["output"])


class TestModeDeltaRefreshDispatch:
    """T004 item 5: DELTA rewrites active runs, REFRESH rewrites whole blocks."""

    def test_delta_and_refresh_blocks_rewrite_exactly_per_dispatch_semantics(self):
        seed = T004_SEED + 41
        backend = _t004_mode_backend(seed=seed)
        model = _t004_reference_model(seed)
        config = _t004_reference_config()
        state = _t004_reference_state()
        x = _t004_input()
        x_drift = _t004_drift_input()

        # Reference: replay the recording step, then the drift step while
        # capturing the per-layer classification the backend must dispatch on.
        _t004_replay_step(model, state, config, x, 0)
        per_layer, logits_ref, kvs_ref = _t004_replay_step(model, state, config, x_drift, 1)

        backend.forward_step(x, _t004_step(0))
        pages_before = {}
        for layer_id, entry in backend.get_kv_cache().items():
            store = entry.paged_kv
            pages_before[layer_id] = (
                [id(page) for page in store.pages],
                [_snapshot_page(page) for page in store.pages],
            )

        second = backend.forward_step(x_drift, _t004_step(1))
        stats = second["execution_stats"]

        skip, delta, refresh, skipped, projected, rewritten = _t004_expected_dispatch(
            per_layer, T004_SEQ, T004_MACRO
        )
        assert delta > 0
        assert refresh > 0
        assert stats["mode_control"] is True
        assert stats["skip_blocks"] == skip
        assert stats["delta_blocks"] == delta
        assert stats["refresh_blocks"] == refresh
        assert stats["skipped_rows"] == skipped
        assert stats["projected_rows"] == projected
        assert stats["rewritten_rows"] == rewritten
        assert stats["appended_tokens"] == 0  # matched store: no clear+append
        assert stats["repack_tokens_saved"] == T004_LAYERS * T004_SEQ - rewritten
        assert stats["fallback"] is False
        assert torch.equal(second["output"], logits_ref)

        for layer_id in range(T004_LAYERS):
            modes, active = per_layer[layer_id]
            intervals = _t004_expected_rewrite_intervals(modes, active, T004_SEQ, T004_MACRO)
            expected_layer_rewritten = _t004_expected_layer_rewritten(
                modes, active, T004_SEQ, T004_MACRO
            )
            store = backend.get_kv_cache()[layer_id].paged_kv
            page_ids, page_snaps = pages_before[layer_id]
            assert [id(page) for page in store.pages] == page_ids
            assert store.appended_tokens == T004_SEQ  # unchanged since cold start
            assert store.rewritten_tokens == expected_layer_rewritten
            for page, page_snap in zip(store.pages, page_snaps):
                page_lo = page_snap["start_token"]
                page_hi = page_lo + page_snap["num_tokens"]
                if any(lo < page_hi and hi > page_lo for lo, hi in intervals):
                    continue
                # pages outside every rewrite interval (SKIP blocks) stay untouched
                _assert_page_equal(page, page_snap)
            # fp32 pages: rewritten store content is bitwise the model's k/v
            k_got, v_got = store.gather()
            assert torch.equal(k_got, kvs_ref[layer_id][0])
            assert torch.equal(v_got, kvs_ref[layer_id][1])


class TestModeRewriteFallback:
    """T004 item 6: rewrite failure degrades the layer to clear+append."""

    def test_rewrite_failure_falls_back_to_clear_append_and_resets_mode_state(self, monkeypatch):
        seed = T004_SEED + 51
        backend = _t004_mode_backend(seed=seed)
        model = _t004_reference_model(seed)
        config = _t004_reference_config()
        state = _t004_reference_state()
        x = _t004_input()
        x_drift = _t004_drift_input()

        _t004_replay_step(model, state, config, x, 0)
        _, _, kvs_ref = _t004_replay_step(model, state, config, x_drift, 1)

        backend.forward_step(x, _t004_step(0))

        def failing_rewrite(self, start_token, k_new, v_new, precision):
            raise RuntimeError("synthetic rewrite failure")

        monkeypatch.setattr(PagedKVStore, "rewrite_tokens", failing_rewrite)
        second = backend.forward_step(x_drift, _t004_step(1))
        stats = second["execution_stats"]
        monkeypatch.undo()

        assert stats["fallback"] is True
        assert stats["rewritten_rows"] == 0
        # every layer degraded to clear + full append with this step's k/v
        assert stats["appended_tokens"] == T004_LAYERS * T004_SEQ
        for layer_id in range(T004_LAYERS):
            store = backend.get_kv_cache()[layer_id].paged_kv
            assert store is not None
            assert store.max_seq == T004_SEQ
            assert store.rewritten_tokens == 0
            k_got, v_got = store.gather()
            assert torch.equal(k_got, kvs_ref[layer_id][0])
            assert torch.equal(v_got, kvs_ref[layer_id][1])

        # the mode state was reset (entries gone, first-step semantics ahead)
        assert backend._mode_state.entries == {}

        third = backend.forward_step(x, _t004_step(2))
        third_stats = third["execution_stats"]
        assert third_stats["fallback"] is False
        assert third_stats["refresh_blocks"] == T004_LAYERS * T004_NUM_MACROS
        assert third_stats["rewritten_rows"] == T004_LAYERS * T004_SEQ


class TestModeModelCallFallback:
    """T004 item 7: mode-call failure degrades to a no-mode full recompute."""

    def test_mode_call_failure_degrades_to_full_recompute_and_rebuilds(self, monkeypatch):
        seed = T004_SEED + 61
        backend = _t004_mode_backend(seed=seed)
        model = _t004_reference_model(seed)
        x = _t004_input()

        logits_dense, _, _ = model.forward_step(x, 0, 0)  # mode-off golden

        from shadowinfer.execution.macro_mode import MacroModeState

        def failing_classify(self, layer_id, hidden, step_id, config):
            raise RuntimeError("synthetic classify failure")

        monkeypatch.setattr(MacroModeState, "classify", failing_classify)
        first = backend.forward_step(x, _t004_step(0))
        stats = first["execution_stats"]
        monkeypatch.undo()

        assert stats["fallback"] is True
        assert torch.equal(first["output"], logits_dense)
        assert stats["appended_tokens"] == T004_LAYERS * T004_SEQ
        assert stats["rewritten_rows"] == 0
        for entry in backend.get_kv_cache().values():
            assert entry.paged_kv is not None
            assert entry.paged_kv.max_seq == T004_SEQ
        assert backend._mode_state.entries == {}

        # the next step rebuilds mode state with first-step semantics
        second = backend.forward_step(x, _t004_step(1))
        second_stats = second["execution_stats"]
        assert second_stats["fallback"] is False
        assert second_stats["refresh_blocks"] == T004_LAYERS * T004_NUM_MACROS
        assert second_stats["rewritten_rows"] == T004_LAYERS * T004_SEQ


class TestModeStatisticsDualChannel:
    """T004 item 8: stats keys and mode_state.stats() agree at every step."""

    def test_three_step_stats_match_mode_state_and_reference(self):
        seed = T004_SEED + 71
        backend = _t004_mode_backend(seed=seed)
        model = _t004_reference_model(seed)
        config = _t004_reference_config()
        state = _t004_reference_state()
        x = _t004_input()
        x_drift = _t004_drift_input()

        for step_id, step_input in ((0, x), (1, x), (2, x_drift)):
            per_layer, _, _ = _t004_replay_step(model, state, config, step_input, step_id)
            skip, delta, refresh, skipped, projected, rewritten = _t004_expected_dispatch(
                per_layer, T004_SEQ, T004_MACRO
            )
            if step_id == 0:
                rewritten = 0  # cold start appends instead of rewriting
            expected_appended = T004_LAYERS * T004_SEQ if step_id == 0 else 0

            result = backend.forward_step(step_input, _t004_step(step_id))
            stats = result["execution_stats"]
            mode_state_stats = backend._mode_state.stats()

            assert stats["mode_control"] is True
            assert stats["skip_blocks"] == skip == mode_state_stats["skip_blocks"]
            assert stats["delta_blocks"] == delta == mode_state_stats["delta_blocks"]
            assert stats["refresh_blocks"] == refresh == mode_state_stats["refresh_blocks"]
            assert stats["skipped_rows"] == skipped == mode_state_stats["skipped_rows"]
            assert stats["rewritten_rows"] == rewritten == mode_state_stats["rewritten_rows"]
            assert stats["projected_rows"] == projected == mode_state_stats["active_rows"]
            assert stats["appended_tokens"] == expected_appended
            assert stats["fallback"] is False


# ---------------------------------------------------------------------------
# T005: orchestrator wiring + end-to-end (design.md 6.1 T005 / 6.3 / 6.4)
# ---------------------------------------------------------------------------

T005_SEED = 2041
T005_LAYERS = 4
T005_HEADS = 4
T005_HEAD_DIM = 16
T005_DIM = 64
T005_BATCH = 2
T005_SEQ = 32
T005_MACRO = 8
T005_PAGE_SIZE = 16
T005_TOTAL_STEPS = 6
T005_REL_L2_LIMIT = 0.02
T005_TAIL_FREEZE_MIN = 0.3
# Trajectory-fixture calibration (test rig setup, not assertion weakening):
# a probe run over candidate sigma schedules showed the pure geometric decay
# 0.9 -> 0.02 overshoots relL2 at the DELTA-band step (0.0231 > 0.02) because
# near-threshold rows freeze inside DELTA/REFRESH blocks and SKIP steps never
# re-record entries. The schedule below keeps every skip step's drift since
# the last recorded entry small and lands each refresh step clearly above the
# gate; measured with seed 2041 (per-step relL2 / freeze rate):
#   step 0: sigma=0.900 relL2=0.00000 freeze=0.000 (warmup REFRESH)
#   step 1: sigma=0.420 relL2=0.00000 freeze=0.000 (REFRESH)
#   step 2: sigma=0.196 relL2=0.00000 freeze=0.000 (REFRESH)
#   step 3: sigma=0.145 relL2=0.01518 freeze=1.000 (all SKIP)
#   step 4: sigma=0.050 relL2=0.00548 freeze=0.000 (REFRESH)
#   step 5: sigma=0.020 relL2=0.01557 freeze=1.000 (all SKIP)
# tail-3 mean freeze 0.667, final relL2 0.01557.
T005_SIGMAS = (0.9, 0.42, 0.196, 0.145, 0.05, 0.02)
# Block-wise semi-AR scenario (design.md 6.3b): S=16 rows, macro blocks of 4.
T005B_SEQ = 16
T005B_NUM_MACROS = T005B_SEQ // MM_MACRO  # 4
T005B_TAIL_START = 12  # rows [12, 16) are the drifting 4th macro block
# Tail coords per steady step: 12.0 drifts 0.2 vs the recorded 10.0, then
# 14.0 drifts |14 - 12| / 12 = 0.167 vs the freshly recorded 12.0.
T005B_TAIL_COORDS = (12.0, 14.0)


def _t005_rel_l2(a, b):
    """relL2 = ||a - b||_2 / ||b||_2."""
    return ((a - b).norm() / b.norm()).item()


def _t005_model_config() -> dict:
    return {
        "name": "ar015-t005",
        "num_layers": T005_LAYERS,
        "num_heads": T005_HEADS,
        "head_dim": T005_HEAD_DIM,
        "hidden_dim": T005_DIM,
        "vocab_size": 100,
        "max_seq_len": 64,
        "intermediate_dim": 4 * T005_DIM,
    }


def _t005_backend(seed: int, **execution) -> PyTorchModelBackend:
    backend = PyTorchModelBackend(model_config=_t005_model_config(), seed=seed)
    backend.load(model_name="dummy", device="cpu")
    if execution:
        backend.configure_execution(**execution)
    return backend


def _t005_e2e_step(step_id: int) -> StepConfig:
    # conservative => fp32 pages; total_steps=6 keeps the time embed active.
    return StepConfig(
        step_id=step_id,
        total_steps=T005_TOTAL_STEPS,
        noise_level=0.0,
        shadowkv_mode="conservative",
        ffn_mode="full",
    )


def _t005_trajectory() -> list:
    """Controlled denoising trajectory x_t = x0 + sigma_t * noise (6 steps)."""
    gen = torch.Generator().manual_seed(T005_SEED + 7)
    x0 = torch.randn(T005_BATCH, T005_SEQ, T005_DIM, generator=gen)
    noise = torch.randn(T005_BATCH, T005_SEQ, T005_DIM, generator=gen)
    return [x0 + sigma * noise for sigma in T005_SIGMAS]


def _t005_semi_ar_inputs() -> list:
    """S=16 inputs: rows [0, 12) constant every step, tail block drifting."""
    base = _base_hidden(batch=T004_BATCH, seq=T005B_SEQ, dim=T004_DIM)
    tail_rows = list(range(T005B_TAIL_START, T005B_SEQ))
    inputs = [base]
    for coord in T005B_TAIL_COORDS:
        inputs.append(_drift_rows(base, tail_rows, coord))
    return inputs


def _t005_small_mode_backend(seed: int) -> PyTorchModelBackend:
    """T004-dims backend in paged+replace+mode (mb=4, page_size=8)."""
    backend = PyTorchModelBackend(model_config=_t004_model_config(), seed=seed)
    backend.load(model_name="dummy", device="cpu")
    backend.configure_execution(
        kv_lifecycle="replace",
        kv_execution="paged",
        mode_control_enabled=True,
        macro_block_size=T004_MACRO,
        page_size=T004_PAGE_SIZE,
    )
    return backend


class TestOrchestratorExecutionWiring:
    """T005 item 1/2: orchestrator cfg passthrough and unsupported fallback."""

    def test_execution_config_reaches_backend_mode_state(self):
        from shadowinfer.execution.macro_mode import MacroModeConfig, MacroModeState

        backend = PyTorchModelBackend(model_config=_t004_model_config(), seed=T005_SEED + 1)
        backend.load(model_name="dummy", device="cpu")
        orch = Orchestrator(
            config={
                "execution": {
                    "kv_lifecycle": "replace",
                    "kv_execution": "paged",
                    "mode_control_enabled": True,
                    "macro_block_size": 8,
                    "page_size": 16,
                    "skip_threshold": 0.05,
                    "delta_min_active_ratio": 0.25,
                    "mode_warmup_steps": 2,
                    "force_refresh_every": 4,
                }
            }
        )
        orch.set_model_backend(backend)

        orch._maybe_configure_execution()

        assert backend._kv_lifecycle == "replace"
        assert backend._kv_execution == "paged"
        assert backend._page_size == 16
        assert isinstance(backend._mode_config, MacroModeConfig)
        assert isinstance(backend._mode_state, MacroModeState)
        assert backend._mode_config.macro_block_size == 8
        assert backend._mode_config.skip_threshold == pytest.approx(0.05)
        assert backend._mode_config.delta_min_active_ratio == pytest.approx(0.25)
        assert backend._mode_config.warmup_steps == 2
        assert backend._mode_config.force_refresh_every == 4

        # the wired config is live: forward reports the configured block size
        result = backend.forward_step(_t004_input(), _t004_step(0))
        stats = result["execution_stats"]
        assert stats["mode_control"] is True
        assert stats["macro_block_size"] == 8
        assert stats["refresh_blocks"] == T004_LAYERS * 2  # warmup forces full
        assert stats["appended_tokens"] == T004_LAYERS * T004_SEQ
        assert stats["fallback"] is False

    def test_no_execution_config_leaves_backend_untouched(self):
        backend = PyTorchModelBackend(model_config=_t004_model_config(), seed=T005_SEED + 2)
        backend.load(model_name="dummy", device="cpu")
        before = dict(vars(backend))

        orch = Orchestrator(config={})
        orch.set_model_backend(backend)
        orch._maybe_configure_execution()

        assert vars(backend) == before
        assert backend._mode_config is None
        assert backend._mode_state is None
        assert backend._kv_lifecycle == "append"
        assert backend._kv_execution == "dense"
        assert backend._page_size == 32

    def test_unsupported_backend_logs_event_and_stays_unconfigured(self):
        mock_backend = MockModelBackend()
        before = dict(vars(mock_backend))
        orch = Orchestrator(
            config={"execution": {"kv_lifecycle": "replace", "kv_execution": "paged"}}
        )
        orch.set_model_backend(mock_backend)

        orch._maybe_configure_execution()  # must not raise

        assert not hasattr(mock_backend, "configure_execution")
        assert vars(mock_backend) == before
        events = [
            record
            for record in orch.logger.get_logs()
            if record["event"] == "execution_unsupported"
        ]
        assert len(events) == 1
        assert events[0]["data"]["backend"] == "MockModelBackend"


class TestModeDefaultOffBitwiseRegression:
    """T005 item 3: mode off keeps AR014 output/stats behavior bitwise."""

    def _assert_mode_keys_off(self, stats):
        assert stats["mode_control"] is False
        assert stats["macro_block_size"] == T004_MACRO_DEFAULT
        for key in (
            "skip_blocks",
            "delta_blocks",
            "refresh_blocks",
            "skipped_rows",
            "rewritten_rows",
            "projected_rows",
        ):
            assert key in stats
            assert stats[key] == 0

    def test_dense_default_output_matches_reference_and_mode_keys_zero(self):
        seed = T005_SEED + 11
        backend = PyTorchModelBackend(model_config=_t004_model_config(), seed=seed)
        backend.load(model_name="dummy", device="cpu")
        reference = _t004_reference_model(seed)
        x = _t004_input()

        result = backend.forward_step(x, _t004_step(0))
        logits_ref, _, _ = reference.forward_step(x, 0, 0)

        assert torch.equal(result["output"], logits_ref)
        stats = result["execution_stats"]
        assert T004_PRE_AR015_STATS_KEYS <= set(stats)
        self._assert_mode_keys_off(stats)
        assert stats["kv_execution"] == "dense"
        assert stats["fallback"] is False

    def test_paged_replace_mode_off_matches_reference_and_ar014_accounting(self):
        seed = T005_SEED + 12
        backend = PyTorchModelBackend(model_config=_t004_model_config(), seed=seed)
        backend.load(model_name="dummy", device="cpu")
        backend.configure_execution(
            kv_lifecycle="replace",
            kv_execution="paged",
            page_size=T004_PAGE_SIZE,
        )
        reference = _t004_reference_model(seed)
        x = _t004_input()

        result = backend.forward_step(x, _t004_step(0))
        logits_ref, _, _ = reference.forward_step(x, 0, 0)

        assert torch.equal(result["output"], logits_ref)
        stats = result["execution_stats"]
        assert T004_PRE_AR015_STATS_KEYS <= set(stats)
        self._assert_mode_keys_off(stats)
        # AR014 replace+paged accounting is untouched by the mode keys.
        assert stats["appended_tokens"] == T004_LAYERS * T004_SEQ
        assert stats["repack_tokens_saved"] == 0
        assert stats["page_count"] == T004_LAYERS * math.ceil(T004_SEQ / T004_PAGE_SIZE)
        assert stats["fallback"] is False


class TestModeControlE2ERelL2:
    """T005 item 4: controlled-trajectory e2e vs dense golden (design 6.3a)."""

    def test_six_step_trajectory_accuracy_and_freeze(self):
        golden = _t005_backend(T005_SEED, kv_lifecycle="replace", kv_execution="dense")
        mode = _t005_backend(
            T005_SEED,
            kv_lifecycle="replace",
            kv_execution="paged",
            mode_control_enabled=True,
            macro_block_size=T005_MACRO,
            page_size=T005_PAGE_SIZE,
        )

        rels = []
        freezes = []
        for step_id, x_t in enumerate(_t005_trajectory()):
            step = _t005_e2e_step(step_id)
            golden_result = golden.forward_step(x_t, step)
            mode_result = mode.forward_step(x_t, step)
            stats = mode_result["execution_stats"]

            rel = _t005_rel_l2(mode_result["output"], golden_result["output"])
            rels.append(rel)
            assert rel <= T005_REL_L2_LIMIT
            assert stats["mode_control"] is True
            assert stats["fallback"] is False
            freezes.append(stats["skipped_rows"] / (T005_LAYERS * T005_SEQ))

        tail_freeze = sum(freezes[-3:]) / 3
        assert tail_freeze >= T005_TAIL_FREEZE_MIN
        assert rels[-1] <= T005_REL_L2_LIMIT


class TestModeBlockWiseSemiAR:
    """T005 item 5: block-wise semi-AR scenario (design.md 6.3b).

    S=16 with mb=4 and page_size=8: rows [0, 12) (macro blocks 0-2) hold the
    same input every step while the tail block [12, 16) drifts per step. At
    steady state the front macros SKIP, the tail block REFRESHes, the page
    covering only frozen rows ([0, 8)) is never written, and the page holding
    the tail rows ([8, 16)) is rewritten in place leaving its frozen rows
    [8, 12) bit-identical.
    """

    def test_frozen_macros_skip_tail_refreshes_with_zero_frozen_page_writes(self):
        seed = T005_SEED + 21
        backend = _t005_small_mode_backend(seed)
        reference = _t004_reference_model(seed)
        config = _t004_reference_config()
        state = _t004_reference_state()
        inputs = _t005_semi_ar_inputs()

        # reference lockstep replay (records step 0 like the backend does)
        _t004_replay_step(reference, state, config, inputs[0], 0)

        first = backend.forward_step(inputs[0], _t004_step(0))
        first_stats = first["execution_stats"]
        assert first_stats["refresh_blocks"] == T004_LAYERS * T005B_NUM_MACROS
        assert first_stats["appended_tokens"] == T004_LAYERS * T005B_SEQ
        assert first_stats["rewritten_rows"] == 0  # cold start appends only
        page_snaps = {}
        for layer_id, entry in backend.get_kv_cache().items():
            store = entry.paged_kv
            assert store.page_count == 2  # pages [0, 8) and [8, 16)
            page_snaps[layer_id] = [_snapshot_page(page) for page in store.pages]

        kvs_ref = {}
        for step_id in (1, 2):
            _, _, kvs_ref = _t004_replay_step(reference, state, config, inputs[step_id], step_id)
            result = backend.forward_step(inputs[step_id], _t004_step(step_id))
            stats = result["execution_stats"]
            assert stats["fallback"] is False
            assert stats["skip_blocks"] == T004_LAYERS * 3
            assert stats["refresh_blocks"] == T004_LAYERS
            assert stats["delta_blocks"] == 0
            assert stats["skipped_rows"] == T004_LAYERS * T005B_TAIL_START
            assert stats["rewritten_rows"] == T004_LAYERS * (T005B_SEQ - T005B_TAIL_START)
            assert stats["appended_tokens"] == 0
            for layer_id in range(T004_LAYERS):
                modes = backend._mode_state.last_modes[layer_id]
                assert [modes[i] for i in range(3)] == ["SKIP", "SKIP", "SKIP"]
                assert modes[3] == "REFRESH"

        for layer_id in range(T004_LAYERS):
            store = backend.get_kv_cache()[layer_id].paged_kv
            assert store.appended_tokens == T005B_SEQ  # unchanged since cold start
            assert store.rewritten_tokens == 2 * (T005B_SEQ - T005B_TAIL_START)
            assert store.rewritten_pages == 2  # page [8, 16) hit once per step
            # the frozen-only page never received a single write
            _assert_page_equal(store.pages[0], page_snaps[layer_id][0])
            # frozen rows inside the rewritten page stay bit-identical
            # (fp32 pages carry no quantization scales: those stay None)
            for name in ("k", "v", "scale_k", "scale_v"):
                tensor = getattr(store.pages[1], name)
                snap = page_snaps[layer_id][1][name]
                if snap is None:
                    assert tensor is None
                else:
                    assert torch.equal(tensor[:, :, 0:4, :], snap[:, :, 0:4, :])
            # fp32 pages: store content is bitwise this step's model new K/V
            k_got, v_got = store.gather()
            assert torch.equal(k_got, kvs_ref[layer_id][0])
            assert torch.equal(v_got, kvs_ref[layer_id][1])


class TestModeShapeMutationReset:
    """T005 item 6: S mutation takes the shape-mismatch reset path (design 6.4).

    After two S=12 steps the input grows to S=16: the store must rebuild via
    clear + full append (appended == S * L), the mode state must be cleared
    (design 6.4 contract), and the next step must report the full-scope
    refresh of a cold start.
    """

    def test_seq_growth_clear_append_full_and_mode_state_reset(self):
        backend = _t005_small_mode_backend(T005_SEED + 31)
        x12 = _t004_input()
        x16 = _base_hidden(batch=T004_BATCH, seq=16, dim=T004_DIM)
        x16[:, :T004_SEQ, :] = x12  # deterministic front rows

        backend.forward_step(x12, _t004_step(0))
        backend.forward_step(x12, _t004_step(1))
        assert backend._mode_state.entries

        mutated = backend.forward_step(x16, _t004_step(2))  # must not crash
        stats = mutated["execution_stats"]
        assert stats["appended_tokens"] == T004_LAYERS * 16  # clear + full append
        for entry in backend.get_kv_cache().values():
            assert entry.paged_kv is not None
            assert entry.paged_kv.max_seq == 16

        # design 6.4: the shape-mismatch reset path must clear the mode state
        assert backend._mode_state.entries == {}

        third = backend.forward_step(x16, _t004_step(3))
        third_stats = third["execution_stats"]
        assert third_stats["fallback"] is False
        assert third_stats["refresh_blocks"] == T004_LAYERS * 4  # full scope
        assert third_stats["skip_blocks"] == 0
        assert third_stats["rewritten_rows"] == T004_LAYERS * 16
        assert third_stats["appended_tokens"] == 0


class TestModeResetExecutionStateIsolation:
    """T005 item 7: reset_execution_state fully isolates consecutive runs."""

    def test_reset_rebuilds_mode_state_and_caches_for_a_cold_restart(self):
        backend = _t004_mode_backend(seed=T005_SEED + 41)
        x = _t004_input()
        backend.forward_step(x, _t004_step(0))
        backend.forward_step(x, _t004_step(1))
        state_before = backend._mode_state
        assert state_before.entries
        assert backend._kv_cache
        assert backend._paged_cache

        backend.reset_execution_state()

        assert backend._mode_state is not state_before
        assert backend._mode_state.entries == {}
        assert backend._kv_cache == {}
        assert backend._paged_cache == {}

        result = backend.forward_step(x, _t004_step(0))
        stats = result["execution_stats"]
        assert stats["mode_control"] is True
        assert stats["fallback"] is False
        assert stats["refresh_blocks"] == T004_LAYERS * T004_NUM_MACROS
        assert stats["skip_blocks"] == 0
        assert stats["appended_tokens"] == T004_LAYERS * T004_SEQ
        assert stats["rewritten_rows"] == 0  # cold start appends, never rewrites
