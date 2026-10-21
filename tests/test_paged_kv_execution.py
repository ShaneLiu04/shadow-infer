"""AR014 T001: paged KV store execution tests.

Exercises the not-yet-implemented ``PagedKVStore``
(``shadowinfer.shadowkv.paged_kv_store``) against the golden
``PackedKVCache.pack(...).unpack()`` reference. Golden comparisons use
``torch.equal`` (bit-exact), which is achievable because
``Quantizer.quantize_int8`` / ``quantize_int4`` compute scales along the
last dim (head_dim / int4 groups), so per-(b,h,t)-row quantization is
independent of how rows are batched into pages.

All tests are real-tensor tests; nothing is mocked.
"""

from __future__ import annotations

import pytest
import torch

from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache
from shadowinfer.shadowkv.paged_kv_store import PagedKVStore

GOLDEN_SEED = 2026
BATCH = 2
HEADS = 4
HEAD_DIM = 32
SEQ_LEN = 19  # deliberately not a multiple of PAGE_SIZE
PAGE_SIZE = 8
APPEND_SPLIT = (5, 5, 5, 4)  # crosses page boundaries at 8 and 16
INT4_GROUP_SIZE = 16


def _make_kv(
    seed: int,
    batch: int = BATCH,
    heads: int = HEADS,
    seq: int = SEQ_LEN,
    head_dim: int = HEAD_DIM,
    scale: float = 2.0,
):
    """Generate non-degenerate K/V (values well away from quantization zero)."""
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn(batch, heads, seq, head_dim, generator=gen) * scale
    v = torch.randn(batch, heads, seq, head_dim, generator=gen) * scale
    return k, v


def _make_store(
    batch: int = BATCH,
    heads: int = HEADS,
    head_dim: int = HEAD_DIM,
    page_size: int = PAGE_SIZE,
) -> PagedKVStore:
    return PagedKVStore(
        batch,
        heads,
        head_dim,
        page_size=page_size,
        device="cpu",
        int4_group_size=INT4_GROUP_SIZE,
    )


def _packed_reference(k_full: torch.Tensor, v_full: torch.Tensor, precision: str):
    """Golden path: PackedKVCache.pack(..., group_size=16).unpack()."""
    seq = k_full.shape[2]
    heads = k_full.shape[1]
    precision_map = {t: {h: precision for h in range(heads)} for t in range(seq)}
    packed = PackedKVCache.pack(k_full, v_full, precision_map, group_size=INT4_GROUP_SIZE)
    return packed["packed_kv"].unpack()


def _stepwise_append(
    store: PagedKVStore,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    precision: str,
    split=APPEND_SPLIT,
) -> None:
    start = 0
    for length in split:
        end = start + length
        store.append(
            k_full[:, :, start:end, :].contiguous(),
            v_full[:, :, start:end, :].contiguous(),
            precision,
        )
        start = end


def _page_tensor_ptrs(page):
    """Snapshot data_ptr() of the page payload tensors.

    Assumes a page object exposes its quantized payload via tensor
    attributes among ``k``, ``v``, ``scale_k``, ``scale_v`` (None-valued
    or missing attributes are skipped).
    """
    ptrs = []
    for name in ("k", "v", "scale_k", "scale_v"):
        tensor = getattr(page, name, None)
        if isinstance(tensor, torch.Tensor):
            ptrs.append((name, tensor.data_ptr()))
    return tuple(ptrs)


# ---------------------------------------------------------------------------
# Group 1: golden equivalence against PackedKVCache (bit-exact torch.equal)
# ---------------------------------------------------------------------------


def test_gather_golden_fp32_stepwise_append():
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    k_ref, v_ref = _packed_reference(k_full, v_full, "fp32")

    store = _make_store()
    _stepwise_append(store, k_full, v_full, "fp32")
    k_paged, v_paged = store.gather()

    assert k_paged.shape == (BATCH, HEADS, SEQ_LEN, HEAD_DIM)
    assert k_paged.dtype == torch.float32
    assert torch.equal(k_paged, k_ref)
    assert torch.equal(v_paged, v_ref)


def test_gather_golden_fp16_stepwise_append():
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    k_ref, v_ref = _packed_reference(k_full, v_full, "fp16")

    store = _make_store()
    _stepwise_append(store, k_full, v_full, "fp16")
    k_paged, v_paged = store.gather()

    assert k_paged.shape == (BATCH, HEADS, SEQ_LEN, HEAD_DIM)
    assert k_paged.dtype == torch.float32
    assert torch.equal(k_paged, k_ref)
    assert torch.equal(v_paged, v_ref)


def test_gather_golden_int8_stepwise_append():
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    k_ref, v_ref = _packed_reference(k_full, v_full, "int8")

    store = _make_store()
    _stepwise_append(store, k_full, v_full, "int8")
    k_paged, v_paged = store.gather()

    assert k_paged.shape == (BATCH, HEADS, SEQ_LEN, HEAD_DIM)
    assert k_paged.dtype == torch.float32
    assert torch.equal(k_paged, k_ref)
    assert torch.equal(v_paged, v_ref)


def test_gather_golden_int4_stepwise_append():
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    k_ref, v_ref = _packed_reference(k_full, v_full, "int4")

    store = _make_store()
    _stepwise_append(store, k_full, v_full, "int4")
    k_paged, v_paged = store.gather()

    assert k_paged.shape == (BATCH, HEADS, SEQ_LEN, HEAD_DIM)
    assert k_paged.dtype == torch.float32
    assert torch.equal(k_paged, k_ref)
    assert torch.equal(v_paged, v_ref)


def test_gather_golden_int8_single_append_spans_pages():
    """One append larger than page_size must split across pages identically."""
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    k_ref, v_ref = _packed_reference(k_full, v_full, "int8")

    store = _make_store()
    store.append(k_full.contiguous(), v_full.contiguous(), "int8")
    k_paged, v_paged = store.gather()

    assert store.appended_tokens == SEQ_LEN
    assert torch.equal(k_paged, k_ref)
    assert torch.equal(v_paged, v_ref)


def test_gather_golden_int4_group_size_fallback_head_dim():
    """head_dim % group_size != 0 must fall back to per-row head_dim groups."""
    head_dim = 24  # 24 % 16 != 0 -> effective group size = 24
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED, seq=11, head_dim=head_dim)
    k_ref, v_ref = _packed_reference(k_full, v_full, "int4")

    store = _make_store(head_dim=head_dim, page_size=4)
    _stepwise_append(store, k_full, v_full, "int4", split=(4, 4, 3))
    k_paged, v_paged = store.gather()

    assert k_paged.shape == (BATCH, HEADS, 11, head_dim)
    assert torch.equal(k_paged, k_ref)
    assert torch.equal(v_paged, v_ref)


# ---------------------------------------------------------------------------
# Group 2: immutability of sealed pages across further appends
# ---------------------------------------------------------------------------


def test_sealed_pages_are_immutable_across_appends():
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    store = _make_store()

    # 10 tokens: page 0 is sealed (8 tokens), page 1 is open (2 tokens).
    _stepwise_append(store, k_full, v_full, "fp16", split=(5, 5))
    assert len(store.pages) == 2
    sealed_page = store.pages[0]
    sealed_ptrs = _page_tensor_ptrs(sealed_page)
    assert sealed_ptrs  # fp16 page must expose k/v payload tensors

    # Grow to 19 tokens: sealed page 0 must keep identity and buffer pointers.
    _stepwise_append(store, k_full, v_full, "fp16", split=(5, 4))
    assert len(store.pages) == 3
    assert store.pages[0] is sealed_page
    assert _page_tensor_ptrs(store.pages[0]) == sealed_ptrs

    # Page 1 is definitely sealed once total tokens exceed 2 * page_size.
    page1 = store.pages[1]
    page1_ptrs = _page_tensor_ptrs(page1)
    gen = torch.Generator().manual_seed(GOLDEN_SEED + 1)
    extra_k = torch.randn(BATCH, HEADS, 3, HEAD_DIM, generator=gen) * 2.0
    extra_v = torch.randn(BATCH, HEADS, 3, HEAD_DIM, generator=gen) * 2.0
    store.append(extra_k, extra_v, "fp16")

    assert store.appended_tokens == 22
    assert store.pages[1] is page1
    assert _page_tensor_ptrs(store.pages[1]) == page1_ptrs


# ---------------------------------------------------------------------------
# Group 3: statistics tracking
# ---------------------------------------------------------------------------


def test_initial_statistics_are_zero():
    store = _make_store()
    assert store.max_seq == 0
    assert store.page_count == 0
    assert store.open_pages == 0
    assert store.appended_tokens == 0
    assert store.page_evictions == 0


def test_page_statistics_track_stepwise_appends():
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    store = _make_store()

    store.append(k_full[:, :, 0:5].contiguous(), v_full[:, :, 0:5].contiguous(), "fp16")
    assert store.page_count == 1
    assert store.open_pages == 1
    assert store.appended_tokens == 5
    assert store.max_seq == 5
    assert store.page_evictions == 0

    store.append(k_full[:, :, 5:10].contiguous(), v_full[:, :, 5:10].contiguous(), "fp16")
    assert store.page_count == 2  # 1 sealed + 1 open
    assert store.open_pages == 1
    assert store.appended_tokens == 10
    assert store.max_seq == 10

    store.append(k_full[:, :, 10:15].contiguous(), v_full[:, :, 10:15].contiguous(), "fp16")
    assert store.page_count == 2
    assert store.open_pages == 1
    assert store.appended_tokens == 15
    assert store.max_seq == 15

    store.append(k_full[:, :, 15:19].contiguous(), v_full[:, :, 15:19].contiguous(), "fp16")
    assert store.page_count == 3
    assert store.open_pages == 1
    assert store.appended_tokens == 19
    assert store.max_seq == 19
    assert store.page_evictions == 0


# ---------------------------------------------------------------------------
# Group 4: byte accounting
# ---------------------------------------------------------------------------


def test_data_plane_bytes_fp16_exact():
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    store = _make_store()
    _stepwise_append(store, k_full, v_full, "fp16")

    expected = BATCH * HEADS * SEQ_LEN * HEAD_DIM * 2 * 2  # k+v, 2 bytes each
    assert store.data_plane_bytes() == expected


def test_data_plane_bytes_fp32_exact():
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    store = _make_store()
    _stepwise_append(store, k_full, v_full, "fp32")

    expected = BATCH * HEADS * SEQ_LEN * HEAD_DIM * 4 * 2  # k+v, 4 bytes each
    assert store.data_plane_bytes() == expected


def test_data_plane_bytes_int8_below_dense_fp16():
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    store = _make_store()
    _stepwise_append(store, k_full, v_full, "int8")

    dense_fp32 = BATCH * HEADS * SEQ_LEN * HEAD_DIM * 4 * 2
    dense_fp16 = BATCH * HEADS * SEQ_LEN * HEAD_DIM * 2 * 2
    assert 0 < store.data_plane_bytes() < dense_fp32
    assert store.data_plane_bytes() < dense_fp16


def test_data_plane_bytes_int4_below_dense_fp16():
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    store = _make_store()
    _stepwise_append(store, k_full, v_full, "int4")

    dense_fp32 = BATCH * HEADS * SEQ_LEN * HEAD_DIM * 4 * 2
    dense_fp16 = BATCH * HEADS * SEQ_LEN * HEAD_DIM * 2 * 2
    assert 0 < store.data_plane_bytes() < dense_fp32
    assert store.data_plane_bytes() < dense_fp16


def test_resident_bytes_include_metadata_overhead():
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    store = _make_store()
    _stepwise_append(store, k_full, v_full, "fp16")

    data = store.data_plane_bytes()
    resident = store.resident_bytes()
    assert resident >= data
    # 48 bytes per token-head metadata estimate plus page table slots.
    assert resident >= data + 48 * BATCH * HEADS * SEQ_LEN


def test_empty_store_accounting_is_zero():
    store = _make_store()
    assert store.data_plane_bytes() == 0
    assert store.resident_bytes() == 0


# ---------------------------------------------------------------------------
# Group 5: validation and boundary behavior
# ---------------------------------------------------------------------------


def test_invalid_precision_raises():
    torch.manual_seed(GOLDEN_SEED)
    store = _make_store()
    k = torch.randn(BATCH, HEADS, 3, HEAD_DIM) * 2.0
    v = torch.randn(BATCH, HEADS, 3, HEAD_DIM) * 2.0
    with pytest.raises(ValueError):
        store.append(k, v, "int3")
    with pytest.raises(ValueError):
        store.append(k, v, "fp8")


def test_append_shape_mismatch_raises_without_mutation():
    torch.manual_seed(GOLDEN_SEED)
    store = _make_store()  # expects [2, 4, T, 32]

    cases = [
        ("wrong head_dim", torch.randn(BATCH, HEADS, 3, 16), torch.randn(BATCH, HEADS, 3, 16)),
        ("wrong heads", torch.randn(BATCH, 5, 3, HEAD_DIM), torch.randn(BATCH, 5, 3, HEAD_DIM)),
        ("wrong batch", torch.randn(3, HEADS, 3, HEAD_DIM), torch.randn(3, HEADS, 3, HEAD_DIM)),
        (
            "missing time dim",
            torch.randn(BATCH, HEADS, HEAD_DIM),
            torch.randn(BATCH, HEADS, HEAD_DIM),
        ),
        (
            "k/v disagreement",
            torch.randn(BATCH, HEADS, 3, HEAD_DIM),
            torch.randn(BATCH, HEADS, 3, HEAD_DIM + 8),
        ),
    ]
    for reason, k_bad, v_bad in cases:
        with pytest.raises(ValueError):
            store.append(k_bad, v_bad, "fp16")

    # Failed appends must not mutate the store.
    assert store.appended_tokens == 0
    assert store.page_count == 0
    assert store.open_pages == 0
    assert store.max_seq == 0


def test_validate_against_accepts_match_and_rejects_mismatch():
    store = _make_store()
    store.validate_against(BATCH, HEADS, HEAD_DIM)  # matching: no raise

    with pytest.raises(ValueError):
        store.validate_against(BATCH + 1, HEADS, HEAD_DIM)
    with pytest.raises(ValueError):
        store.validate_against(BATCH, HEADS + 1, HEAD_DIM)
    with pytest.raises(ValueError):
        store.validate_against(BATCH, HEADS, HEAD_DIM + 8)


def test_zero_token_append_is_noop():
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED, seq=3)
    store = _make_store()
    store.append(k_full, v_full, "fp16")

    before = store.gather()
    stats = (store.max_seq, store.page_count, store.open_pages, store.appended_tokens)

    empty_k = torch.randn(BATCH, HEADS, 0, HEAD_DIM)
    empty_v = torch.randn(BATCH, HEADS, 0, HEAD_DIM)
    store.append(empty_k, empty_v, "int8")

    assert (store.max_seq, store.page_count, store.open_pages, store.appended_tokens) == stats
    after = store.gather()
    assert torch.equal(after[0], before[0])
    assert torch.equal(after[1], before[1])


def test_empty_store_gather_returns_zero_length():
    store = _make_store()
    k, v = store.gather()
    assert k.shape == (BATCH, HEADS, 0, HEAD_DIM)
    assert v.shape == (BATCH, HEADS, 0, HEAD_DIM)
    assert k.dtype == torch.float32
    assert v.dtype == torch.float32


# ---------------------------------------------------------------------------
# Group 6: mixed-precision pages in one store
# ---------------------------------------------------------------------------


def test_mixed_precision_pages_match_per_segment_references():
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED, seq=8)

    store = _make_store(page_size=4)
    store.append(k_full[:, :, 0:4].contiguous(), v_full[:, :, 0:4].contiguous(), "fp16")
    store.append(k_full[:, :, 4:8].contiguous(), v_full[:, :, 4:8].contiguous(), "int8")
    k_paged, v_paged = store.gather()

    assert k_paged.shape == (BATCH, HEADS, 8, HEAD_DIM)
    assert store.appended_tokens == 8

    k_ref_fp16, v_ref_fp16 = _packed_reference(
        k_full[:, :, 0:4].contiguous(), v_full[:, :, 0:4].contiguous(), "fp16"
    )
    k_ref_int8, v_ref_int8 = _packed_reference(
        k_full[:, :, 4:8].contiguous(), v_full[:, :, 4:8].contiguous(), "int8"
    )

    assert torch.equal(k_paged[:, :, 0:4, :], k_ref_fp16)
    assert torch.equal(v_paged[:, :, 0:4, :], v_ref_fp16)
    assert torch.equal(k_paged[:, :, 4:8, :], k_ref_int8)
    assert torch.equal(v_paged[:, :, 4:8, :], v_ref_int8)


# ---------------------------------------------------------------------------
# Group 7 (AR014 T002): token-head eviction and free-list management
# ---------------------------------------------------------------------------


def _evict_pairs(full_tokens=(), partial_heads=()):
    """Build a (token, head) eviction set.

    ``full_tokens`` lists tokens whose HEADS heads are all evicted;
    ``partial_heads`` is an iterable of ``(token, heads)`` pairs evicting
    only the given heads of that token.
    """
    pairs = set()
    for token in full_tokens:
        for head in range(HEADS):
            pairs.add((token, head))
    for token, heads in partial_heads:
        for head in heads:
            pairs.add((token, head))
    return pairs


def _evicted_reference(k_full: torch.Tensor, v_full: torch.Tensor, precision, evict_set):
    """Golden eviction path: PackedKVCache.pack -> evict(set) -> unpack."""
    seq = k_full.shape[2]
    heads = k_full.shape[1]
    precision_map = {t: {h: precision for h in range(heads)} for t in range(seq)}
    packed = PackedKVCache.pack(k_full, v_full, precision_map, group_size=INT4_GROUP_SIZE)
    cache = packed["packed_kv"]
    cache.evict(evict_set)
    return cache.unpack()


def test_evict_golden_fp16_mixed_full_partial_untouched():
    """Mixed eviction set matches the pack-evict-unpack golden bit-for-bit."""
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    evict_set = _evict_pairs(
        full_tokens=(2, 4, 5, 17),
        partial_heads=((10, (0, 1)), (12, (2, 3))),
    )
    k_ref, v_ref = _evicted_reference(k_full, v_full, "fp16", set(evict_set))

    store = _make_store()
    _stepwise_append(store, k_full, v_full, "fp16")
    k_before, v_before = store.gather()
    removed = store.evict_tokens(set(evict_set))
    k_paged, v_paged = store.gather()

    assert removed == 20  # 4 fully evicted tokens * 4 heads + 2 + 2 partial pairs
    assert k_paged.shape == (BATCH, HEADS, SEQ_LEN, HEAD_DIM)
    assert k_paged.dtype == torch.float32
    assert torch.equal(k_paged, k_ref)
    assert torch.equal(v_paged, v_ref)

    # Fully evicted tokens are exactly zero at every head.
    for token in (2, 4, 5, 17):
        assert torch.count_nonzero(k_paged[:, :, token, :]) == 0
        assert torch.count_nonzero(v_paged[:, :, token, :]) == 0
    # Partially evicted tokens: evicted heads zero, surviving heads intact.
    assert torch.count_nonzero(k_paged[:, (0, 1), 10, :]) == 0
    assert torch.count_nonzero(k_paged[:, (2, 3), 12, :]) == 0
    assert torch.equal(k_paged[:, (2, 3), 10, :], k_before[:, (2, 3), 10, :])
    assert torch.equal(v_paged[:, (2, 3), 10, :], v_before[:, (2, 3), 10, :])
    assert torch.equal(k_paged[:, (0, 1), 12, :], k_before[:, (0, 1), 12, :])
    assert torch.equal(v_paged[:, (0, 1), 12, :], v_before[:, (0, 1), 12, :])
    # Untouched tokens keep their values.
    assert torch.equal(k_paged[:, :, 0, :], k_before[:, :, 0, :])
    assert torch.equal(k_paged[:, :, 18, :], k_before[:, :, 18, :])


def test_evict_int8_golden_and_zeroed_heads():
    """INT8 eviction: evicted heads decode to 0.0, survivors bit-identical."""
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    evict_set = _evict_pairs(
        full_tokens=(3, 16),
        partial_heads=((9, (0, 2)),),
    )
    k_ref, v_ref = _evicted_reference(k_full, v_full, "int8", set(evict_set))

    store = _make_store()
    _stepwise_append(store, k_full, v_full, "int8")
    k_before, v_before = store.gather()
    removed = store.evict_tokens(set(evict_set))
    k_paged, v_paged = store.gather()

    assert removed == 10
    assert torch.equal(k_paged, k_ref)
    assert torch.equal(v_paged, v_ref)
    assert torch.count_nonzero(k_paged[:, :, 3, :]) == 0
    assert torch.count_nonzero(v_paged[:, :, 16, :]) == 0
    assert torch.count_nonzero(k_paged[:, (0, 2), 9, :]) == 0
    assert torch.equal(k_paged[:, (1, 3), 9, :], k_before[:, (1, 3), 9, :])
    assert torch.equal(v_paged[:, (1, 3), 9, :], v_before[:, (1, 3), 9, :])
    assert torch.equal(k_paged[:, :, 0, :], k_before[:, :, 0, :])
    assert torch.equal(k_paged[:, :, 18, :], k_before[:, :, 18, :])


def test_evict_int4_golden_and_zeroed_heads():
    """INT4 eviction: evicted heads decode to 0.0, survivors bit-identical."""
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    evict_set = _evict_pairs(
        full_tokens=(5, 15),
        partial_heads=((11, (1, 3)),),
    )
    k_ref, v_ref = _evicted_reference(k_full, v_full, "int4", set(evict_set))

    store = _make_store()
    _stepwise_append(store, k_full, v_full, "int4")
    k_before, v_before = store.gather()
    removed = store.evict_tokens(set(evict_set))
    k_paged, v_paged = store.gather()

    assert removed == 10
    assert torch.equal(k_paged, k_ref)
    assert torch.equal(v_paged, v_ref)
    assert torch.count_nonzero(k_paged[:, :, 5, :]) == 0
    assert torch.count_nonzero(v_paged[:, :, 15, :]) == 0
    assert torch.count_nonzero(k_paged[:, (1, 3), 11, :]) == 0
    assert torch.equal(k_paged[:, (0, 2), 11, :], k_before[:, (0, 2), 11, :])
    assert torch.equal(v_paged[:, (0, 2), 11, :], v_before[:, (0, 2), 11, :])
    assert torch.equal(k_paged[:, :, 0, :], k_before[:, :, 0, :])
    assert torch.equal(k_paged[:, :, 18, :], k_before[:, :, 18, :])


def test_evict_full_tokens_release_exact_bytes():
    """Full-token eviction frees exact fp16 payload bytes and bumps stats."""
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    store = _make_store()
    _stepwise_append(store, k_full, v_full, "fp16")

    data_before = store.data_plane_bytes()
    resident_before = store.resident_bytes()
    per_token = BATCH * HEADS * HEAD_DIM * 2 * 2  # k+v fp16 bytes per token

    evict_set = _evict_pairs(full_tokens=(4, 5), partial_heads=((6, (0, 1)),))
    removed = store.evict_tokens(set(evict_set))

    assert removed == 10  # 2 full tokens (8 pairs) + 2 partial pairs
    assert store.data_plane_bytes() == data_before - 2 * per_token
    assert store.resident_bytes() < resident_before
    assert store.page_evictions >= 1  # only page 0 (tokens 4-6) was affected


def test_evict_partial_heads_save_no_page_bytes():
    """Partial-head eviction zeroes rows but frees no page-granularity bytes."""
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    store = _make_store()
    _stepwise_append(store, k_full, v_full, "fp16")

    data_before = store.data_plane_bytes()
    resident_before = store.resident_bytes()

    evict_set = _evict_pairs(partial_heads=((7, (0, 1)), (11, (2, 3))))
    removed = store.evict_tokens(set(evict_set))

    assert removed == 4
    assert store.data_plane_bytes() == data_before
    assert store.resident_bytes() == resident_before
    assert store.page_evictions >= 2  # pages holding tokens 7 and 11 affected

    k_paged, _ = store.gather()
    assert torch.count_nonzero(k_paged[:, (0, 1), 7, :]) == 0
    assert torch.count_nonzero(k_paged[:, (2, 3), 11, :]) == 0


def test_evict_full_page_returns_slot_to_free_list():
    """Whole-page eviction recycles the slot; later appends reuse it."""
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED, seq=12)
    store = _make_store(page_size=4)
    _stepwise_append(store, k_full, v_full, "fp16", split=(4, 4, 4))
    assert len(store.pages) == 3

    per_token = BATCH * HEADS * HEAD_DIM * 2 * 2
    removed = store.evict_tokens(set(_evict_pairs(full_tokens=(4, 5, 6, 7))))
    assert removed == 16
    assert len(store.pages) == 3  # slot pool keeps the recycled slot
    assert store.page_count == 2  # the evicted page left the token order
    assert store.data_plane_bytes() == 8 * per_token

    k_mid, v_mid = store.gather()
    assert k_mid.shape == (BATCH, HEADS, 12, HEAD_DIM)
    assert torch.equal(k_mid[:, :, 4:8, :], torch.zeros(BATCH, HEADS, 4, HEAD_DIM))
    assert torch.equal(k_mid[:, :, 0:4, :], k_full[:, :, 0:4, :].half().float())
    assert torch.equal(k_mid[:, :, 8:12, :], k_full[:, :, 8:12, :].half().float())

    gen = torch.Generator().manual_seed(GOLDEN_SEED + 2)
    extra_k = torch.randn(BATCH, HEADS, 2, HEAD_DIM, generator=gen) * 2.0
    extra_v = torch.randn(BATCH, HEADS, 2, HEAD_DIM, generator=gen) * 2.0
    store.append(extra_k.contiguous(), extra_v.contiguous(), "fp16")

    assert len(store.pages) == 3  # free slot reused instead of growing the pool
    assert store.page_count == 3
    assert store.max_seq == 14
    assert store.appended_tokens == 14

    k_after, v_after = store.gather()
    assert k_after.shape == (BATCH, HEADS, 14, HEAD_DIM)
    assert torch.equal(k_after[:, :, 12:14, :], extra_k.half().float())
    assert torch.equal(v_after[:, :, 12:14, :], extra_v.half().float())
    assert torch.equal(k_after[:, :, 4:8, :], torch.zeros(BATCH, HEADS, 4, HEAD_DIM))
    assert torch.equal(k_after[:, :, 0:4, :], k_mid[:, :, 0:4, :])
    assert torch.equal(k_after[:, :, 8:12, :], k_mid[:, :, 8:12, :])


def test_evict_middle_token_splits_page_into_segments():
    """A mid-page hole splits the page; segment start tokens stay in place."""
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    store = _make_store()
    _stepwise_append(store, k_full, v_full, "fp16")
    k_before, v_before = store.gather()
    page_count_before = store.page_count

    removed = store.evict_tokens(set(_evict_pairs(full_tokens=(3,))))
    assert removed == 4

    k_paged, v_paged = store.gather()
    assert k_paged.shape == (BATCH, HEADS, SEQ_LEN, HEAD_DIM)
    assert torch.count_nonzero(k_paged[:, :, 3, :]) == 0
    assert torch.count_nonzero(v_paged[:, :, 3, :]) == 0
    assert torch.equal(k_paged[:, :, 0:3, :], k_before[:, :, 0:3, :])
    assert torch.equal(v_paged[:, :, 0:3, :], v_before[:, :, 0:3, :])
    assert torch.equal(k_paged[:, :, 4:SEQ_LEN, :], k_before[:, :, 4:SEQ_LEN, :])
    assert torch.equal(v_paged[:, :, 4:SEQ_LEN, :], v_before[:, :, 4:SEQ_LEN, :])

    assert store.page_count >= page_count_before  # holed page became 2 segments
    segments = sorted(
        (page.start_token, page.end_token) for page in store.pages if page is not None
    )
    assert segments == [(0, 3), (4, 8), (8, 16), (16, 19)]


def test_evict_empty_set_is_noop():
    """Evicting an empty set changes nothing."""
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    store = _make_store()
    _stepwise_append(store, k_full, v_full, "fp16")
    before = store.gather()
    stats = (
        store.max_seq,
        store.page_count,
        store.open_pages,
        store.appended_tokens,
        store.data_plane_bytes(),
        store.resident_bytes(),
    )

    removed = store.evict_tokens(set())

    assert removed == 0
    assert store.page_evictions == 0
    stats_after = (
        store.max_seq,
        store.page_count,
        store.open_pages,
        store.appended_tokens,
        store.data_plane_bytes(),
        store.resident_bytes(),
    )
    assert stats_after == stats
    after = store.gather()
    assert torch.equal(after[0], before[0])
    assert torch.equal(after[1], before[1])


def test_evict_absent_and_already_evicted_positions_returns_zero():
    """Absent or already-evicted positions are not counted and never revive."""
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    store = _make_store()
    _stepwise_append(store, k_full, v_full, "fp16")

    removed = store.evict_tokens(set(_evict_pairs(full_tokens=(5,))))
    assert removed == 4
    after_first = store.gather()

    removed_again = store.evict_tokens({(5, 0), (5, 3), (SEQ_LEN + 5, 0), (0, HEADS + 3)})
    assert removed_again == 0

    after_second = store.gather()
    assert torch.equal(after_second[0], after_first[0])
    assert torch.equal(after_second[1], after_first[1])
    assert torch.count_nonzero(after_second[0][:, :, 5, :]) == 0


def test_evict_preserves_max_seq_and_supports_further_append():
    """max_seq survives eviction; appends continue at max_seq over the holes."""
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    store = _make_store()
    _stepwise_append(store, k_full, v_full, "fp16")
    k_before, v_before = store.gather()

    removed = store.evict_tokens(set(_evict_pairs(full_tokens=(3,))))
    assert removed == 4
    assert store.max_seq == SEQ_LEN
    k_holed, _ = store.gather()
    assert k_holed.shape[2] == SEQ_LEN

    gen = torch.Generator().manual_seed(GOLDEN_SEED + 3)
    extra_k = torch.randn(BATCH, HEADS, 2, HEAD_DIM, generator=gen) * 2.0
    extra_v = torch.randn(BATCH, HEADS, 2, HEAD_DIM, generator=gen) * 2.0
    store.append(extra_k.contiguous(), extra_v.contiguous(), "fp16")

    assert store.max_seq == SEQ_LEN + 2
    assert store.appended_tokens == SEQ_LEN + 2
    k_after, v_after = store.gather()
    assert k_after.shape == (BATCH, HEADS, SEQ_LEN + 2, HEAD_DIM)
    assert torch.equal(k_after[:, :, SEQ_LEN:, :], extra_k.half().float())
    assert torch.equal(v_after[:, :, SEQ_LEN:, :], extra_v.half().float())
    assert torch.count_nonzero(k_after[:, :, 3, :]) == 0  # hole stays zero
    assert torch.equal(k_after[:, :, 0:3, :], k_before[:, :, 0:3, :])
    assert torch.equal(k_after[:, :, 4:SEQ_LEN, :], k_before[:, :, 4:SEQ_LEN, :])


def test_evict_leaves_unaffected_pages_identity_intact():
    """Pages outside the eviction range keep object and buffer identity."""
    torch.manual_seed(GOLDEN_SEED)
    k_full, v_full = _make_kv(GOLDEN_SEED)
    store = _make_store()
    _stepwise_append(store, k_full, v_full, "fp16")
    assert len(store.pages) == 3  # pages cover [0,8) [8,16) [16,19)
    page1 = store.pages[1]
    page2 = store.pages[2]
    ptrs1 = _page_tensor_ptrs(page1)
    ptrs2 = _page_tensor_ptrs(page2)
    data_before = store.data_plane_bytes()
    per_token = BATCH * HEADS * HEAD_DIM * 2 * 2

    removed = store.evict_tokens(set(_evict_pairs(full_tokens=(1,))))
    assert removed == 4

    assert store.pages[1] is page1
    assert store.pages[2] is page2
    assert _page_tensor_ptrs(store.pages[1]) == ptrs1
    assert _page_tensor_ptrs(store.pages[2]) == ptrs2
    assert store.data_plane_bytes() == data_before - per_token
