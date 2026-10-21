"""AR014 T003-T005: paged backend execution wiring tests.

Exercises the not-yet-implemented ``kv_execution="paged"`` path of
``PyTorchModelBackend`` (configure/forward_step/execution_stats/reset/
fallback/shape-mismatch) plus the orchestrator ``execution`` config
passthrough. The paged path must be logits-equivalent to the packed path
(same seed, same inputs, same precision) because both quantize rows
identically; fp16 round-trips are idempotent, so ``torch.equal`` is the
bar for the fp16 group.

All tests use the real SimpleDiffusionLLM forward path with real tensors.
No mathematical operation is mocked; only failure injection uses
monkeypatch (AR013 ``TestPackedFallback`` pattern).
"""

from __future__ import annotations

import math

import pytest
import torch

from shadowinfer.core.backends.pytorch_backend import PyTorchModelBackend
from shadowinfer.core.structs import KVCacheEntry, StepConfig
from shadowinfer.orchestrator import Orchestrator
from shadowinfer.shadowkv.paged_kv_store import PagedKVStore

SEQ_LEN = 8
NUM_LAYERS = 2
NUM_HEADS = 2
HEAD_DIM = 16
HIDDEN_DIM = 32
VOCAB_SIZE = 64
INTERMEDIATE_DIM = 64
MAX_SEQ_LEN = 64
PAGE_SIZE = 4
# Crosses page boundaries: 5 -> pages [0,4)+[4,5); 9 -> 3 pages; 13 -> 4 pages.
STEP_LENGTHS = (5, 4, 4)


def _tiny_model_config() -> dict:
    return {
        "name": "ar014-tiny",
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "hidden_dim": HIDDEN_DIM,
        "vocab_size": VOCAB_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "intermediate_dim": INTERMEDIATE_DIM,
    }


def _backend(seed: int) -> PyTorchModelBackend:
    backend = PyTorchModelBackend(model_config=_tiny_model_config(), seed=seed)
    backend.load(model_name="ar014-tiny", device="cpu")
    return backend


def _paged_backend(seed: int, lifecycle: str = "append") -> PyTorchModelBackend:
    backend = _backend(seed)
    backend.configure_execution(
        kv_lifecycle=lifecycle,
        kv_execution="paged",
        packed_precision="fp16",
        page_size=PAGE_SIZE,
    )
    return backend


def _step(step_id: int, total_steps: int, mode: str = "balanced") -> StepConfig:
    return StepConfig(
        step_id=step_id,
        total_steps=total_steps,
        noise_level=0.0,
        shadowkv_mode=mode,
        ffn_mode="full",
    )


def _make_inputs(seed: int, lengths=STEP_LENGTHS) -> list:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(1, length, HIDDEN_DIM, generator=gen) for length in lengths]


def _execution_setting(backend: PyTorchModelBackend, name: str):
    attr = f"_{name}"
    if hasattr(backend, attr):
        return getattr(backend, attr)
    cfg = getattr(backend, "_execution_config", None)
    if cfg is None:
        return None
    if hasattr(cfg, "get"):
        return cfg.get(name)
    return cfg[name]


def _page_tensor_ptrs(page):
    """Snapshot data_ptr() of a page payload (same helper as AR014 T001)."""
    ptrs = []
    for name in ("k", "v", "scale_k", "scale_v"):
        tensor = getattr(page, name, None)
        if isinstance(tensor, torch.Tensor):
            ptrs.append((name, tensor.data_ptr()))
    return tuple(ptrs)


class TestPagedConfigValidation:
    def test_paged_execution_is_accepted(self):
        backend = _backend(seed=201)
        backend.configure_execution(kv_execution="paged", page_size=4)
        assert _execution_setting(backend, "kv_execution") == "paged"
        assert _execution_setting(backend, "page_size") == 4

        default_backend = _backend(seed=202)
        default_backend.configure_execution(kv_execution="paged")
        assert _execution_setting(default_backend, "page_size") == 32

    def test_bogus_kv_execution_still_raises(self):
        backend = _backend(seed=203)
        with pytest.raises(ValueError):
            backend.configure_execution(kv_execution="bogus")

    def test_page_size_below_one_raises(self):
        backend = _backend(seed=204)
        with pytest.raises(ValueError):
            backend.configure_execution(kv_execution="paged", page_size=0)
        with pytest.raises(ValueError):
            backend.configure_execution(page_size=-1)


class TestPagedAppendEquivalence:
    def test_paged_matches_packed_logits_fp16(self):
        packed_backend = _backend(seed=211)
        packed_backend.configure_execution(
            kv_lifecycle="append", kv_execution="packed", packed_precision="fp16"
        )
        paged_backend = _paged_backend(seed=211)

        inputs = _make_inputs(seed=31)
        for step_id, x in enumerate(inputs):
            packed_result = packed_backend.forward_step(x, _step(step_id, len(inputs)))
            paged_result = paged_backend.forward_step(x, _step(step_id, len(inputs)))
            assert torch.equal(paged_result["output"], packed_result["output"])

        total = sum(STEP_LENGTHS)
        for layer_id in range(NUM_LAYERS):
            packed_entry = packed_backend.get_kv_cache()[layer_id]
            paged_entry = paged_backend.get_kv_cache()[layer_id]
            assert paged_entry.paged_kv is not None
            assert paged_entry.packed_kv is None
            assert paged_entry.k_tensor is None
            store = paged_entry.paged_kv
            assert store.max_seq == total
            assert store.appended_tokens == total
            # fp16 half -> float -> half is idempotent, so incremental pages
            # must decode bit-identically to the full re-pack.
            k_packed, v_packed = packed_entry.packed_kv.unpack()
            k_paged, v_paged = store.gather()
            assert torch.equal(k_paged, k_packed)
            assert torch.equal(v_paged, v_packed)

    def test_paged_matches_packed_logits_int8(self):
        packed_backend = _backend(seed=212)
        packed_backend.configure_execution(
            kv_lifecycle="append", kv_execution="packed", packed_precision="int8"
        )
        paged_backend = _backend(seed=212)
        paged_backend.configure_execution(
            kv_lifecycle="append",
            kv_execution="paged",
            packed_precision="int8",
            page_size=PAGE_SIZE,
        )

        inputs = _make_inputs(seed=32)
        for step_id, x in enumerate(inputs):
            packed_result = packed_backend.forward_step(x, _step(step_id, len(inputs)))
            paged_result = paged_backend.forward_step(x, _step(step_id, len(inputs)))
            # Packed re-quantizes past rows each step; paged keeps the original
            # codes, so equality degrades to a tight numeric tolerance.
            assert torch.allclose(paged_result["output"], packed_result["output"], atol=2e-3)

        total = sum(STEP_LENGTHS)
        for entry in paged_backend.get_kv_cache().values():
            assert entry.paged_kv is not None
            assert entry.paged_kv.max_seq == total
            assert entry.paged_kv.appended_tokens == total


class TestPagedExecutionStats:
    def test_stats_track_paged_append_accounting(self):
        backend = _paged_backend(seed=221)
        inputs = _make_inputs(seed=33)
        total = 0
        page_counts = []

        for step_id, x in enumerate(inputs):
            result = backend.forward_step(x, _step(step_id, len(inputs)))
            stats = result["execution_stats"]
            new_tokens = STEP_LENGTHS[step_id]
            past_tokens = total
            total += new_tokens

            assert stats["kv_execution"] == "paged"
            assert stats["paged"] is True
            assert stats["page_size"] == PAGE_SIZE
            assert stats["fallback"] is False
            assert stats["reset_reason"] is None
            assert stats["page_evictions"] == 0
            assert stats["appended_tokens"] == NUM_LAYERS * new_tokens
            assert stats["appended_tokens_total"] == NUM_LAYERS * total
            assert stats["repack_tokens_saved"] == NUM_LAYERS * past_tokens

            expected_pages = NUM_LAYERS * math.ceil(total / PAGE_SIZE)
            assert stats["page_count"] == expected_pages
            expected_open = NUM_LAYERS * (1 if total % PAGE_SIZE else 0)
            assert stats["open_pages"] == expected_open
            page_counts.append(stats["page_count"])

            # fp16 pages carry exactly the dense fp16 baseline payload bytes.
            assert stats["data_plane_bytes"] == stats["dense_bytes"]
            assert stats["data_plane_ratio"] == pytest.approx(1.0)
            assert stats["metadata_bytes"] > 0
            assert stats["resident_bytes"] == (stats["data_plane_bytes"] + stats["metadata_bytes"])
            assert stats["resident_bytes"] > stats["data_plane_bytes"]

            # The dual ledger must equal the per-layer store ledgers.
            stores = [entry.paged_kv for entry in backend.get_kv_cache().values()]
            assert len(stores) == NUM_LAYERS
            assert stats["data_plane_bytes"] == sum(s.data_plane_bytes() for s in stores)
            assert stats["resident_bytes"] == sum(s.resident_bytes() for s in stores)

        assert page_counts == sorted(page_counts)
        assert len(set(page_counts)) == len(page_counts)  # grows every step


class TestPagedIncrementalImmutability:
    def test_sealed_page_identity_survives_next_append(self):
        backend = _paged_backend(seed=231)
        inputs = _make_inputs(seed=34, lengths=(5, 4))

        backend.forward_step(inputs[0], _step(0, 2))
        store = backend.get_kv_cache()[0].paged_kv
        assert store is not None
        assert store.page_count == 2  # [0,4) sealed + [4,5) open
        sealed_page = store.pages[0]
        sealed_ptrs = _page_tensor_ptrs(sealed_page)
        assert sealed_page.num_tokens == PAGE_SIZE

        backend.forward_step(inputs[1], _step(1, 2))
        store_after = backend.get_kv_cache()[0].paged_kv
        assert store_after is store  # the per-layer store instance persists
        assert store_after.pages[0] is sealed_page
        assert _page_tensor_ptrs(store_after.pages[0]) == sealed_ptrs


class TestPagedReplaceLifecycle:
    def test_replace_paged_repages_each_step(self):
        backend = _paged_backend(seed=241, lifecycle="replace")
        x = torch.randn(1, SEQ_LEN, HIDDEN_DIM)

        for step_id in range(2):
            result = backend.forward_step(x, _step(step_id, 2))
            stats = result["execution_stats"]
            assert stats["paged"] is True
            assert stats["fallback"] is False
            assert torch.isfinite(result["output"]).all()
            # Replace rewrites the whole page set each step: full append, no past.
            assert stats["appended_tokens"] == NUM_LAYERS * SEQ_LEN
            assert stats["repack_tokens_saved"] == 0
            assert stats["page_count"] == NUM_LAYERS * math.ceil(SEQ_LEN / PAGE_SIZE)
            for entry in result["kv_cache"].values():
                assert entry.paged_kv is not None
                assert entry.paged_kv.max_seq == SEQ_LEN


class TestPagedReset:
    def test_reset_clears_paged_state(self):
        backend = _paged_backend(seed=251)
        inputs = _make_inputs(seed=35, lengths=(5, 4))

        first = backend.forward_step(inputs[0], _step(0, 2))
        backend.forward_step(inputs[1], _step(1, 2))
        assert backend.get_kv_cache()
        assert getattr(backend, "_paged_cache", None)

        backend.reset_execution_state()

        assert backend.get_kv_cache() == {}
        assert not getattr(backend, "_paged_cache", None)

        rerun = backend.forward_step(inputs[0], _step(0, 2))
        rerun_stats = rerun["execution_stats"]
        assert rerun_stats["fallback"] is False
        assert rerun_stats["page_count"] == first["execution_stats"]["page_count"]
        assert (
            rerun_stats["appended_tokens_total"]
            == first["execution_stats"]["appended_tokens_total"]
        )


class TestPagedShapeMismatch:
    def test_paged_shape_mismatch_clears_cache_with_reason(self):
        backend = _paged_backend(seed=261)
        inputs = _make_inputs(seed=36, lengths=(5, 4))

        backend.forward_step(inputs[0], _step(0, 2))
        assert backend.get_kv_cache()

        bad_store = PagedKVStore(1, NUM_HEADS, HEAD_DIM + 8, page_size=PAGE_SIZE)
        bad_entry = KVCacheEntry(k_tensor=None, v_tensor=None, precision="fp32")
        bad_entry.paged_kv = bad_store
        backend.set_kv_cache({0: bad_entry})

        result = backend.forward_step(inputs[1], _step(1, 2))
        stats = result["execution_stats"]
        assert stats["reset_reason"] == "shape_mismatch"
        assert stats["paged"] is True
        assert stats["fallback"] is False
        # The cleared cache restarts from scratch: no past, full fresh append.
        assert stats["repack_tokens_saved"] == 0
        assert stats["appended_tokens"] == NUM_LAYERS * STEP_LENGTHS[1]
        assert torch.isfinite(result["output"]).all()


class TestPagedFallback:
    def test_append_failure_falls_back_to_dense_and_recovers(self, monkeypatch):
        paged_backend = _paged_backend(seed=271)
        dense_backend = _backend(seed=271)
        dense_backend.configure_execution(kv_lifecycle="append", kv_execution="dense")

        def failing_append(self, k_new, v_new, precision):
            raise RuntimeError("synthetic page append failure")

        monkeypatch.setattr(PagedKVStore, "append", failing_append)

        inputs = _make_inputs(seed=37)
        first_paged = paged_backend.forward_step(inputs[0], _step(0, 3))
        first_dense = dense_backend.forward_step(inputs[0], _step(0, 3))

        stats = first_paged["execution_stats"]
        assert stats["fallback"] is True
        for entry in first_paged["kv_cache"].values():
            assert entry.k_tensor is not None
            assert entry.paged_kv is None
        assert not getattr(paged_backend, "_paged_cache", None)
        assert torch.equal(first_paged["output"], first_dense["output"])

        monkeypatch.undo()

        second_paged = paged_backend.forward_step(inputs[1], _step(1, 3))
        second_stats = second_paged["execution_stats"]
        assert second_stats["fallback"] is False
        assert second_stats["paged"] is True
        total_after_recovery = STEP_LENGTHS[0] + STEP_LENGTHS[1]
        for entry in second_paged["kv_cache"].values():
            store = entry.paged_kv
            assert store is not None
            # Recovery re-pages the full history so no context is lost.
            assert store.max_seq == total_after_recovery

        dense_backend.forward_step(inputs[1], _step(1, 3))
        third_paged = paged_backend.forward_step(inputs[2], _step(2, 3))
        third_dense = dense_backend.forward_step(inputs[2], _step(2, 3))
        assert third_paged["execution_stats"]["fallback"] is False
        assert torch.allclose(third_paged["output"], third_dense["output"], atol=2e-3)


class TestOrchestratorPagedWiring:
    def test_paged_execution_config_reaches_backend_and_results(self, tmp_path):
        backend = _backend(seed=281)
        config = {
            "execution": {
                "kv_execution": "paged",
                "page_size": 4,
                "packed_precision": "fp16",
            },
            "output_dir": str(tmp_path),
            "log_dir": str(tmp_path / "logs"),
            "early_stop": {"enabled": False},
        }
        orchestrator = Orchestrator(config=config, model_backend=backend)
        model_config = _tiny_model_config()
        model_config.update({"batch_size": 1, "seq_len": SEQ_LEN})
        orchestrator.initialize(model_config=model_config)

        assert _execution_setting(backend, "kv_execution") == "paged"
        assert _execution_setting(backend, "page_size") == 4

        orchestrator.run_optimized("ar014-paged", num_steps=2)

        assert orchestrator.step_results
        last = orchestrator.step_results[-1]
        execution = last["shadowkv"]["execution"]
        assert execution["kv_execution"] == "paged"
        assert execution["paged"] is True
        assert execution["page_size"] == 4
        assert "repack_tokens_saved" in execution
        assert "appended_tokens" in execution

        profiler_execution = last["profiler"]["execution"]
        assert profiler_execution["kv_execution"] == "paged"
        assert "page_size" in profiler_execution


class TestDenseDefaultRegression:
    def test_default_execution_stays_dense(self):
        backend = _backend(seed=291)
        x = torch.randn(1, SEQ_LEN, HIDDEN_DIM)
        result = backend.forward_step(x, _step(0, 1))
        stats = result["execution_stats"]
        assert stats["kv_execution"] == "dense"
        assert stats.get("paged", False) is False
        for entry in result["kv_cache"].values():
            assert entry.k_tensor is not None
            assert entry.packed_kv is None
            assert getattr(entry, "paged_kv", None) is None
