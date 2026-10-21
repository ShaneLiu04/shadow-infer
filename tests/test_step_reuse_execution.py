"""AR013: cross-step block reuse and packed KV execution tests.

These tests exercise the real SimpleDiffusionLLM forward path with real
tensors. No mathematical operation is mocked.
"""

from __future__ import annotations

import pytest
import torch

from shadowinfer.core.backends.pytorch_backend import PyTorchModelBackend
from shadowinfer.core.structs import StepConfig
from shadowinfer.execution.step_reuse import StepReuseConfig, StepReuseState
from shadowinfer.models import SimpleDiffusionLLM
from shadowinfer.orchestrator import Orchestrator

SEQ_LEN = 8
NUM_LAYERS = 2
NUM_HEADS = 2
HEAD_DIM = 16
HIDDEN_DIM = 32
VOCAB_SIZE = 64
INTERMEDIATE_DIM = 64
MAX_SEQ_LEN = 32


def _tiny_model_config() -> dict:
    return {
        "name": "ar013-tiny",
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "hidden_dim": HIDDEN_DIM,
        "vocab_size": VOCAB_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "intermediate_dim": INTERMEDIATE_DIM,
    }


def _backend(seed: int = 123, num_layers: int = NUM_LAYERS) -> PyTorchModelBackend:
    cfg = _tiny_model_config()
    cfg["num_layers"] = num_layers
    backend = PyTorchModelBackend(model_config=cfg, seed=seed)
    backend.load(model_name="ar013-tiny", device="cpu")
    return backend


def _step(step_id: int, total_steps: int, mode: str = "conservative") -> StepConfig:
    return StepConfig(
        step_id=step_id,
        total_steps=total_steps,
        noise_level=0.0,
        shadowkv_mode=mode,
        ffn_mode="full",
    )


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


def _relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        (actual - reference).norm() / reference.norm().clamp_min(torch.finfo(torch.float32).tiny)
    )


class TestKVLifecycle:
    def test_replace_keeps_current_step_length(self):
        backend = _backend()
        backend.configure_execution(kv_lifecycle="replace")
        x = torch.randn(1, SEQ_LEN, HIDDEN_DIM)

        for step_id in range(3):
            result = backend.forward_step(x, _step(step_id, 3))
            for layer_id in range(NUM_LAYERS):
                entry = result["kv_cache"][layer_id]
                if entry.k_tensor is not None:
                    assert entry.k_tensor.shape[2] == SEQ_LEN
                else:
                    assert entry.packed_kv is not None
                    assert entry.packed_kv.seq_len == SEQ_LEN

    def test_append_preserves_legacy_growth(self):
        backend = _backend()
        x = torch.randn(1, SEQ_LEN, HIDDEN_DIM)

        first = backend.forward_step(x, _step(0, 2))
        second = backend.forward_step(x, _step(1, 2))

        for layer_id in range(NUM_LAYERS):
            assert first["kv_cache"][layer_id].k_tensor.shape[2] == SEQ_LEN
            assert second["kv_cache"][layer_id].k_tensor.shape[2] == 2 * SEQ_LEN

    def test_invalid_lifecycle_raises(self):
        backend = _backend()
        with pytest.raises(ValueError):
            backend.configure_execution(kv_lifecycle="grow")


class TestReuseDisabled:
    def test_two_calls_identical_and_no_state(self):
        backend = _backend()
        backend.configure_execution(kv_lifecycle="replace")
        x = torch.randn(1, SEQ_LEN, HIDDEN_DIM)

        first = backend.forward_step(x, _step(0, 2))
        second = backend.forward_step(x, _step(0, 2))

        assert torch.equal(first["output"], second["output"])
        assert getattr(backend, "_reuse_state", None) is None
        assert list(second["execution_stats"]["reused_layers"]) == []


class TestReuseMechanics:
    def test_controlled_trajectory_reuses_all_layers(self):
        reuse_backend = _backend(seed=321)
        reuse_backend.configure_execution(
            kv_lifecycle="replace",
            step_reuse_enabled=True,
            step_reuse_threshold=0.1,
            step_reuse_warmup_steps=0,
        )
        reference_backend = _backend(seed=321)
        reference_backend.configure_execution(kv_lifecycle="replace")

        gen = torch.Generator().manual_seed(7)
        x0 = torch.randn(1, SEQ_LEN, HIDDEN_DIM, generator=gen) * 3.0
        noise = torch.randn(1, SEQ_LEN, HIDDEN_DIM, generator=gen)
        sigmas = (0.05, 0.04, 0.03)
        trajectory = [x0 + sigma * noise for sigma in sigmas]

        reuse_backend.forward_step(trajectory[0], _step(0, 3))
        reuse_result = reuse_backend.forward_step(trajectory[1], _step(1, 3))

        reference_backend.forward_step(trajectory[0], _step(0, 3))
        reference_result = reference_backend.forward_step(trajectory[1], _step(1, 3))

        stats = reuse_result["execution_stats"]
        assert sorted(stats["reused_layers"]) == list(range(NUM_LAYERS))
        assert stats["reused_ratio"] >= 0.1
        assert set(stats["drifts"]) == set(range(NUM_LAYERS))
        assert all(d is not None and d <= 0.1 for d in stats["drifts"].values())
        assert _relative_l2(reuse_result["output"], reference_result["output"]) <= 0.02

    def test_warmup_blocks_reuse(self):
        backend = _backend(seed=11)
        backend.configure_execution(
            kv_lifecycle="replace",
            step_reuse_enabled=True,
            step_reuse_threshold=0.1,
            step_reuse_warmup_steps=2,
        )

        gen = torch.Generator().manual_seed(9)
        x0 = torch.randn(1, SEQ_LEN, HIDDEN_DIM, generator=gen) * 3.0
        noise = torch.randn(1, SEQ_LEN, HIDDEN_DIM, generator=gen)
        trajectory = [x0 + sigma * noise for sigma in (0.05, 0.04, 0.03)]

        for step_id in (0, 1):
            result = backend.forward_step(trajectory[step_id], _step(step_id, 3))
            assert list(result["execution_stats"]["reused_layers"]) == []


class TestReuseCorrectness:
    def test_manual_cached_delta_formula(self):
        cfg = _tiny_model_config()
        cfg.pop("name")
        cfg["num_layers"] = 1
        model = SimpleDiffusionLLM(**cfg)
        state = StepReuseState()
        reuse_cfg = StepReuseConfig(enabled=True, threshold=1e6, warmup_steps=0)

        gen = torch.Generator().manual_seed(13)
        x0 = torch.randn(1, SEQ_LEN, HIDDEN_DIM, generator=gen)
        model.forward_step(x0, step_t=0, total_steps=2, reuse_state=state, reuse_config=reuse_cfg)
        assert 0 in state.entries

        entry = state.entries[0]
        cached_in = torch.full((1, SEQ_LEN, HIDDEN_DIM), 0.123)
        cached_out = torch.full((1, SEQ_LEN, HIDDEN_DIM), -0.456)
        entry.hidden_in = cached_in
        entry.hidden_out = cached_out

        x1 = x0 + 0.001
        logits, _, _ = model.forward_step(
            x1, step_t=1, total_steps=2, reuse_state=state, reuse_config=reuse_cfg
        )
        assert 0 in state.last_reused_layers

        positions = torch.arange(SEQ_LEN, dtype=torch.long)
        ratio = torch.tensor([[1.0 / 2.0]])
        hidden_in = x1 + model.pos_embed(positions).unsqueeze(0)
        hidden_in = hidden_in + model.time_embed(ratio).unsqueeze(1)

        expected_out = cached_out + (hidden_in - cached_in)
        expected_logits = model.output_proj(model.out_norm(expected_out))

        assert torch.allclose(logits, expected_logits, atol=1e-6)


class TestPackedExecution:
    def test_packed_entries_and_quality(self):
        packed_backend = _backend(seed=55)
        packed_backend.configure_execution(
            kv_lifecycle="replace", kv_execution="packed", packed_precision="fp16"
        )
        dense_backend = _backend(seed=55)
        dense_backend.configure_execution(kv_lifecycle="replace", kv_execution="dense")

        gen = torch.Generator().manual_seed(17)
        inputs = [
            torch.randn(1, SEQ_LEN, HIDDEN_DIM, generator=gen),
            torch.randn(1, SEQ_LEN, HIDDEN_DIM, generator=gen),
        ]

        for step_id, x in enumerate(inputs):
            packed_result = packed_backend.forward_step(x, _step(step_id, 2, "balanced"))
            dense_result = dense_backend.forward_step(x, _step(step_id, 2, "balanced"))

            for layer_id in range(NUM_LAYERS):
                entry = packed_result["kv_cache"][layer_id]
                assert entry.k_tensor is None
                assert entry.v_tensor is None
                assert entry.packed_kv is not None

            stats = packed_result["execution_stats"]
            assert stats["data_plane_bytes"] <= stats["dense_bytes"]
            assert stats["resident_bytes"] == (stats["data_plane_bytes"] + stats["metadata_bytes"])
            assert stats["metadata_bytes"] > 0
            assert torch.allclose(packed_result["output"], dense_result["output"], atol=2e-3)

    def test_packed_next_step_has_no_dense_tensors(self):
        backend = _backend(seed=56)
        backend.configure_execution(
            kv_lifecycle="replace", kv_execution="packed", packed_precision="fp16"
        )
        x = torch.randn(1, SEQ_LEN, HIDDEN_DIM)

        for step_id in range(2):
            result = backend.forward_step(x, _step(step_id, 2, "balanced"))
            assert result["execution_stats"]["packed_layers"] == NUM_LAYERS
            for entry in result["kv_cache"].values():
                assert entry.k_tensor is None
                assert entry.v_tensor is None
                assert entry.packed_kv is not None


class TestPackedAppendExecution:
    def test_packed_past_is_consumed_in_append_mode(self):
        """Packed K/V from step 1 must feed step 2 attention (real closed loop)."""
        packed_backend = _backend(seed=61)
        packed_backend.configure_execution(
            kv_lifecycle="append", kv_execution="packed", packed_precision="fp16"
        )
        dense_backend = _backend(seed=61)
        dense_backend.configure_execution(kv_lifecycle="append", kv_execution="dense")

        gen = torch.Generator().manual_seed(29)
        inputs = [
            torch.randn(1, 4, HIDDEN_DIM, generator=gen),
            torch.randn(1, 5, HIDDEN_DIM, generator=gen),
        ]

        for step_id, x in enumerate(inputs):
            packed_result = packed_backend.forward_step(x, _step(step_id, 2, "balanced"))
            dense_result = dense_backend.forward_step(x, _step(step_id, 2, "balanced"))
            assert torch.allclose(packed_result["output"], dense_result["output"], atol=2e-3)

        for entry in packed_result["kv_cache"].values():
            assert entry.k_tensor is None
            assert entry.packed_kv is not None
            assert entry.packed_kv.seq_len == 4 + 5
        assert packed_result["execution_stats"]["packed_layers"] == NUM_LAYERS


class TestAggressivePacked:
    WIDE_HIDDEN = 128

    def _wide_backend(self, seed: int) -> PyTorchModelBackend:
        cfg = _tiny_model_config()
        cfg["head_dim"] = 64
        cfg["hidden_dim"] = self.WIDE_HIDDEN
        cfg["max_seq_len"] = 64
        backend = PyTorchModelBackend(model_config=cfg, seed=seed)
        backend.load(model_name="ar013-tiny", device="cpu")
        return backend

    def test_int8_two_steps_resident_and_quality(self):
        packed_backend = self._wide_backend(seed=77)
        packed_backend.configure_execution(
            kv_lifecycle="append", kv_execution="packed", packed_precision="int8"
        )
        dense_backend = self._wide_backend(seed=77)
        dense_backend.configure_execution(kv_lifecycle="append", kv_execution="dense")

        gen = torch.Generator().manual_seed(19)
        inputs = [
            torch.randn(2, 4, self.WIDE_HIDDEN, generator=gen),
            torch.randn(2, 5, self.WIDE_HIDDEN, generator=gen),
        ]

        for step_id, x in enumerate(inputs):
            packed_result = packed_backend.forward_step(x, _step(step_id, 2, "aggressive"))
            dense_result = dense_backend.forward_step(x, _step(step_id, 2, "aggressive"))
            stats = packed_result["execution_stats"]
            assert stats["precision"] == "int8"
            assert stats["data_plane_ratio"] <= 0.6
            assert _relative_l2(packed_result["output"], dense_result["output"]) <= 0.02

        for entry in packed_result["kv_cache"].values():
            assert entry.k_tensor is None
            assert entry.packed_kv is not None
            assert entry.packed_kv.seq_len == 4 + 5


class TestCombinedReusePacked:
    def test_skipped_layers_keep_packed_object(self):
        backend = _backend(seed=91)
        backend.configure_execution(
            kv_lifecycle="replace",
            kv_execution="packed",
            packed_precision="fp16",
            step_reuse_enabled=True,
            step_reuse_threshold=1e6,
            step_reuse_warmup_steps=0,
        )

        gen = torch.Generator().manual_seed(23)
        x0 = torch.randn(1, SEQ_LEN, HIDDEN_DIM, generator=gen)
        x1 = x0 + 1e-4 * torch.randn(1, SEQ_LEN, HIDDEN_DIM, generator=gen)

        first = backend.forward_step(x0, _step(0, 2, "balanced"))
        first_packed = {layer_id: entry.packed_kv for layer_id, entry in first["kv_cache"].items()}
        first_resident = first["execution_stats"]["resident_bytes"]

        second = backend.forward_step(x1, _step(1, 2, "balanced"))
        stats = second["execution_stats"]
        reused = sorted(stats["reused_layers"])

        assert reused == list(range(NUM_LAYERS))
        for layer_id in reused:
            assert second["kv_cache"][layer_id].packed_kv is first_packed[layer_id]
        assert stats["packed_layers"] == len(stats["computed_layers"])
        assert stats["skipped_repack"] == len(stats["reused_layers"])
        assert stats["resident_bytes"] == first_resident


class TestInvalidExecutionConfig:
    def test_negative_threshold_raises(self):
        backend = _backend()
        with pytest.raises(ValueError):
            backend.configure_execution(step_reuse_threshold=-0.01)
        with pytest.raises(ValueError):
            StepReuseConfig(enabled=True, threshold=-0.01, warmup_steps=0).validate()

    def test_bad_kv_execution_raises(self):
        backend = _backend()
        with pytest.raises(ValueError):
            backend.configure_execution(kv_execution="sparse")

    def test_bad_packed_precision_raises(self):
        backend = _backend()
        with pytest.raises(ValueError):
            backend.configure_execution(packed_precision="int3")


class TestResetExecutionState:
    def test_reset_clears_all_execution_state(self):
        backend = _backend(seed=101)
        backend.configure_execution(
            kv_lifecycle="replace",
            kv_execution="packed",
            packed_precision="fp16",
            step_reuse_enabled=True,
            step_reuse_threshold=1e6,
            step_reuse_warmup_steps=0,
        )
        x = torch.randn(1, SEQ_LEN, HIDDEN_DIM)

        backend.forward_step(x, _step(0, 2, "balanced"))
        backend.forward_step(x + 1e-4, _step(1, 2, "balanced"))

        assert backend.get_kv_cache()
        state = getattr(backend, "_reuse_state", None)
        assert state is not None and state.entries

        backend.reset_execution_state()

        assert backend.get_kv_cache() == {}
        assert not getattr(backend, "_packed_cache", None)
        reset_state = getattr(backend, "_reuse_state", None)
        assert reset_state is None or not reset_state.entries

        result = backend.forward_step(x, _step(0, 2, "balanced"))
        assert list(result["execution_stats"]["reused_layers"]) == []


class TestOrchestratorExecutionWiring:
    def test_initialize_applies_execution_and_runs_reset(self, tmp_path, monkeypatch):
        backend = _backend(seed=131)
        config = {
            "execution": {
                "kv_lifecycle": "replace",
                "step_reuse_enabled": True,
                "step_reuse_threshold": 0.5,
            },
            "output_dir": str(tmp_path),
            "log_dir": str(tmp_path / "logs"),
            "early_stop": {"enabled": False},
        }
        orchestrator = Orchestrator(config=config, model_backend=backend)
        model_config = _tiny_model_config()
        model_config.update({"batch_size": 1, "seq_len": SEQ_LEN})
        orchestrator.initialize(model_config=model_config)

        assert _execution_setting(backend, "kv_lifecycle") == "replace"
        assert _execution_setting(backend, "step_reuse_enabled") is True
        assert float(_execution_setting(backend, "step_reuse_threshold")) == pytest.approx(0.5)

        calls = {"count": 0}
        original_reset = backend.reset_execution_state

        def counting_reset(*args, **kwargs):
            calls["count"] += 1
            return original_reset(*args, **kwargs)

        monkeypatch.setattr(backend, "reset_execution_state", counting_reset)

        orchestrator.run_optimized("ar013", num_steps=2)
        orchestrator.run_optimized("ar013", num_steps=2)

        assert calls["count"] >= 2


class TestPackedFallback:
    def test_pack_failure_falls_back_to_dense(self, monkeypatch):
        from shadowinfer.core.backends import pytorch_backend as backend_mod

        packed_backend = _backend(seed=141)
        packed_backend.configure_execution(
            kv_lifecycle="replace", kv_execution="packed", packed_precision="fp16"
        )
        dense_backend = _backend(seed=141)
        dense_backend.configure_execution(kv_lifecycle="replace", kv_execution="dense")

        def failing_pack(*args, **kwargs):
            raise RuntimeError("synthetic pack failure")

        monkeypatch.setattr(backend_mod.PackedKVCache, "pack", failing_pack)

        x = torch.randn(1, SEQ_LEN, HIDDEN_DIM)
        packed_result = packed_backend.forward_step(x, _step(0, 2, "balanced"))
        dense_result = dense_backend.forward_step(x, _step(0, 2, "balanced"))

        stats = packed_result["execution_stats"]
        assert stats["fallback"] is True
        for entry in packed_result["kv_cache"].values():
            assert entry.k_tensor is not None
            assert entry.packed_kv is None
        assert not getattr(packed_backend, "_packed_cache", None)
        assert torch.equal(packed_result["output"], dense_result["output"])

        monkeypatch.undo()
        recovered = packed_backend.forward_step(x, _step(1, 2, "balanced"))
        assert recovered["execution_stats"]["fallback"] is False
        assert recovered["execution_stats"]["packed_layers"] == NUM_LAYERS


class TestShapeMismatchReset:
    def test_shape_mismatch_clears_cache_with_reason(self):
        backend = _backend(seed=151)
        backend.configure_execution(kv_lifecycle="replace")
        x = torch.randn(1, SEQ_LEN, HIDDEN_DIM)

        backend.forward_step(x, _step(0, 2, "balanced"))
        assert backend.get_kv_cache()

        bad_k = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM + 8)
        bad_v = torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM + 8)
        from shadowinfer.core.structs import KVCacheEntry

        backend.set_kv_cache({0: KVCacheEntry(k_tensor=bad_k, v_tensor=bad_v, precision="fp32")})

        result = backend.forward_step(x, _step(1, 2, "balanced"))

        stats = result["execution_stats"]
        assert stats["reset_reason"] == "shape_mismatch"
        for entry in backend.get_kv_cache().values():
            kv = backend._entry_kv_tensors(entry)
            assert kv is not None
            assert kv[0].shape[3] == HEAD_DIM
        assert torch.isfinite(result["output"]).all()

    def test_matching_shapes_do_not_trigger_reset(self):
        backend = _backend(seed=152)
        backend.configure_execution(kv_lifecycle="replace")
        x = torch.randn(1, SEQ_LEN, HIDDEN_DIM)

        backend.forward_step(x, _step(0, 2, "balanced"))
        result = backend.forward_step(x, _step(1, 2, "balanced"))

        assert result["execution_stats"]["reset_reason"] is None


class TestRunStreamReset:
    def test_run_stream_resets_execution_state(self, tmp_path, monkeypatch):
        backend = _backend(seed=161)
        config = {
            "execution": {
                "kv_lifecycle": "replace",
                "step_reuse_enabled": True,
                "step_reuse_threshold": 0.1,
            },
            "output_dir": str(tmp_path),
            "log_dir": str(tmp_path / "logs"),
            "early_stop": {"enabled": False},
        }
        orchestrator = Orchestrator(config=config, model_backend=backend)
        model_config = _tiny_model_config()
        model_config.update({"batch_size": 1, "seq_len": SEQ_LEN})
        orchestrator.initialize(model_config=model_config)

        calls = {"count": 0}
        original_reset = backend.reset_execution_state

        def counting_reset(*args, **kwargs):
            calls["count"] += 1
            return original_reset(*args, **kwargs)

        monkeypatch.setattr(backend, "reset_execution_state", counting_reset)

        list(orchestrator.run_stream("ar013", num_steps=2))
        list(orchestrator.run_stream("ar013", num_steps=2))

        assert calls["count"] >= 2


class TestOrchestratorFallbackMarker:
    def test_backend_forward_failure_sets_fallback_marker(self, tmp_path, monkeypatch):
        backend = _backend(seed=181)
        config = {
            "execution": {
                "kv_lifecycle": "replace",
                "step_reuse_enabled": True,
                "step_reuse_threshold": 0.1,
            },
            "output_dir": str(tmp_path),
            "log_dir": str(tmp_path / "logs"),
            "early_stop": {"enabled": False},
        }
        orchestrator = Orchestrator(config=config, model_backend=backend)
        model_config = _tiny_model_config()
        model_config.update({"batch_size": 1, "seq_len": SEQ_LEN})
        orchestrator.initialize(model_config=model_config)

        def failing_forward_step(*args, **kwargs):
            raise RuntimeError("synthetic backend failure")

        monkeypatch.setattr(backend, "forward_step", failing_forward_step)

        orchestrator.run_optimized("ar013", num_steps=1)

        last = orchestrator.step_results[-1]
        execution = last["shadowkv"]["execution"]
        assert execution["fallback"] is True
        assert "backend_forward_failed" in execution["fallback_reason"]


class TestExecutionStatsVisibility:
    def test_execution_stats_reach_shadowkv_and_profiler(self, tmp_path):
        backend = _backend(seed=171)
        config = {
            "execution": {
                "kv_lifecycle": "replace",
                "kv_execution": "packed",
                "packed_precision": "fp16",
                "step_reuse_enabled": True,
                "step_reuse_threshold": 0.1,
            },
            "output_dir": str(tmp_path),
            "log_dir": str(tmp_path / "logs"),
            "early_stop": {"enabled": False},
        }
        orchestrator = Orchestrator(config=config, model_backend=backend)
        model_config = _tiny_model_config()
        model_config.update({"batch_size": 1, "seq_len": SEQ_LEN})
        orchestrator.initialize(model_config=model_config)

        orchestrator.run_optimized("ar013", num_steps=2)

        assert orchestrator.step_results
        last = orchestrator.step_results[-1]
        kv_execution = last["shadowkv"]["execution"]
        assert kv_execution["kv_lifecycle"] == "replace"
        assert kv_execution["kv_execution"] == "packed"
        assert "drifts" in kv_execution
        assert "resident_bytes" in kv_execution
        assert "data_plane_ratio" in kv_execution
        profiler_execution = last["profiler"]["execution"]
        assert profiler_execution["kv_execution"] == "packed"
        assert profiler_execution["reused_ratio"] == kv_execution["reused_ratio"]
