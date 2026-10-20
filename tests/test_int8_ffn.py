"""Tests for AR011 executable INT8 FFN (INT8Linear, model and orchestrator wiring)."""

from __future__ import annotations

import gc
import logging
import weakref
from unittest import mock

import pytest
import torch
import torch.nn as nn

from shadowinfer.core.backends.pytorch_backend import PyTorchModelBackend
from shadowinfer.core.model_backend import ModelBackend
from shadowinfer.ffn_optimizer.int8_linear import INT8Linear
from shadowinfer.models import SimpleDiffusionLLM
from shadowinfer.orchestrator import Orchestrator


def _relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    actual_f = actual.detach().float()
    reference_f = reference.detach().float()
    denom = torch.linalg.vector_norm(reference_f).clamp_min(1e-12)
    return (torch.linalg.vector_norm(actual_f - reference_f) / denom).item()


def _tiny_model_kwargs() -> dict:
    return {
        "num_layers": 2,
        "num_heads": 2,
        "head_dim": 8,
        "hidden_dim": 16,
        "vocab_size": 32,
        "max_seq_len": 8,
        "intermediate_dim": 12,
    }


def _tiny_backend_config() -> dict:
    cfg = _tiny_model_kwargs()
    cfg.update({"name": "tiny-int8", "batch_size": 1, "seq_len": 8})
    return cfg


def _make_int8_linear(
    in_features: int = 128,
    out_features: int = 96,
    bias: bool = False,
    seed: int = 0,
) -> tuple[nn.Linear, INT8Linear]:
    torch.manual_seed(seed)
    linear = nn.Linear(in_features, out_features, bias=bias)
    return linear, INT8Linear.from_linear(linear)


class TestINT8Linear:
    def test_relative_error_shape_and_dtype(self):
        linear, int8_linear = _make_int8_linear(128, 96, bias=False, seed=0)
        x = torch.randn(64, 128)
        reference = linear(x)
        output = int8_linear(x)

        assert output.shape == (64, 96)
        assert output.dtype == x.dtype
        assert _relative_l2(output, reference) <= 0.03

        assert int8_linear.qweight.shape == (128, 96)
        assert int8_linear.qweight.dtype == torch.int8
        assert int8_linear.weight_scale.shape == (96,)
        assert int8_linear.weight_scale.dtype == torch.float32
        assert int8_linear.in_features == 128
        assert int8_linear.out_features == 96

    def test_relative_error_with_bias(self):
        linear, int8_linear = _make_int8_linear(128, 96, bias=True, seed=1)
        x = torch.randn(64, 128)
        reference = linear(x)
        output = int8_linear(x)

        assert output.shape == (64, 96)
        assert output.dtype == x.dtype
        assert int8_linear.bias is not None
        assert int8_linear.bias.shape == (96,)
        assert int8_linear.bias.dtype == torch.float32
        assert _relative_l2(output, reference) <= 0.03

    def test_batched_and_one_dimensional_inputs(self):
        _, int8_linear = _make_int8_linear(128, 96, bias=True, seed=2)
        x = torch.randn(2, 5, 128)

        batched = int8_linear(x)
        flattened = int8_linear(x.reshape(-1, 128)).reshape(2, 5, 96)
        single = int8_linear(x[0, 0])

        assert batched.shape == (2, 5, 96)
        assert torch.equal(batched, flattened)
        assert single.shape == (96,)
        assert torch.equal(single, batched[0, 0])

    @pytest.mark.parametrize(
        "x",
        [
            torch.zeros(4, 128),
            torch.full((4, 128), 0.5),
            torch.randn(4, 128) * 1e4,
        ],
        ids=["zero", "constant", "large"],
    )
    def test_extreme_inputs_are_finite(self, x):
        _, int8_linear = _make_int8_linear(128, 96, bias=True, seed=3)
        output = int8_linear(x)

        assert bool(torch.isfinite(output).all())
        assert not bool(torch.isnan(output).any())

    def test_empty_and_invalid_inputs(self):
        _, int8_linear = _make_int8_linear(128, 96, bias=True, seed=6)
        empty = int8_linear(torch.empty(0, 128))
        assert empty.shape == (0, 96)

        with pytest.raises(ValueError):
            int8_linear(torch.randn(4, 127))
        with pytest.raises(ValueError):
            INT8Linear(torch.zeros(128, dtype=torch.int8), torch.ones(96))
        with pytest.raises(ValueError):
            INT8Linear(torch.zeros(128, 96, dtype=torch.int8), torch.ones(95))
        with pytest.raises(ValueError):
            INT8Linear(
                torch.zeros(128, 96, dtype=torch.int8),
                torch.ones(96),
                bias=torch.zeros(95),
            )

    def test_memory_accounting(self):
        _, with_bias = _make_int8_linear(128, 96, bias=True, seed=4)
        _, without_bias = _make_int8_linear(128, 96, bias=False, seed=4)

        expected_with_bias = (
            with_bias.qweight.numel()
            + 4 * with_bias.weight_scale.numel()
            + 4 * with_bias.bias.numel()
        )
        expected_dense_with_bias = 4 * 128 * 96 + 4 * 96
        assert with_bias.memory_bytes() == expected_with_bias
        assert with_bias.dense_equivalent_bytes() == expected_dense_with_bias

        expected_without_bias = without_bias.qweight.numel() + 4 * without_bias.weight_scale.numel()
        expected_dense_without_bias = 4 * 128 * 96
        assert without_bias.memory_bytes() == expected_without_bias
        assert without_bias.dense_equivalent_bytes() == expected_dense_without_bias
        assert without_bias.dense_equivalent_bytes() / without_bias.memory_bytes() >= 3.8

    def test_int_mm_failure_falls_back_once(self, monkeypatch, caplog):
        _, int8_linear = _make_int8_linear(64, 48, bias=True, seed=5)
        x = torch.randn(8, 64)

        normal = int8_linear(x).clone()
        assert int8_linear.using_int_mm is True

        def _raise(*args, **kwargs):
            raise RuntimeError("injected _int_mm failure")

        caplog.clear()
        monkeypatch.setattr(torch, "_int_mm", _raise)
        with caplog.at_level(logging.WARNING):
            first = int8_linear(x)
            second = int8_linear(x)

        assert int8_linear.using_int_mm is False
        assert (first - normal).abs().max().item() <= 1e-6
        assert (second - normal).abs().max().item() <= 1e-6

        warnings = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING and "int8" in record.name.lower()
        ]
        assert len(warnings) == 1


class TestModelINT8FFN:
    def test_replaces_all_ffn_modules_and_is_idempotent(self):
        torch.manual_seed(11)
        model = SimpleDiffusionLLM(**_tiny_model_kwargs())
        up_refs = [weakref.ref(block.up_proj) for block in model.blocks]
        down_refs = [weakref.ref(block.down_proj) for block in model.blocks]

        stats = model.enable_int8_ffn()

        assert stats["layers"] == 2
        assert stats["modules"] == 4
        assert stats["weight_bytes"] > 0
        assert stats["dense_bytes"] > stats["weight_bytes"]
        assert isinstance(stats["fallback"], bool)
        for block in model.blocks:
            assert isinstance(block.up_proj, INT8Linear)
            assert isinstance(block.down_proj, INT8Linear)

        second = model.enable_int8_ffn()
        assert second["layers"] == stats["layers"]
        assert second["modules"] == stats["modules"]
        assert second["weight_bytes"] == stats["weight_bytes"]
        assert second["dense_bytes"] == stats["dense_bytes"]
        for block in model.blocks:
            assert isinstance(block.up_proj, INT8Linear)
            assert isinstance(block.down_proj, INT8Linear)

        gc.collect()
        assert all(ref() is None for ref in up_refs)
        assert all(ref() is None for ref in down_refs)

    def test_keep_fp32_reference_accuracy(self):
        torch.manual_seed(12)
        model = SimpleDiffusionLLM(**_tiny_model_kwargs())
        model.eval()
        inputs = [torch.randn(1, 8, 16) for _ in range(5)]

        with torch.no_grad():
            references = [model.forward_step(x, step_t=1, total_steps=4)[0].clone() for x in inputs]

        model.enable_int8_ffn(keep_fp32_reference=True)

        with torch.no_grad():
            outputs = [model.forward_step(x, step_t=1, total_steps=4)[0] for x in inputs]

        reference = torch.cat([tensor.reshape(-1) for tensor in references])
        output = torch.cat([tensor.reshape(-1) for tensor in outputs])
        assert _relative_l2(output, reference) <= 0.03

        matches = sum(
            int((out.argmax(dim=-1) == ref.argmax(dim=-1)).sum().item())
            for out, ref in zip(outputs, references)
        )
        total = sum(out.argmax(dim=-1).numel() for out in outputs)
        assert matches / total >= 0.95

    def test_restore_with_reference_is_exact(self):
        torch.manual_seed(13)
        model = SimpleDiffusionLLM(**_tiny_model_kwargs())
        original_up = model.blocks[0].up_proj
        original_down = model.blocks[1].down_proj

        model.enable_int8_ffn(keep_fp32_reference=True)
        assert model.blocks[0].up_proj is not original_up
        assert model.blocks[1].down_proj is not original_down

        model.restore_fp32_ffn()

        assert model.blocks[0].up_proj is original_up
        assert model.blocks[1].down_proj is original_down
        for block in model.blocks:
            assert isinstance(block.up_proj, nn.Linear)
            assert isinstance(block.down_proj, nn.Linear)

    def test_restore_without_reference_raises(self):
        torch.manual_seed(14)
        model = SimpleDiffusionLLM(**_tiny_model_kwargs())
        model.enable_int8_ffn()

        with pytest.raises(RuntimeError):
            model.restore_fp32_ffn()

    def test_repeated_keep_reference_preserves_restore(self):
        torch.manual_seed(16)
        model = SimpleDiffusionLLM(**_tiny_model_kwargs())
        original = model.blocks[0].up_proj

        model.enable_int8_ffn(keep_fp32_reference=True)
        model.enable_int8_ffn(keep_fp32_reference=True)
        model.restore_fp32_ffn()

        assert model.blocks[0].up_proj is original
        assert isinstance(model.blocks[1].down_proj, nn.Linear)

    def test_keep_then_release_clears_reference(self):
        torch.manual_seed(17)
        model = SimpleDiffusionLLM(**_tiny_model_kwargs())
        model.enable_int8_ffn(keep_fp32_reference=True)
        model.enable_int8_ffn(keep_fp32_reference=False)

        with pytest.raises(RuntimeError):
            model.restore_fp32_ffn()


def test_quantize_ffn_modules_generic_model():
    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Linear(16, 24)
            self.down_proj = nn.Linear(24, 16)
            self.other = nn.Linear(16, 16)

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _MLP()

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Layer()])

    from shadowinfer.ffn_optimizer.int8_linear import quantize_ffn_modules

    model = _Model()
    stats = quantize_ffn_modules(model)
    assert stats["supported"] is True
    assert stats["modules"] == 2
    assert isinstance(model.layers[0].mlp.up_proj, INT8Linear)
    assert isinstance(model.layers[0].mlp.down_proj, INT8Linear)
    assert isinstance(model.layers[0].mlp.other, nn.Linear)


class TestBackendAndOrchestratorWiring:
    def test_legacy_ffn_optimizer_key_enables_int8(self, tmp_path):
        backend = self._load_backend()
        config = {
            "ffn_optimizer": {"int8_execution_enabled": True},
            "output_dir": str(tmp_path),
            "warmup_steps": 0,
        }
        orchestrator = Orchestrator(config=config, model_backend=backend)
        orchestrator.initialize(model_config=_tiny_backend_config())
        assert orchestrator._int8_ffn_stats is not None
        assert isinstance(backend._model.blocks[0].up_proj, INT8Linear)

    def _load_backend(self) -> PyTorchModelBackend:
        backend = PyTorchModelBackend(model_config=_tiny_backend_config(), seed=15)
        backend.load(model_name="tiny-int8", device="cpu")
        return backend

    def test_base_backend_default_is_unsupported(self):
        class _MinimalBackend(ModelBackend):
            def load(self, model_name, device=None, **kwargs):
                pass

            def forward_step(self, x, step_cfg, kv_cache=None):
                return {}

            def get_kv_cache(self):
                return {}

            def set_kv_cache(self, kv_cache):
                pass

            def get_model_config(self):
                return {}

        stats = _MinimalBackend().enable_int8_ffn()
        assert stats == {"supported": False}

    def test_enabled_config_uses_int8_exec_without_agent(self, tmp_path):
        backend = self._load_backend()
        direct_stats = backend.enable_int8_ffn()

        assert direct_stats["supported"] is True
        assert direct_stats["modules"] == 4
        assert direct_stats["weight_bytes"] > 0

        config = {
            "ffn": {"int8_execution_enabled": True},
            "output_dir": str(tmp_path),
            "warmup_steps": 0,
        }
        model_config = _tiny_backend_config()
        orchestrator = Orchestrator(config=config, model_backend=backend)

        with mock.patch.object(
            backend, "enable_int8_ffn", wraps=backend.enable_int8_ffn
        ) as enable_spy:
            orchestrator.initialize(model_config=model_config)
        assert enable_spy.call_count >= 1

        with mock.patch.object(
            orchestrator.ffn_optimizer,
            "on_step",
            wraps=orchestrator.ffn_optimizer.on_step,
        ) as agent_spy:
            orchestrator.run_optimized("tiny prompt", num_steps=1)

        assert agent_spy.call_count == 0
        assert len(orchestrator.step_results) == 1
        ffn_result = orchestrator.step_results[0]["ffn"]
        assert ffn_result["compute_path"] == "int8_exec"

    def test_disabled_config_keeps_fp32_and_agent_path(self, tmp_path):
        backend = self._load_backend()
        original_up = backend._model.blocks[0].up_proj
        config = {
            "ffn": {
                "int8_execution_enabled": False,
                "dense_precision": "int8",
                "mixed_precision": True,
            },
            "output_dir": str(tmp_path),
            "warmup_steps": 0,
        }
        model_config = _tiny_backend_config()
        orchestrator = Orchestrator(config=config, model_backend=backend)
        orchestrator.initialize(model_config=model_config)

        assert backend._model.blocks[0].up_proj is original_up
        assert orchestrator.ffn_optimizer.config.get("mixed_precision") is True
        for block in backend._model.blocks:
            assert isinstance(block.up_proj, nn.Linear)
            assert not isinstance(block.up_proj, INT8Linear)
            assert isinstance(block.down_proj, nn.Linear)
            assert not isinstance(block.down_proj, INT8Linear)

        with mock.patch.object(
            orchestrator.ffn_optimizer,
            "on_step",
            wraps=orchestrator.ffn_optimizer.on_step,
        ) as agent_spy:
            orchestrator.run_optimized("tiny prompt", num_steps=1)

        assert agent_spy.call_count > 0
        assert len(orchestrator.step_results) == 1
        ffn_result = orchestrator.step_results[0]["ffn"]
        assert ffn_result["compute_path"] != "int8_exec"
