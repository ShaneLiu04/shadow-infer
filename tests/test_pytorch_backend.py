"""Tests for SimpleDiffusionLLM and the PyTorchModelBackend."""

from __future__ import annotations

import pytest
import torch

from shadowinfer.core.backends.pytorch_backend import PyTorchModelBackend
from shadowinfer.core.model_backend import ModelBackend
from shadowinfer.core.structs import KVCacheEntry, StepConfig
from shadowinfer.models import SimpleDiffusionLLM


class TestSimpleDiffusionLLM:
    """Unit tests for the tiny Diffusion LLM implementation."""

    @pytest.fixture
    def tiny_config(self):
        return {
            "num_layers": 2,
            "num_heads": 2,
            "head_dim": 16,
            "hidden_dim": 32,
            "vocab_size": 64,
            "max_seq_len": 32,
            "intermediate_dim": 64,
        }

    def test_default_init(self):
        model = SimpleDiffusionLLM()
        assert model.num_layers == 4
        assert model.num_heads == 4
        assert model.hidden_dim == 128

    def test_custom_init(self, tiny_config):
        model = SimpleDiffusionLLM(**tiny_config)
        assert model.num_layers == tiny_config["num_layers"]
        assert model.num_heads == tiny_config["num_heads"]
        assert model.hidden_dim == tiny_config["hidden_dim"]

    def test_forward_token_ids(self, tiny_config):
        model = SimpleDiffusionLLM(**tiny_config)
        tokens = torch.randint(0, tiny_config["vocab_size"], (2, 5))
        logits, kv_cache, attn = model.forward_step(tokens, step_t=0, total_steps=10)

        assert logits.shape == (2, 5, tiny_config["vocab_size"])
        assert len(kv_cache) == tiny_config["num_layers"]
        assert attn.shape == (2, tiny_config["num_heads"], 5, 5)

    def test_forward_hidden_state(self, tiny_config):
        model = SimpleDiffusionLLM(**tiny_config)
        hidden = torch.randn(2, 5, tiny_config["hidden_dim"])
        logits, kv_cache, attn = model.forward_step(hidden, step_t=0, total_steps=10)

        assert logits.shape == (2, 5, tiny_config["vocab_size"])
        assert len(kv_cache) == tiny_config["num_layers"]
        assert attn.shape == (2, tiny_config["num_heads"], 5, 5)

    def test_deterministic_with_seed(self, tiny_config):
        torch.manual_seed(42)
        model1 = SimpleDiffusionLLM(**tiny_config)
        tokens = torch.randint(0, tiny_config["vocab_size"], (1, 4))
        logits1, _, _ = model1.forward_step(tokens, step_t=1, total_steps=10)

        torch.manual_seed(42)
        model2 = SimpleDiffusionLLM(**tiny_config)
        logits2, _, _ = model2.forward_step(tokens, step_t=1, total_steps=10)

        assert torch.allclose(logits1, logits2)

    def test_kv_cache_grows_across_steps(self, tiny_config):
        model = SimpleDiffusionLLM(**tiny_config)
        first_tokens = torch.randint(0, tiny_config["vocab_size"], (1, 4))
        _, kv1, _ = model.forward_step(first_tokens, step_t=0, total_steps=10)

        second_tokens = torch.randint(0, tiny_config["vocab_size"], (1, 3))
        _, kv2, _ = model.forward_step(second_tokens, step_t=1, total_steps=10, kv_cache=kv1)

        for layer_id in range(tiny_config["num_layers"]):
            k1, v1 = kv1[layer_id]
            k2, v2 = kv2[layer_id]
            assert k2.shape[2] == k1.shape[2] + 3
            assert v2.shape[2] == v1.shape[2] + 3

    def test_bad_hidden_dim_raises(self, tiny_config):
        model = SimpleDiffusionLLM(**tiny_config)
        with pytest.raises(ValueError):
            model.forward_step(torch.randn(1, 2, tiny_config["hidden_dim"] + 1), 0, 1)


class TestPyTorchModelBackend:
    """Unit tests for the PyTorch-backed ModelBackend."""

    @pytest.fixture
    def tiny_config(self):
        return {
            "name": "simple-test",
            "num_layers": 2,
            "num_heads": 2,
            "head_dim": 16,
            "hidden_dim": 32,
            "vocab_size": 64,
            "max_seq_len": 32,
            "intermediate_dim": 64,
        }

    @pytest.fixture
    def backend(self, tiny_config):
        backend = PyTorchModelBackend(model_config=tiny_config, seed=123)
        backend.load(model_name="simple-test", device="cpu")
        return backend

    def test_is_model_backend(self):
        assert issubclass(PyTorchModelBackend, ModelBackend)

    def test_load_and_get_config(self, backend, tiny_config):
        assert isinstance(backend.get_model_config(), dict)
        assert backend.get_model_config()["name"] == "simple-test"
        assert backend.get_model_config()["hidden_dim"] == tiny_config["hidden_dim"]

    def test_forward_step_returns_expected_keys(self, backend, tiny_config):
        tokens = torch.randint(0, tiny_config["vocab_size"], (1, 5))
        step_cfg = StepConfig(
            step_id=0,
            total_steps=10,
            noise_level=0.5,
            shadowkv_mode="balanced",
        )
        result = backend.forward_step(tokens, step_cfg)

        assert "output" in result
        assert "kv_cache" in result
        assert "attention_scores" in result
        assert result["output"].shape == (1, 5, tiny_config["vocab_size"])
        assert len(result["kv_cache"]) == tiny_config["num_layers"]

    def test_kv_cache_state_persists(self, backend, tiny_config):
        step_cfg = StepConfig(
            step_id=0,
            total_steps=10,
            noise_level=0.0,
            shadowkv_mode="balanced",
        )
        backend.forward_step(torch.randint(0, tiny_config["vocab_size"], (1, 1)), step_cfg)
        first_cache = backend.get_kv_cache()
        first_len = first_cache[0].k_tensor.shape[2]

        step_cfg.step_id = 1
        backend.forward_step(torch.randint(0, tiny_config["vocab_size"], (1, 1)), step_cfg)
        second_cache = backend.get_kv_cache()
        second_len = second_cache[0].k_tensor.shape[2]

        assert second_len == first_len + 1

    def test_kv_cache_argument_overrides_internal_state(self, backend, tiny_config):
        step_cfg = StepConfig(
            step_id=0,
            total_steps=10,
            noise_level=0.0,
            shadowkv_mode="balanced",
        )
        backend.forward_step(torch.randint(0, tiny_config["vocab_size"], (1, 4)), step_cfg)
        internal_cache = backend.get_kv_cache()

        external_cache = {
            layer_id: KVCacheEntry(
                k_tensor=torch.randn(1, tiny_config["num_heads"], 2, tiny_config["head_dim"]),
                v_tensor=torch.randn(1, tiny_config["num_heads"], 2, tiny_config["head_dim"]),
                precision="fp32",
            )
            for layer_id in range(tiny_config["num_layers"])
        }

        step_cfg.step_id = 1
        result = backend.forward_step(
            torch.randint(0, tiny_config["vocab_size"], (1, 1)),
            step_cfg,
            kv_cache=external_cache,
        )

        assert len(result["kv_cache"]) == tiny_config["num_layers"]
        # After overriding with external cache the backend state should be updated.
        assert backend.get_kv_cache() is result["kv_cache"]
        assert backend.get_kv_cache() is not internal_cache

    def test_reuse_layers_marked(self, backend, tiny_config):
        step_cfg = StepConfig(
            step_id=0,
            total_steps=10,
            noise_level=0.0,
            shadowkv_mode="balanced",
            reuse_layers=[0],
        )
        backend.forward_step(torch.randint(0, tiny_config["vocab_size"], (1, 3)), step_cfg)
        first_cache = backend.get_kv_cache()

        step_cfg.step_id = 1
        backend.forward_step(torch.randint(0, tiny_config["vocab_size"], (1, 3)), step_cfg)
        second_cache = backend.get_kv_cache()

        assert second_cache[0].is_reused is True
        assert second_cache[0].reuse_step == 0
        assert second_cache[1].is_reused is False
        assert second_cache[1].reuse_step == -1
        # Reused entry should keep the old tensors.
        assert torch.equal(second_cache[0].k_tensor, first_cache[0].k_tensor)

    @pytest.mark.parametrize(
        "mode, expected_precision",
        [
            ("conservative", "fp32"),
            ("balanced", "fp16"),
            ("aggressive", "int8"),
        ],
    )
    def test_shadowkv_precision_hint(self, backend, tiny_config, mode, expected_precision):
        step_cfg = StepConfig(
            step_id=0,
            total_steps=10,
            noise_level=0.0,
            shadowkv_mode=mode,
        )
        result = backend.forward_step(torch.randint(0, tiny_config["vocab_size"], (1, 3)), step_cfg)
        for layer_id in range(tiny_config["num_layers"]):
            entry = result["kv_cache"][layer_id]
            assert entry.precision == expected_precision
            if expected_precision == "fp16":
                assert entry.k_tensor.dtype == torch.float16
            elif expected_precision == "fp32":
                assert entry.k_tensor.dtype == torch.float32
            elif expected_precision == "int8":
                assert entry.k_tensor.dtype == torch.int8
                assert entry.scale_k is not None
                assert entry.scale_v is not None
