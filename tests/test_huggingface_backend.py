"""Tests for the HuggingFace backend adapter."""

from __future__ import annotations

import sys
from unittest import mock

import pytest
import torch

from shadowinfer.core.backends.huggingface_backend import HuggingFaceModelBackend
from shadowinfer.core.model_backend import ModelBackend
from shadowinfer.core.structs import KVCacheEntry, StepConfig

try:
    import transformers  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    _HAS_TRANSFORMERS = False

pytestmark = pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")


def _huggingface_reachable(timeout: float = 3.0) -> bool:
    """Check whether the HuggingFace Hub can be reached."""
    import socket

    try:
        with socket.create_connection(("huggingface.co", 443), timeout=timeout):
            return True
    except OSError:
        return False


_HF_REACHABLE = _huggingface_reachable()


class TestHuggingFaceModelBackend:
    """Unit tests for ``HuggingFaceModelBackend``."""

    """Unit tests for ``HuggingFaceModelBackend``."""

    @pytest.fixture
    def backend(self, tmp_path):
        """A backend loaded with the tiny local fallback model."""
        backend = HuggingFaceModelBackend(seed=42)
        # Internal test identifiers always resolve to the local fallback without
        # touching the network.
        backend.load("__shadowinfer_tiny_test_model__", device="cpu")
        return backend

    def test_is_model_backend(self):
        assert issubclass(HuggingFaceModelBackend, ModelBackend)

    def test_load_and_get_config(self, backend):
        cfg = backend.get_model_config()
        assert isinstance(cfg, dict)
        assert "num_layers" in cfg
        assert "num_heads" in cfg
        assert "hidden_dim" in cfg
        assert "vocab_size" in cfg

    def test_forward_step_with_hidden_state(self, backend):
        cfg = backend.get_model_config()
        hidden_dim = cfg["hidden_dim"]
        vocab_size = cfg["vocab_size"]
        seq_len = 4

        x = torch.randn(1, seq_len, hidden_dim)
        step_cfg = StepConfig(
            step_id=0,
            total_steps=10,
            noise_level=0.5,
            shadowkv_mode="balanced",
        )
        result = backend.forward_step(x, step_cfg)

        assert "output" in result
        assert "kv_cache" in result
        assert "attention_scores" in result
        assert result["output"].shape == (1, seq_len, vocab_size)
        assert len(result["kv_cache"]) == cfg["num_layers"]
        assert isinstance(result["attention_scores"], torch.Tensor)

    def test_forward_step_with_token_ids(self, backend):
        cfg = backend.get_model_config()
        vocab_size = cfg["vocab_size"]

        tokens = torch.randint(0, vocab_size, (1, 5))
        step_cfg = StepConfig(
            step_id=0,
            total_steps=10,
            noise_level=0.0,
            shadowkv_mode="conservative",
        )
        result = backend.forward_step(tokens, step_cfg)

        assert result["output"].shape == (1, 5, vocab_size)
        assert len(result["kv_cache"]) == cfg["num_layers"]

    def test_kv_cache_grows_across_steps(self, backend):
        cfg = backend.get_model_config()
        hidden_dim = cfg["hidden_dim"]

        step_cfg = StepConfig(
            step_id=0,
            total_steps=10,
            noise_level=0.0,
            shadowkv_mode="balanced",
        )
        backend.forward_step(torch.randn(1, 3, hidden_dim), step_cfg)
        first_len = backend.get_kv_cache()[0].k_tensor.shape[2]

        step_cfg.step_id = 1
        backend.forward_step(torch.randn(1, 2, hidden_dim), step_cfg)
        second_len = backend.get_kv_cache()[0].k_tensor.shape[2]

        assert second_len == first_len + 2

    def test_kv_cache_argument_overrides_internal_state(self, backend):
        cfg = backend.get_model_config()
        hidden_dim = cfg["hidden_dim"]

        step_cfg = StepConfig(
            step_id=0,
            total_steps=10,
            noise_level=0.0,
            shadowkv_mode="balanced",
        )
        backend.forward_step(torch.randn(1, 4, hidden_dim), step_cfg)
        internal_cache = backend.get_kv_cache()

        external_cache = {
            layer_id: KVCacheEntry(
                k_tensor=torch.randn(1, cfg["num_heads"], 2, cfg["head_dim"]),
                v_tensor=torch.randn(1, cfg["num_heads"], 2, cfg["head_dim"]),
                precision="fp32",
            )
            for layer_id in range(cfg["num_layers"])
        }

        step_cfg.step_id = 1
        result = backend.forward_step(
            torch.randn(1, 1, hidden_dim),
            step_cfg,
            kv_cache=external_cache,
        )

        assert len(result["kv_cache"]) == cfg["num_layers"]
        assert backend.get_kv_cache() is result["kv_cache"]
        assert backend.get_kv_cache() is not internal_cache

    def test_reuse_layers_marked(self, backend):
        cfg = backend.get_model_config()
        hidden_dim = cfg["hidden_dim"]

        step_cfg = StepConfig(
            step_id=0,
            total_steps=10,
            noise_level=0.0,
            shadowkv_mode="balanced",
            reuse_layers=[0],
        )
        backend.forward_step(torch.randn(1, 3, hidden_dim), step_cfg)
        first_cache = backend.get_kv_cache()

        step_cfg.step_id = 1
        backend.forward_step(torch.randn(1, 3, hidden_dim), step_cfg)
        second_cache = backend.get_kv_cache()

        assert second_cache[0].is_reused is True
        assert second_cache[0].reuse_step == 0
        assert second_cache[1].is_reused is False
        assert torch.equal(second_cache[0].k_tensor, first_cache[0].k_tensor)

    @pytest.mark.parametrize(
        "mode, expected_precision",
        [
            ("conservative", "fp32"),
            ("balanced", "fp16"),
            ("aggressive", "int8"),
        ],
    )
    def test_shadowkv_precision_hint(self, backend, mode, expected_precision):
        cfg = backend.get_model_config()
        hidden_dim = cfg["hidden_dim"]

        step_cfg = StepConfig(
            step_id=0,
            total_steps=10,
            noise_level=0.0,
            shadowkv_mode=mode,
        )
        result = backend.forward_step(torch.randn(1, 3, hidden_dim), step_cfg)
        for layer_id in range(cfg["num_layers"]):
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

    def test_fallback_when_network_unavailable(self, monkeypatch):
        """If the Hub cannot be reached, a local tiny model is used."""
        original_model_from_pretrained = AutoModelForCausalLM.from_pretrained
        original_tokenizer_from_pretrained = AutoTokenizer.from_pretrained

        def fake_model_from_pretrained(pretrained_model_name_or_path, **kwargs):
            if str(pretrained_model_name_or_path).startswith("unreachable"):
                raise OSError("simulated network failure")
            return original_model_from_pretrained(pretrained_model_name_or_path, **kwargs)

        def fake_tokenizer_from_pretrained(pretrained_model_name_or_path, **kwargs):
            if str(pretrained_model_name_or_path).startswith("unreachable"):
                raise OSError("simulated network failure")
            return original_tokenizer_from_pretrained(pretrained_model_name_or_path, **kwargs)

        monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained)
        monkeypatch.setattr(AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained)

        backend = HuggingFaceModelBackend(seed=0)
        backend.load("unreachable/model-id", device="cpu")

        assert backend._model is not None
        cfg = backend.get_model_config()
        assert cfg["num_layers"] > 0

    def test_transformers_missing_raises_import_error(self, monkeypatch):
        """The backend raises a clear ImportError when transformers is absent."""
        import shadowinfer.core.backends.huggingface_backend as hf_module

        monkeypatch.setattr(hf_module, "_TRANSFORMERS_AVAILABLE", False)
        monkeypatch.setattr(
            hf_module, "_HF_IMPORT_ERROR", ImportError("no transformers"), raising=False
        )

        with pytest.raises(ImportError):
            HuggingFaceModelBackend()

    def test_load_local_directory(self, backend, tmp_path):
        """Loading from a saved local directory works without the network."""
        model = backend._model
        local_path = tmp_path / "local_hf_model"
        model.save_pretrained(local_path)

        new_backend = HuggingFaceModelBackend(seed=42)
        new_backend.load(str(local_path), device="cpu")

        assert new_backend._model is not None

    def test_load_local_files_only_skips_network(self):
        """local_files_only=True always resolves to the local fallback."""
        backend = HuggingFaceModelBackend(seed=0)
        backend.load("some/online-model", device="cpu", local_files_only=True)
        assert backend._model is not None
        cfg = backend.get_model_config()
        assert cfg["num_layers"] > 0

    @pytest.mark.skipif(not _HF_REACHABLE, reason="HuggingFace Hub is not reachable")
    def test_load_online_real_tiny_model(self):
        """When the network is available, a real tiny Hub model can be loaded."""
        backend = HuggingFaceModelBackend(seed=0)
        backend.load("sshleifer/tiny-gpt2", device="cpu")
        assert backend._model is not None
        cfg = backend.get_model_config()
        assert cfg["num_layers"] > 0
        assert cfg["vocab_size"] > 0

    @pytest.mark.skipif(
        _HF_REACHABLE, reason="HuggingFace Hub is reachable; online path already tested"
    )
    def test_load_online_fallback_when_hub_unreachable(self):
        """If the Hub cannot be reached, loading falls back to the tiny model."""
        backend = HuggingFaceModelBackend(seed=0)
        backend.load("sshleifer/tiny-gpt2", device="cpu")
        assert backend._model is not None
        cfg = backend.get_model_config()
        assert cfg["num_layers"] > 0


class TestHuggingFaceBackendOptionalImport:
    """Verify that the backend stays optional at import time."""

    def test_can_import_without_transformers_installed(self):
        """Importing the class should not fail even if transformers is missing.

        The actual instantiation is what raises ``ImportError``.
        """
        # Simulate a missing transformers by temporarily hiding it.
        real_module = sys.modules.get("transformers")
        if "shadowinfer.core.backends.huggingface_backend" in sys.modules:
            del sys.modules["shadowinfer.core.backends.huggingface_backend"]

        with mock.patch.dict("sys.modules", {"transformers": None}):
            # A fresh import should still succeed because the top-level import is
            # wrapped in a try/except.
            from shadowinfer.core.backends.huggingface_backend import (
                HuggingFaceModelBackend as ImportedBackend,
            )

            assert issubclass(ImportedBackend, ModelBackend)

        # Restore the real transformers module and reload the backend module so
        # that later tests see a fully functional implementation.
        if real_module is not None:
            sys.modules["transformers"] = real_module
        import importlib

        import shadowinfer.core.backends.huggingface_backend as hf_module

        importlib.reload(hf_module)
