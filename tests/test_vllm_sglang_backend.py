"""Tests for the vLLM and SGLang backend scaffolding.

These tests do not require vLLM or SGLang to be installed; they verify that the
adapters are importable, report availability correctly, and fail gracefully.
"""

import pytest
import torch

from shadowinfer.core.backends.sglang_backend import _SGLANG_AVAILABLE, SGLangModelBackend
from shadowinfer.core.backends.vllm_backend import _VLLM_AVAILABLE, VLLMModelBackend
from shadowinfer.core.structs import StepConfig


def test_vllm_availability_reflects_environment():
    """is_available must match whether vLLM can be imported."""
    assert VLLMModelBackend.is_available() == _VLLM_AVAILABLE


def test_sglang_availability_reflects_environment():
    """is_available must match whether SGLang can be imported."""
    assert SGLangModelBackend.is_available() == _SGLANG_AVAILABLE


def test_vllm_load_fails_when_unavailable():
    """load() must raise RuntimeError when vLLM is not installed."""
    backend = VLLMModelBackend()
    if _VLLM_AVAILABLE:
        pytest.skip("vLLM is installed; skipping unavailable test")
    with pytest.raises(RuntimeError, match="vLLM backend is not available"):
        backend.load("gpt2")


def test_sglang_load_fails_when_unavailable():
    """load() must raise RuntimeError when SGLang is not installed."""
    backend = SGLangModelBackend()
    if _SGLANG_AVAILABLE:
        pytest.skip("SGLang is installed; skipping unavailable test")
    with pytest.raises(RuntimeError, match="SGLang backend is not available"):
        backend.load("gpt2")


def test_vllm_forward_step_raises_not_implemented(monkeypatch):
    """forward_step must raise NotImplementedError in the scaffold."""
    monkeypatch.setattr("shadowinfer.core.backends.vllm_backend._VLLM_AVAILABLE", True)
    backend = VLLMModelBackend()
    backend.load("gpt2")
    x = torch.randn(1, 4, 8)
    cfg = StepConfig(step_id=0, total_steps=1, noise_level=0.5, shadowkv_mode="balanced")
    with pytest.raises(NotImplementedError):
        backend.forward_step(x, cfg)


def test_sglang_forward_step_raises_not_implemented(monkeypatch):
    """forward_step must raise NotImplementedError in the scaffold."""
    monkeypatch.setattr("shadowinfer.core.backends.sglang_backend._SGLANG_AVAILABLE", True)
    backend = SGLangModelBackend()
    backend.load("gpt2")
    x = torch.randn(1, 4, 8)
    cfg = StepConfig(step_id=0, total_steps=1, noise_level=0.5, shadowkv_mode="balanced")
    with pytest.raises(NotImplementedError):
        backend.forward_step(x, cfg)


def test_vllm_get_set_kv_cache():
    """KV cache getter/setter must round-trip."""
    backend = VLLMModelBackend()
    assert backend.get_kv_cache() == {}
    backend.set_kv_cache({1: {"k": None, "v": None}})
    assert 1 in backend.get_kv_cache()


def test_sglang_get_set_kv_cache():
    """KV cache getter/setter must round-trip."""
    backend = SGLangModelBackend()
    assert backend.get_kv_cache() == {}
    backend.set_kv_cache({1: {"k": None, "v": None}})
    assert 1 in backend.get_kv_cache()
