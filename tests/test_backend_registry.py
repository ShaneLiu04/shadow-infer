"""Tests for the backend plugin registry.

Covers built-in registration, plugin discovery via entry points, availability
checks, and instantiation.
"""

from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest
import torch

from shadowinfer.core.backends.registry import BackendRegistry, get_backend_registry
from shadowinfer.core.model_backend import ModelBackend
from shadowinfer.core.structs import KVCacheEntry, StepConfig


class FakePluginBackend(ModelBackend):
    """Fake plugin backend for registry testing."""

    def __init__(self, model_config: Optional[Dict[str, Any]] = None) -> None:
        self.model_config = model_config or {}

    @classmethod
    def is_available(cls) -> bool:
        return True

    def load(self, model_name: str, device: Optional[str] = None) -> None:
        self.model_config["name"] = model_name

    def forward_step(
        self,
        x: torch.Tensor,
        step_cfg: StepConfig,
        kv_cache: Optional[Dict[int, KVCacheEntry]] = None,
    ) -> Dict[str, Any]:
        return {"output": x}

    def get_kv_cache(self) -> Dict[int, KVCacheEntry]:
        return {}

    def set_kv_cache(self, kv_cache: Dict[int, KVCacheEntry]) -> None:
        pass

    def get_model_config(self) -> Dict[str, Any]:
        return self.model_config


def test_registry_includes_builtins():
    """The global registry must contain all built-in backends."""
    registry = get_backend_registry(discover_plugins=False)
    names = registry.list_names()
    assert "mock" in names
    assert "pytorch" in names
    assert "huggingface" in names
    assert "vllm" in names
    assert "sglang" in names


def test_builtin_availability():
    """Built-in backends are always available; optional deps may not be."""
    registry = get_backend_registry(discover_plugins=False)
    assert registry.is_available("mock") is True
    assert registry.is_available("pytorch") is True
    assert registry.is_available("huggingface") is True
    # vLLM/SGLang are not installed in this environment.
    assert registry.is_available("vllm") is False
    assert registry.is_available("sglang") is False


def test_create_backend():
    """Registry.create must instantiate a backend with kwargs."""
    registry = BackendRegistry()
    registry.register_builtin_backends()
    backend = registry.create("mock", model_config={"name": "test"})
    assert isinstance(backend, ModelBackend)
    assert backend.get_model_config()["name"] == "test"


def test_create_unavailable_backend_raises():
    """Creating an unavailable backend must raise RuntimeError."""
    registry = BackendRegistry()
    registry.register_builtin_backends()
    with pytest.raises(RuntimeError):
        registry.create("vllm")


def test_register_invalid_class_raises():
    """Registering a non-ModelBackend class must raise TypeError."""
    registry = BackendRegistry()
    with pytest.raises(TypeError):
        registry.register("bad", str)


def test_duplicate_registration_raises():
    """Duplicate registration without overwrite must raise ValueError."""
    registry = BackendRegistry()
    registry.register("x", FakePluginBackend)
    with pytest.raises(ValueError):
        registry.register("x", FakePluginBackend)


def test_plugin_discovery(monkeypatch):
    """Backends registered via entry points must be discovered."""
    fake_ep = SimpleNamespace(
        name="fake",
        value="tests.test_backend_registry:FakePluginBackend",
        load=lambda: FakePluginBackend,
    )

    def fake_entry_points(*, group):
        if group == "shadowinfer.backends":
            return [fake_ep]
        return []

    monkeypatch.setattr("shadowinfer.core.backends.registry.entry_points", fake_entry_points)

    registry = BackendRegistry()
    registry.register_builtin_backends()
    registry.discover_plugins()
    assert "fake" in registry.list_names()
    assert registry.is_available("fake") is True
    backend = registry.create("fake")
    assert isinstance(backend, FakePluginBackend)


def test_plugin_discovery_skips_invalid(monkeypatch):
    """Broken or invalid entry points must be skipped with a warning."""
    bad_ep = SimpleNamespace(
        name="bad",
        value="tests.test_backend_registry:NotABackend",
        load=lambda: str,
    )

    def fake_entry_points(*, group):
        if group == "shadowinfer.backends":
            return [bad_ep]
        return []

    monkeypatch.setattr("shadowinfer.core.backends.registry.entry_points", fake_entry_points)

    registry = BackendRegistry()
    registry.register_builtin_backends()
    registry.discover_plugins()
    assert "bad" not in registry.list_names()


def test_help_text():
    """help_text must list registered backends."""
    registry = BackendRegistry()
    registry.register_builtin_backends()
    text = registry.help_text()
    assert "mock" in text
    assert "pytorch" in text
