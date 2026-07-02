"""Tests for the agent plugin registry."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from shadowinfer.core.agent_plugin_registry import (
    AgentPluginRegistry,
    get_agent_plugin_registry,
)
from shadowinfer.core.base_agent import BaseAgent
from shadowinfer.core.structs import ProfileResult, StepConfig


class _DummyAgent(BaseAgent):
    """A minimal agent implementation for testing."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, config.get("name", "dummy"))
        self.init_calls: int = 0
        self.step_calls: int = 0
        self.shutdown_calls: int = 0

    def on_init(self, model_config: Dict[str, Any]) -> None:
        self.init_calls += 1

    def on_step(self, step_config: StepConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.step_calls += 1
        return {"ok": True, "step_id": step_config.step_id}

    def on_shutdown(self) -> Optional[ProfileResult]:
        self.shutdown_calls += 1
        return None


@pytest.fixture
def registry() -> AgentPluginRegistry:
    """Return a fresh registry with built-in agents registered."""
    reg = AgentPluginRegistry()
    reg.register_builtin_agents()
    return reg


def test_builtin_agents_registered(registry: AgentPluginRegistry) -> None:
    names = registry.list_names()
    assert "profiler" in names
    assert "shadowkv" in names
    assert "qdrift" in names
    assert "ffn_optimizer" in names


def test_create_builtin_agent(registry: AgentPluginRegistry) -> None:
    agent = registry.create("profiler", {"model_name": "test", "run_id": "r1"})
    assert agent.name == "profiler"


def test_unknown_agent_raises(registry: AgentPluginRegistry) -> None:
    with pytest.raises(KeyError, match="Unknown agent"):
        registry.create("not_exists", {})


def test_register_invalid_class(registry: AgentPluginRegistry) -> None:
    with pytest.raises(TypeError, match="must inherit from BaseAgent"):
        registry.register("bad", object)  # type: ignore[arg-type]


def test_duplicate_registration(registry: AgentPluginRegistry) -> None:
    registry.register("dummy", _DummyAgent)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("dummy", _DummyAgent)

    registry.register("dummy", _DummyAgent, overwrite=True)
    assert registry.get("dummy") is _DummyAgent


def test_list_available(registry: AgentPluginRegistry) -> None:
    registry.register("dummy", _DummyAgent)
    assert "dummy" in registry.list_available()


class _UnavailableAgent(BaseAgent):
    """Agent that reports itself unavailable."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, "unavailable")

    @classmethod
    def is_available(cls) -> bool:
        return False

    def on_init(self, model_config: Dict[str, Any]) -> None:
        pass

    def on_step(self, step_config: StepConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def on_shutdown(self) -> Optional[ProfileResult]:
        return None


def test_unavailable_agent_cannot_be_created(registry: AgentPluginRegistry) -> None:
    registry.register("unavailable", _UnavailableAgent)
    with pytest.raises(RuntimeError, match="not available"):
        registry.create("unavailable", {})


def test_plugin_discovery(monkeypatch, registry: AgentPluginRegistry) -> None:
    """Mock entry-point discovery and verify the plugin is loaded."""

    class _FakeEntryPoint:
        name = "dummy_plugin"
        value = "tests.test_agent_plugin_registry:_DummyAgent"

        def load(self):
            return _DummyAgent

    def _fake_entry_points(*, group: str) -> List[_FakeEntryPoint]:
        if group == registry.ENTRY_POINT_GROUP:
            return [_FakeEntryPoint()]
        return []

    monkeypatch.setattr("shadowinfer.core.agent_plugin_registry.entry_points", _fake_entry_points)
    registry.discover_plugins()

    assert registry.is_registered("dummy_plugin")
    agent = registry.create("dummy_plugin", {"name": "dp"})
    assert isinstance(agent, _DummyAgent)


def test_plugin_discovery_skips_invalid(monkeypatch, registry: AgentPluginRegistry) -> None:
    """Invalid entry points should be skipped with a warning."""

    class _BadEntryPoint:
        name = "bad_plugin"
        value = "tests.test_agent_plugin_registry:object"

        def load(self):
            return object

    def _fake_entry_points(*, group: str) -> List[_BadEntryPoint]:
        if group == registry.ENTRY_POINT_GROUP:
            return [_BadEntryPoint()]
        return []

    monkeypatch.setattr("shadowinfer.core.agent_plugin_registry.entry_points", _fake_entry_points)
    with pytest.warns(UserWarning, match="does not inherit from BaseAgent"):
        registry.discover_plugins()

    assert not registry.is_registered("bad_plugin")


def test_help_text(registry: AgentPluginRegistry) -> None:
    text = registry.help_text()
    assert "profiler" in text
    assert "Available agents:" in text


def test_global_registry_singleton() -> None:
    reg1 = get_agent_plugin_registry()
    reg2 = get_agent_plugin_registry()
    assert reg1 is reg2
