"""Tests for Orchestrator integration with agent plugins."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from shadowinfer.core.agent_plugin_registry import AgentPluginRegistry
from shadowinfer.core.base_agent import BaseAgent
from shadowinfer.core.structs import ProfileResult, StepConfig
from shadowinfer.orchestrator import Orchestrator


class _CounterAgent(BaseAgent):
    """Plugin agent that counts how many times it is invoked."""

    def __init__(self, config: Dict[str, Any]) -> None:
        name = config.get("name", "counter")
        super().__init__(config, name)
        self.step_count: int = 0
        self.model_config: Optional[Dict[str, Any]] = None

    def on_init(self, model_config: Dict[str, Any]) -> None:
        self.model_config = model_config

    def on_step(self, step_config: StepConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.step_count += 1
        return {
            "count": self.step_count,
            "plugin": True,
            "step_id": step_config.step_id,
        }

    def on_shutdown(self) -> Optional[ProfileResult]:
        return None


def _make_registry_with_counter() -> AgentPluginRegistry:
    """Return a fresh registry that includes the counter plugin."""
    reg = AgentPluginRegistry()
    reg.register_builtin_agents()
    reg.register("counter", _CounterAgent, help="Test counter plugin")
    return reg


def _default_model_config() -> Dict[str, Any]:
    return {
        "name": "test",
        "num_layers": 2,
        "num_heads": 2,
        "head_dim": 8,
        "hidden_dim": 16,
        "intermediate_dim": 32,
        "batch_size": 1,
        "seq_len": 4,
    }


def test_extra_agent_initialized_and_executed() -> None:
    """An extra agent declared in config is initialized and runs each step."""
    config = {
        "extra_agents": [
            {
                "name": "my_counter",
                "agent": "counter",
                "config": {"custom_key": 42},
            }
        ]
    }
    orch = Orchestrator(config=config)
    orch.agent_registry = _make_registry_with_counter()
    orch.initialize(model_config=_default_model_config())

    result = orch.run_optimized(prompt="test", num_steps=3)
    assert result is not None
    assert "my_counter" in orch.registry

    counter = orch.registry.get("my_counter")
    assert isinstance(counter, _CounterAgent)
    assert counter.model_config is not None
    assert counter.model_config["name"] == "test"
    assert counter.step_count == 3

    # Each step result should carry the plugin output.
    assert len(orch.step_results) == 3
    for step in orch.step_results:
        assert "my_counter" in step.get("outputs", {})
        assert step["outputs"]["my_counter"]["plugin"] is True


def test_extra_agent_broadcasted_in_payload() -> None:
    """The STEP_RESULT broadcast payload includes plugin outputs."""
    config = {
        "extra_agents": [
            {"name": "counter_2", "agent": "counter", "config": {}}
        ]
    }
    orch = Orchestrator(config=config)
    orch.agent_registry = _make_registry_with_counter()
    orch.initialize(model_config=_default_model_config())

    captured: list = []
    orch.bus.subscribe("captor", lambda msg: captured.append(msg))

    orch.run_optimized(prompt="test", num_steps=1)

    step_result_msgs = [m for m in captured if m.message_type == "STEP_RESULT"]
    assert len(step_result_msgs) == 1
    payload = step_result_msgs[0].payload
    assert "outputs" in payload
    assert payload["outputs"].get("counter_2", {}).get("plugin") is True


def test_malformed_extra_agent_entries_are_skipped() -> None:
    """Malformed extra agent entries are logged and ignored, not crashing."""
    config = {
        "extra_agents": [
            {"name": "ok_counter", "agent": "counter", "config": {}},
            {"missing_name": True},  # should be skipped
            "not_a_dict",  # should be skipped
        ]
    }
    orch = Orchestrator(config=config)
    orch.agent_registry = _make_registry_with_counter()
    orch.initialize(model_config=_default_model_config())

    assert "ok_counter" in orch.registry
    assert "missing_name" not in orch.registry
    orch.run_optimized(prompt="test", num_steps=1)
    assert orch.registry.get("ok_counter").step_count == 1


def test_unknown_extra_agent_raises() -> None:
    """An unknown agent plugin name raises a clear error at init time."""
    config = {
        "extra_agents": [
            {"name": "bad", "agent": "not_registered", "config": {}}
        ]
    }
    orch = Orchestrator(config=config)
    orch.agent_registry = _make_registry_with_counter()
    with pytest.raises(KeyError, match="Unknown agent plugin"):
        orch.initialize(model_config=_default_model_config())
