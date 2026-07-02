"""测试 BaseAgent 生命周期和状态转换。"""

import pytest

from shadowinfer.core.base_agent import AgentRegistry, BaseAgent
from shadowinfer.core.structs import AgentState


class DummyAgent(BaseAgent):
    """用于测试的最小 Agent 实现。"""

    def on_init(self, model_config):
        self.transition_to(AgentState.READY)

    def on_step(self, step_config, inputs):
        self.transition_to(AgentState.RUNNING)
        return {}

    def on_shutdown(self):
        self.transition_to(AgentState.SHUTDOWN)
        return None


class TestBaseAgent:
    def test_init_state(self):
        """验证初始状态为 INIT。"""
        agent = DummyAgent(config={}, name="test")
        assert agent.get_status() == AgentState.INIT

    def test_valid_transitions(self):
        """验证合法状态转换路径。"""
        agent = DummyAgent(config={}, name="test")
        # INIT -> READY
        assert agent.transition_to(AgentState.READY) is True
        assert agent.get_status() == AgentState.READY
        # READY -> RUNNING
        assert agent.transition_to(AgentState.RUNNING) is True
        assert agent.get_status() == AgentState.RUNNING
        # RUNNING -> PAUSED
        assert agent.transition_to(AgentState.PAUSED) is True
        assert agent.get_status() == AgentState.PAUSED
        # PAUSED -> RUNNING
        assert agent.transition_to(AgentState.RUNNING) is True
        assert agent.get_status() == AgentState.RUNNING
        # RUNNING -> SHUTDOWN
        assert agent.transition_to(AgentState.SHUTDOWN) is True
        assert agent.get_status() == AgentState.SHUTDOWN

    def test_invalid_transition(self):
        """验证非法状态转换被阻止。"""
        agent = DummyAgent(config={}, name="test")
        # INIT -> RUNNING (不合法，必须经过 READY)
        assert agent.transition_to(AgentState.RUNNING) is False
        assert agent.get_status() == AgentState.INIT
        # SHUTDOWN -> 任何状态都不合法
        agent.set_state(AgentState.SHUTDOWN)
        assert agent.transition_to(AgentState.INIT) is False
        assert agent.transition_to(AgentState.READY) is False

    def test_error_state(self):
        """验证错误状态处理。"""
        agent = DummyAgent(config={}, name="test")
        agent.transition_to(AgentState.READY)
        agent.transition_to(AgentState.RUNNING)
        # 模拟错误
        try:
            raise ValueError("test error")
        except Exception as e:
            agent.log_error(e, step_id=5)
        assert agent.get_status() == AgentState.ERROR
        assert len(agent.error_log) == 1
        assert agent.error_log[0].error_type == "ValueError"
        assert agent.error_log[0].step_id == 5
        # ERROR 可以转到 INIT 或 SHUTDOWN
        assert agent.transition_to(AgentState.INIT) is True
        assert agent.transition_to(AgentState.SHUTDOWN) is True

    def test_step_stats_recording(self):
        """验证 per-step 统计记录。"""
        agent = DummyAgent(config={}, name="test")
        agent.record_step_stat(
            step_id=0,
            stats={
                "latency_ms": 12.5,
                "memory_mb": 1024.0,
                "flops": 1e9,
                "accuracy_delta": 0.001,
                "kv_compression_ratio": 0.5,
                "ffn_sparse_ratio": 0.3,
            },
        )
        agent.record_step_stat(
            step_id=1,
            stats={
                "latency_ms": 13.0,
                "memory_mb": 1050.0,
                "flops": 1e9,
                "accuracy_delta": 0.002,
                "kv_compression_ratio": 0.48,
                "ffn_sparse_ratio": 0.35,
            },
        )
        assert len(agent.step_stats) == 2
        assert agent.step_stats[0].latency_ms == 12.5
        assert agent.step_stats[1].memory_mb == 1050.0
        assert agent.step_stats[0].timestamp != ""

    def test_performance_summary(self):
        """验证性能汇总。"""
        agent = DummyAgent(config={}, name="test")
        agent.record_step_stat(
            step_id=0,
            stats={
                "latency_ms": 10.0,
                "memory_mb": 1000.0,
                "flops": 1e9,
                "accuracy_delta": 0.0,
                "kv_compression_ratio": 0.5,
                "ffn_sparse_ratio": 0.0,
            },
        )
        agent.record_step_stat(
            step_id=1,
            stats={
                "latency_ms": 20.0,
                "memory_mb": 2000.0,
                "flops": 2e9,
                "accuracy_delta": 0.0,
                "kv_compression_ratio": 0.6,
                "ffn_sparse_ratio": 0.1,
            },
        )
        summary = agent.get_performance_summary()
        assert summary["total_steps"] == 2.0
        assert summary["total_latency_ms"] == 30.0
        assert summary["avg_latency_ms"] == 15.0
        assert summary["avg_memory_mb"] == 1500.0
        assert summary["avg_kv_compression_ratio"] == 0.55
        assert summary["avg_ffn_sparse_ratio"] == 0.05
        assert summary["error_count"] == 0.0


class TestAgentRegistry:
    def test_register_and_get(self):
        """验证注册和获取 Agent。"""
        registry = AgentRegistry()
        agent1 = DummyAgent(config={}, name="a1")
        agent2 = DummyAgent(config={}, name="a2")
        registry.register(agent1)
        registry.register(agent2)
        assert registry.get("a1") is agent1
        assert registry.get("a2") is agent2
        assert len(registry) == 2

    def test_duplicate_register_raises(self):
        """验证重复注册抛出 ValueError。"""
        registry = AgentRegistry()
        agent = DummyAgent(config={}, name="a1")
        registry.register(agent)
        with pytest.raises(ValueError):
            registry.register(agent)

    def test_remove_and_clear(self):
        """验证移除和清空。"""
        registry = AgentRegistry()
        agent = DummyAgent(config={}, name="a1")
        registry.register(agent)
        assert registry.remove("a1") is True
        assert registry.remove("a1") is False
        assert len(registry) == 0
        registry.register(agent)
        registry.clear()
        assert len(registry) == 0

    def test_contains_and_iter(self):
        """验证包含检查和迭代。"""
        registry = AgentRegistry()
        agent = DummyAgent(config={}, name="a1")
        registry.register(agent)
        assert "a1" in registry
        assert "a2" not in registry
        agents = list(registry)
        assert len(agents) == 1
        assert agents[0].name == "a1"
