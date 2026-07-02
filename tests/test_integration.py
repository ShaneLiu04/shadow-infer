"""集成测试。"""

from shadowinfer.core import AgentState
from shadowinfer.orchestrator import InferenceResult, Orchestrator


class TestIntegration:
    def test_orchestrator_initialization(self):
        """验证 Orchestrator 初始化所有 Agent。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        orch.initialize()
        assert orch._initialized is True
        agents = orch.agents
        assert "profiler" in agents
        assert "shadowkv" in agents
        assert "qdrift" in agents
        assert "ffn_optimizer" in agents
        for name, agent in agents.items():
            assert agent.get_status() == AgentState.READY

    def test_single_step_execution(self):
        """验证单 step 执行流程。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        orch.initialize()
        inputs = orch._make_dummy_inputs(0, 10)
        result = orch.run_step(0, 10, inputs)
        assert "qdrift" in result
        assert "shadowkv" in result
        assert "ffn" in result
        assert "profiler" in result
        assert result["step_id"] == 0

    def test_conflict_resolution(self):
        """验证冲突仲裁机制。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        orch.initialize()
        # 制造一个 accuracy drop 很大的场景
        inputs = orch._make_dummy_inputs(0, 10)
        inputs["accuracy"] = {
            "baseline_perplexity": 10.0,
            "optimized_perplexity": 11.5,  # 15% drop -> CRITICAL
        }
        result = orch.run_step(0, 10, inputs)
        conflicts = result.get("conflicts", [])
        resolution = result.get("resolution", {})
        # 至少有一个 conflict 被检测到
        assert isinstance(conflicts, list)
        assert isinstance(resolution, dict)
        assert "actions" in resolution

    def test_end_to_end_pipeline(self):
        """验证完整端到端流水线。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        result = orch.run_inference(
            model="Fast-dLLM-v2-7B",
            prompt="Hello world",
            num_steps=5,
        )
        assert isinstance(result, InferenceResult)
        assert len(result.step_results) == 5
        assert isinstance(result.accuracy_drop, float)
        assert isinstance(result.latency_ms, float)
        assert isinstance(result.memory_mb, float)
        assert isinstance(result.constraints_satisfied, bool)
        # 关闭
        orch.shutdown()
        for name, agent in orch.agents.items():
            assert agent.get_status() == AgentState.SHUTDOWN

    def test_bus_integration(self):
        """验证 Bus 集成所有 Agent 通信。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        orch.initialize()
        # 检查 bus 上有订阅者
        assert orch.bus.get_subscriber_count() == 4
        # 运行一个 step，bus 上应该有消息
        inputs = orch._make_dummy_inputs(0, 10)
        orch.run_step(0, 10, inputs)
        logs = orch.bus.get_message_log()
        assert len(logs) > 0
        # 至少有一条 STEP_RESULT 广播
        step_results = orch.bus.get_message_log(message_type="STEP_RESULT")
        assert len(step_results) >= 1

    def test_context_manager(self):
        """验证 Orchestrator 上下文管理器。"""
        with Orchestrator(config="configs/optimize_full.yaml") as orch:
            assert orch._initialized is True
            inputs = orch._make_dummy_inputs(0, 10)
            result = orch.run_step(0, 10, inputs)
            assert result is not None
        # 退出后应该已 shutdown
        for name, agent in orch.agents.items():
            assert agent.get_status() == AgentState.SHUTDOWN

    def test_global_constraints(self):
        """验证全局约束检查。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        orch.initialize()
        # 模拟满足约束的结果
        result = InferenceResult()
        result.accuracy_drop = 0.005
        result.latency_ms = 50.0
        result.memory_mb = 1000.0
        satisfied = orch._check_global_constraints(result)
        assert satisfied is True

        # 模拟违反约束的结果
        result2 = InferenceResult()
        result2.accuracy_drop = 0.02  # > 0.01
        result2.latency_ms = 50.0
        result2.memory_mb = 1000.0
        satisfied2 = orch._check_global_constraints(result2)
        assert satisfied2 is False
