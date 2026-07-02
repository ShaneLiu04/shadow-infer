"""测试 Orchestrator。"""

import json

from shadowinfer.core import AgentState, Config, MockModelBackend
from shadowinfer.engineering.degradation_circuit import ProductionSafetyNet
from shadowinfer.orchestrator import InferenceResult, Orchestrator


class TestOrchestrator:
    def test_load_config(self):
        """验证配置加载。"""
        # 通过字符串路径加载
        orch = Orchestrator(config="configs/optimize_full.yaml")
        assert orch.config is not None
        assert "optimization" in orch.config
        assert orch.config.get("shadowkv", {}).get("enabled") is True

        # 通过 Config 对象加载
        cfg = Config.from_yaml("configs/optimize_full.yaml")
        orch2 = Orchestrator(config=cfg)
        assert orch2.config is not None

        # 通过 dict 加载
        raw = cfg.to_dict()
        orch3 = Orchestrator(config=raw)
        assert orch3.config is not None

    def test_initialize_agents(self):
        """验证初始化所有 Agent。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        orch.initialize()
        assert orch._initialized is True
        agents = orch.agents
        assert len(agents) == 4
        for name, agent in agents.items():
            assert agent.get_status() == AgentState.READY

    def test_single_step(self):
        """验证单 step 执行。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        orch.initialize()
        inputs = orch._make_dummy_inputs(0, 10)
        result = orch.run_step(0, 10, inputs)
        assert result["step_id"] == 0
        assert "qdrift" in result
        assert "shadowkv" in result
        assert "ffn" in result
        assert "profiler" in result

    def test_conflict_resolution_accuracy(self):
        """验证 accuracy 冲突仲裁。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        orch.initialize()
        # 模拟 profiler 输出包含 accuracy CRITICAL
        profiler_output = {
            "alerts": [
                {
                    "level": "CRITICAL",
                    "metric": "accuracy_drop",
                    "message": "Accuracy drop exceeds 1.0%",
                    "value": 0.015,
                }
            ]
        }
        conflicts = orch._check_conflicts(profiler_output)
        resolution = orch._resolve_conflicts(conflicts)
        assert resolution["accuracy_priority"] is True
        assert "rollback_to_full_precision" in resolution["actions"]

    def test_conflict_resolution_memory(self):
        """验证 memory 冲突仲裁。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        orch.initialize()
        profiler_output = {
            "alerts": [
                {
                    "level": "WARNING",
                    "metric": "memory",
                    "message": "Memory exceeds 90% budget",
                    "value": 7500.0,
                }
            ]
        }
        conflicts = orch._check_conflicts(profiler_output)
        resolution = orch._resolve_conflicts(conflicts)
        assert resolution["memory_priority"] is True
        assert "increase_kv_compression" in resolution["actions"]

    def test_context_manager(self):
        """验证上下文管理器自动初始化和关闭。"""
        with Orchestrator(config="configs/optimize_full.yaml") as orch:
            assert orch._initialized is True
            inputs = orch._make_dummy_inputs(0, 10)
            result = orch.run_step(0, 10, inputs)
            assert result is not None
        for agent in orch.agents.values():
            assert agent.get_status() == AgentState.SHUTDOWN

    def test_global_constraints(self):
        """验证全局约束检查。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        orch.initialize()
        # 满足约束
        r1 = InferenceResult()
        r1.accuracy_drop = 0.005
        r1.latency_ms = 50.0
        r1.memory_mb = 1000.0
        assert orch._check_global_constraints(r1) is True

        # 违反 latency
        r2 = InferenceResult()
        r2.accuracy_drop = 0.005
        r2.latency_ms = 200.0  # > 100
        r2.memory_mb = 1000.0
        assert orch._check_global_constraints(r2) is False

        # 违反 memory
        r3 = InferenceResult()
        r3.accuracy_drop = 0.005
        r3.latency_ms = 50.0
        r3.memory_mb = 9000.0  # > 8192
        assert orch._check_global_constraints(r3) is False

    def test_run_inference(self):
        """验证完整推理运行。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        result = orch.run_inference(model="test", prompt="hello", num_steps=3)
        assert isinstance(result, InferenceResult)
        assert len(result.step_results) == 3
        assert result.latency_ms > 0.0

    def test_run_baseline_and_optimized(self):
        """验证基线和优化运行。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        baseline = orch.run_baseline(prompt="test", num_steps=2)
        optimized = orch.run_optimized(prompt="test", num_steps=2)
        assert isinstance(baseline, dict)
        assert isinstance(optimized, dict)
        assert "latency" in baseline
        assert "latency" in optimized

    def test_bus_available(self):
        """验证 Bus 可用。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        orch.initialize()
        assert orch.bus is not None
        assert orch.bus.get_subscriber_count() == 4

    def test_compute_total_latency(self):
        """验证总延迟计算。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        result = InferenceResult()
        result.step_results = [
            {"profiler": {"performance_metrics": {"latency": {"e2e_ms": 10.0}}}},
            {"profiler": {"performance_metrics": {"latency": {"e2e_ms": 15.0}}}},
            {"profiler": {"performance_metrics": {"latency": {"e2e_ms": 20.0}}}},
        ]
        total = orch._compute_total_latency(result)
        assert total == 45.0

    def test_compute_peak_memory(self):
        """验证峰值内存计算。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        result = InferenceResult()
        result.step_results = [
            {"profiler": {"performance_metrics": {"memory": {"total_mb": 1000.0}}}},
            {"profiler": {"performance_metrics": {"memory": {"total_mb": 2000.0}}}},
            {"profiler": {"performance_metrics": {"memory": {"total_mb": 1500.0}}}},
        ]
        peak = orch._compute_peak_memory(result)
        assert peak == 2000.0


class TestOrchestratorModelBackend:
    def test_accepts_mock_model_backend(self):
        """验证 Orchestrator 可接受 MockModelBackend。"""
        backend = MockModelBackend(
            {
                "name": "test-model",
                "num_layers": 4,
                "num_heads": 8,
                "head_dim": 32,
                "hidden_dim": 256,
                "intermediate_dim": 512,
                "batch_size": 1,
                "seq_len": 16,
            }
        )
        orch = Orchestrator(config="configs/optimize_full.yaml", model_backend=backend)
        assert orch.model_backend is backend
        orch.initialize()
        assert orch._get_model_backend() is backend

    def test_set_model_backend(self):
        """验证 set_model_backend 生效。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        backend = MockModelBackend()
        orch.set_model_backend(backend)
        assert orch.model_backend is backend
        assert orch._get_model_backend() is backend

    def test_run_baseline_with_model_backend(self):
        """验证使用模型后端运行基线。"""
        backend = MockModelBackend(
            {
                "name": "test-model",
                "num_layers": 2,
                "num_heads": 4,
                "head_dim": 16,
                "hidden_dim": 64,
                "intermediate_dim": 128,
                "batch_size": 1,
                "seq_len": 8,
            }
        )
        orch = Orchestrator(config="configs/optimize_full.yaml", model_backend=backend)
        baseline = orch.run_baseline(prompt="test", num_steps=2)
        assert isinstance(baseline, dict)
        assert "latency" in baseline


class TestOrchestratorSafetyNet:
    def test_enable_safety_net(self):
        """验证 enable_safety_net 可启用安全网。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        safety_net = ProductionSafetyNet()
        orch.enable_safety_net(safety_net)
        assert orch.safety_net is safety_net
        assert orch.safety_net._enabled is True

    def test_enable_safety_net_default(self):
        """验证未传入安全网时自动创建默认实例。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        orch.enable_safety_net()
        assert isinstance(orch.safety_net, ProductionSafetyNet)
        assert orch.safety_net._enabled is True

    def test_safety_net_triggers_checks(self):
        """验证安全网在 step 前后触发检查。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        safety_net = ProductionSafetyNet()
        orch.enable_safety_net(safety_net)
        orch.initialize()

        pre_calls = []
        post_calls = []

        original_pre = safety_net.pre_flight_check

        def patched_pre(*args, **kwargs):
            pre_calls.append((args, kwargs))
            return original_pre(*args, **kwargs)

        original_post = safety_net.post_flight_check

        def patched_post(*args, **kwargs):
            post_calls.append((args, kwargs))
            return original_post(*args, **kwargs)

        safety_net.pre_flight_check = patched_pre
        safety_net.post_flight_check = patched_post

        inputs = orch._make_dummy_inputs(0, 2)
        result = orch.run_step(0, 2, inputs)
        assert result["step_id"] == 0
        assert len(pre_calls) == 1
        assert len(post_calls) == 1

    def test_safety_net_pre_flight_degrade(self):
        """验证安全网 pre-flight 失败时降级到保守设置。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        safety_net = ProductionSafetyNet()
        orch.enable_safety_net(safety_net)
        orch.initialize()

        def failing_pre(*args, **kwargs):
            return False, "forced degradation"

        safety_net.pre_flight_check = failing_pre

        inputs = orch._make_dummy_inputs(0, 2)
        result = orch.run_step(0, 2, inputs)
        assert result["step_id"] == 0
        assert result["step_config"]["shadowkv_mode"] == "conservative"
        assert result["step_config"]["ffn_mode"] == "full"

    def test_safety_net_pre_flight_exception_degrade(self):
        """验证安全网 pre-flight 抛异常时降级到保守设置。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        safety_net = ProductionSafetyNet()
        orch.enable_safety_net(safety_net)
        orch.initialize()

        def raising_pre(*args, **kwargs):
            raise RuntimeError("forced error")

        safety_net.pre_flight_check = raising_pre

        inputs = orch._make_dummy_inputs(0, 2)
        result = orch.run_step(0, 2, inputs)
        assert result["step_id"] == 0
        assert result["step_config"]["shadowkv_mode"] == "conservative"
        assert result["step_config"]["ffn_mode"] == "full"


class TestOrchestratorProductionFeatures:
    """Orchestrator 生产级新特性测试。"""

    def test_run_stream_yields_steps(self):
        """流式运行应逐 step 生成完整结果字典。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        try:
            steps = list(orch.run_stream(prompt="test", num_steps=3))
            assert len(steps) == 3
            for s in steps:
                assert "step_id" in s
                assert "qdrift" in s
                assert "shadowkv" in s
                assert "ffn" in s
                assert "profiler" in s
        finally:
            orch.shutdown()

    def test_cancel_stops_pipeline(self):
        """取消请求应中断流式推理。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        try:
            stream = orch.run_stream(prompt="test", num_steps=20)
            first = next(stream)
            assert first["step_id"] == 0
            orch.cancel()
            remaining = list(stream)
            assert len(remaining) < 19
        finally:
            orch.shutdown()

    def test_on_step_callback(self):
        """on_step 回调应在每个优化 step 后被调用。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        called = []

        def cb(step):
            called.append(step["step_id"])

        try:
            orch.run_optimized(prompt="test", num_steps=2, on_step=cb)
            assert len(called) == 2
            assert called == [0, 1]
        finally:
            orch.shutdown()

    def test_step_snapshot_written(self, tmp_path):
        """启用快照时应将每个 step 状态写入 JSON。"""
        snapshot_dir = tmp_path / "snapshots"
        cfg = {
            "output_dir": str(tmp_path / "outputs"),
            "snapshot_dir": str(snapshot_dir),
            "enable_snapshots": True,
        }
        orch = Orchestrator(config=cfg)
        try:
            orch.run_optimized(prompt="test", num_steps=1)
            files = list(snapshot_dir.glob("*.json"))
            assert len(files) == 1
            data = json.loads(files[0].read_text(encoding="utf-8"))
            assert data["step_id"] == 0
            assert "qdrift" in data
            assert "shadowkv" in data
            assert "ffn" in data
            assert "profiler" in data
        finally:
            orch.shutdown()
