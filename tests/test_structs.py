"""测试核心数据结构。"""

import torch

from shadowinfer.core.structs import (
    AgentState,
    ErrorRecord,
    KVCacheEntry,
    Message,
    ProfileResult,
    StepConfig,
    StepStats,
)


class TestStructs:
    def test_kv_cache_entry(self):
        """验证 KVCacheEntry 创建和字段。"""
        k = torch.randn(1, 32, 128, 128)
        v = torch.randn(1, 32, 128, 128)
        entry = KVCacheEntry(
            k_tensor=k,
            v_tensor=v,
            precision="fp16",
            scale_k=None,
            scale_v=None,
            importance_score=0.85,
            is_reused=False,
            reuse_step=-1,
        )
        assert entry.precision == "fp16"
        assert entry.importance_score == 0.85
        assert entry.is_reused is False
        assert entry.reuse_step == -1
        assert torch.equal(entry.k_tensor, k)
        assert torch.equal(entry.v_tensor, v)

    def test_step_config(self):
        """验证 StepConfig 创建。"""
        sc = StepConfig(
            step_id=5,
            total_steps=50,
            noise_level=0.1,
            shadowkv_mode="aggressive",
            reuse_layers=[0, 1, 2],
            compression_target=0.6,
            ffn_mode="sparse",
            weight_precision_map={0: "fp16", 1: "int8"},
            compute_path="sparse",
            sensitivity_score=0.2,
            drift_score=0.15,
        )
        assert sc.step_id == 5
        assert sc.total_steps == 50
        assert sc.noise_level == 0.1
        assert sc.shadowkv_mode == "aggressive"
        assert sc.reuse_layers == [0, 1, 2]
        assert sc.compression_target == 0.6
        assert sc.ffn_mode == "sparse"
        assert sc.compute_path == "sparse"
        assert sc.sensitivity_score == 0.2
        assert sc.drift_score == 0.15

    def test_profile_result(self):
        """验证 ProfileResult 创建。"""
        pr = ProfileResult(
            model_name="Fast-dLLM-v2-7B",
            run_id="run-001",
            kv_precision_distribution={0: {0: "fp32", 1: "fp16"}},
            kv_reuse_rate={0: 0.5},
            kv_memory_mb={0: 100.0},
            q_drift_hit_rate={0: 0.9},
            activation_delta={0: {"query": 0.01}},
            ffn_compute_load={0: {"flops": 1e9}},
            ffn_sparse_update_ratio=0.3,
            latency_e2e_ms=500.0,
            latency_per_step_ms={0: 10.0, 1: 12.0},
            throughput_tokens_per_sec=50.0,
            perplexity_delta=0.005,
            bleu_drop=0.001,
            accuracy_metrics={"em": 0.95},
        )
        assert pr.model_name == "Fast-dLLM-v2-7B"
        assert pr.run_id == "run-001"
        assert pr.latency_e2e_ms == 500.0
        assert pr.perplexity_delta == 0.005
        assert pr.accuracy_metrics["em"] == 0.95

    def test_message_creation(self):
        """验证 Message 创建和工厂方法。"""
        msg = Message.create(
            source="profiler",
            target="orchestrator",
            message_type="PROFILE_DATA",
            payload={"latency_ms": 15.0},
            step_id=3,
        )
        assert msg.source == "profiler"
        assert msg.target == "orchestrator"
        assert msg.message_type == "PROFILE_DATA"
        assert msg.payload["latency_ms"] == 15.0
        assert msg.step_id == 3
        assert msg.version == "1.0"
        assert msg.message_id != ""
        assert msg.timestamp != ""

    def test_agent_state_enum(self):
        """验证 AgentState 枚举值。"""
        assert AgentState.INIT == "INIT"
        assert AgentState.READY == "READY"
        assert AgentState.RUNNING == "RUNNING"
        assert AgentState.PAUSED == "PAUSED"
        assert AgentState.ERROR == "ERROR"
        assert AgentState.SHUTDOWN == "SHUTDOWN"

    def test_step_stats(self):
        """验证 StepStats 创建。"""
        ss = StepStats(
            step_id=0,
            latency_ms=10.0,
            memory_mb=512.0,
            flops=1e9,
            accuracy_delta=0.001,
            kv_compression_ratio=0.5,
            ffn_sparse_ratio=0.2,
            custom_metrics={"layer_id": 0},
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert ss.step_id == 0
        assert ss.latency_ms == 10.0
        assert ss.custom_metrics["layer_id"] == 0

    def test_error_record(self):
        """验证 ErrorRecord 创建。"""
        er = ErrorRecord.from_exception(
            error=ValueError("test error"),
            step_id=5,
            traceback_str="traceback line 1\ntraceback line 2",
        )
        assert er.error_type == "ValueError"
        assert er.message == "test error"
        assert er.step_id == 5
        assert "traceback line 1" in er.traceback
        assert er.timestamp != ""
