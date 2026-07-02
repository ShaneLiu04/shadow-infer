"""pytest fixtures for ShadowInfer test suite."""

import pytest
import torch


@pytest.fixture
def mock_config():
    """返回标准测试配置。"""
    return {
        "model_name": "Fast-dLLM-v2-7B",
        "run_id": "test-run",
        "num_layers": 4,
        "hidden_dim": 4096,
        "num_heads": 32,
        "head_dim": 128,
        "intermediate_dim": 11008,
        "max_seq_len": 4096,
        "log_dir": "logs/",
        "compression_target": 0.5,
        "importance_thresholds": {"fp32": 0.8, "fp16": 0.5, "int8": 0.2, "int4": 0.0},
        "reuse_base_threshold": 0.15,
        "reuse_adaptive": True,
        "learning_rate": 0.05,
        "noise_schedule": "cosine",
        "sensitivity_temperature": 1.0,
        "drift_method": "relative_l2",
        "mixed_precision": True,
        "channel_importance_threshold": 0.7,
        "sparse_update": True,
        "delta_threshold": 0.05,
        "compute_paths": ["reuse", "incremental", "sparse", "full"],
    }


@pytest.fixture
def mock_model_config():
    """返回标准测试模型配置。"""
    return {
        "model_name": "Fast-dLLM-v2-7B",
        "num_layers": 4,
        "hidden_dim": 4096,
        "num_heads": 32,
        "head_dim": 128,
        "intermediate_dim": 11008,
        "vocab_size": 32000,
        "max_seq_len": 4096,
        "max_latency_ms": 100.0,
        "max_memory_mb": 8192.0,
    }


@pytest.fixture
def mock_step_config():
    """返回标准测试 step 配置。"""
    from shadowinfer.core.structs import StepConfig

    return StepConfig(
        step_id=0,
        total_steps=10,
        noise_level=0.0,
        shadowkv_mode="balanced",
        reuse_layers=[0, 1],
        compression_target=0.5,
        ffn_mode="full",
        weight_precision_map={},
        compute_path="full",
        sensitivity_score=0.3,
        drift_score=0.1,
    )


@pytest.fixture
def mock_attention_scores():
    """返回模拟 attention scores。"""
    return torch.randn(1, 32, 128, 128)


@pytest.fixture
def mock_kv_tensors():
    """返回模拟 KV tensors。"""
    return {
        "k": torch.randn(1, 32, 128, 128),
        "v": torch.randn(1, 32, 128, 128),
    }


@pytest.fixture
def mock_ffn_weights():
    """返回模拟 FFN 权重。"""
    return {
        "up": torch.randn(11008, 4096),
        "down": torch.randn(4096, 11008),
    }


@pytest.fixture
def mock_ffn_inputs():
    """返回模拟 FFN 输入。"""
    return torch.randn(1, 128, 4096)
