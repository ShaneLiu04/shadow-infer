"""
Tests for distributed inference module.
"""

import torch

from shadowinfer.distributed import (
    DistributedInferenceEngine,
    DistributedKVCacheManager,
    PipelineParallelEngine,
    PipelineStage,
    TensorParallelAttention,
)


class TestPipelineStage:
    def test_forward_shape(self):
        """Test pipeline stage forward pass."""
        layers = [torch.nn.Linear(16, 32), torch.nn.ReLU()]
        stage = PipelineStage(0, 4, layers, torch.device("cpu"))

        x = torch.randn(2, 4, 16)
        output = stage.forward(x)
        assert output.shape == (2, 4, 32)

    def test_device_placement(self):
        """Test tensors are on correct device."""
        layers = [torch.nn.Linear(8, 16)]
        stage = PipelineStage(0, 2, layers, torch.device("cpu"))

        x = torch.randn(1, 2, 8)
        output = stage.forward(x)
        assert output.device == torch.device("cpu")


class TestPipelineParallelEngine:
    def test_forward(self):
        """Test pipeline parallel forward."""
        stages = [
            PipelineStage(0, 3, [torch.nn.Linear(16, 32)], torch.device("cpu")),
            PipelineStage(1, 3, [torch.nn.Linear(32, 64)], torch.device("cpu")),
            PipelineStage(2, 3, [torch.nn.Linear(64, 8)], torch.device("cpu")),
        ]

        engine = PipelineParallelEngine(stages, num_micro_batches=4)
        x = torch.randn(2, 4, 16)
        output = engine.forward(x)
        assert output.shape == (2, 4, 8)

    def test_bubble_estimation(self):
        """Test pipeline bubble estimation."""
        stages = [
            PipelineStage(0, 4, [torch.nn.Identity()], torch.device("cpu")),
        ] * 4

        engine = PipelineParallelEngine(stages, num_micro_batches=4)
        bubble = engine.estimate_pipeline_bubble(10.0)

        # With 4 stages and 4 micro-batches: bubble = 3*10 / (4*4) = 7.5/16 = 1.875
        expected = (4 - 1) * 10.0 / (4 * 4)
        assert abs(bubble - expected) < 1e-6

    def test_more_microbatches_reduce_bubble(self):
        """Test more micro-batches reduce bubble."""
        stages = [PipelineStage(0, 4, [torch.nn.Identity()], torch.device("cpu"))] * 4

        engine_4 = PipelineParallelEngine(stages, num_micro_batches=4)
        engine_8 = PipelineParallelEngine(stages, num_micro_batches=8)

        bubble_4 = engine_4.estimate_pipeline_bubble(10.0)
        bubble_8 = engine_8.estimate_pipeline_bubble(10.0)

        assert bubble_8 < bubble_4


class TestTensorParallelAttention:
    def test_split_heads(self):
        """Test head splitting."""
        tp = TensorParallelAttention(num_heads=32, num_gpus=4, head_dim=64)

        B, H, S, D = 2, 32, 8, 64
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        splits = tp.split_heads(q, k, v)

        assert len(splits) == 4
        for q_local, k_local, v_local in splits:
            assert q_local.shape[1] == 8  # 32/4 = 8 heads per GPU
            assert k_local.shape[1] == 8
            assert v_local.shape[1] == 8

    def test_parallel_compute_shape(self):
        """Test tensor parallel attention shape."""
        tp = TensorParallelAttention(num_heads=32, num_gpus=4, head_dim=64)

        B, H, S, D = 2, 32, 8, 64
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        output = tp.compute_parallel(q, k, v)
        assert output.shape == (B, H, S, D)

    def test_communication_volume(self):
        """Test communication volume estimate."""
        tp = TensorParallelAttention(num_heads=32, num_gpus=4, head_dim=64)

        volume = tp.estimate_communication_volume(batch_size=2, seq_len=512)

        # Expected: 2 * B * S * H * D * sizeof(fp16) = 2 * 2 * 512 * 32 * 64 * 2
        expected = 2 * 2 * 512 * 32 * 64 * 2
        assert volume == expected

    def test_heads_per_gpu(self):
        """Test heads per GPU calculation."""
        tp = TensorParallelAttention(num_heads=32, num_gpus=4, head_dim=64)
        assert tp.heads_per_gpu == 8

        tp2 = TensorParallelAttention(num_heads=16, num_gpus=2, head_dim=64)
        assert tp2.heads_per_gpu == 8


class TestDistributedKVCacheManager:
    def test_sync_kv_cache(self):
        """Test KV cache sync."""
        manager = DistributedKVCacheManager(num_gpus=4, max_seq_len=512, head_dim=64)

        k = torch.randn(1, 32, 512, 64)
        v = torch.randn(1, 32, 512, 64)

        manager.sync_kv_cache(0, k, v)

        cached = manager.get_consistent_kv(0)
        assert cached is not None
        assert torch.equal(cached[0], k)
        assert torch.equal(cached[1], v)

    def test_estimate_sync_cost(self):
        """Test sync cost estimation."""
        manager = DistributedKVCacheManager(num_gpus=4, max_seq_len=512, head_dim=64)

        cost = manager.estimate_sync_cost(batch_size=2, seq_len=512, num_heads=32)

        # Expected: 2 * B * S * H * D * num_gpus * sizeof(fp16)
        expected = 2 * 2 * 512 * 32 * 64 * 2 * 4
        assert cost == expected

    def test_multiple_gpus_sync(self):
        """Test sync to multiple GPUs."""
        manager = DistributedKVCacheManager(num_gpus=4, max_seq_len=512, head_dim=64)

        for gpu_id in range(4):
            k = torch.randn(1, 32, 512, 64)
            v = torch.randn(1, 32, 512, 64)
            manager.sync_kv_cache(gpu_id, k, v)

        assert len(manager.kv_cache) == 4


class TestDistributedInferenceEngine:
    def test_forward(self):
        """Test distributed engine forward."""
        stages = [
            PipelineStage(0, 2, [torch.nn.Linear(16, 32)], torch.device("cpu")),
            PipelineStage(1, 2, [torch.nn.Linear(32, 8)], torch.device("cpu")),
        ]

        engine = DistributedInferenceEngine(pipeline_stages=stages)
        x = torch.randn(2, 4, 16)
        output = engine.forward(x)
        assert output.shape == (2, 4, 8)

    def test_efficiency_report(self):
        """Test efficiency report generation."""
        stages = [PipelineStage(0, 2, [torch.nn.Identity()], torch.device("cpu"))] * 2
        tp = TensorParallelAttention(num_heads=32, num_gpus=4, head_dim=64)
        kv_manager = DistributedKVCacheManager(num_gpus=4, max_seq_len=512, head_dim=64)

        engine = DistributedInferenceEngine(
            pipeline_stages=stages,
            tensor_parallel=tp,
            kv_manager=kv_manager,
        )

        report = engine.get_efficiency_report()

        assert "pipeline_bubble" in report
        assert "tensor_comm_volume" in report
        assert "kv_sync_cost" in report
        assert report["pipeline_bubble"] > 0
        assert report["tensor_comm_volume"] > 0
        assert report["kv_sync_cost"] > 0

    def test_no_pipeline(self):
        """Test engine without pipeline parallelism."""
        engine = DistributedInferenceEngine(pipeline_stages=None)
        x = torch.randn(2, 4, 16)
        output = engine.forward(x)
        assert torch.equal(output, x)
