"""FFN Optimizer Agent — FFN 层计算优化专家。

对应文档：FFN_OPTIMIZER_AGENT.md, TECHNICAL_SPEC.md §2.3 / §3.2
版本：v3.0
"""

from __future__ import annotations

__version__ = "3.0"

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from shadowinfer.core.base_agent import BaseAgent
from shadowinfer.core.structs import AgentState, ProfileResult, StepConfig
from shadowinfer.ffn_optimizer.packed_weight import PackedFFNWeight
from shadowinfer.kernels import get_kernel_status, sparse_gemm_ffn
from shadowinfer.utils.logging_utils import StructuredLogger
from shadowinfer.utils.metrics import Metrics


class FFNOptimizerAgent(BaseAgent):
    """FFN Optimizer Agent — FFN 层计算优化专家。

    对应文档：FFN_OPTIMIZER_AGENT.md, TECHNICAL_SPEC.md §2.3 / §3.2
    版本：v3.0
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, name="ffn_optimizer")
        self.weight_up_quantized = None  # 量化后的 up 权重
        self.weight_down_quantized = None  # 量化后的 down 权重
        self.channel_importance = None  # 通道重要性
        self.precision_scales = {}  # 量化 scale
        self.logger = StructuredLogger("ffn_optimizer", config.get("log_dir", "logs/"))

        # Packing configuration
        self.use_packed_weights: bool = config.get("use_packed_weights", True)
        self._pack_group_size: int = config.get("ffn_pack_group_size", 64)

        # Dynamic channel pruning configuration
        self.channel_pruning_enabled: bool = config.get("channel_pruning_enabled", False)
        self.channel_pruning_ratio: float = config.get("channel_pruning_ratio", 0.0)
        self.channel_pruning_dynamic: bool = config.get("channel_pruning_dynamic", False)
        self.channel_pruning_method: str = config.get(
            "channel_pruning_method", "static_importance"
        )
        self._channel_mask: Optional[torch.Tensor] = None
        self._active_channels: int = 0

        # Cumulative compute statistics
        self._total_flops_saved: float = 0.0
        self._total_flops: float = 0.0
        self._sparse_ratios: List[float] = []
        self._mixed_precision_ratios: List[float] = []
        self._step_count: int = 0

        # Cumulative memory statistics from packed weights
        self._total_memory_bytes_saved: float = 0.0
        self._total_memory_bytes_original: float = 0.0
        self._total_memory_bytes_packed: float = 0.0

        # Cached weights (PyTorch standard: [out_features, in_features])
        self._weight_up: Optional[torch.Tensor] = None
        self._weight_down: Optional[torch.Tensor] = None
        self._model_name: str = config.get("model_name", "unknown")
        self._run_id: str = config.get("run_id", "default")

    # ------------------------------------------------------------------
    # 生命周期接口
    # ------------------------------------------------------------------

    def on_init(self, model_config: Dict[str, Any]) -> None:
        """初始化：加载权重，分析通道重要性，预量化。

        如果 model_config 中提供了 weights，则缓存并预计算。
        否则在 on_step 中首次遇到 weights 时再进行计算。

        Args:
            model_config: 模型级配置字典（如层数、hidden_dim 等）。
        """
        self._model_name = model_config.get("model_name", self._model_name)
        self._run_id = model_config.get("run_id", self._run_id)

        if "weights" in model_config:
            weights = model_config["weights"]
            weight_up = weights.get("up")
            weight_down = weights.get("down")
            if weight_up is not None and weight_down is not None:
                self._precompute_weights(weight_up, weight_down)

        self.transition_to(AgentState.READY)
        self.logger.log_event(
            "agent_init",
            "FFNOptimizerAgent initialized.",
            data={"model_name": self._model_name},
        )

    def on_step(self, step_config: StepConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个 step。

        输入 inputs 包含：
        - 'ffn_input_current': Tensor[batch, seq_len, hidden_dim]
        - 'ffn_input_previous': Tensor (可能为 None)
        - 'ffn_output_previous': Tensor (可能为 None)
        - 'weights': Dict with 'up' and 'down' tensors
        - 'qdrift_signal': Dict with 'sensitivity_score', 'ffn_mode'
        - 'layer_id': int

        返回：{'compute_path': str, 'output': Tensor, 'quantization': Dict,
                 'sparse_update': Dict, 'compute_stats': Dict}

        Args:
            step_config: 当前 step 的优化配置。
            inputs: 输入数据字典。

        Returns:
            输出数据字典。
        """
        start_time = time.perf_counter()
        step_id = step_config.step_id
        layer_id = inputs.get("layer_id", 0)

        ffn_input_current = inputs["ffn_input_current"]
        ffn_input_previous = inputs.get("ffn_input_previous")
        ffn_output_previous = inputs.get("ffn_output_previous")
        weights = inputs.get("weights", {})
        qdrift_signal = inputs.get("qdrift_signal", {})

        weight_up = weights.get("up")
        weight_down = weights.get("down")

        if weight_up is None or weight_down is None:
            raise ValueError("Missing 'up' or 'down' weights in inputs.")

        # 缓存权重（仅在首次遇到时）
        if self._weight_up is None:
            self._precompute_weights(weight_up, weight_down)

        # 使用缓存的权重
        weight_up = self._weight_up
        weight_down = self._weight_down

        # Step 1: 解析 Q-drift 信号
        sensitivity_score = qdrift_signal.get("sensitivity_score", 0.0)
        ffn_mode = qdrift_signal.get("ffn_mode", step_config.ffn_mode)

        # 如果 sensitivity_score >= 0.7，强制使用 full 模式，并禁用通道剪枝
        if sensitivity_score >= 0.7:
            ffn_mode = "full"

        # Pre-compute channel importance if channel pruning may be used.
        if (
            self.channel_pruning_enabled
            and self.channel_pruning_ratio > 0.0
            and sensitivity_score < 0.7
            and self.channel_importance is None
        ):
            self.channel_importance = self.analyze_channel_importance(
                ffn_input_current, weight_up, weight_down, method="weight_magnitude"
            )

        # Dynamic channel pruning: only when enabled and sensitivity is low enough.
        channel_mask: Optional[torch.Tensor] = None
        active_ratio = 1.0
        if (
            self.channel_pruning_enabled
            and self.channel_pruning_ratio > 0.0
            and sensitivity_score < 0.7
        ):
            try:
                channel_mask = self.compute_channel_mask(
                    ffn_input_current, weight_up, method=self.channel_pruning_method
                )
                active_channels = int(channel_mask.sum().item())
                intermediate_dim = weight_up.shape[0]
                active_ratio = active_channels / max(1, intermediate_dim)
                self._channel_mask = channel_mask
                self._active_channels = active_channels
            except Exception as exc:
                self.logger.log_alert(
                    "warning",
                    f"Channel pruning failed at step {step_id}: {exc}",
                    recommendation="Falling back to full channels.",
                    step_id=step_id,
                )
                channel_mask = None

        # Step 2: 选择计算路径
        compute_path = self.select_compute_path(
            ffn_input_current,
            ffn_input_previous,
            ffn_output_previous,
            sensitivity_score,
            ffn_mode,
        )

        # 安全回退：如果缺少 previous 数据但路径需要复用
        if compute_path in ("reuse", "incremental", "sparse") and (
            ffn_input_previous is None or ffn_output_previous is None
        ):
            compute_path = "full"

        # Step 3: 通道重要性分析与量化（仅非 full 路径需要）
        quantization_info = {}
        if compute_path in ("mixed_full", "sparse", "incremental"):
            if self.channel_importance is None:
                self.channel_importance = self.analyze_channel_importance(
                    ffn_input_current, weight_up, weight_down, method="weight_magnitude"
                )

            if self.weight_up_quantized is None:
                self.weight_up_quantized, self.weight_down_quantized, self.precision_scales = (
                    self.quantize_mixed_precision(weight_up, weight_down, self.channel_importance)
                )

            num_fp16 = sum(1 for p in self.precision_scales.values() if p[0] == "fp16")
            num_int8 = sum(1 for p in self.precision_scales.values() if p[0] == "int8")
            num_int4 = sum(1 for p in self.precision_scales.values() if p[0] == "int4")
            total_channels = max(len(self.precision_scales), 1)

            quantization_info = {
                "weight_precision_map": {
                    f"channel_{k}": v[0] for k, v in self.precision_scales.items()
                },
                "num_fp16_channels": num_fp16,
                "num_int8_channels": num_int8,
                "num_int4_channels": num_int4,
                "compression_ratio": 1.0
                - (num_fp16 * 2 + num_int8 * 1 + num_int4 * 0.5) / (total_channels * 2),
            }

            # Track real memory savings from byte-level packing.
            if isinstance(self.weight_up_quantized, PackedFFNWeight) and isinstance(
                self.weight_down_quantized, PackedFFNWeight
            ):
                packed_memory = (
                    self.weight_up_quantized.memory_bytes()
                    + self.weight_down_quantized.memory_bytes()
                )
                original_memory = (
                    self.weight_up_quantized.original_bytes()
                    + self.weight_down_quantized.original_bytes()
                )
                memory_saved = original_memory - packed_memory
                self._total_memory_bytes_original += original_memory
                self._total_memory_bytes_packed += packed_memory
                self._total_memory_bytes_saved += memory_saved
                quantization_info.update(
                    {
                        "packed_memory_bytes": packed_memory,
                        "original_memory_bytes": original_memory,
                        "memory_savings_bytes": memory_saved,
                        "compression_ratio": (
                            memory_saved / original_memory if original_memory else 0.0
                        ),
                    }
                )

        # Step 4: 执行 FFN 计算
        sparse_update_info = {}
        changed_ratio = 1.0

        # Use packed/quantized weights for optimized paths when available.
        up_weight = self.weight_up_quantized if self.weight_up_quantized is not None else weight_up
        down_weight = (
            self.weight_down_quantized if self.weight_down_quantized is not None else weight_down
        )

        if compute_path == "reuse":
            output = ffn_output_previous
            changed_ratio = 0.0

        elif compute_path == "incremental":
            delta_output = self.incremental_reconstruct(
                ffn_input_current, ffn_input_previous, up_weight, down_weight
            )
            output = ffn_output_previous + delta_output
            # 估算变化比例
            delta_norm = torch.norm(ffn_input_current - ffn_input_previous, dim=-1)
            changed_ratio = (delta_norm > 0.05).float().mean().item()

        elif compute_path == "sparse":
            output, sparse_stats = self.sparse_update(
                ffn_input_current,
                ffn_input_previous,
                ffn_output_previous,
                up_weight,
                down_weight,
            )
            sparse_update_info = sparse_stats
            changed_ratio = sparse_stats.get("changed_tokens_ratio", 1.0)

        elif compute_path == "mixed_full":
            output = self._mixed_compute(ffn_input_current, up_weight, down_weight, channel_mask)
            changed_ratio = 1.0

        else:  # full
            output = self._full_compute(ffn_input_current, weight_up, weight_down, channel_mask)
            changed_ratio = 1.0

        # Step 5: 计算统计
        batch_size, seq_len, hidden_dim = ffn_input_current.shape
        intermediate_dim = weight_up.shape[0] if weight_up is not None else 0

        full_flops = self._compute_flops("full", batch_size, seq_len, hidden_dim, intermediate_dim)
        actual_flops = self._compute_flops(
            compute_path,
            batch_size,
            seq_len,
            hidden_dim,
            intermediate_dim,
            changed_ratio,
            active_ratio,
        )
        flops_saved = full_flops - actual_flops

        compute_time_ms = (time.perf_counter() - start_time) * 1000.0

        self._total_flops += full_flops
        self._total_flops_saved += flops_saved
        self._sparse_ratios.append(changed_ratio)
        self._mixed_precision_ratios.append(1.0 if compute_path == "mixed_full" else 0.0)
        self._step_count += 1

        # Record per-step stats
        self.record_step_stat(
            step_id,
            {
                "latency_ms": compute_time_ms,
                "memory_mb": 0.0,
                "flops": float(actual_flops),
                "accuracy_delta": 0.0,
                "kv_compression_ratio": 0.0,
                "ffn_sparse_ratio": 1.0 - changed_ratio,
                "custom_metrics": {
                    "compute_path": compute_path,
                    "layer_id": layer_id,
                    "sensitivity_score": sensitivity_score,
                    "ffn_mode": ffn_mode,
                    "changed_ratio": changed_ratio,
                    "active_ratio": active_ratio,
                    "channel_pruning_enabled": self.channel_pruning_enabled,
                },
            },
        )

        self.logger.log_metric(
            "compute_path", compute_path, step_id=step_id, tags={"layer_id": str(layer_id)}
        )
        self.logger.log_metric(
            "flops_saved", float(flops_saved), step_id=step_id, tags={"layer_id": str(layer_id)}
        )

        memory_saved = quantization_info.get("memory_savings_bytes", 0.0)
        compute_stats = {
            "flops_saved": float(flops_saved),
            "flops_total": float(full_flops),
            "compute_time_ms": compute_time_ms,
            "memory_savings_bytes": float(memory_saved),
            "active_ratio": active_ratio,
            "channel_mask": channel_mask,
            "channel_pruning_enabled": self.channel_pruning_enabled,
        }

        channel_pruning_info = {
            "enabled": self.channel_pruning_enabled,
            "active_ratio": active_ratio,
            "active_channels": self._active_channels,
            "method": self.channel_pruning_method,
        }

        return {
            "compute_path": compute_path,
            "output": output,
            "quantization": quantization_info,
            "sparse_update": sparse_update_info,
            "compute_stats": compute_stats,
            "channel_pruning": channel_pruning_info,
        }

    def on_shutdown(self) -> Optional[ProfileResult]:
        """关闭：返回计算统计。

        Returns:
            可选的 ProfileResult 汇总对象，包含 FFN 计算负载统计。
        """
        self.transition_to(AgentState.SHUTDOWN)
        summary = self.get_performance_summary()

        # Build ffn_compute_load from step_stats
        ffn_compute_load: Dict[int, Dict[str, float]] = {}
        for sid, stats in self.step_stats.items():
            custom = stats.custom_metrics or {}
            ffn_compute_load[sid] = {
                "flops": stats.flops,
                "sparse_ratio": custom.get("changed_ratio", 0.0),
            }

        # Build latency_per_step_ms
        latency_per_step_ms: Dict[int, float] = {
            sid: stats.latency_ms for sid, stats in self.step_stats.items()
        }

        avg_sparse_ratio = sum(self._sparse_ratios) / max(len(self._sparse_ratios), 1)
        avg_mixed_ratio = sum(self._mixed_precision_ratios) / max(
            len(self._mixed_precision_ratios), 1
        )

        self.logger.log_event(
            "agent_shutdown",
            "FFNOptimizerAgent shutdown complete.",
            data={
                "performance_summary": summary,
                "avg_sparse_ratio": avg_sparse_ratio,
                "avg_mixed_precision_ratio": avg_mixed_ratio,
            },
        )

        return ProfileResult(
            model_name=self._model_name,
            run_id=self._run_id,
            ffn_compute_load=ffn_compute_load,
            ffn_sparse_update_ratio=avg_sparse_ratio,
            latency_per_step_ms=latency_per_step_ms,
            latency_e2e_ms=sum(latency_per_step_ms.values()),
            throughput_tokens_per_sec=0.0,
            perplexity_delta=0.0,
            bleu_drop=0.0,
            accuracy_metrics={},
            kv_precision_distribution={},
            kv_reuse_rate={},
            kv_memory_mb={},
            q_drift_hit_rate={},
            activation_delta={},
        )

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _precompute_weights(self, weight_up: torch.Tensor, weight_down: torch.Tensor) -> None:
        """缓存权重。

        假设传入的权重为 PyTorch 标准格式 [out_features, in_features]：
        - weight_up: [intermediate_dim, hidden_dim]
        - weight_down: [hidden_dim, intermediate_dim]
        """
        self._weight_up = weight_up
        self._weight_down = weight_down
        self.logger.log_event(
            "weights_cached",
            "FFN weights cached for optimization.",
            data={
                "weight_up_shape": list(weight_up.shape),
                "weight_down_shape": list(weight_down.shape),
            },
        )

    # ------------------------------------------------------------------
    # 核心算法 1: 通道重要性分析
    # ------------------------------------------------------------------

    def analyze_channel_importance(
        self,
        ffn_input: torch.Tensor,
        weight_up: torch.Tensor,
        weight_down: torch.Tensor,
        method: str = "activation_magnitude",
    ) -> torch.Tensor:
        """
        分析 FFN 中间层各通道的重要性。

        算法（对应 FFN_OPTIMIZER_AGENT.md §核心算法 1）：
        - activation_magnitude: 中间激活的绝对值均值
            intermediate = linear(ffn_input, weight_up)  # [batch, seq_len, intermediate_dim]
            importance = mean(abs(intermediate), dim=[0, 1])
        - weight_magnitude: up 和 down 权重联合幅度
            importance_up = mean(abs(weight_up), dim=1)
            importance_down = mean(abs(weight_down), dim=0)
            importance = importance_up * importance_down

        归一化：importance = importance / (importance.max() + 1e-8)
        返回: [intermediate_dim]

        Args:
            ffn_input: FFN 输入张量 [batch, seq_len, hidden_dim]。
            weight_up: 上投影权重 [intermediate_dim, hidden_dim]。
            weight_down: 下投影权重 [hidden_dim, intermediate_dim]。
            method: 重要性分析方法，"activation_magnitude" | "weight_magnitude"。

        Returns:
            通道重要性分数 [intermediate_dim]。
        """
        if method == "activation_magnitude":
            intermediate = F.linear(ffn_input, weight_up)  # [batch, seq_len, intermediate_dim]
            importance = intermediate.abs().mean(dim=[0, 1])  # [intermediate_dim]

        elif method == "weight_magnitude":
            # 使用 PyTorch 标准 [out, in] 权重格式
            importance_up = weight_up.abs().mean(dim=1)  # [intermediate_dim]
            importance_down = weight_down.abs().mean(dim=0)  # [intermediate_dim]
            importance = importance_up * importance_down

        else:
            # 默认回退到 weight_magnitude
            importance_up = weight_up.abs().mean(dim=1)
            importance_down = weight_down.abs().mean(dim=0)
            importance = importance_up * importance_down

        # 归一化到 [0, 1]
        importance = importance / (importance.max() + 1e-8)
        return importance

    # ------------------------------------------------------------------
    # 核心算法 2: 混合精度量化
    # ------------------------------------------------------------------

    def quantize_mixed_precision(
        self,
        weight_up: torch.Tensor,
        weight_down: torch.Tensor,
        channel_importance: torch.Tensor,
        high_precision_threshold: float = 0.7,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        对 FFN 权重进行混合精度量化。

        算法（对应 FFN_OPTIMIZER_AGENT.md §核心算法 2）：
        对每个 channel_idx:
        - importance >= 0.7: FP16（直接 half().float()）
        - 0.3 <= importance < 0.7: INT8（scale = max(abs) / 127, round, clamp）
        - importance < 0.3: INT4（scale = max(abs) / 7, round, clamp）

        返回: (quantized_up, quantized_down, scales_dict)

        Args:
            weight_up: 上投影权重 [intermediate_dim, hidden_dim]。
            weight_down: 下投影权重 [hidden_dim, intermediate_dim]。
            channel_importance: 通道重要性分数 [intermediate_dim]。
            high_precision_threshold: 高精度阈值。

        Returns:
            (quantized_up, quantized_down, precision_scales_dict)
        """
        intermediate_dim = weight_up.shape[0]  # PyTorch 标准: [out, in]

        quantized_up = torch.zeros_like(weight_up)
        quantized_down = torch.zeros_like(weight_down)
        scales = {}

        for channel_idx in range(intermediate_dim):
            importance = channel_importance[channel_idx].item()

            if importance >= high_precision_threshold:
                # 高重要性 -> FP16
                quantized_up[channel_idx, :] = weight_up[channel_idx, :].half().float()
                quantized_down[:, channel_idx] = weight_down[:, channel_idx].half().float()
                scales[channel_idx] = ("fp16", None)

            elif importance >= 0.3:
                # 中重要性 -> INT8
                w_up = weight_up[channel_idx, :]
                scale_up = w_up.abs().max() / 127.0
                if scale_up > 0:
                    q_up = torch.round(w_up / scale_up).clamp(-128, 127).to(torch.int8)
                    quantized_up[channel_idx, :] = q_up.float() * scale_up
                else:
                    quantized_up[channel_idx, :] = w_up

                w_down = weight_down[:, channel_idx]
                scale_down = w_down.abs().max() / 127.0
                if scale_down > 0:
                    q_down = torch.round(w_down / scale_down).clamp(-128, 127).to(torch.int8)
                    quantized_down[:, channel_idx] = q_down.float() * scale_down
                else:
                    quantized_down[:, channel_idx] = w_down

                scales[channel_idx] = ("int8", (float(scale_up.item()), float(scale_down.item())))

            else:
                # 低重要性 -> INT4
                w_up = weight_up[channel_idx, :]
                scale_up = w_up.abs().max() / 7.0
                if scale_up > 0:
                    q_up = torch.round(w_up / scale_up).clamp(-8, 7).to(torch.int8)
                    quantized_up[channel_idx, :] = q_up.float() * scale_up
                else:
                    quantized_up[channel_idx, :] = w_up

                w_down = weight_down[:, channel_idx]
                scale_down = w_down.abs().max() / 7.0
                if scale_down > 0:
                    q_down = torch.round(w_down / scale_down).clamp(-8, 7).to(torch.int8)
                    quantized_down[:, channel_idx] = q_down.float() * scale_down
                else:
                    quantized_down[:, channel_idx] = w_down

                scales[channel_idx] = ("int4", (float(scale_up.item()), float(scale_down.item())))

        if self.use_packed_weights:
            precision_map = {ch: info[0] for ch, info in scales.items()}
            packed_up = PackedFFNWeight.pack(
                weight_up,
                precision_map,
                pack_dim=0,
                group_size=self._pack_group_size,
            )
            packed_down = PackedFFNWeight.pack(
                weight_down,
                precision_map,
                pack_dim=1,
                group_size=self._pack_group_size,
            )
            return packed_up, packed_down, scales

        return quantized_up, quantized_down, scales

    # ------------------------------------------------------------------
    # 核心算法 3: 稀疏更新
    # ------------------------------------------------------------------

    def sparse_update(
        self,
        ffn_input_current: torch.Tensor,
        ffn_input_previous: torch.Tensor,
        ffn_output_previous: torch.Tensor,
        weight_up: torch.Tensor,
        weight_down: torch.Tensor,
        delta_threshold: float = 0.05,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        稀疏更新 FFN 输出。

        算法（对应 FFN_OPTIMIZER_AGENT.md §核心算法 3）：
        1. input_delta = abs(ffn_input_current - ffn_input_previous)
        2. relative_delta = mean(input_delta, dim=-1) / (mean(abs(current), dim=-1) + 1e-8)
        3. changed_mask = relative_delta > delta_threshold
        4. output = ffn_output_previous.clone()
        5. 如果 changed_mask.any():
           changed_input = ffn_input_current[changed_mask]
           intermediate = gelu(linear(changed_input, weight_up))
           changed_output = linear(intermediate, weight_down)
           output[changed_mask] = changed_output
        6. 返回 output 和 stats

        Args:
            ffn_input_current: 当前 FFN 输入 [batch, seq_len, hidden_dim]。
            ffn_input_previous: 上一 step FFN 输入 [batch, seq_len, hidden_dim]。
            ffn_output_previous: 上一 step FFN 输出 [batch, seq_len, hidden_dim]。
            weight_up: 上投影权重。
            weight_down: 下投影权重。
            delta_threshold: 变化阈值。

        Returns:
            (output, sparse_stats_dict)
        """
        # 1. 计算输入变化
        input_delta = torch.abs(ffn_input_current - ffn_input_previous)

        # 2. 计算相对变化
        input_magnitude = torch.abs(ffn_input_current).mean(dim=-1) + 1e-8
        relative_delta = input_delta.mean(dim=-1) / input_magnitude  # [batch, seq_len]

        # 3. 标记变化显著的 token
        changed_mask = relative_delta > delta_threshold  # [batch, seq_len]

        # 4. 构建输出（先复用旧输出）
        output = ffn_output_previous.clone()

        # 5. 仅对变化的 token 重新计算
        if changed_mask.any():
            changed_input = ffn_input_current[changed_mask]  # [num_changed, hidden_dim]

            up_output = self._linear_forward(
                changed_input, weight_up, use_sparse_gemm=True
            )  # [num_changed, intermediate_dim]
            intermediate = F.gelu(up_output)
            changed_output = self._linear_forward(
                intermediate, weight_down, use_sparse_gemm=True
            )  # [num_changed, hidden_dim]

            output[changed_mask] = changed_output

        # 6. 统计
        changed_tokens = int(changed_mask.sum().item())
        total_tokens = changed_mask.numel()
        unchanged_tokens = total_tokens - changed_tokens

        stats = {
            "changed_tokens_ratio": changed_tokens / max(total_tokens, 1),
            "changed_tokens": changed_tokens,
            "unchanged_tokens": unchanged_tokens,
        }

        return output, stats

    # ------------------------------------------------------------------
    # 核心算法 4: 增量重构
    # ------------------------------------------------------------------

    def incremental_reconstruct(
        self,
        ffn_input_current: torch.Tensor,
        ffn_input_previous: torch.Tensor,
        weight_up: torch.Tensor,
        weight_down: torch.Tensor,
        delta_threshold: float = 0.05,
    ) -> torch.Tensor:
        """
        增量重构 FFN 输出。

        算法（对应 FFN_OPTIMIZER_AGENT.md §核心算法 4）：
        1. input_delta = ffn_input_current - ffn_input_previous
        2. significant_delta = where(||delta|| > threshold, delta, 0)
        3. delta_intermediate = gelu(linear(significant_delta, weight_up))
        4. delta_output = linear(delta_intermediate, weight_down)
        5. 返回 delta_output（由调用方与旧输出相加）

        Args:
            ffn_input_current: 当前 FFN 输入 [batch, seq_len, hidden_dim]。
            ffn_input_previous: 上一 step FFN 输入 [batch, seq_len, hidden_dim]。
            weight_up: 上投影权重。
            weight_down: 下投影权重。
            delta_threshold: 变化阈值。

        Returns:
            delta_output 张量 [batch, seq_len, hidden_dim]。
        """
        # 1. 计算输入差值
        input_delta = ffn_input_current - ffn_input_previous

        # 2. 只保留显著差值
        delta_norm = torch.norm(input_delta, dim=-1, keepdim=True)  # [batch, seq_len, 1]
        significant_delta = torch.where(
            delta_norm > delta_threshold,
            input_delta,
            torch.zeros_like(input_delta),
        )

        # 3. 计算增量贡献
        up_output = self._linear_forward(
            significant_delta, weight_up, use_sparse_gemm=True
        )  # [batch, seq_len, intermediate_dim]
        delta_intermediate = F.gelu(up_output)
        delta_output = self._linear_forward(
            delta_intermediate, weight_down, use_sparse_gemm=True
        )  # [batch, seq_len, hidden_dim]

        return delta_output

    # ------------------------------------------------------------------
    # 核心算法 5: 计算路径选择
    # ------------------------------------------------------------------

    def select_compute_path(
        self,
        ffn_input_current: torch.Tensor,
        ffn_input_previous: Optional[torch.Tensor],
        ffn_output_previous: Optional[torch.Tensor],
        sensitivity_score: float,
        mode: str,
    ) -> str:
        """
        选择 FFN 计算路径。

        对应 FFN_OPTIMIZER_AGENT.md §核心算法 5：
        - mode == "full": 直接返回 "full"
        - mode == "mixed": 返回 "mixed_full"（量化但完整计算）
        - mode == "sparse":
          - 如果 ffn_input_previous 为 None: 返回 "full"（首次计算）
          - 计算 relative_delta = mean(||current - previous||) / mean(||current||)
          - 如果 relative_delta < 0.02: "reuse"（直接复用旧输出）
          - elif relative_delta < 0.05: "incremental"（增量重构）
          - elif relative_delta < 0.15: "sparse"（稀疏更新）
          - else: "full"（完整计算）

        Args:
            ffn_input_current: 当前 FFN 输入。
            ffn_input_previous: 上一 step FFN 输入（可能为 None）。
            ffn_output_previous: 上一 step FFN 输出（可能为 None）。
            sensitivity_score: 敏感度分数。
            mode: FFN 模式，"full" | "mixed" | "sparse"。

        Returns:
            计算路径字符串。
        """
        if mode == "full":
            return "full"

        if mode == "mixed":
            return "mixed_full"

        if mode == "sparse":
            if ffn_input_previous is None or ffn_output_previous is None:
                return "full"

            # 计算相对变化
            delta_norm = torch.norm(
                ffn_input_current - ffn_input_previous, dim=-1
            )  # [batch, seq_len]
            current_norm = torch.norm(ffn_input_current, dim=-1) + 1e-8  # [batch, seq_len]
            relative_delta = (delta_norm / current_norm).mean().item()

            if relative_delta < 0.02:
                return "reuse"
            elif relative_delta < 0.05:
                return "incremental"
            elif relative_delta < 0.15:
                return "sparse"
            else:
                return "full"

        return "full"

    # ------------------------------------------------------------------
    # 辅助计算方法
    # ------------------------------------------------------------------

    def compute_channel_mask(
        self,
        ffn_input: torch.Tensor,
        weight_up: torch.Tensor,
        method: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute a boolean mask of active intermediate channels.

        Args:
            ffn_input: FFN input [batch, seq_len, hidden_dim].
            weight_up: Up-projection weight [intermediate_dim, hidden_dim].
            method: ``static_importance`` uses the cached channel_importance
                scores; ``dynamic_activation`` uses the current input's up-proj
                activation energy.

        Returns:
            Bool tensor [intermediate_dim], ``True`` for kept channels.
        """
        method = method or self.channel_pruning_method
        intermediate_dim = weight_up.shape[0]
        device = weight_up.device

        if method == "dynamic_activation":
            with torch.no_grad():
                pre_act = F.linear(ffn_input, weight_up)  # [B, S, I]
                channel_energy = pre_act.abs().mean(dim=[0, 1])  # [I]
            ratio = max(0.0, min(1.0, self.channel_pruning_ratio))
            if ratio == 0.0:
                threshold = -1.0
            else:
                threshold = torch.quantile(channel_energy, ratio).item()
            return channel_energy > threshold

        # Static importance: keep top-(1 - ratio) most important channels.
        if self.channel_importance is None:
            raise RuntimeError(
                "Static channel pruning requires channel_importance; "
                "run analyze_channel_importance first or use dynamic_activation."
            )
        k = max(1, int(intermediate_dim * (1.0 - self.channel_pruning_ratio)))
        keep_indices = torch.topk(self.channel_importance.to(device), k, largest=True).indices
        mask = torch.zeros(intermediate_dim, dtype=torch.bool, device=device)
        mask[keep_indices] = True
        return mask

    def _full_compute(
        self,
        ffn_input: torch.Tensor,
        weight_up: torch.Tensor,
        weight_down: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """标准 FFN 计算：gelu(linear(x, weight_up)) @ weight_down

        使用 PyTorch 标准 F.linear 实现。

        Args:
            ffn_input: 输入张量 [..., hidden_dim]。
            weight_up: 上投影权重 [intermediate_dim, hidden_dim]。
            weight_down: 下投影权重 [hidden_dim, intermediate_dim]。
            channel_mask: 可选的动态通道剪枝掩码 [intermediate_dim]。

        Returns:
            FFN 输出张量 [..., hidden_dim]。
        """
        if channel_mask is not None:
            weight_up = weight_up[channel_mask, :]
            weight_down = weight_down[:, channel_mask]
        intermediate = F.gelu(F.linear(ffn_input, weight_up))  # [..., active_dim]
        output = F.linear(intermediate, weight_down)  # [..., hidden_dim]
        return output

    def _mixed_compute(
        self,
        ffn_input: torch.Tensor,
        weight_up: torch.Tensor,
        weight_down: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """使用量化权重但完整计算。

        Args:
            ffn_input: 输入张量 [..., hidden_dim]。
            weight_up: 量化后的上投影权重。
            weight_down: 量化后的下投影权重。
            channel_mask: 可选的动态通道剪枝掩码。

        Returns:
            FFN 输出张量 [..., hidden_dim]。
        """
        up_output = self._linear_forward(
            ffn_input, weight_up, use_sparse_gemm=True, channel_mask=channel_mask
        )
        intermediate = F.gelu(up_output)
        return self._linear_forward(
            intermediate, weight_down, use_sparse_gemm=True, channel_mask=None
        )

    def _linear_forward(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        use_sparse_gemm: bool = False,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Linear forward with optional sparse GEMM kernel and packed-weight fallback.

        If ``weight`` is a ``PackedFFNWeight`` it is dequantized on-the-fly.
        When CUDA is available and the sparse GEMM kernel is loaded, the
        existing ``sparse_gemm_ffn`` path is used for ``sparse`` / ``mixed_full``
        execution paths; otherwise a standard dense ``F.linear`` fallback is used.

        Args:
            channel_mask: Optional bool mask [out_features] applied after
                dequantization. Used for dynamic channel pruning.
        """
        is_packed = isinstance(weight, PackedFFNWeight)
        device = input_tensor.device

        dense_weight = weight.dequantize_for_matmul() if is_packed else weight
        if dense_weight.device != device:
            dense_weight = dense_weight.to(device)

        if channel_mask is not None:
            dense_weight = dense_weight[channel_mask]

        cuda_ready = (
            use_sparse_gemm
            and input_tensor.is_cuda
            and get_kernel_status()["sparse_gemm_kernel"] == "cuda"
        )

        if cuda_ready:
            changed_mask = torch.ones(dense_weight.shape[0], dtype=torch.bool, device=device)
            return sparse_gemm_ffn(input_tensor, dense_weight, changed_mask)

        return F.linear(input_tensor, dense_weight)

    def _compute_flops(
        self,
        path: str,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        intermediate_dim: int,
        changed_ratio: float = 1.0,
        active_ratio: float = 1.0,
    ) -> int:
        """计算当前路径的 FLOPs。

        Args:
            path: 计算路径。
            batch_size: 批次大小。
            seq_len: 序列长度。
            hidden_dim: 隐藏维度。
            intermediate_dim: 中间维度。
            changed_ratio: 变化 token 比例（仅 sparse/incremental 路径使用）。
            active_ratio: 活跃通道比例（仅 channel pruning 使用）。

        Returns:
            FLOPs 数量。
        """
        up_macs = Metrics.compute_flops_macs(hidden_dim, intermediate_dim, batch_size, seq_len)
        down_macs = Metrics.compute_flops_macs(intermediate_dim, hidden_dim, batch_size, seq_len)
        full_flops = (up_macs + down_macs) * 2

        if path in ("full", "mixed_full"):
            return int(full_flops * active_ratio)
        elif path == "reuse":
            return 0
        elif path in ("incremental", "sparse"):
            return int(full_flops * changed_ratio * active_ratio)
        else:
            return int(full_flops * active_ratio)

    def get_compute_stats(self) -> Dict[str, float]:
        """获取计算统计：flops_saved, flops_total, sparse_update_ratio, mixed_precision_ratio。

        Returns:
            计算统计字典。
        """
        avg_sparse = sum(self._sparse_ratios) / max(len(self._sparse_ratios), 1)
        avg_mixed = sum(self._mixed_precision_ratios) / max(len(self._mixed_precision_ratios), 1)

        return {
            "flops_saved": self._total_flops_saved,
            "flops_total": self._total_flops,
            "sparse_update_ratio": avg_sparse,
            "mixed_precision_ratio": avg_mixed,
            "memory_savings_bytes": self._total_memory_bytes_saved,
            "memory_original_bytes": self._total_memory_bytes_original,
            "memory_packed_bytes": self._total_memory_bytes_packed,
        }
