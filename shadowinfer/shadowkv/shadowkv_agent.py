"""ShadowKV Agent — KV Cache 压缩与复用专家。

对应文档：SHADOWKV_AGENT.md, TECHNICAL_SPEC.md §2.1 / §3.2
版本：v3.0
"""

from __future__ import annotations

__version__ = "3.0"

import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from shadowinfer.core.base_agent import BaseAgent
from shadowinfer.core.structs import AgentState, KVCacheEntry, ProfileResult, StepConfig
from shadowinfer.shadowkv.decision_plane import KVDecisionPlane
from shadowinfer.shadowkv.eviction_policy import EvictionPolicy, LeastImportantEvictionPolicy
from shadowinfer.shadowkv.importance_model import ImportanceModel
from shadowinfer.utils.logging_utils import StructuredLogger
from shadowinfer.utils.metrics import Metrics
from shadowinfer.utils.quantization import Quantizer

from .kv_cache_manager import KVCacheManager
from .packed_kv_cache import PackedKVCache


class ShadowKVAgent(BaseAgent):
    """ShadowKV Agent — KV Cache 压缩与复用专家。

    对应文档：SHADOWKV_AGENT.md, TECHNICAL_SPEC.md §2.1 / §3.2
    版本：v3.0
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, name="shadowkv")
        self.kv_cache: Dict[int, Dict[str, Any]] = (
            {}
        )  # layer_id -> {k: Tensor, v: Tensor, precision: str}
        self.precision_map: Dict[int, Dict[int, Dict[int, str]]] = (
            {}
        )  # layer_id -> token -> head -> precision
        self.importance_history: Dict[int, List[torch.Tensor]] = (
            {}
        )  # layer_id -> [importance_map, ...]
        self.logger = StructuredLogger("shadowkv", config.get("log_dir", "logs/"))

        # New abstractions (lazy-initialized in on_init).
        self.importance_model: Optional[ImportanceModel] = None
        self.decision_plane: Optional[KVDecisionPlane] = None

        self.num_layers = config.get("num_layers", 32)
        self.num_heads = config.get("num_heads", 32)
        self.head_dim = config.get("head_dim", 128)
        self.model_name = config.get("model_name", "unknown")
        self.run_id = config.get("run_id", "default")
        self.use_packed_cache = config.get("use_packed_cache", True)
        self.use_decision_plane = config.get("use_decision_plane", False)
        self.prefetch_enabled = config.get("prefetch_enabled", False)

        # Memory budget / eviction.
        memory_budget_mb = config.get("memory_budget_mb")
        memory_budget_bytes = (
            int(memory_budget_mb * 1024 * 1024) if memory_budget_mb is not None else None
        )
        eviction_policy_cfg = config.get("eviction_policy")
        eviction_policy: Optional[EvictionPolicy] = None
        if eviction_policy_cfg == "age":
            from shadowinfer.shadowkv.eviction_policy import ImportanceAgeEvictionPolicy

            eviction_policy = ImportanceAgeEvictionPolicy()
        elif eviction_policy_cfg is not None:
            eviction_policy = LeastImportantEvictionPolicy()

        self.cache_manager = KVCacheManager(
            self.num_layers,
            memory_budget_bytes=memory_budget_bytes,
            eviction_policy=eviction_policy,
        )
        self._compression_stats = {
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "step_count": 0,
        }
        self._precision_distribution: Dict[str, int] = {
            "fp32": 0,
            "fp16": 0,
            "int8": 0,
            "int4": 0,
        }
        self._reuse_stats = {
            "full_reuse_count": 0,
            "partial_reuse_count": 0,
            "no_reuse_count": 0,
        }

    def on_init(self, model_config: Dict[str, Any]) -> None:
        """初始化：加载配置，设置精度阈值和复用参数。"""
        self.num_layers = model_config.get("num_layers", self.num_layers)
        self.num_heads = model_config.get("num_heads", self.num_heads)
        self.head_dim = model_config.get("head_dim", self.head_dim)
        self.model_name = model_config.get("model_name", self.model_name)
        self.cache_manager = KVCacheManager(self.num_layers)

        # 加载优化配置
        self.compression_target = self.config.get("compression_target", 0.5)
        self.importance_thresholds = self.config.get(
            "importance_thresholds",
            {"fp32": 0.8, "fp16": 0.5, "int8": 0.2, "int4": 0.0},
        )
        self.reuse_base_threshold = self.config.get("reuse_base_threshold", 0.15)
        self.reuse_adaptive = self.config.get("reuse_adaptive", True)

        # Initialize new abstractions.
        self.importance_model = ImportanceModel(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        self.decision_plane = KVDecisionPlane(
            importance_thresholds=dict(self.importance_thresholds),
            reuse_base_threshold=self.reuse_base_threshold,
            memory_budget_bytes=self.cache_manager.memory_budget_bytes,
        )

        self.transition_to(AgentState.READY)
        self.logger.log_event(
            "agent_init",
            f"ShadowKVAgent initialized for {self.model_name} "
            f"({self.num_layers} layers, {self.num_heads} heads)",
            data={"model_config": model_config},
        )

    def on_step(self, step_config: StepConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个 step。

        输入 inputs 包含：
        - 'attention_scores': Tensor[batch, num_heads, seq_len, seq_len]
        - 'kv_current': Dict with 'k' and 'v' tensors
        - 'kv_previous': Dict with 'k' and 'v' tensors (可能为 None)
        - 'layer_id': int
        - 'qdrift_signal': Dict with 'sensitivity_score', 'drift_score', 'shadowkv_mode'

        返回：{'compressed_kv': ..., 'precision_map': ..., 'reuse_decision': ...,
                 'importance_stats': ...}
        """
        start_time = time.perf_counter()
        step_id = step_config.step_id
        total_steps = step_config.total_steps
        layer_id = inputs.get("layer_id", 0)

        # 提取输入
        attention_scores = inputs["attention_scores"]
        kv_current = inputs["kv_current"]
        kv_previous = inputs.get("kv_previous", None)
        qdrift_signal = inputs.get("qdrift_signal", {})

        sensitivity_score = qdrift_signal.get("sensitivity_score", 0.0)
        drift_score = qdrift_signal.get("drift_score", 0.0)
        mode = qdrift_signal.get("shadowkv_mode", step_config.shadowkv_mode)

        # Step 1: 评估 Q-drift 信号，必要时强制切换模式
        if sensitivity_score > 0.7:
            mode = "conservative"
        elif drift_score > 0.5:
            # 增加压缩率但保留 top 20% head 为 FP16
            mode = "balanced"

        # Step 2: 计算重要性分数
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        importance_map = torch.zeros((seq_len, num_heads), device=attention_scores.device)

        for token_idx in range(seq_len):
            for head_idx in range(num_heads):
                importance_map[token_idx, head_idx] = self.compute_importance_score(
                    attention_scores, token_idx, head_idx, layer_id, self.num_layers
                )

        # 记录历史
        if layer_id not in self.importance_history:
            self.importance_history[layer_id] = []
        self.importance_history[layer_id].append(importance_map.detach().cpu())

        # Step 3: 分配精度
        precision_map = self.allocate_precision(importance_map, mode)

        # 如果 drift_score > 0.5，保留每 token top 20% head 为 FP16
        if drift_score > 0.5:
            num_fp16_heads = max(1, int(num_heads * 0.2))
            for token_idx in range(seq_len):
                top_heads = torch.topk(importance_map[token_idx], num_fp16_heads).indices.tolist()
                for head_idx in top_heads:
                    precision_map[token_idx][head_idx] = "fp16"

        # Step 4: 复用决策
        reuse_decision = self._build_reuse_decision(
            kv_current,
            kv_previous,
            step_id,
            total_steps,
            drift_score,
            mode,
            importance_map,
        )

        # Step 5: 压缩 KV
        compressed_kv = self.compress_kv(
            kv_current,
            precision_map,
            reuse_decision=reuse_decision,
            reuse_mask=reuse_decision.get("reuse_mask"),
            kv_previous=kv_previous,
        )

        # 若启用 packed cache 且有复用决策，将复用 token-head 从上一个 packed cache 复制过来
        if self.use_packed_cache and compressed_kv.get("packed_kv") is not None:
            prev_packed = self.kv_cache.get(layer_id, {}).get("packed_kv")
            reuse_mask = reuse_decision.get("reuse_mask")
            if prev_packed is not None and reuse_mask is not None:
                compressed_kv["packed_kv"].apply_reuse_mask(prev_packed, reuse_mask)

        # Step 6: 更新缓存
        self._store_kv(
            layer_id,
            compressed_kv["k"],
            compressed_kv["v"],
            precision_map,
            reuse_decision,
            packed_kv=compressed_kv.get("packed_kv"),
        )
        self.precision_map[layer_id] = precision_map

        # 记录 step 统计
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        compression_ratio = compressed_kv.get("compression_ratio", 0.0)
        memory_mb = compressed_kv.get("memory_mb", 0.0)

        self.record_step_stat(
            step_id,
            {
                "latency_ms": latency_ms,
                "memory_mb": memory_mb,
                "flops": 0.0,  # 压缩主要是访存开销
                "accuracy_delta": 0.0,
                "kv_compression_ratio": compression_ratio,
                "ffn_sparse_ratio": 0.0,
                "custom_metrics": {
                    "layer_id": layer_id,
                    "mode": mode,
                    "drift_score": drift_score,
                    "sensitivity_score": sensitivity_score,
                    "reuse_strategy": reuse_decision["strategy"],
                },
            },
        )

        # 构建重要性统计
        importance_stats = {
            "mean": float(importance_map.mean()),
            "std": float(importance_map.std()),
            "max": float(importance_map.max()),
            "min": float(importance_map.min()),
        }

        # 更新 precision distribution
        for token_map in precision_map.values():
            for prec in token_map.values():
                self._precision_distribution[prec] = self._precision_distribution.get(prec, 0) + 1

        # 更新 reuse stats
        if reuse_decision["strategy"] == "full_reuse":
            self._reuse_stats["full_reuse_count"] += 1
        elif reuse_decision["strategy"] == "partial_reuse":
            self._reuse_stats["partial_reuse_count"] += 1
        else:
            self._reuse_stats["no_reuse_count"] += 1

        return {
            "compressed_kv": compressed_kv,
            "precision_map": precision_map,
            "reuse_decision": reuse_decision,
            "importance_stats": importance_stats,
        }

    def on_shutdown(self) -> Optional[ProfileResult]:
        """关闭：返回汇总统计。"""
        self.transition_to(AgentState.SHUTDOWN)
        summary = self.get_performance_summary()

        # 构建 ProfileResult
        kv_precision_distribution = {}
        for layer_id, pmap in self.precision_map.items():
            kv_precision_distribution[layer_id] = {}
            for token_idx, head_map in pmap.items():
                for head_idx, prec in head_map.items():
                    if head_idx not in kv_precision_distribution[layer_id]:
                        kv_precision_distribution[layer_id][head_idx] = prec

        # 复用率：逐层计算
        kv_reuse_rate = self.cache_manager.get_reuse_stats()
        layer_reuse_rates = {}
        for k, v in kv_reuse_rate.items():
            if k.startswith("layer_") and k.endswith("_reuse_rate"):
                try:
                    layer_id = int(k.replace("layer_", "").replace("_reuse_rate", ""))
                    layer_reuse_rates[layer_id] = v
                except ValueError:
                    continue

        # 逐层内存
        kv_memory_mb = {}
        for layer_id in range(self.num_layers):
            entry = self.cache_manager.retrieve(layer_id)
            if entry is None:
                kv_memory_mb[layer_id] = 0.0
            elif entry.packed_kv is not None:
                kv_memory_mb[layer_id] = entry.packed_kv.memory_mb()
            else:
                kv_memory_mb[layer_id] = (
                    self._tensor_bytes(entry.k_tensor) + self._tensor_bytes(entry.v_tensor)
                ) / (1024.0 * 1024.0)

        profile_result = ProfileResult(
            model_name=self.model_name,
            run_id=self.run_id,
            kv_precision_distribution=kv_precision_distribution,
            kv_reuse_rate=layer_reuse_rates,
            kv_memory_mb=kv_memory_mb,
            q_drift_hit_rate={},
            activation_delta={},
            ffn_compute_load={},
            ffn_sparse_update_ratio=0.0,
            latency_e2e_ms=summary.get("total_latency_ms", 0.0),
            latency_per_step_ms={s.step_id: s.latency_ms for s in self.step_stats.values()},
            throughput_tokens_per_sec=0.0,
            perplexity_delta=0.0,
            bleu_drop=0.0,
            accuracy_metrics={},
        )

        self.logger.log_event(
            "agent_shutdown",
            "ShadowKVAgent shutdown complete",
            data={
                "performance_summary": summary,
                "compression_stats": self.get_compression_stats(),
                "reuse_stats": self._reuse_stats,
                "precision_distribution": self._precision_distribution,
            },
        )

        return profile_result

    # === 核心算法 ===

    def compute_importance_score(
        self,
        attention_scores: torch.Tensor,
        token_index: int,
        head_index: int,
        layer_index: int,
        num_layers: int,
    ) -> float:
        """
        计算 per-token-head 重要性分数。

        算法（对应 SHADOWKV_AGENT.md §核心算法 1）：
        1. 取该 token 在当前 head 上的 attention 权重分布
        2. 计算 entropy（高 entropy → 关注范围广 → 高重要性）
        3. 层深度因子（浅层关注局部，深层关注语义）
        4. 位置因子（开头和结尾的 token 通常更重要）
        5. 综合打分 ∈ [0, 1]
        """
        # 1. 提取该 token 的 attention 权重
        attn_weights = attention_scores[:, head_index, token_index, :]  # [batch, seq_len]
        weights = F.softmax(attn_weights, dim=-1)

        # 2. 计算 entropy
        entropy = Metrics.compute_entropy(weights)
        normalized_entropy = entropy / math.log(weights.shape[-1])

        # 3. 层深度因子
        layer_factor = 1.0 + 0.1 * (layer_index / max(num_layers, 1))

        # 4. 位置因子
        seq_len = weights.shape[-1]
        pos_factor = 1.0
        if token_index < 5 or token_index > seq_len - 5:
            pos_factor = 1.2

        score = normalized_entropy * layer_factor * pos_factor
        return float(min(score, 1.0))

    def allocate_precision(
        self, importance_map: torch.Tensor, mode: str
    ) -> Dict[int, Dict[int, str]]:
        """
        分配精度。

        对应 SHADOWKV_AGENT.md 精度映射策略：
        - ≥ 0.8: FP32
        - 0.5 - 0.8: FP16
        - 0.2 - 0.5: INT8
        - < 0.2: INT4

        参数：
        - importance_map: Tensor[seq_len, num_heads] 或 Dict
        - mode: "aggressive" | "balanced" | "conservative"

        aggressive 模式下阈值降低 0.1（更激进），conservative 模式下提高 0.1。
        """
        mode_adjustment = {
            "aggressive": -0.1,
            "balanced": 0.0,
            "conservative": 0.1,
        }.get(mode, 0.0)

        thresholds = {
            "fp32": max(0.0, 0.8 + mode_adjustment),
            "fp16": max(0.0, 0.5 + mode_adjustment),
            "int8": max(0.0, 0.2 + mode_adjustment),
            "int4": 0.0,
        }

        if isinstance(importance_map, dict):
            seq_len = max(importance_map.keys()) + 1 if importance_map else 0
            num_heads = max(len(v) for v in importance_map.values()) if importance_map else 0
        else:
            seq_len = importance_map.shape[0]
            num_heads = importance_map.shape[1]

        precision_map: Dict[int, Dict[int, str]] = {}

        for token_idx in range(seq_len):
            precision_map[token_idx] = {}
            for head_idx in range(num_heads):
                if isinstance(importance_map, dict):
                    score = importance_map.get(token_idx, {}).get(head_idx, 0.0)
                else:
                    score = float(importance_map[token_idx, head_idx].item())

                if score >= thresholds["fp32"]:
                    precision_map[token_idx][head_idx] = "fp32"
                elif score >= thresholds["fp16"]:
                    precision_map[token_idx][head_idx] = "fp16"
                elif score >= thresholds["int8"]:
                    precision_map[token_idx][head_idx] = "int8"
                else:
                    precision_map[token_idx][head_idx] = "int4"

        return precision_map

    def decide_reuse(
        self,
        kv_current_k: torch.Tensor,
        kv_current_v: torch.Tensor,
        kv_previous_k: torch.Tensor,
        kv_previous_v: torch.Tensor,
        step_id: int,
        total_steps: int,
        qdrift_drift_score: float,
        mode: str,
    ) -> Tuple[bool, str, List[int]]:
        """
        判断 KV 是否可复用。

        算法（对应 SHADOWKV_AGENT.md §核心算法 3）：
        1. 计算 L2 变化幅度: delta_k = ||k_current - k_previous|| / ||k_current||
        2. 阈值动态调整：
           - aggressive: base=0.20, balanced: base=0.15, conservative: base=0.10
           - 后期更保守：adaptive_threshold = base * (1.0 - 0.5 * progress)
        3. 结合 Q-drift：effective_threshold = adaptive * (1.0 - drift_score * 0.3)
        4. 如果 delta < threshold * 0.5: full_reuse
           如果 delta < threshold: partial_reuse（选择变化最小的 70% head）
           否则: no_reuse
        """
        # 1. 计算 L2 变化幅度
        delta_k = torch.norm(kv_current_k - kv_previous_k) / (torch.norm(kv_current_k) + 1e-8)
        delta_v = torch.norm(kv_current_v - kv_previous_v) / (torch.norm(kv_current_v) + 1e-8)
        delta = (delta_k + delta_v) / 2.0

        # 2. 阈值动态调整
        progress = step_id / max(total_steps, 1)
        base_threshold = {
            "aggressive": 0.20,
            "balanced": 0.15,
            "conservative": 0.10,
        }.get(mode, 0.15)

        adaptive_threshold = base_threshold * (1.0 - 0.5 * progress)

        # 3. 结合 Q-drift
        effective_threshold = adaptive_threshold * (1.0 - qdrift_drift_score * 0.3)

        # 4. 复用决策
        if delta < effective_threshold:
            if delta < effective_threshold * 0.5:
                num_heads = kv_current_k.shape[1]
                return True, "full_reuse", list(range(num_heads))
            else:
                # 部分复用：选择变化最小的 70% head
                head_deltas = self._compute_head_deltas(
                    kv_current_k, kv_current_v, kv_previous_k, kv_previous_v
                )
                sorted_heads = sorted(range(len(head_deltas)), key=lambda x: head_deltas[x])
                num_reuse = max(1, int(len(sorted_heads) * 0.7))
                reused_heads = sorted_heads[:num_reuse]
                return True, "partial_reuse", reused_heads
        else:
            return False, "no_reuse", []

    def decide_reuse_per_token(
        self,
        kv_current_k: torch.Tensor,
        kv_current_v: torch.Tensor,
        kv_previous_k: torch.Tensor,
        kv_previous_v: torch.Tensor,
        step_id: int,
        total_steps: int,
        qdrift_drift_score: float,
        mode: str,
        importance_map: torch.Tensor,
    ) -> Tuple[Dict[Tuple[int, int], bool], Dict[str, float]]:
        """
        逐 (token, head) 判断 KV 是否可复用。

        算法：
        1. 计算每个 token/head 的相对 L2 漂移：
           delta = (||k_cur - k_prev|| + ||v_cur - v_prev||)
                   / (||k_cur|| + ||v_cur|| + eps)
        2. 动态阈值：
           - aggressive: 0.20, balanced: 0.15, conservative: 0.10
           - 后期更保守：base * (1.0 - 0.5 * progress)
           - 结合 Q-drift：* (1.0 - drift_score * 0.3)
           - 结合重要性：重要性越高越严格，* (1.0 - 0.3 * importance)
        3. delta < threshold 的 token-head 判定为复用。

        返回：
            reuse_mask: {(token_idx, head_idx): True/False}
            stats: {
                "full_reuse_count": 完全复用的 token 数,
                "partial_reuse_count": 部分复用的 token 数,
                "no_reuse_count": 无复用的 token 数,
                "reuse_ratio": 复用 token-head 比例,
            }
        """
        batch_size, num_heads, seq_len, head_dim = kv_current_k.shape

        # 1. 计算每个 head/token 的相对 L2 漂移
        diff_k = kv_current_k - kv_previous_k
        diff_v = kv_current_v - kv_previous_v
        norm_diff_k = torch.norm(diff_k, dim=(0, 3))
        norm_diff_v = torch.norm(diff_v, dim=(0, 3))
        norm_cur_k = torch.norm(kv_current_k, dim=(0, 3)) + 1e-8
        norm_cur_v = torch.norm(kv_current_v, dim=(0, 3)) + 1e-8

        delta_k = norm_diff_k / norm_cur_k
        delta_v = norm_diff_v / norm_cur_v
        delta_map = ((delta_k + delta_v) / 2.0).detach().cpu()  # [num_heads, seq_len]

        # 2. 动态阈值
        progress = step_id / max(total_steps, 1)
        base_threshold = {
            "aggressive": 0.20,
            "balanced": 0.15,
            "conservative": 0.10,
        }.get(mode, 0.15)

        adaptive_threshold = base_threshold * (1.0 - 0.5 * progress)
        effective_threshold = adaptive_threshold * (1.0 - qdrift_drift_score * 0.3)

        reuse_mask: Dict[Tuple[int, int], bool] = {}
        full_reuse_count = 0
        partial_reuse_count = 0
        no_reuse_count = 0
        reused_entries = 0
        total_entries = seq_len * num_heads

        importance_map_cpu = importance_map.detach().cpu()

        for token_idx in range(seq_len):
            reused_in_token = 0
            for head_idx in range(num_heads):
                delta = float(delta_map[head_idx, token_idx])
                importance = float(importance_map_cpu[token_idx, head_idx].clamp(0.0, 1.0))
                threshold = effective_threshold * (1.0 - 0.3 * importance)
                threshold = max(threshold, 1e-5)

                should_reuse = delta < threshold
                reuse_mask[(token_idx, head_idx)] = should_reuse
                if should_reuse:
                    reused_in_token += 1
                    reused_entries += 1

            if reused_in_token == num_heads:
                full_reuse_count += 1
            elif reused_in_token == 0:
                no_reuse_count += 1
            else:
                partial_reuse_count += 1

        reuse_ratio = reused_entries / max(total_entries, 1)

        stats = {
            "full_reuse_count": float(full_reuse_count),
            "partial_reuse_count": float(partial_reuse_count),
            "no_reuse_count": float(no_reuse_count),
            "reuse_ratio": float(reuse_ratio),
        }

        return reuse_mask, stats

    def _build_reuse_decision(
        self,
        kv_current: Dict[str, torch.Tensor],
        kv_previous: Optional[Dict[str, torch.Tensor]],
        step_id: int,
        total_steps: int,
        drift_score: float,
        mode: str,
        importance_map: torch.Tensor,
    ) -> Dict[str, Any]:
        """构建包含 layer-level 与 token-head-level 的复用决策。"""
        k_tensor = kv_current["k"]
        num_heads = k_tensor.shape[1]
        seq_len = k_tensor.shape[2]

        if kv_previous is None:
            reuse_mask = {(t, h): False for t in range(seq_len) for h in range(num_heads)}
            stats = {
                "full_reuse_count": 0.0,
                "partial_reuse_count": 0.0,
                "no_reuse_count": float(seq_len),
                "reuse_ratio": 0.0,
            }
        else:
            reuse_mask, stats = self.decide_reuse_per_token(
                kv_current["k"],
                kv_current["v"],
                kv_previous["k"],
                kv_previous["v"],
                step_id,
                total_steps,
                drift_score,
                mode,
                importance_map,
            )

        reuse_ratio = stats["reuse_ratio"]
        if reuse_ratio >= 1.0 - 1e-9:
            strategy = "full_reuse"
        elif reuse_ratio > 0.0:
            strategy = "partial_reuse"
        else:
            strategy = "no_reuse"

        should_reuse = strategy != "no_reuse"

        # layer-level reused_heads：在所有 token 上都被复用的 head
        reused_heads = [
            h for h in range(num_heads) if all(reuse_mask[(t, h)] for t in range(seq_len))
        ]
        updated_heads = [h for h in range(num_heads) if h not in reused_heads]

        return {
            "should_reuse": should_reuse,
            "strategy": strategy,
            "reused_heads": reused_heads,
            "updated_heads": updated_heads,
            "reuse_mask": reuse_mask,
            "reuse_stats": stats,
        }

    def compress_kv(
        self,
        kv_current: Dict[str, torch.Tensor],
        precision_map: Dict[int, Dict[int, str]],
        reuse_decision: Optional[Dict[str, Any]] = None,
        reuse_mask: Optional[Dict[Tuple[int, int], bool]] = None,
        kv_previous: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        压缩 KV cache。

        对每个 token/head 按 precision_map 进行量化：
        - FP32: 保持原样
        - FP16: half()
        - INT8: Quantizer.quantize_int8()
        - INT4: Quantizer.quantize_int4()

        复用决策支持两种格式（保持向后兼容）：
        1. 旧格式：reuse_decision 包含全局 `reused_heads` 列表，对所有 token 生效。
        2. 新格式：reuse_decision 包含 `reuse_mask`，或显式传入 `reuse_mask`，
           键为 (token_idx, head_idx)，值为是否复用。
        """
        k_tensor = kv_current["k"]
        v_tensor = kv_current["v"]
        batch_size, num_heads, seq_len, head_dim = k_tensor.shape

        # 初始化输出
        k_out = torch.zeros_like(k_tensor)
        v_out = torch.zeros_like(v_tensor)

        scale_k_map: Dict[Tuple[int, int], Optional[torch.Tensor]] = {}
        scale_v_map: Dict[Tuple[int, int], Optional[torch.Tensor]] = {}

        # 解析复用掩码（新格式优先）
        effective_reuse_mask: Optional[Dict[Tuple[int, int], bool]] = None
        if reuse_mask is not None:
            effective_reuse_mask = reuse_mask
        elif reuse_decision is not None and "reuse_mask" in reuse_decision:
            effective_reuse_mask = reuse_decision["reuse_mask"]

        # 旧格式回退：全局 reused_heads
        reused_heads: Set[int] = set()
        if effective_reuse_mask is None and reuse_decision is not None:
            if reuse_decision.get("should_reuse", False):
                reused_heads = set(reuse_decision.get("reused_heads", []))

        # 逐 token/head 处理
        for token_idx in range(seq_len):
            token_precision = precision_map.get(token_idx, {})
            for head_idx in range(num_heads):
                precision = token_precision.get(head_idx, "fp16")

                should_reuse = False
                if effective_reuse_mask is not None:
                    should_reuse = effective_reuse_mask.get((token_idx, head_idx), False)
                elif head_idx in reused_heads:
                    should_reuse = True

                if should_reuse and kv_previous is not None:
                    # 复用旧值
                    k_out[:, head_idx, token_idx, :] = kv_previous["k"][:, head_idx, token_idx, :]
                    v_out[:, head_idx, token_idx, :] = kv_previous["v"][:, head_idx, token_idx, :]
                    continue

                # 量化当前值
                k_slice = k_tensor[:, head_idx, token_idx, :]
                v_slice = v_tensor[:, head_idx, token_idx, :]

                k_quantized, scale_k = Quantizer.quantize_tensor(k_slice.unsqueeze(0), precision)
                v_quantized, scale_v = Quantizer.quantize_tensor(v_slice.unsqueeze(0), precision)

                k_out[:, head_idx, token_idx, :] = k_quantized.squeeze(0)
                v_out[:, head_idx, token_idx, :] = v_quantized.squeeze(0)

                scale_k_map[(token_idx, head_idx)] = scale_k
                scale_v_map[(token_idx, head_idx)] = scale_v

        # 计算压缩比
        original_bytes = (
            k_tensor.numel() * k_tensor.element_size() + v_tensor.numel() * v_tensor.element_size()
        )

        result: Dict[str, Any] = {
            "k": k_out,
            "v": v_out,
            "scale_k_map": scale_k_map,
            "scale_v_map": scale_v_map,
        }

        if self.use_packed_cache:
            packed_result = PackedKVCache.pack(k_tensor, v_tensor, precision_map)
            compressed_bytes = packed_result["memory_bytes"]
            memory_mb = packed_result["memory_mb"]
            compression_ratio = (
                1.0 - compressed_bytes / original_bytes if original_bytes > 0 else 0.0
            )
            result["packed_kv"] = packed_result["packed_kv"]
            result["memory_bytes"] = compressed_bytes
        else:
            compressed_bytes = self._estimate_compressed_bytes(k_tensor, v_tensor, precision_map)
            memory_mb = compressed_bytes / (1024.0 * 1024.0)
            compression_ratio = (
                1.0 - compressed_bytes / original_bytes if original_bytes > 0 else 0.0
            )

        result["compression_ratio"] = compression_ratio
        result["memory_mb"] = memory_mb

        # 更新统计
        self._compression_stats["total_original_bytes"] += original_bytes
        self._compression_stats["total_compressed_bytes"] += compressed_bytes
        self._compression_stats["step_count"] += 1

        return result

    def _compute_head_deltas(self, kv_current_k, kv_current_v, kv_previous_k, kv_previous_v):
        """计算每个 head 的变化幅度。"""
        num_heads = kv_current_k.shape[1]
        head_deltas = []
        for h in range(num_heads):
            d_k = torch.norm(kv_current_k[:, h] - kv_previous_k[:, h])
            d_v = torch.norm(kv_current_v[:, h] - kv_previous_v[:, h])
            head_deltas.append(float((d_k + d_v) / 2.0))
        return head_deltas

    def _store_kv(
        self,
        layer_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
        precision: Any,
        reuse_decision: Optional[Dict[str, Any]] = None,
        packed_kv: Optional[PackedKVCache] = None,
    ) -> None:
        """存储 KV 到缓存。"""
        precision_str = "mixed"
        if isinstance(precision, dict):
            counts = {}
            for tmap in precision.values():
                for p in tmap.values():
                    counts[p] = counts.get(p, 0) + 1
            if counts:
                precision_str = max(counts, key=counts.get)

        is_reused = False
        reuse_step = -1
        if reuse_decision is not None:
            is_reused = reuse_decision.get("should_reuse", False)

        entry = KVCacheEntry(
            k_tensor=k,
            v_tensor=v,
            precision=precision_str,
            importance_score=0.0,
            is_reused=is_reused,
            reuse_step=reuse_step,
            packed_kv=packed_kv,
        )
        self.cache_manager.store(layer_id, entry)
        self.kv_cache[layer_id] = {
            "k": k,
            "v": v,
            "precision": precision_str,
            "packed_kv": packed_kv,
        }

    def _load_kv(self, layer_id: int) -> Dict[str, torch.Tensor]:
        """从缓存加载 KV。"""
        entry = self.cache_manager.retrieve(layer_id)
        if entry is None:
            return {}
        return {
            "k": entry.k_tensor,
            "v": entry.v_tensor,
            "precision": entry.precision,
        }

    def prefetch_next_step(
        self,
        current_inputs: Dict[str, Any],
        current_kv_result: Dict[str, Any],
        predicted_sensitivity: float,
        predicted_drift: float,
        predicted_mode: str = "balanced",
    ) -> Dict[str, Any]:
        """基于 Q-drift 敏感度预测预取下一 step 的 KV cache。

        算法：
        1. 如果预测敏感度 >= 0.7，不预取（避免复用压缩缓存）。
        2. 否则用当前 KV 作为下一 step 的代理，生成预测复用 mask。
        3. 将预测会复用的 token-head 从当前 packed cache 复制到预取缓冲区。

        Args:
            current_inputs: 当前 step 的输入字典，含 ``kv_current`` / ``kv_previous``。
            current_kv_result: 当前 step 的 ShadowKV 输出（含 ``compressed_kv``）。
            predicted_sensitivity: Q-drift 预测的下一 step 敏感度。
            predicted_drift: Q-drift 预测的下一 step 漂移。
            predicted_mode: 预测的 ShadowKV 模式。

        Returns:
            预取结果字典，包含 ``prefetched_count`` 与 ``reuse_mask``。
        """
        if not self.prefetch_enabled or predicted_sensitivity >= 0.7:
            return {"prefetched_count": 0, "reuse_mask": {}}

        kv_current = current_inputs.get("kv_current")
        kv_previous = current_inputs.get("kv_previous")
        if kv_current is None:
            return {"prefetched_count": 0, "reuse_mask": {}}

        compressed_kv = current_kv_result.get("compressed_kv", {})
        packed_kv = compressed_kv.get("packed_kv")
        if packed_kv is None:
            return {"prefetched_count": 0, "reuse_mask": {}}

        # Use current importance map as a proxy for the next step.
        attention_scores = current_inputs.get("attention_scores")
        if attention_scores is None or attention_scores.numel() == 0:
            return {"prefetched_count": 0, "reuse_mask": {}}

        layer_id = current_inputs.get("layer_id", 0)
        importance_map = self.importance_model.score(
            attention_scores,
            kv_current=kv_current,
            kv_previous=kv_previous,
            layer_id=layer_id,
        )

        # Compute drift proxy: if no previous KV, assume low drift.
        if kv_previous is not None:
            drift_map = KVDecisionPlane.compute_drift_map(
                kv_current["k"], kv_current["v"], kv_previous["k"], kv_previous["v"]
            )
        else:
            drift_map = torch.zeros_like(importance_map)

        # Force conservative reuse under high predicted drift.
        if predicted_drift > 0.5:
            predicted_mode = "conservative"

        reuse_mask, _ = self.decide_reuse_per_token(
            kv_current["k"],
            kv_current["v"],
            kv_current["k"],  # proxy previous
            kv_current["v"],
            step_id=current_inputs.get("step_id", 0) + 1,
            total_steps=current_inputs.get("total_steps", 1),
            qdrift_drift_score=predicted_drift,
            mode=predicted_mode,
            importance_map=importance_map,
        )

        # Copy reused token-heads into a staging PackedKVCache.
        staged = packed_kv  # start from current packed cache
        # Create a shallow copy by re-packing with the same precision map.
        precision_map = self.precision_map.get(layer_id, {})
        staged_result = PackedKVCache.pack(compressed_kv["k"], compressed_kv["v"], precision_map)
        staged_packed = staged_result["packed_kv"]
        staged_packed.apply_reuse_mask(packed_kv, reuse_mask)

        prefetched_entry = KVCacheEntry(
            k_tensor=compressed_kv["k"],
            v_tensor=compressed_kv["v"],
            precision="mixed",
            packed_kv=staged_packed,
        )
        self.cache_manager.store_prefetched(layer_id, "next", prefetched_entry)

        prefetched_count = sum(1 for v in reuse_mask.values() if v)
        return {"prefetched_count": prefetched_count, "reuse_mask": reuse_mask}

    def set_memory_budget(self, memory_budget_mb: Optional[float]) -> None:
        """Runtime update of the KV cache memory budget."""
        bytes_budget = int(memory_budget_mb * 1024 * 1024) if memory_budget_mb is not None else None
        self.cache_manager.memory_budget_bytes = bytes_budget
        if self.decision_plane is not None:
            self.decision_plane.memory_budget_bytes = bytes_budget

    def get_compression_stats(self) -> Dict[str, float]:
        """获取压缩统计：compression_ratio, memory_savings_mb, avg_precision。"""
        total_orig = self._compression_stats["total_original_bytes"]
        total_comp = self._compression_stats["total_compressed_bytes"]
        steps = self._compression_stats["step_count"]

        if total_orig > 0:
            overall_compression_ratio = 1.0 - total_comp / total_orig
        else:
            overall_compression_ratio = 0.0

        memory_savings_mb = (total_orig - total_comp) / (1024.0 * 1024.0)

        total_precision_bits = 0
        total_count = 0
        precision_bits = {"fp32": 32, "fp16": 16, "int8": 8, "int4": 4}
        for prec, count in self._precision_distribution.items():
            total_precision_bits += precision_bits.get(prec, 16) * count
            total_count += count

        avg_precision = total_precision_bits / max(total_count, 1)

        return {
            "compression_ratio": overall_compression_ratio,
            "memory_savings_mb": memory_savings_mb,
            "avg_precision_bits": avg_precision,
            "step_count": float(steps),
        }

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _estimate_compressed_bytes(
        self,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        precision_map: Dict[int, Dict[int, str]],
    ) -> int:
        """估算压缩后的字节数。"""
        batch_size, num_heads, seq_len, head_dim = k_tensor.shape
        total_bits = 0
        precision_bits = {"fp32": 32, "fp16": 16, "int8": 8, "int4": 4}

        for token_idx in range(seq_len):
            token_map = precision_map.get(token_idx, {})
            for head_idx in range(num_heads):
                bits = precision_bits.get(token_map.get(head_idx, "fp16"), 16)
                total_bits += bits * head_dim * 2  # k + v

        total_bytes = total_bits * batch_size // 8
        return total_bytes
