"""ShadowKV Agent — KV Cache 压缩与复用专家。

对应文档：README.md《核心模块》ShadowKV Agent
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
from shadowinfer.shadowkv.importance_model import ImportanceModel, compute_irap_importance
from shadowinfer.utils.logging_utils import StructuredLogger
from shadowinfer.utils.metrics import Metrics
from shadowinfer.utils.quantization import Quantizer

from .kv_cache_manager import KVCacheManager
from .packed_kv_cache import PackedKVCache
from .precision_plan import PRECISION_ORDER, PrecisionPlan
from .rd_allocator import EVICTED, admit_budget, entry_bytes, kv_sensitivity


class ShadowKVAgent(BaseAgent):
    """ShadowKV Agent — KV Cache 压缩与复用专家。

    对应文档：README.md《核心模块》ShadowKV Agent
    版本：v3.0
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, name="shadowkv")
        self.kv_cache: Dict[
            int, Dict[str, Any]
        ] = {}  # layer_id -> {k: Tensor, v: Tensor, precision: str}
        self.precision_map: Dict[
            int, Dict[int, Dict[int, str]]
        ] = {}  # layer_id -> token -> head -> precision
        # AR003 T004：IRAP 跨步 EMA 状态（layer_id -> [seq_k, heads]，常数内存）；
        # 取代原无界 importance_history（KV-P9）
        self._irap_state: Dict[int, torch.Tensor] = {}
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
        self.rd_allocation_enabled = bool(config.get("rd_allocation_enabled", True))
        self.hard_eviction_enabled = bool(config.get("hard_eviction_enabled", False))

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
        # AR002 M5：保留初始化配置，on_init 重建 cache_manager 时不再丢失
        self._memory_budget_bytes = memory_budget_bytes
        self._eviction_policy = eviction_policy
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
        # AR002 M5：决策缓存（(step_id, layer_id) 定位键 + 输入指纹校验）
        self._decision_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._decision_cache_step: Optional[int] = None
        # AR003 T002：attention 输入契约（KV-P2）。显式配置优先于缺省告警路径
        self._attn_contract_configured = "attn_is_probabilities" in config
        self.attn_is_probabilities = bool(config.get("attn_is_probabilities", False))
        self._attn_contract_warned = False
        # AR003 T003/T004：IRAP 状态（跨步 EMA，per-layer 常数内存）
        self._irap_processed = set()
        self._sink_warned = False
        self._pressure_warned = False
        self._pressure_step: Optional[int] = None

    def on_init(self, model_config: Dict[str, Any]) -> None:
        """初始化：加载配置，设置精度阈值和复用参数。"""
        self.num_layers = model_config.get("num_layers", self.num_layers)
        self.num_heads = model_config.get("num_heads", self.num_heads)
        self.head_dim = model_config.get("head_dim", self.head_dim)
        self.model_name = model_config.get("model_name", self.model_name)
        # AR002 M5：重建 cache_manager 时保留 memory_budget / eviction 配置
        # （修复原实现丢弃配置的缺陷），并全量重置累计状态与决策缓存
        self.cache_manager = KVCacheManager(
            self.num_layers,
            memory_budget_bytes=self._memory_budget_bytes,
            eviction_policy=self._eviction_policy,
        )
        self._compression_stats = {
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "step_count": 0,
        }
        self._precision_distribution = {
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
        self._decision_cache = {}
        self._decision_cache_step = None
        self._attn_contract_warned = False
        self._irap_state = {}
        self._irap_processed = set()
        self._sink_warned = False
        self._pressure_warned = False
        self._pressure_step = None
        self._eviction_warned = False

        # 加载优化配置
        self.compression_target = self.config.get("compression_target", 0.5)
        self.rd_allocation_enabled = bool(self.config.get("rd_allocation_enabled", True))
        self.hard_eviction_enabled = bool(self.config.get("hard_eviction_enabled", False))
        self.importance_thresholds = self.config.get(
            "importance_thresholds",
            {"fp32": 0.8, "fp16": 0.5, "int8": 0.2, "int4": 0.0},
        )
        self.reuse_base_threshold = self.config.get("reuse_base_threshold", 0.15)
        self.reuse_adaptive = self.config.get("reuse_adaptive", True)
        # AR003 T003：IRAP 公式参数（sink 硬保护 / 熵辅助权重 / EMA 衰减）
        self.sink_tokens = int(self.config.get("sink_tokens", 4))
        self.entropy_weight = float(self.config.get("entropy_weight", 0.3))
        self.irap_decay = float(self.config.get("irap_decay", 0.9))
        # AR003 T004：参数校验（srs FR3.3 / design 4.3.1）
        if not 0.0 < self.irap_decay < 1.0:
            raise ValueError(f"irap_decay must be in (0, 1), got {self.irap_decay}")
        if not 0.0 <= self.entropy_weight <= 1.0:
            raise ValueError(f"entropy_weight must be in [0, 1], got {self.entropy_weight}")
        if self.sink_tokens < 0:
            raise ValueError(f"sink_tokens must be >= 0, got {self.sink_tokens}")
        # AR003 T005：预算准入与全局驱逐阈值
        self.eviction_pressure_threshold = float(
            self.config.get("eviction_pressure_threshold", 0.9)
        )
        if not 0.0 < self.eviction_pressure_threshold <= 1.0:
            raise ValueError(
                "eviction_pressure_threshold must be in (0, 1], "
                f"got {self.eviction_pressure_threshold}"
            )
        self._eviction_warned = False

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

        # Step 2-4: 决策计算（AR002 M5：(step_id, layer_id) 决策缓存 + 输入指纹校验；
        # 命中且指纹一致时直接复用，指纹异常旁路，step 滚动自动失效）
        batch_size, num_heads, seq_len, _ = attention_scores.shape

        # AR003 T002：契约解析（inputs 键 > config 键 > 缺省告警路径）
        attn_is_probabilities = self._resolve_attn_contract(inputs)
        memory_pressure = inputs.get("memory_pressure", 0.0)
        # review I10：sink 配置覆盖全序列时告警（配置错误提示，一次）
        if self.sink_tokens >= seq_len and not self._sink_warned:
            self.logger.log_alert(
                "warning",
                f"sink_tokens ({self.sink_tokens}) >= seq_len ({seq_len}); "
                "all tokens protected at top tier",
            )
            self._sink_warned = True

        if step_id != self._decision_cache_step:
            self._decision_cache = {}
            self._decision_cache_step = step_id

        cache_key = (step_id, layer_id)
        fingerprint = self._decision_input_fingerprint(
            attention_scores,
            kv_current,
            kv_previous,
            total_steps,
            sensitivity_score,
            drift_score,
            mode,
            attn_is_probabilities=attn_is_probabilities,
            memory_pressure=memory_pressure,
        )
        cached = self._decision_cache.get(cache_key) if fingerprint is not None else None
        step_delta = None
        new_ema: Optional[torch.Tensor] = None
        if cached is not None and cached.get("fingerprint") == fingerprint:
            importance_map = cached["importance_map"]
            # review M4：副本隔离，避免准入/调用方修改污染缓存对象
            precision_map = {t: dict(m) for t, m in cached["precision_map"].items()}
            reuse_decision = cached["reuse_decision"]
            step_delta = cached.get("importance_drift")
        else:
            # Step 2: 计算重要性分数（AR002 M1 向量化；AR003 T003/T004 IRAP）
            # review I3：EMA/ΔA 通过局部变量传递，不落全局单槽（异常中断不残留）
            importance_map, new_ema, step_delta = self._compute_importance_with_state(
                attention_scores,
                layer_id,
                self.num_layers,
                attn_is_probabilities=attn_is_probabilities,
                received_ema=self._irap_state.get(layer_id),
                irap_decay=self.irap_decay,
                entropy_weight=self.entropy_weight,
                sink_tokens=self.sink_tokens,
            )

            # Step 3: 分配精度
            precision_map = self.allocate_precision(importance_map, mode)

            # 如果 drift_score > 0.5，保留每 token top 20% head 为 FP16
            # （review I1：sink 前缀硬保护优先，永不降档）
            if drift_score > 0.5:
                num_fp16_heads = max(1, int(num_heads * 0.2))
                for token_idx in range(seq_len):
                    if token_idx < self.sink_tokens:
                        continue
                    top_heads = torch.topk(
                        importance_map[token_idx], num_fp16_heads
                    ).indices.tolist()
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

            if fingerprint is not None:
                # review M4：缓存副本（准入在缓存之外原地进行）
                self._decision_cache[cache_key] = {
                    "fingerprint": fingerprint,
                    "importance_map": importance_map,
                    "precision_map": {t: dict(m) for t, m in precision_map.items()},
                    "reuse_decision": reuse_decision,
                    "importance_drift": step_delta,
                }

        # AR003 T005：预算准入（pre-pack 降档阶梯，先于压缩）
        # AR009：复用条目作为保护集，避免被降档/驱逐
        protected_reuse: Optional[Set[Tuple[int, int]]] = None
        if kv_previous is not None and isinstance(reuse_decision, dict):
            reuse_mask = reuse_decision.get("reuse_mask") or {}
            protected_reuse = {key for key, flag in reuse_mask.items() if flag}
        precision_map = self._admit_precision_budget(
            importance_map, precision_map, kv_current, protected=protected_reuse
        )

        # 记录状态（AR003 T004；review I2：按 (step, layer) 幂等，pressure/契约
        # 变化导致的重放不得二次推进 EMA）
        processed_key = (step_id, layer_id)
        if new_ema is not None and processed_key not in self._irap_processed:
            prev_state = self._irap_state.get(layer_id)
            if prev_state is not None and prev_state.shape[0] > new_ema.shape[0]:
                self.logger.log_alert(
                    "warning",
                    f"IRAP state truncated: seq_k {prev_state.shape[0]} -> {new_ema.shape[0]}",
                )
            self._irap_state[layer_id] = new_ema
            self._irap_processed.add(processed_key)

        # Step 5: 压缩 KV
        compressed_kv = self.compress_kv(
            kv_current,
            precision_map,
            reuse_decision=reuse_decision,
            reuse_mask=reuse_decision.get("reuse_mask"),
            kv_previous=kv_previous,
            build_scale_map=False,
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

        # AR003 T005：写入张量级重要性元数据（全局驱逐依据）
        self.cache_manager.set_importance_matrix(layer_id, importance_map.detach().cpu())
        # AR003 T005：消费 memory_pressure（超阈跨层驱逐；无预算仅告警一次）
        # review I4：每步最多触发一次全局驱逐（逐层循环内去重）
        self._consume_memory_pressure(memory_pressure, step_id)

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
                    # AR003 T006：ΔA 跨步差分信号
                    "importance_drift": step_delta,
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
            # AR003 T006：ΔA 跨步差分信号（FR3.5；首步 None）
            "importance_drift": step_delta,
            "irap_stats": {
                "delta_a": step_delta,
                "gamma": self.irap_decay,
                "seq_k": int(seq_len),
            },
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

    PRECISION_BITS = {"fp32": 32, "fp16": 16, "int8": 8, "int4": 4}
    DOWNGRADE_LADDER = {"fp32": "fp16", "fp16": "int8", "int8": "int4"}

    def _entry_bytes(self, precision: str, batch: int, head_dim: int) -> float:
        """单条 token-head 的 k+v 存储字节（含 int8/int4 scale，与 packed 一致）。

        AR009：成本模型统一由 ``rd_allocator.entry_bytes`` 提供，``evicted`` 为 0。
        """
        return entry_bytes(precision, batch, head_dim)

    def _estimate_precision_bytes(
        self, precision_map: Dict[int, Dict[int, str]], batch: int, seq_len: int, head_dim: int
    ) -> int:
        """按 per-precision bits 口径估算字节（修复 KV-P10 的 8x 低估）。"""
        total = 0.0
        for token_map in precision_map.values():
            for precision in token_map.values():
                total += self._entry_bytes(precision, batch, head_dim)
        return int(math.ceil(total))

    def _admit_precision_budget(
        self,
        importance_map: torch.Tensor,
        precision_map: Dict[int, Dict[int, str]],
        kv_current: Dict[str, torch.Tensor],
        protected: Optional[Set[Tuple[int, int]]] = None,
    ) -> Dict[int, Dict[int, str]]:
        """预算准入（AR003 T005 / AR009）：率失真排序 + 可选硬驱逐。

        - 优先级为 ``importance × KV 幅值敏感度``（RD），可通过
          ``rd_allocation_enabled=False`` 回退到纯重要性排序；
        - sink 前缀与 ``protected``（复用条目）不降档、不驱逐；
        - 降档至 int4 仍超预算且启用 ``hard_eviction_enabled`` 时，按同序
          驱逐低价值 token-head（零字节）。
        """
        budget = self.cache_manager.memory_budget_bytes
        if not budget or budget <= 0:
            return precision_map

        layer_budget = max(1, int(budget) // max(1, self.num_layers))
        k_tensor = kv_current["k"]
        v_tensor = kv_current["v"]
        batch, _, seq_len, head_dim = k_tensor.shape

        predicted = self._estimate_precision_bytes(precision_map, batch, seq_len, head_dim)
        if predicted <= layer_budget:
            return precision_map

        try:
            sensitivity = kv_sensitivity(k_tensor, v_tensor)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.log_alert(
                "warning", f"kv sensitivity unavailable, falling back to importance: {exc}"
            )
            sensitivity = torch.ones(
                importance_map.shape[0], importance_map.shape[1], dtype=torch.float32
            )

        result, predicted, evicted = admit_budget(
            importance_map,
            sensitivity,
            precision_map,
            layer_budget,
            batch=batch,
            head_dim=head_dim,
            sink_tokens=self.sink_tokens,
            rd_allocation=self.rd_allocation_enabled,
            allow_eviction=self.hard_eviction_enabled,
            protected=protected,
        )

        if evicted:
            self.logger.log_event(
                "hard_eviction",
                f"evicted {evicted} token-heads to satisfy layer budget",
                data={"layer_budget": layer_budget, "predicted": predicted},
            )
        if predicted > layer_budget:
            self.logger.log_alert(
                "warning",
                f"layer budget floor reached: predicted={predicted} "
                f"layer_budget={layer_budget} (all candidates at int4)",
            )
        return result

    def _consume_memory_pressure(
        self, memory_pressure: float, step_id: Optional[int] = None
    ) -> None:
        """消费 orchestrator 的 memory_pressure 信号（AR003 T005，FR3.4/KV-P8）。

        review I4：同一 step 内最多执行一次全局驱逐（逐层循环去重）。
        review M2：非有限 / 非法压力值忽略并告警一次。
        """
        if step_id is not None:
            if self._pressure_step == step_id:
                return
            self._pressure_step = step_id
        try:
            pressure = float(memory_pressure)
        except (TypeError, ValueError):
            pressure = float("nan")
        if not math.isfinite(pressure) or pressure < 0.0:
            if not self._pressure_warned:
                self.logger.log_alert(
                    "warning", f"invalid memory_pressure ignored: {memory_pressure!r}"
                )
                self._pressure_warned = True
            return
        if pressure < self.eviction_pressure_threshold:
            return

        budget = self.cache_manager.memory_budget_bytes
        if budget and budget > 0:
            evicted = self.cache_manager.enforce_global_budget(budget, self.sink_tokens)
            if evicted:
                self.logger.log_event(
                    "global_eviction",
                    f"memory_pressure={pressure:.3f} evicted {evicted} token-heads",
                    data={"evicted": evicted, "pressure": pressure},
                )
        elif not self._eviction_warned:
            self.logger.log_alert(
                "warning",
                "memory_pressure >= threshold but memory_budget_mb is not configured; "
                "no eviction action taken",
            )
            self._eviction_warned = True

    def reset_state(self) -> None:
        """重置步级/跨步决策状态（review I7：跨 run 复用 orchestrator 时冷启动）。

        保留配置与累计统计（on_init 负责完整重置）。
        """
        self._irap_state = {}
        self._irap_processed = set()
        self._decision_cache = {}
        self._decision_cache_step = None
        self._pressure_step = None
        self.precision_map = {}
        if self.cache_manager is not None:
            self.cache_manager.clear()

    def _resolve_attn_contract(self, inputs: Dict[str, Any]) -> bool:
        """解析 attention 输入契约（AR003 T002，FR3.1）。

        优先级：on_step inputs 键 > agent 配置键 > 缺省（logits，mock 语义）。
        缺省路径每次 agent 生命周期内仅告警一次。
        """
        if "attn_is_probabilities" in inputs:
            return bool(inputs["attn_is_probabilities"])
        if self._attn_contract_configured:
            return self.attn_is_probabilities
        if not self._attn_contract_warned:
            self.logger.log_alert(
                "warning",
                "attn_is_probabilities not configured; assuming logits (mock semantics)",
            )
            self._attn_contract_warned = True
        return False

    def compute_importance_score(
        self,
        attention_scores: torch.Tensor,
        token_index: int,
        head_index: int,
        layer_index: int,
        num_layers: int,
        attn_is_probabilities: bool = False,
        received_ema=None,
        irap_decay: float = 0.9,
        entropy_weight: float = 0.3,
        sink_tokens: int = 4,
    ) -> float:
        """
        计算 per-token-head 重要性分数（逐元素参考实现 / golden 基准）。

        AR003 T003 新公式（IRAP，与共享核心 compute_irap_importance 等价）：
        1. 契约归一化权重（概率输入跳过 softmax，仅行和安全校正）
        2. 列向接收注意力 received[t,h]（KV token 被关注程度，修复 KV-P3）
        3. 逐 head 升序 dense-rank 归一化 r ∈ (0, 1]
        4. 行熵辅助项（权重 β = entropy_weight）
        5. imp = clamp(((1-β)·r + β·归一化熵) · 层因子, max=1)；
           sink 前缀硬保护为 1.0；旧位置加成已移除（AR003）
        """
        # 1. 契约归一化
        if attn_is_probabilities:
            weights = self._row_normalize(attention_scores)
        else:
            weights = F.softmax(attention_scores, dim=-1)

        seq_len = weights.shape[-1]

        # 2. 列向接收注意力 [heads, seq_k]（可选 EMA，与向量化路径一致）
        received = weights.mean(dim=(0, 2))
        if received_ema is not None:
            prev = received_ema
            if prev.shape[0] != seq_len:
                if prev.shape[0] < seq_len:
                    pad = torch.zeros(
                        seq_len - prev.shape[0],
                        prev.shape[1],
                        dtype=prev.dtype,
                        device=prev.device,
                    )
                    prev = torch.cat([prev, pad], dim=0)
                else:
                    prev = prev[:seq_len]
            received = irap_decay * prev.transpose(0, 1) + (1.0 - irap_decay) * received

        # 3. 逐 head 升序 dense-rank（排序复刻，独立于向量化 argsort 实现）
        head_received = received[head_index]
        sorted_idx = torch.sort(head_received).indices
        rank_of = torch.empty(sorted_idx.shape, dtype=torch.float64)
        rank_of[sorted_idx] = torch.arange(1, seq_len + 1, dtype=torch.float64)
        rank_norm = float(rank_of[token_index] / seq_len)

        # 4. 行熵辅助项（Metrics.compute_entropy 语义：clamp + 全局重归一化）
        #    非方形输入下 KV token t ≥ seq_q 无对应 query 行，熵项取 0
        if token_index < weights.shape[2]:
            row = weights[:, head_index, token_index, :]
            entropy = Metrics.compute_entropy(row)
            normalized_entropy = entropy / math.log(max(seq_len, 2))
        else:
            normalized_entropy = 0.0

        # 5. 综合评分 + sink 硬保护
        layer_factor = 1.0 + 0.1 * (layer_index / max(num_layers, 1))
        score = ((1.0 - entropy_weight) * rank_norm + entropy_weight * normalized_entropy) * (
            layer_factor
        )
        if token_index < sink_tokens:
            return 1.0
        return float(min(score, 1.0))

    @staticmethod
    def _row_normalize(tensor: torch.Tensor) -> torch.Tensor:
        """行和数值安全校正（AR003 T002）：概率契约输入不再 softmax。"""
        return tensor / tensor.sum(dim=-1, keepdim=True).clamp(min=1e-10)

    def _compute_importance_matrix(
        self,
        attention_scores: torch.Tensor,
        layer_index: int,
        num_layers: int,
        attn_is_probabilities: bool = False,
        received_ema=None,
        irap_decay: float = 0.9,
        entropy_weight: float = 0.3,
        sink_tokens: int = 4,
    ) -> torch.Tensor:
        """
        向量化计算全 (token, head) 重要性矩阵（仅返回重要性）。

        AR003 T003（FR3.2/FP6）：委托共享核心 ``compute_irap_importance``
        （单一公式来源）——列向接收注意力 EMA 状态的 rank 归一化为主信号、
        行熵辅助、sink 前缀硬保护；旧位置加成已移除。与逐元素
        ``compute_importance_score`` 等价（容差 1e-6，golden 双实现对照）。

        非方形输入口径：attention_scores 形状 (batch, heads, seq_q, seq_k) 时，
        重要性矩阵按 KV token 维（seq_k）输出——接收注意力、rank 归一化与
        sink 保护均以 seq_k 为基准（AR003 起重要性主体为 KV 列；方形自注意力
        下 seq_q == seq_k，行为不变）。
        """
        importance_map, _, _ = self._compute_importance_with_state(
            attention_scores,
            layer_index,
            num_layers,
            attn_is_probabilities=attn_is_probabilities,
            received_ema=received_ema,
            irap_decay=irap_decay,
            entropy_weight=entropy_weight,
            sink_tokens=sink_tokens,
        )
        return importance_map

    def _compute_importance_with_state(
        self,
        attention_scores: torch.Tensor,
        layer_index: int,
        num_layers: int,
        attn_is_probabilities: bool = False,
        received_ema=None,
        irap_decay: float = 0.9,
        entropy_weight: float = 0.3,
        sink_tokens: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[float]]:
        """向量化重要性 + 新 EMA 状态 + ΔA（review I3：局部返回，无全局单槽）。"""
        if attn_is_probabilities:
            weights = self._row_normalize(attention_scores)
        else:
            # 数值安全：logits 含 NaN/Inf 时置零后再 softmax
            weights = F.softmax(torch.nan_to_num(attention_scores), dim=-1)

        return compute_irap_importance(
            weights,
            received_ema_prev=received_ema,
            irap_decay=irap_decay,
            entropy_weight=entropy_weight,
            sink_tokens=sink_tokens,
            layer_index=layer_index,
            num_layers=num_layers,
        )

    def allocate_precision(
        self, importance_map: torch.Tensor, mode: str
    ) -> Dict[int, Dict[int, str]]:
        """
        分配精度。

        对应 README.md《核心算法》精度映射策略：
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
            return self._allocate_precision_reference(importance_map, mode)

        seq_len = importance_map.shape[0]
        num_heads = importance_map.shape[1]

        scores = importance_map.detach().cpu().double()

        code = torch.where(
            scores >= thresholds["fp32"],
            torch.full_like(scores, 3.0),
            torch.where(
                scores >= thresholds["fp16"],
                torch.full_like(scores, 2.0),
                torch.where(
                    scores >= thresholds["int8"],
                    torch.full_like(scores, 1.0),
                    torch.zeros_like(scores),
                ),
            ),
        )

        code_names = {0: "int4", 1: "int8", 2: "fp16", 3: "fp32"}
        precision_map: Dict[int, Dict[int, str]] = {}
        code_list = code.to(torch.int32).tolist()
        for token_idx in range(seq_len):
            token_map: Dict[int, str] = {}
            row = code_list[token_idx]
            for head_idx in range(num_heads):
                token_map[head_idx] = code_names[row[head_idx]]
            precision_map[token_idx] = token_map

        return precision_map

    def _allocate_precision_reference(
        self, importance_map: Any, mode: str
    ) -> Dict[int, Dict[int, str]]:
        """原逐元素循环实现（golden 基准 / dict 输入路径）。"""
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

        算法（对应 README.md《核心算法》ShadowKV 分层压缩 3）：
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

        # 3. 向量化逐 (token, head) 阈值比较（AR002 M1）：
        #    float64 精确复刻逐元素 float()/python 浮点比较语义
        delta64 = delta_map.double()  # [num_heads, seq_len]
        importance64 = importance_map.detach().cpu().double().clamp(0.0, 1.0)  # [seq, heads]
        threshold = effective_threshold * (1.0 - 0.3 * importance64)
        threshold = torch.clamp(threshold, min=1e-5)
        mask = delta64.t() < threshold  # [seq_len, num_heads]

        per_token_reused = mask.sum(dim=1)
        full_reuse_count = int((per_token_reused == num_heads).sum().item())
        no_reuse_count = int((per_token_reused == 0).sum().item())
        partial_reuse_count = seq_len - full_reuse_count - no_reuse_count
        reused_entries = int(mask.sum().item())
        total_entries = seq_len * num_heads

        reuse_mask: Dict[Tuple[int, int], bool] = {}
        mask_list = mask.tolist()
        for token_idx in range(seq_len):
            row = mask_list[token_idx]
            for head_idx in range(num_heads):
                reuse_mask[(token_idx, head_idx)] = bool(row[head_idx])

        reuse_ratio = reused_entries / max(total_entries, 1)

        stats = {
            "full_reuse_count": float(full_reuse_count),
            "partial_reuse_count": float(partial_reuse_count),
            "no_reuse_count": float(no_reuse_count),
            "reuse_ratio": float(reuse_ratio),
        }

        return reuse_mask, stats

    def _decide_reuse_per_token_reference(
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
        """原逐元素循环实现（golden 基准）。"""
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

    def _decision_input_fingerprint(
        self,
        attention_scores: torch.Tensor,
        kv_current: Dict[str, torch.Tensor],
        kv_previous: Optional[Dict[str, torch.Tensor]],
        total_steps: int,
        sensitivity_score: float,
        drift_score: float,
        mode: str,
        attn_is_probabilities: bool = False,
        memory_pressure: float = 0.0,
    ) -> Optional[tuple]:
        """
        决策输入指纹（AR002 M5，design 4.2.2 点 5；AR003 T002 扩展契约与压力标量）。

        各输入张量的 (shape 元组, dtype, data_ptr) + 标量输入元组；
        计算抛异常时返回 None（调用方旁路缓存：不命中、不写入）。
        """
        try:
            parts: List[Any] = []
            for tensor in (attention_scores, kv_current["k"], kv_current["v"]):
                parts.append((tuple(tensor.shape), str(tensor.dtype), tensor.data_ptr()))
            if kv_previous is not None:
                parts.append(
                    (
                        tuple(kv_previous["k"].shape),
                        kv_previous["k"].data_ptr(),
                        tuple(kv_previous["v"].shape),
                        kv_previous["v"].data_ptr(),
                    )
                )
            else:
                parts.append(None)
            parts.extend(
                (
                    total_steps,
                    float(sensitivity_score),
                    float(drift_score),
                    mode,
                    bool(attn_is_probabilities),
                    float(memory_pressure),
                )
            )
            return tuple(parts)
        except Exception:
            return None

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
        build_scale_map: bool = True,
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

        # 解析复用掩码（新格式优先）
        effective_reuse_mask: Optional[Dict[Tuple[int, int], bool]] = None
        if reuse_mask is not None:
            effective_reuse_mask = reuse_mask
        elif reuse_decision is not None and "reuse_mask" in reuse_decision:
            effective_reuse_mask = reuse_decision["reuse_mask"]

        # 旧格式回退：全局 reused_heads 展开为逐 (token, head) 掩码
        if effective_reuse_mask is None and reuse_decision is not None:
            if reuse_decision.get("should_reuse", False):
                reused_head_set: Set[int] = set(reuse_decision.get("reused_heads", []))
                if reused_head_set:
                    effective_reuse_mask = {
                        (token_idx, head_idx): head_idx in reused_head_set
                        for token_idx in range(seq_len)
                        for head_idx in range(num_heads)
                    }

        k_prev = kv_previous["k"] if kv_previous is not None else None
        v_prev = kv_previous["v"] if kv_previous is not None else None

        # AR008：plan/run 分离——一次构建，compress 与 pack 共享。
        # 复用命中仅在存在上一步张量时生效（与 legacy 语义一致）。
        plan: Optional[PrecisionPlan] = None
        try:
            plan = PrecisionPlan.build(
                precision_map,
                seq_len,
                num_heads,
                reuse_mask=effective_reuse_mask if k_prev is not None else None,
                device=k_tensor.device,
            )
            core = self._compress_kv_vectorized(
                k_tensor,
                v_tensor,
                k_prev,
                v_prev,
                precision_map,
                effective_reuse_mask,
                plan=plan,
                build_scale_map=build_scale_map,
            )
        except Exception as exc:
            self.logger.log_alert(
                "error",
                f"vectorized compress_kv failed, falling back to reference: {exc}",
            )
            plan = None
            core = self._compress_kv_reference(
                k_tensor, v_tensor, k_prev, v_prev, precision_map, effective_reuse_mask
            )

        k_out = core["k"]
        v_out = core["v"]
        scale_k_map = core["scale_k_map"]
        scale_v_map = core["scale_v_map"]

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
            packed_result = PackedKVCache.pack(
                k_tensor,
                v_tensor,
                precision_map,
                plan=plan,
                precomputed=core.get("group_payload"),
            )
            compressed_bytes = packed_result["memory_bytes"]
            compression_ratio = (
                1.0 - compressed_bytes / original_bytes if original_bytes > 0 else 0.0
            )
            result["packed_kv"] = packed_result["packed_kv"]
            result["memory_bytes"] = compressed_bytes
        else:
            compressed_bytes = self._estimate_compressed_bytes(k_tensor, v_tensor, precision_map)
            compression_ratio = (
                1.0 - compressed_bytes / original_bytes if original_bytes > 0 else 0.0
            )

        # AR002 M5（srs FR3.3）：memory_mb 口径修正为真实节省
        # （未压缩基准字节 - 压缩后字节），禁止以元素数近似
        memory_mb = (original_bytes - compressed_bytes) / (1024.0 * 1024.0)

        result["compression_ratio"] = compression_ratio
        result["memory_mb"] = memory_mb
        result["uncompressed_bytes"] = original_bytes
        result["compressed_bytes"] = compressed_bytes

        # 更新统计
        self._compression_stats["total_original_bytes"] += original_bytes
        self._compression_stats["total_compressed_bytes"] += compressed_bytes
        self._compression_stats["step_count"] += 1

        return result

    def _compress_kv_vectorized(
        self,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        k_prev: Optional[torch.Tensor],
        v_prev: Optional[torch.Tensor],
        precision_map: Dict[int, Dict[int, str]],
        reuse_mask: Optional[Dict[Tuple[int, int], bool]],
        plan: Optional[PrecisionPlan] = None,
        build_scale_map: bool = True,
    ) -> Dict[str, Any]:
        """
        向量化压缩核心（AR002 M1）。

        与 _compress_kv_reference 逐元素等价：
        - 按精度分组，advanced indexing 批量取 (N, batch, head_dim) 后整组量化
        - 复用条目批量拷贝自 k_prev/v_prev
        - int8/int4 的整数码与 per-entry scale 与 reference bitwise 一致

        已知包络差异（预存缺陷，文档记录）：reference 的 int4 写回依赖
        (batch, g, gs) 向 (batch, head_dim) 的赋值，仅在 batch==1 且
        g==1（head_dim==group_size=64）时成立，其余情形 RuntimeError；
        本路径统一 reshape(N, batch, head_dim) 写回，在 reference 可用的
        包络内 bitwise 一致，其余情形按语义意图产出。
        """
        if plan is not None:
            return self._compress_kv_with_plan(
                k_tensor, v_tensor, k_prev, v_prev, plan, build_scale_map
            )

        batch_size, num_heads, seq_len, head_dim = k_tensor.shape
        k_out = torch.zeros_like(k_tensor)
        v_out = torch.zeros_like(v_tensor)
        scale_k_map: Dict[Tuple[int, int], Optional[torch.Tensor]] = {}
        scale_v_map: Dict[Tuple[int, int], Optional[torch.Tensor]] = {}
        # AR003：INT4 分组自适应（与 PackedKVCache 规则一致）——head_dim 不可
        # 被 64 整除时以 head_dim 为组（修复小 head_dim 模型 int4 直接抛错）
        int4_group = 64 if head_dim % 64 == 0 else head_dim

        groups: Dict[str, List[Tuple[int, int]]] = {
            "fp32": [],
            "fp16": [],
            "int8": [],
            "int4": [],
        }
        reuse_ts: List[int] = []
        reuse_hs: List[int] = []

        for token_idx in range(seq_len):
            token_precision = precision_map.get(token_idx, {})
            for head_idx in range(num_heads):
                precision = token_precision.get(head_idx, "fp16")
                p = precision.lower() if hasattr(precision, "lower") else precision
                if p not in groups:
                    raise ValueError(f"Unsupported precision: {precision}")
                if (
                    reuse_mask is not None
                    and reuse_mask.get((token_idx, head_idx), False)
                    and k_prev is not None
                ):
                    reuse_ts.append(token_idx)
                    reuse_hs.append(head_idx)
                else:
                    groups[p].append((token_idx, head_idx))

        device = k_tensor.device

        if reuse_ts:
            h_idx = torch.tensor(reuse_hs, dtype=torch.long, device=device)
            t_idx = torch.tensor(reuse_ts, dtype=torch.long, device=device)
            k_out[:, h_idx, t_idx, :] = k_prev[:, h_idx, t_idx, :]
            v_out[:, h_idx, t_idx, :] = v_prev[:, h_idx, t_idx, :]

        for precision, entries in groups.items():
            if not entries:
                continue

            h_idx = torch.tensor([h for _, h in entries], dtype=torch.long, device=device)
            t_idx = torch.tensor([t for t, _ in entries], dtype=torch.long, device=device)
            k_group = k_tensor[:, h_idx, t_idx, :].permute(1, 0, 2)  # (N, batch, head_dim)
            v_group = v_tensor[:, h_idx, t_idx, :].permute(1, 0, 2)

            if precision in ("fp32", "fp16"):
                k_quantized = k_group if precision == "fp32" else k_group.half()
                v_quantized = v_group if precision == "fp32" else v_group.half()
                k_out[:, h_idx, t_idx, :] = k_quantized.permute(1, 0, 2).to(k_out.dtype)
                v_out[:, h_idx, t_idx, :] = v_quantized.permute(1, 0, 2).to(v_out.dtype)
                for token_idx, head_idx in entries:
                    scale_k_map[(token_idx, head_idx)] = None
                    scale_v_map[(token_idx, head_idx)] = None
            else:
                if precision == "int8":
                    qk, sk = Quantizer.quantize_int8(k_group)
                    qv, sv = Quantizer.quantize_int8(v_group)
                else:
                    qk, sk = Quantizer.quantize_int4(k_group, group_size=int4_group)
                    qv, sv = Quantizer.quantize_int4(v_group, group_size=int4_group)
                k_codes = qk.reshape(qk.shape[0], batch_size, head_dim).permute(1, 0, 2)
                v_codes = qv.reshape(qv.shape[0], batch_size, head_dim).permute(1, 0, 2)
                k_out[:, h_idx, t_idx, :] = k_codes.to(k_out.dtype)
                v_out[:, h_idx, t_idx, :] = v_codes.to(v_out.dtype)
                for n, (token_idx, head_idx) in enumerate(entries):
                    scale_k_map[(token_idx, head_idx)] = sk[n : n + 1]
                    scale_v_map[(token_idx, head_idx)] = sv[n : n + 1]

        return {
            "k": k_out,
            "v": v_out,
            "scale_k_map": scale_k_map,
            "scale_v_map": scale_v_map,
        }

    def _compress_kv_with_plan(
        self,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        k_prev: Optional[torch.Tensor],
        v_prev: Optional[torch.Tensor],
        plan: PrecisionPlan,
        build_scale_map: bool,
    ) -> Dict[str, Any]:
        """Plan-driven compression (AR008).

        Equivalent to the legacy grouped path: reuse entries are copied from
        the previous step, all other entries are quantized per precision using
        flat gather/scatter indices. The plan is shared with ``PackedKVCache``
        through ``group_payload`` so packing does not rebuild groups or repeat
        the int8/fp quantization.
        """
        batch_size, num_heads, seq_len, head_dim = k_tensor.shape
        if (
            plan.seq_len != seq_len
            or plan.num_heads != num_heads
            or v_tensor.shape != k_tensor.shape
        ):
            raise ValueError(
                "PrecisionPlan shape mismatch: plan "
                f"(seq_len={plan.seq_len}, num_heads={plan.num_heads}) vs "
                f"tensor (seq_len={seq_len}, num_heads={num_heads})"
            )

        k_out = torch.zeros_like(k_tensor)
        v_out = torch.zeros_like(v_tensor)
        k_flat = k_tensor.reshape(batch_size, num_heads * seq_len, head_dim)
        v_flat = v_tensor.reshape(batch_size, num_heads * seq_len, head_dim)
        k_out_flat = k_out.view(batch_size, num_heads * seq_len, head_dim)
        v_out_flat = v_out.view(batch_size, num_heads * seq_len, head_dim)

        reuse_active = k_prev is not None and plan.reuse_flat.numel() > 0
        if reuse_active and v_prev is None:
            raise ValueError("v_prev is required when reuse entries are active")
        if reuse_active and (k_prev.shape != k_tensor.shape or v_prev.shape != v_tensor.shape):
            raise ValueError("previous KV tensors must match the current KV shape")
        if plan.codes.device.type != k_tensor.device.type:
            raise ValueError(
                f"PrecisionPlan device ({plan.codes.device}) does not match "
                f"KV device ({k_tensor.device})"
            )
        reuse_set = plan.reuse_set() if reuse_active else set()

        int4_group = 64 if head_dim % 64 == 0 else head_dim
        scale_k_map: Optional[Dict[Tuple[int, int], Optional[torch.Tensor]]] = (
            {} if build_scale_map else None
        )
        scale_v_map: Optional[Dict[Tuple[int, int], Optional[torch.Tensor]]] = (
            {} if build_scale_map else None
        )
        payload: Dict[str, Dict[str, Any]] = {}

        for precision in PRECISION_ORDER:
            entries = plan.groups.get(precision, [])
            if not entries:
                continue
            flat = plan.flat_indices[precision]
            k_group = k_flat.index_select(1, flat).permute(1, 0, 2)
            v_group = v_flat.index_select(1, flat).permute(1, 0, 2)

            if precision in ("fp32", "fp16"):
                qk = k_group if precision == "fp32" else k_group.half()
                qv = v_group if precision == "fp32" else v_group.half()
                sk = sv = None
            elif precision == "int8":
                qk, sk = Quantizer.quantize_int8(k_group)
                qv, sv = Quantizer.quantize_int8(v_group)
            else:
                qk, sk = Quantizer.quantize_int4(k_group, group_size=int4_group)
                qv, sv = Quantizer.quantize_int4(v_group, group_size=int4_group)
                qk = qk.reshape(qk.shape[0], batch_size, head_dim)
                qv = qv.reshape(qv.shape[0], batch_size, head_dim)

            k_out_flat.index_copy_(1, flat, qk.permute(1, 0, 2).to(k_out.dtype).contiguous())
            v_out_flat.index_copy_(1, flat, qv.permute(1, 0, 2).to(v_out.dtype).contiguous())

            payload[precision] = {
                "k_group": k_group,
                "v_group": v_group,
                "qk": qk,
                "qv": qv,
                "scale_k": sk,
                "scale_v": sv,
            }

            if build_scale_map:
                for index, entry in enumerate(entries):
                    if entry in reuse_set:
                        continue
                    scale_k_map[entry] = None if sk is None else sk[index : index + 1]
                    scale_v_map[entry] = None if sv is None else sv[index : index + 1]

        if reuse_active:
            # Legacy writes reuse entries after grouping (its groups exclude
            # them); overwrite the quantized placeholders with previous values.
            k_prev_flat = k_prev.reshape(batch_size, num_heads * seq_len, head_dim)
            v_prev_flat = v_prev.reshape(batch_size, num_heads * seq_len, head_dim)
            k_out_flat.index_copy_(1, plan.reuse_flat, k_prev_flat.index_select(1, plan.reuse_flat))
            v_out_flat.index_copy_(1, plan.reuse_flat, v_prev_flat.index_select(1, plan.reuse_flat))

        return {
            "k": k_out,
            "v": v_out,
            "scale_k_map": scale_k_map,
            "scale_v_map": scale_v_map,
            "group_payload": payload,
        }

    def _compress_kv_reference(
        self,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        k_prev: Optional[torch.Tensor],
        v_prev: Optional[torch.Tensor],
        precision_map: Dict[int, Dict[int, str]],
        reuse_mask: Optional[Dict[Tuple[int, int], bool]],
    ) -> Dict[str, Any]:
        """原逐元素循环实现（golden 基准 / 异常回退路径）。"""
        batch_size, num_heads, seq_len, head_dim = k_tensor.shape
        k_out = torch.zeros_like(k_tensor)
        v_out = torch.zeros_like(v_tensor)
        scale_k_map: Dict[Tuple[int, int], Optional[torch.Tensor]] = {}
        scale_v_map: Dict[Tuple[int, int], Optional[torch.Tensor]] = {}
        # AR003：INT4 分组自适应（与向量化路径一致）
        int4_group = 64 if head_dim % 64 == 0 else head_dim

        for token_idx in range(seq_len):
            token_precision = precision_map.get(token_idx, {})
            for head_idx in range(num_heads):
                precision = token_precision.get(head_idx, "fp16")
                if isinstance(precision, str) and precision.lower() == EVICTED:
                    continue

                should_reuse = False
                if reuse_mask is not None:
                    should_reuse = reuse_mask.get((token_idx, head_idx), False)

                if should_reuse and k_prev is not None:
                    k_out[:, head_idx, token_idx, :] = k_prev[:, head_idx, token_idx, :]
                    v_out[:, head_idx, token_idx, :] = v_prev[:, head_idx, token_idx, :]
                    continue

                k_slice = k_tensor[:, head_idx, token_idx, :]
                v_slice = v_tensor[:, head_idx, token_idx, :]

                k_quantized, scale_k = Quantizer.quantize_tensor(
                    k_slice.unsqueeze(0), precision, group_size=int4_group
                )
                v_quantized, scale_v = Quantizer.quantize_tensor(
                    v_slice.unsqueeze(0), precision, group_size=int4_group
                )

                k_out[:, head_idx, token_idx, :] = k_quantized.squeeze(0)
                v_out[:, head_idx, token_idx, :] = v_quantized.squeeze(0)

                scale_k_map[(token_idx, head_idx)] = scale_k
                scale_v_map[(token_idx, head_idx)] = scale_v

        return {
            "k": k_out,
            "v": v_out,
            "scale_k_map": scale_k_map,
            "scale_v_map": scale_v_map,
        }

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

        # Reuse decisions rely on the Q-drift predicted drift score below.

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

        # Build a staging PackedKVCache for the prefetched entry.
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
        precision_bits = {"fp32": 32, "fp16": 16, "int8": 8, "int4": 4, EVICTED: 0}
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
