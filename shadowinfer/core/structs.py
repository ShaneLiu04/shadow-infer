"""ShadowInfer Core Data Structures.

Implements all core dataclasses defined in TECHNICAL_SPEC.md §3.1.

Version: 3.0
Corresponds to: TECHNICAL_SPEC.md v2.0
"""

from __future__ import annotations

__version__ = "3.0"
__doc_version__ = "TECHNICAL_SPEC.md v2.0"

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from torch import Tensor


class AgentState(str, Enum):
    """Agent 生命周期状态枚举。

    定义在 TECHNICAL_SPEC.md §3.2 (BaseAgent 使用)
    """

    INIT = "INIT"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    SHUTDOWN = "SHUTDOWN"


@dataclass
class KVCacheEntry:
    """单个 KV cache 条目。

    对应 TECHNICAL_SPEC.md §3.1 核心数据结构 — KVCacheEntry。

    Attributes:
        k_tensor: 存储的 key tensor（可能已量化）。
        v_tensor: 存储的 value tensor（可能已量化）。
        precision: 存储精度，取值 "fp32" | "fp16" | "int8" | "int4"。
        scale_k: 量化 scale（如适用，仅 int8/int4 时非 None）。
        scale_v: 量化 scale（如适用，仅 int8/int4 时非 None）。
        importance_score: 基于 attention 分布的重要性分数，范围 [0, 1]。
        is_reused: 是否复用自上一 denoising step。
        reuse_step: 复用来源的 step id；若未复用则为 -1。
    """

    k_tensor: Tensor
    v_tensor: Tensor
    precision: str
    scale_k: Optional[Tensor] = None
    scale_v: Optional[Tensor] = None
    importance_score: float = 0.0
    is_reused: bool = False
    reuse_step: int = -1
    packed_kv: Optional[Any] = None


@dataclass
class StepConfig:
    """单 step 的优化配置。

    对应 TECHNICAL_SPEC.md §3.1 核心数据结构 — StepConfig。

    Attributes:
        step_id: 当前 denoising step 的序号。
        total_steps: 整个 denoising 过程的总 step 数。
        noise_level: 当前 step 的噪声水平（与 diffusion schedule 相关）。
        shadowkv_mode: ShadowKV 优化策略，取值 "aggressive" | "balanced" | "conservative"。
        reuse_layers: 在当前 step 中启用 KV 复用的层索引列表。
        compression_target: 目标压缩比（例如 0.5 表示压缩至 50% 存储）。
        ffn_mode: FFN 计算模式，取值 "sparse" | "mixed" | "full"。
        weight_precision_map: 逐通道（per-channel）FFN 权重精度映射。
            键为通道索引，值为精度字符串（如 "fp16" / "int8" / "int4"）。
        compute_path: FFN 计算路径，取值 "reuse" | "incremental" | "sparse" | "full"。
        sensitivity_score: Q-drift 计算得到的 step 敏感度 ∈ [0, 1]。
        drift_score: Q-drift 检测得到的激活漂移量 ∈ [0, 1]。
    """

    step_id: int
    total_steps: int
    noise_level: float
    shadowkv_mode: str
    reuse_layers: List[int] = field(default_factory=list)
    compression_target: float = 0.5
    ffn_mode: str = "full"
    weight_precision_map: Dict[int, str] = field(default_factory=dict)
    compute_path: str = "full"
    sensitivity_score: float = 0.0
    drift_score: float = 0.0


@dataclass
class ProfileResult:
    """Profiling 结果汇总。

    对应 TECHNICAL_SPEC.md §3.1 核心数据结构 — ProfileResult。

    Attributes:
        model_name: 被测模型名称（如 "Fast-dLLM-v2-7B"）。
        run_id: 本次运行唯一标识符。
        kv_precision_distribution: 逐层逐头的精度分布。
            外层字典 key 为 layer id；内层字典 key 为 head id，value 为精度字符串。
        kv_reuse_rate: 逐层 KV 复用率。
        kv_memory_mb: 逐层 KV cache 占用显存（MB）。
        q_drift_hit_rate: 逐层 Q-drift 命中/检测率。
        activation_delta: 逐层各类激活漂移指标。
            外层字典 key 为 layer id；内层字典 key 为指标名称（如 "query" / "activation"）。
        ffn_compute_load: 逐层 FFN 计算负载详情。
            外层字典 key 为 layer id；内层字典 key 为指标名称（如 "flops" / "sparse_ratio"）。
        ffn_sparse_update_ratio: 全局 FFN 稀疏更新比例。
        latency_e2e_ms: 端到端推理延迟（ms）。
        latency_per_step_ms: 逐 step 延迟（ms）。
        throughput_tokens_per_sec: 吞吐量（tokens / 秒）。
        perplexity_delta: 相对于 FP32 基线的困惑度增量。
        bleu_drop: 相对于 FP32 基线的 BLEU 分数下降值。
        accuracy_metrics: 各维度精度指标（如 "em" / "f1" / "exact_match"）。
    """

    model_name: str
    run_id: str
    kv_precision_distribution: Dict[int, Dict[int, str]] = field(default_factory=dict)
    kv_reuse_rate: Dict[int, float] = field(default_factory=dict)
    kv_memory_mb: Dict[int, float] = field(default_factory=dict)
    q_drift_hit_rate: Dict[int, float] = field(default_factory=dict)
    activation_delta: Dict[int, Dict[str, float]] = field(default_factory=dict)
    ffn_compute_load: Dict[int, Dict[str, float]] = field(default_factory=dict)
    ffn_sparse_update_ratio: float = 0.0
    latency_e2e_ms: float = 0.0
    latency_per_step_ms: Dict[int, float] = field(default_factory=dict)
    throughput_tokens_per_sec: float = 0.0
    perplexity_delta: float = 0.0
    bleu_drop: float = 0.0
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Message:
    """Profiling Bus 通信消息。

    用于 ShadowInfer 多 Agent 之间的事件传递与状态同步。
    定义在 TECHNICAL_SPEC.md §3.2 Agent 接口规范上下文（BaseAgent 通信）。

    Attributes:
        version: 消息协议版本号。
        message_id: 消息唯一标识符。
        message_type: 消息类型（如 "profiling" / "shadowkv" / "qdrift" / "ffn" / "control"）。
        source: 发送方 Agent 标识符。
        target: 目标 Agent 标识符（"broadcast" 表示广播）。
        step_id: 关联的 denoising step id（全局控制消息可为 -1）。
        timestamp: 消息发送时间戳（UNIX 时间或 ISO 格式字符串）。
        payload: 消息负载，包含任意业务数据字典。
    """

    version: str = "1.0"
    message_id: str = ""
    message_type: str = ""
    source: str = ""
    target: str = "broadcast"
    step_id: int = -1
    timestamp: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        source: str,
        target: str,
        message_type: str,
        payload: Dict[str, Any],
        step_id: int = -1,
    ) -> "Message":
        """构造一条 Profiling Bus 消息。

        对应 bus.py 中 Message.create 调用。
        """
        import uuid
        from datetime import datetime, timezone

        return cls(
            version="1.0",
            message_id=str(uuid.uuid4())[:12],
            message_type=message_type,
            source=source,
            target=target,
            step_id=step_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload=payload,
        )


@dataclass
class StepState:
    """单个 denoising step 的运行时状态。

    由 Orchestrator 维护，贯穿 Q-drift → ShadowKV → FFN → Profiler 的全流程，
    也是事件溯源（event sourcing）和流式输出的最小单元。
    """

    step_id: int
    total_steps: int
    inputs: Dict[str, Any] = field(default_factory=dict)
    step_config: Optional[StepConfig] = None
    qdrift_result: Dict[str, Any] = field(default_factory=dict)
    kv_result: Dict[str, Any] = field(default_factory=dict)
    ffn_result: Dict[str, Any] = field(default_factory=dict)
    profiler_result: Dict[str, Any] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    resolution: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    cancelled: bool = False
    outputs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为测试与 Bus 兼容的字典（保留 StepConfig 的简化字段）。"""
        sc = self.step_config
        return {
            "step_id": self.step_id,
            "step_config": {
                "shadowkv_mode": sc.shadowkv_mode if sc else "balanced",
                "ffn_mode": sc.ffn_mode if sc else "mixed",
                "noise_level": sc.noise_level if sc else 0.0,
            },
            "qdrift": self.qdrift_result,
            "shadowkv": self.kv_result,
            "ffn": self.ffn_result,
            "profiler": self.profiler_result,
            "alerts": self.alerts,
            "conflicts": self.conflicts,
            "resolution": self.resolution,
            "outputs": self.outputs,
        }


@dataclass
class PipelineContext:
    """单次推理运行的上下文。

    承载全局预算、回调、取消信号与事件溯源配置。
    """

    run_id: str
    start_time: float = field(default_factory=lambda: 0.0)
    latency_budget_ms: float = 100.0
    memory_budget_mb: float = 8192.0
    snapshot_dir: Optional[str] = None
    enable_snapshots: bool = False
    on_step: Optional[Callable[[StepState], None]] = None
    close_loop: bool = False
    cancelled: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def cancel(self) -> None:
        """请求取消本次推理运行。"""
        with self._lock:
            self.cancelled = True

    def is_cancelled(self) -> bool:
        """查询是否已请求取消。"""
        with self._lock:
            return self.cancelled


@dataclass
class StepStats:
    """单 step 统计条目。

    对应 BaseAgent.record_step_stat 使用。
    """

    step_id: int
    latency_ms: float = 0.0
    memory_mb: float = 0.0
    flops: float = 0.0
    accuracy_delta: float = 0.0
    kv_compression_ratio: float = 0.0
    ffn_sparse_ratio: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class ErrorRecord:
    """错误记录条目。

    对应 BaseAgent.log_error 使用。
    """

    step_id: int
    error_type: str
    message: str
    traceback: str
    timestamp: str = ""

    @classmethod
    def from_exception(
        cls,
        error: Exception,
        step_id: int = -1,
        traceback_str: str = "",
    ) -> "ErrorRecord":
        """从异常对象构造 ErrorRecord。"""
        from datetime import datetime, timezone

        return cls(
            step_id=step_id,
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback_str,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
