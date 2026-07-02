"""QDrift Agent — Step-aware 调度策略专家。

对应文档：QDRIFT_AGENT.md, TECHNICAL_SPEC.md §2.2 / §3.2
版本：v3.0
"""

from __future__ import annotations

__version__ = "3.0"

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from shadowinfer.core.base_agent import BaseAgent
from shadowinfer.core.structs import ProfileResult, StepConfig
from shadowinfer.utils.logging_utils import StructuredLogger
from shadowinfer.utils.metrics import Metrics


class QDriftAgent(BaseAgent):
    """QDrift Agent — Step-aware 调度策略专家。

    对应文档：QDRIFT_AGENT.md, TECHNICAL_SPEC.md §2.2 / §3.2
    版本：v3.0
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, name="qdrift")
        self.sensitivity_history: List[float] = []  # 敏感度历史
        self.drift_history: List[float] = []  # 漂移历史
        self.dispatch_history: List[Dict] = []  # 调度历史
        self.learning_rate: float = config.get("learning_rate", 0.05)
        self.noise_schedule: str = config.get("noise_schedule", "cosine")
        self.temperature: float = config.get("sensitivity_temperature", 1.0)
        self.drift_method: str = config.get("drift_method", "relative_l2")
        self.logger: StructuredLogger = StructuredLogger("qdrift", config.get("log_dir", "logs/"))

    # ------------------------------------------------------------------
    # 生命周期接口
    # ------------------------------------------------------------------

    def on_init(self, model_config: Dict[str, Any]) -> None:
        """初始化：加载配置，设置调度参数。

        Args:
            model_config: 模型级配置字典（如层数、hidden_dim 等）。
        """
        self.logger.log_event(
            event_type="agent_init",
            message="QDriftAgent initialized.",
            data={
                "learning_rate": self.learning_rate,
                "noise_schedule": self.noise_schedule,
                "temperature": self.temperature,
                "drift_method": self.drift_method,
                "model_config_keys": list(model_config.keys()),
            },
        )

    def on_step(self, step_config: StepConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个 step。

        输入 inputs 包含：
        - 'step_id': int
        - 'total_steps': int
        - 'query_current': Tensor[batch, seq_len, num_heads, head_dim]
        - 'query_previous': Tensor (可能为 None)
        - 'activation_current': Tensor[batch, seq_len, hidden_dim]
        - 'activation_previous': Tensor (可能为 None)
        - 'noise_level': float
        - 'profiler_feedback': Dict (可能为 None，含 accuracy_delta, latency_ms)

        返回：{'sensitivity_score': float, 'drift_score': float, 'dispatch': Dict,
                 'step_phase': str, 'learning_state': Dict}

        Args:
            step_config: 当前 step 的优化配置。
            inputs: 输入数据字典。

        Returns:
            输出数据字典，包含 sensitivity_score、drift_score、dispatch、step_phase、learning_state。
        """
        step_id: int = inputs.get("step_id", step_config.step_id)
        total_steps: int = inputs.get("total_steps", step_config.total_steps)
        noise_level: float = inputs.get("noise_level", step_config.noise_level)
        query_current = inputs.get("query_current")
        query_previous = inputs.get("query_previous")
        activation_current = inputs.get("activation_current")
        activation_previous = inputs.get("activation_previous")
        profiler_feedback: Optional[Dict] = inputs.get("profiler_feedback")

        # 1. Step Sensitivity Estimation
        sensitivity_score = self.estimate_sensitivity(
            step_id=step_id,
            total_steps=total_steps,
            noise_level=noise_level,
            noise_schedule=self.noise_schedule,
            temperature=self.temperature,
            profiler_feedback=profiler_feedback,
        )

        # 2. Activation Drift Detection
        drift_score = self.detect_drift(
            query_current=query_current,
            query_previous=query_previous,
            activation_current=activation_current,
            activation_previous=activation_previous,
            method=self.drift_method,
        )

        # 3. Dispatch Strategy Generation
        dispatch = self.generate_dispatch(sensitivity_score, drift_score)

        # 4. Adaptive Learning Adjustment
        adjusted_sensitivity = self.adaptive_adjustment(sensitivity_score, profiler_feedback or {})

        # 5. Phase Detection
        step_phase = self.get_phase(step_id, total_steps)

        # Record histories
        self.sensitivity_history.append(sensitivity_score)
        self.drift_history.append(drift_score)
        self.dispatch_history.append(
            {
                "step_id": step_id,
                "sensitivity_score": sensitivity_score,
                "drift_score": drift_score,
                "dispatch": dispatch,
                "phase": step_phase,
                "adjusted_sensitivity": adjusted_sensitivity,
            }
        )

        # Record per-step stats
        self.record_step_stat(
            step_id=step_id,
            stats={
                "latency_ms": (
                    profiler_feedback.get("latency_ms", 0.0) if profiler_feedback else 0.0
                ),
                "memory_mb": 0.0,
                "flops": 0.0,
                "accuracy_delta": (
                    profiler_feedback.get("accuracy_delta", 0.0) if profiler_feedback else 0.0
                ),
                "kv_compression_ratio": 0.0,
                "ffn_sparse_ratio": 0.0,
                "custom_metrics": {
                    "sensitivity_score": sensitivity_score,
                    "drift_score": drift_score,
                    "shadowkv_mode": dispatch.get("shadowkv_mode"),
                    "ffn_mode": dispatch.get("ffn_mode"),
                },
            },
        )

        self.logger.log_metric(
            metric_name="sensitivity_score",
            value=sensitivity_score,
            step_id=step_id,
            tags={"phase": step_phase, "shadowkv_mode": dispatch.get("shadowkv_mode")},
        )
        self.logger.log_metric(
            metric_name="drift_score",
            value=drift_score,
            step_id=step_id,
            tags={"phase": step_phase, "ffn_mode": dispatch.get("ffn_mode")},
        )

        return {
            "sensitivity_score": sensitivity_score,
            "drift_score": drift_score,
            "dispatch": dispatch,
            "step_phase": step_phase,
            "learning_state": {
                "adjusted_sensitivity": adjusted_sensitivity,
                "learning_rate": self.learning_rate,
                "temperature": self.temperature,
                "noise_schedule": self.noise_schedule,
                "drift_method": self.drift_method,
            },
        }

    def on_shutdown(self) -> Optional[ProfileResult]:
        """关闭：返回调度历史统计。

        Returns:
            可选的 ProfileResult 汇总对象，包含 Q-drift 相关统计。
        """
        self.logger.log_event(
            event_type="agent_shutdown",
            message="QDriftAgent shutting down.",
            data={
                "total_steps_processed": len(self.sensitivity_history),
                "sensitivity_mean": self.get_sensitivity_stats().get("mean", 0.0),
                "sensitivity_std": self.get_sensitivity_stats().get("std", 0.0),
            },
        )

        # Build latency_per_step_ms from step_stats
        latency_per_step_ms: Dict[int, float] = {}
        for step_id, stats in self.step_stats.items():
            latency_per_step_ms[step_id] = stats.latency_ms

        # Build q_drift_hit_rate from dispatch_history (per-step dispatch record count)
        q_drift_hit_rate: Dict[int, float] = {}
        for record in self.dispatch_history:
            sid = record.get("step_id", -1)
            if sid >= 0:
                q_drift_hit_rate[sid] = 1.0  # each step has a dispatch decision

        # Build activation_delta from drift_history
        activation_delta: Dict[int, Dict[str, float]] = {}
        for record in self.dispatch_history:
            sid = record.get("step_id", -1)
            if sid >= 0:
                activation_delta[sid] = {
                    "query": 0.0,
                    "activation": record.get("drift_score", 0.0),
                }

        # Build ffn_compute_load placeholder
        ffn_compute_load: Dict[int, Dict[str, float]] = {}
        for record in self.dispatch_history:
            sid = record.get("step_id", -1)
            if sid >= 0:
                ffn_compute_load[sid] = {
                    "flops": 0.0,
                    "sparse_ratio": 0.0,
                }

        model_name = self.config.get("model_name", "unknown")
        run_id = self.config.get("run_id", "qdrift-run")

        return ProfileResult(
            model_name=model_name,
            run_id=run_id,
            q_drift_hit_rate=q_drift_hit_rate,
            activation_delta=activation_delta,
            ffn_compute_load=ffn_compute_load,
            latency_per_step_ms=latency_per_step_ms,
            ffn_sparse_update_ratio=0.0,
            latency_e2e_ms=sum(latency_per_step_ms.values()),
            throughput_tokens_per_sec=0.0,
            perplexity_delta=0.0,
            bleu_drop=0.0,
            accuracy_metrics=self.get_sensitivity_stats(),
        )

    # ------------------------------------------------------------------
    # 核心算法 1: Step Sensitivity Estimation
    # ------------------------------------------------------------------

    def estimate_sensitivity(
        self,
        step_id: int,
        total_steps: int,
        noise_level: float,
        noise_schedule: str = "cosine",
        temperature: float = 1.0,
        profiler_feedback: Optional[Dict] = None,
    ) -> float:
        """估计当前 step 对计算误差的敏感度。

        算法（对应 QDRIFT_AGENT.md §核心算法 1）：
        1. progress = step_id / total_steps
        2. 基于 noise_schedule 计算基线敏感度：
           - linear: sensitivity = progress
           - cosine: sensitivity = 0.5 * (1 - cos(progress * pi))
           - sigmoid: sensitivity = 1 / (1 + exp(-10 * (progress - 0.5)))
        3. noise_correction = 1.0 - noise_level
        4. sensitivity = (base_sensitivity * 0.7 + noise_correction * 0.3) ** (1.0 / temperature)
        5. 如果有 profiler_feedback 且 accuracy_delta > 0.008: sensitivity *= 1.3（上限 1.0）
           如果 accuracy_delta < 0.003: sensitivity *= 0.9
        6. return clamp(sensitivity, 0.0, 1.0)

        Args:
            step_id: 当前 step id。
            total_steps: 总 step 数。
            noise_level: 当前噪声水平。
            noise_schedule: 噪声调度策略，"linear" | "cosine" | "sigmoid"。
            temperature: 敏感度温度参数。
            profiler_feedback: 可选的 profiler 反馈字典，含 accuracy_delta 等。

        Returns:
            敏感度分数，范围 [0.0, 1.0]。
        """
        if total_steps <= 0:
            progress = 0.0
        else:
            progress = step_id / total_steps

        # 2. 基于 noise_schedule 计算基线敏感度
        if noise_schedule == "linear":
            base_sensitivity = progress
        elif noise_schedule == "cosine":
            base_sensitivity = 0.5 * (1.0 - math.cos(progress * math.pi))
        elif noise_schedule == "sigmoid":
            base_sensitivity = 1.0 / (1.0 + math.exp(-10.0 * (progress - 0.5)))
        else:
            # 默认回退到 cosine
            base_sensitivity = 0.5 * (1.0 - math.cos(progress * math.pi))

        # 3. noise_correction
        noise_correction = 1.0 - noise_level

        # 4. 综合敏感度
        sensitivity = (base_sensitivity * 0.7 + noise_correction * 0.3) ** (1.0 / temperature)

        # 5. 基于 profiler feedback 调整
        if profiler_feedback is not None:
            accuracy_delta = profiler_feedback.get("accuracy_delta", 0.0)
            if accuracy_delta > 0.008:
                sensitivity *= 1.3
                sensitivity = min(sensitivity, 1.0)
            elif accuracy_delta < 0.003:
                sensitivity *= 0.9

        # 6. clamp
        return float(max(0.0, min(1.0, sensitivity)))

    # ------------------------------------------------------------------
    # 核心算法 2: Activation Drift Detection
    # ------------------------------------------------------------------

    def detect_drift(
        self,
        query_current: torch.Tensor,
        query_previous: torch.Tensor,
        activation_current: torch.Tensor,
        activation_previous: torch.Tensor,
        method: str = "relative_l2",
    ) -> float:
        """检测相邻 step 间的激活漂移。

        算法（对应 QDRIFT_AGENT.md §核心算法 2）：
        - relative_l2:
            query_drift = ||Q_t - Q_{t-1}|| / ||Q_t||
            activation_drift = ||A_t - A_{t-1}|| / ||A_t||
        - cosine_similarity:
            query_drift = 1 - cosine_similarity(Q_t, Q_{t-1})
            activation_drift = 1 - cosine_similarity(A_t, A_{t-1})
        - kl_divergence:
            query_current_norm = softmax(Q_t.flatten(-2))
            query_prev_norm = softmax(Q_{t-1}.flatten(-2))
            query_drift = KL(query_current_norm || query_prev_norm)
            activation_drift = 0.0

        综合：drift_score = 0.6 * query_drift + 0.4 * activation_drift
        return clamp(drift_score, 0.0, 1.0)

        Args:
            query_current: 当前 step 的 query tensor。
            query_previous: 上一 step 的 query tensor（可能为 None）。
            activation_current: 当前 step 的 activation tensor。
            activation_previous: 上一 step 的 activation tensor（可能为 None）。
            method: 漂移检测方法，"relative_l2" | "cosine_similarity" | "kl_divergence"。

        Returns:
            漂移分数，范围 [0.0, 1.0]。
        """
        # 如果缺少 previous tensor，无法计算漂移，返回 0.0
        if query_previous is None or activation_previous is None:
            return 0.0

        if method == "relative_l2":
            query_drift = Metrics.compute_relative_error(query_current, query_previous)
            activation_drift = Metrics.compute_relative_error(
                activation_current, activation_previous
            )
        elif method == "cosine_similarity":
            query_cos = Metrics.compute_cosine_similarity(query_current, query_previous)
            activation_cos = Metrics.compute_cosine_similarity(
                activation_current, activation_previous
            )
            query_drift = 1.0 - query_cos
            activation_drift = 1.0 - activation_cos
        elif method == "kl_divergence":
            # query_current_norm = softmax(Q_t.flatten(-2))
            query_current_flat = query_current.flatten(-2)
            query_prev_flat = query_previous.flatten(-2)
            query_current_norm = F.softmax(query_current_flat, dim=-1)
            query_prev_norm = F.softmax(query_prev_flat, dim=-1)
            query_drift = Metrics.compute_kl_divergence(query_current_norm, query_prev_norm)
            activation_drift = 0.0
        else:
            # 默认回退到 relative_l2
            query_drift = Metrics.compute_relative_error(query_current, query_previous)
            activation_drift = Metrics.compute_relative_error(
                activation_current, activation_previous
            )

        # 综合漂移分数
        drift_score = 0.6 * query_drift + 0.4 * activation_drift
        return float(max(0.0, min(1.0, drift_score)))

    # ------------------------------------------------------------------
    # 核心算法 3: 调度策略生成
    # ------------------------------------------------------------------

    def generate_dispatch(
        self,
        sensitivity_score: float,
        drift_score: float,
    ) -> Dict[str, str]:
        """生成调度决策。

        对应 QDRIFT_AGENT.md 调度策略矩阵：
        | sensitivity | drift | shadowkv_mode | ffn_mode | 说明 |
        | < 0.3 | < 0.2 | aggressive | sparse | 低敏感度+低漂移 → 最大优化 |
        | < 0.3 | ≥ 0.2 | balanced | sparse | 低敏感度+漂移大 → 适度优化 |
        | 0.3-0.7 | < 0.2 | balanced | mixed | 中等敏感度 → 平衡 |
        | 0.3-0.7 | ≥ 0.2 | conservative | mixed | 中等敏感度+漂移 → 保守 |
        | ≥ 0.7 | any | conservative | full | 高敏感度 → 全精度 |

        Args:
            sensitivity_score: 敏感度分数，范围 [0.0, 1.0]。
            drift_score: 漂移分数，范围 [0.0, 1.0]。

        Returns:
            调度决策字典，包含 'shadowkv_mode' 和 'ffn_mode'。
        """
        if sensitivity_score >= 0.7:
            # 高敏感度 → 全精度
            return {"shadowkv_mode": "conservative", "ffn_mode": "full"}
        elif sensitivity_score < 0.3:
            if drift_score < 0.2:
                # 低敏感度+低漂移 → 最大优化
                return {"shadowkv_mode": "aggressive", "ffn_mode": "sparse"}
            else:
                # 低敏感度+漂移大 → 适度优化
                return {"shadowkv_mode": "balanced", "ffn_mode": "sparse"}
        else:
            # 0.3 <= sensitivity < 0.7
            if drift_score < 0.2:
                # 中等敏感度 → 平衡
                return {"shadowkv_mode": "balanced", "ffn_mode": "mixed"}
            else:
                # 中等敏感度+漂移 → 保守
                return {"shadowkv_mode": "conservative", "ffn_mode": "mixed"}

    # ------------------------------------------------------------------
    # 核心算法 4: 自适应学习
    # ------------------------------------------------------------------

    def adaptive_adjustment(
        self,
        current_sensitivity: float,
        profiler_feedback: Dict,
    ) -> float:
        """基于 accuracy feedback 自适应调整敏感度。

        算法：
        - 如果 accuracy_delta > 0.005: 提高敏感度（更保守），调整量 = learning_rate * (acc_delta / 0.01)
        - 如果 accuracy_delta < 0.001: 降低敏感度（更激进），调整量 = learning_rate * 0.5
        - 否则保持不变

        Args:
            current_sensitivity: 当前敏感度分数。
            profiler_feedback: profiler 反馈字典，必须包含 'accuracy_delta'。

        Returns:
            调整后的敏感度分数，范围 [0.0, 1.0]。
        """
        accuracy_delta = profiler_feedback.get("accuracy_delta", 0.0)
        adjusted = current_sensitivity

        if accuracy_delta > 0.005:
            # 提高敏感度（更保守）
            adjustment = self.learning_rate * (accuracy_delta / 0.01)
            adjusted = current_sensitivity + adjustment
        elif accuracy_delta < 0.001:
            # 降低敏感度（更激进）
            adjustment = self.learning_rate * 0.5
            adjusted = current_sensitivity - adjustment
        # 否则保持不变

        return float(max(0.0, min(1.0, adjusted)))

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def get_phase(self, step_id: int, total_steps: int) -> str:
        """返回当前 step 阶段：early (<0.33) / mid / late (>0.66)

        Args:
            step_id: 当前 step id。
            total_steps: 总 step 数。

        Returns:
            阶段字符串："early" | "mid" | "late"。
        """
        if total_steps <= 0:
            return "early"
        progress = step_id / total_steps
        if progress < 0.33:
            return "early"
        elif progress > 0.66:
            return "late"
        else:
            return "mid"

    def get_dispatch_history(self) -> List[Dict]:
        """获取调度历史。

        Returns:
            调度历史记录列表。
        """
        return list(self.dispatch_history)

    def predict_next_sensitivity(
        self,
        step_id: int,
        total_steps: int,
    ) -> float:
        """Predict the sensitivity of the next denoising step.

        Uses the same analytical schedule as ``estimate_sensitivity`` with the
        next step's noise level. This is intentionally cheap so it can be used
        for KV-cache prefetching without running the model.
        """
        next_step_id = min(step_id + 1, total_steps)
        next_noise = next_step_id / max(total_steps, 1)
        return self.estimate_sensitivity(
            step_id=next_step_id,
            total_steps=total_steps,
            noise_level=next_noise,
            noise_schedule=self.noise_schedule,
            temperature=self.temperature,
            profiler_feedback=None,
        )

    def predict_next_drift(
        self,
        query_current: Optional[torch.Tensor] = None,
        activation_current: Optional[torch.Tensor] = None,
    ) -> float:
        """Predict next-step drift using a simple momentum heuristic.

        When current tensors are available, assume next drift is similar to the
        current drift estimate. When unavailable, fall back to the mean of the
        drift history or 0.0.
        """
        if query_current is not None and activation_current is not None:
            # Use current tensors as a proxy for the next step's previous state.
            return self.detect_drift(
                query_current=query_current,
                query_previous=query_current,
                activation_current=activation_current,
                activation_previous=activation_current,
                method=self.drift_method,
            )
        if self.drift_history:
            return float(sum(self.drift_history) / len(self.drift_history))
        return 0.0

    def get_sensitivity_stats(self) -> Dict[str, float]:
        """获取敏感度统计：mean, std, min, max。

        Returns:
            统计字典，包含 mean、std、min、max。若无历史数据则全部返回 0.0。
        """
        if not self.sensitivity_history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        import statistics

        mean = statistics.mean(self.sensitivity_history)
        std = (
            statistics.stdev(self.sensitivity_history) if len(self.sensitivity_history) > 1 else 0.0
        )
        min_val = min(self.sensitivity_history)
        max_val = max(self.sensitivity_history)

        return {
            "mean": float(mean),
            "std": float(std),
            "min": float(min_val),
            "max": float(max_val),
        }
