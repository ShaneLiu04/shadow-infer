"""Profiler Agent — 性能分析专家。

对应文档：PROFILER_AGENT.md, TECHNICAL_SPEC.md §3.2
版本：v3.0
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch

from shadowinfer.core.base_agent import BaseAgent
from shadowinfer.core.bus import MESSAGE_TYPES, ProfilingBus
from shadowinfer.core.structs import AgentState, Message, ProfileResult, StepConfig
from shadowinfer.utils.logging_utils import StructuredLogger
from shadowinfer.utils.memory_utils import MemoryTracker
from shadowinfer.utils.metrics import Metrics

from .reporter import HTMLReporter


class ProfilerAgent(BaseAgent):
    """Profiler Agent — 性能分析专家。

    对应文档：PROFILER_AGENT.md, TECHNICAL_SPEC.md §3.2
    版本：v3.0
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, name="profiler")
        self.logger = StructuredLogger("profiler", config.get("log_dir", "logs/"))
        self.baseline_data: Dict[str, Any] = {}  # 基线性能数据
        self.optimized_data: Dict[str, Any] = {}  # 优化后性能数据
        self.alert_thresholds: Dict[str, Any] = {}  # 告警阈值

        # 内部状态
        self._latency_history: List[float] = []  # 用于检测 latency 连续递增
        self._accuracy_history: List[Dict[str, Any]] = []  # 用于检测 accuracy 抖动
        self._step_counter: int = 0
        self._run_id: str = str(uuid.uuid4())[:8]
        self._model_name: str = config.get("model_name", "unknown")
        self._bus: Optional[ProfilingBus] = None  # 由外部注入
        self._baseline_collected: bool = False

        # per-step 数据聚合
        self._kv_metrics_per_step: Dict[int, Dict[str, Any]] = {}
        self._qdrift_metrics_per_step: Dict[int, Dict[str, Any]] = {}
        self._ffn_metrics_per_step: Dict[int, Dict[str, Any]] = {}
        self._perf_metrics_per_step: Dict[int, Dict[str, Any]] = {}
        self._accuracy_metrics_per_step: Dict[int, Dict[str, Any]] = {}
        self._alerts_per_step: Dict[int, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # 生命周期接口
    # ------------------------------------------------------------------

    def on_init(self, model_config: Dict[str, Any]) -> None:
        """初始化：加载配置，设置告警阈值。"""
        self.transition_to(AgentState.READY)

        self._model_name = model_config.get("model_name", self._model_name)

        self.alert_thresholds = {
            "accuracy_warning": 0.005,  # 0.5%
            "accuracy_critical": 0.01,  # 1.0%
            "latency_budget_ms": model_config.get("max_latency_ms", 100.0),
            "memory_budget_mb": model_config.get("max_memory_mb", 8192.0),
            "latency_warning_ratio": 1.5,  # 150%
            "memory_warning_ratio": 0.9,  # 90%
            "latency_consecutive_increase": 3,  # 连续 3 个 step
            "accuracy_jitter_threshold": 0.002,  # 0.2% 抖动阈值
        }

        self.logger.log_event(
            "init",
            f"ProfilerAgent initialized. model={self._model_name}, run_id={self._run_id}",
            data={"thresholds": self.alert_thresholds},
        )

        # 如果配置中有预计算基线数据，直接加载
        if "baseline_data" in self.config:
            self.baseline_data = self.config["baseline_data"]
            self._baseline_collected = True
            self.logger.log_event("init", "Baseline data loaded from config.")

    def on_step(self, step_config: StepConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个 step。

        输入 inputs 包含：
        - 'kv_metrics': KV cache 相关数据
        - 'qdrift_metrics': Q-drift 相关数据
        - 'ffn_metrics': FFN 相关数据
        - 'performance': 性能数据
        - 'accuracy': 精度数据
        - 'baseline_output': 基线输出（用于对比）
        - 'optimized_output': 优化输出（用于对比）

        返回：metrics dict + alerts list
        """
        step_id = step_config.step_id
        self._step_counter = max(self._step_counter, step_id)
        self.transition_to(AgentState.RUNNING)

        # 提取各维度输入
        kv_data = inputs.get("kv_metrics", {})
        qdrift_data = inputs.get("qdrift_metrics", {})
        ffn_data = inputs.get("ffn_metrics", {})
        perf_data = inputs.get("performance", {})
        accuracy_data = inputs.get("accuracy", {})
        baseline_output = inputs.get("baseline_output")
        optimized_output = inputs.get("optimized_output")

        # 如果提供了张量输出，追加到 accuracy_data 供相对误差计算
        if baseline_output is not None and optimized_output is not None:
            accuracy_data["baseline_output"] = baseline_output
            accuracy_data["optimized_output"] = optimized_output

        # 收集各维度指标
        kv_metrics = self._collect_kv_metrics(kv_data)
        qdrift_metrics = self._collect_qdrift_metrics(qdrift_data)
        ffn_metrics = self._collect_ffn_metrics(ffn_data)
        perf_metrics = self._collect_performance_metrics(perf_data)
        accuracy_metrics = self._collect_accuracy_metrics(accuracy_data)

        metrics = {
            "step_id": step_id,
            "kv_metrics": kv_metrics,
            "qdrift_metrics": qdrift_metrics,
            "ffn_metrics": ffn_metrics,
            "performance_metrics": perf_metrics,
            "accuracy_metrics": accuracy_metrics,
        }

        # 检查告警
        alerts = self._check_alerts(metrics)
        metrics["alerts"] = alerts

        # 记录 step stat（兼容 BaseAgent）
        self.record_step_stat(
            step_id,
            {
                "latency_ms": perf_metrics.get("latency", {}).get("e2e_ms", 0.0),
                "memory_mb": perf_metrics.get("memory", {}).get("total_mb", 0.0),
                "flops": sum(
                    layer.get("flops", 0.0)
                    for layer in ffn_metrics.get("compute_load", {}).get("per_layer", {}).values()
                ),
                "accuracy_delta": accuracy_metrics.get("perplexity", {}).get("delta_percent", 0.0),
                "kv_compression_ratio": kv_metrics.get("memory_mb", {}).get("savings_ratio", 0.0),
                "ffn_sparse_ratio": ffn_metrics.get("sparse_update_ratio", {}).get("overall", 0.0),
            },
        )

        # 记录关键指标到 logger
        self.logger.log_metric(
            "step_latency_ms",
            perf_metrics.get("latency", {}).get("e2e_ms", 0.0),
            step_id=step_id,
            tags={"model": self._model_name},
        )
        self.logger.log_metric(
            "step_memory_mb",
            perf_metrics.get("memory", {}).get("total_mb", 0.0),
            step_id=step_id,
            tags={"model": self._model_name},
        )
        self.logger.log_metric(
            "accuracy_delta_percent",
            accuracy_metrics.get("perplexity", {}).get("delta_percent", 0.0),
            step_id=step_id,
            tags={"model": self._model_name},
        )

        # 存储 per-step 数据
        self._kv_metrics_per_step[step_id] = kv_metrics
        self._qdrift_metrics_per_step[step_id] = qdrift_metrics
        self._ffn_metrics_per_step[step_id] = ffn_metrics
        self._perf_metrics_per_step[step_id] = perf_metrics
        self._accuracy_metrics_per_step[step_id] = accuracy_metrics
        self._alerts_per_step[step_id] = alerts

        # 若总线已注入，发布 PROFILE_DATA 消息
        if self._bus is not None:
            msg = Message(
                version="1.0",
                message_id=f"profile-{self._run_id}-{step_id}",
                message_type=MESSAGE_TYPES.PROFILE_DATA,
                source="profiler",
                target="orchestrator",
                step_id=step_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                payload={
                    "kv_metrics": kv_metrics,
                    "qdrift_metrics": qdrift_metrics,
                    "ffn_metrics": ffn_metrics,
                    "performance_metrics": perf_metrics,
                    "accuracy_metrics": accuracy_metrics,
                    "alerts": alerts,
                },
            )
            self._bus.send(msg)

        return metrics

    def on_shutdown(self) -> Optional[ProfileResult]:
        """关闭：生成最终报告。"""
        self.transition_to(AgentState.SHUTDOWN)

        self.optimized_data = self._aggregate_step_data()

        result = self._generate_profile_result()

        output_dir = self.config.get("output_dir", "outputs/")
        os.makedirs(output_dir, exist_ok=True)

        self._export_baseline(os.path.join(output_dir, "profile_baseline.json"))
        self._export_optimized(os.path.join(output_dir, "profile_optimized.json"))

        self._generate_comparison_html(
            self.baseline_data,
            self.optimized_data,
            os.path.join(output_dir, "profile_comparison.html"),
        )

        summary = self._generate_summary(self.baseline_data, self.optimized_data)
        summary_path = os.path.join(output_dir, "profile_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        self.logger.log_event(
            "shutdown",
            "ProfilerAgent shutdown complete.",
            data={
                "run_id": self._run_id,
                "total_steps": self._step_counter,
                "output_dir": output_dir,
            },
        )
        self.logger.flush()

        return result

    # ------------------------------------------------------------------
    # 维度收集方法
    # ------------------------------------------------------------------

    def _collect_baseline(self, num_steps: int, warmup_steps: int) -> Dict[str, Any]:
        """基线收集：运行无优化模型，收集完整性能数据。

        注：基线收集通常需要实际模型执行。此处提供可扩展框架：
        若 config 中提供 ``baseline_runner`` 可调用对象，则调用它；
        否则返回空框架，等待外部数据填充。
        """
        self.logger.log_event(
            "baseline",
            f"Starting baseline collection: warmup={warmup_steps}, profile={num_steps}",
        )

        baseline_runner = self.config.get("baseline_runner")
        if baseline_runner is not None and callable(baseline_runner):
            baseline_data = baseline_runner(num_steps=num_steps, warmup_steps=warmup_steps)
            self.baseline_data = baseline_data
            self._baseline_collected = True
            self.logger.log_event("baseline", "Baseline collected via runner.")
            return baseline_data

        self.logger.log_event(
            "baseline",
            "No baseline_runner provided. Returning empty baseline framework.",
        )
        return {
            "model": self._model_name,
            "run_id": self._run_id,
            "num_steps": num_steps,
            "warmup_steps": warmup_steps,
            "kv_cache": {},
            "q_drift": {},
            "ffn": {},
            "latency": {},
            "accuracy": {},
        }

    def _collect_kv_metrics(self, kv_data: Dict[str, Any]) -> Dict[str, Any]:
        """收集 KV Cache 维度指标：precision_distribution, reuse_rate, memory_mb, access_pattern."""
        precision_map = kv_data.get("precision_map", {})
        reuse_decision = kv_data.get("reuse_decision", {})
        memory_mb = kv_data.get("memory_mb", 0.0)

        # precision_distribution
        per_token_head: Dict[int, Dict[int, str]] = {}
        histogram = {"fp32": 0, "fp16": 0, "int8": 0, "int4": 0}
        for token_id, heads in precision_map.items():
            tid = int(token_id) if isinstance(token_id, str) else token_id
            per_token_head[tid] = {}
            for head_id, precision in heads.items():
                hid = int(head_id) if isinstance(head_id, str) else head_id
                p = str(precision).lower()
                per_token_head[tid][hid] = p
                if p in histogram:
                    histogram[p] += 1

        # reuse_rate
        per_layer_reuse: Dict[int, float] = {}
        reuse_count = 0
        total_count = 0
        for layer_id, decision in reuse_decision.items():
            lid = int(layer_id) if isinstance(layer_id, str) else layer_id
            if isinstance(decision, dict):
                reused = bool(decision.get("reused", False))
            else:
                reused = bool(decision)
            per_layer_reuse[lid] = 1.0 if reused else 0.0
            total_count += 1
            if reused:
                reuse_count += 1
        overall_reuse = reuse_count / total_count if total_count > 0 else 0.0

        # memory_mb
        per_layer_memory: Dict[int, float] = {}
        per_layer_raw = kv_data.get("per_layer_memory", {})
        if isinstance(per_layer_raw, dict):
            for k, v in per_layer_raw.items():
                lid = int(k) if isinstance(k, str) else k
                per_layer_memory[lid] = float(v)
        baseline_total = kv_data.get("baseline_total_memory", memory_mb)
        savings_ratio = 0.0
        if baseline_total > 0 and memory_mb > 0:
            savings_ratio = 1.0 - (memory_mb / baseline_total)

        # access_pattern
        access_pattern = {
            "read_bandwidth_gb": float(kv_data.get("read_bandwidth_gb", 0.0)),
            "write_bandwidth_gb": float(kv_data.get("write_bandwidth_gb", 0.0)),
            "cache_hit_rate": float(kv_data.get("cache_hit_rate", 0.0)),
        }

        return {
            "precision_distribution": {
                "per_token_head": per_token_head,
                "histogram": histogram,
            },
            "reuse_rate": {
                "per_layer": per_layer_reuse,
                "overall": overall_reuse,
            },
            "memory_mb": {
                "per_layer": per_layer_memory,
                "total": float(memory_mb),
                "baseline_total": float(baseline_total),
                "savings_ratio": savings_ratio,
            },
            "access_pattern": access_pattern,
        }

    def _collect_qdrift_metrics(self, qdrift_data: Dict[str, Any]) -> Dict[str, Any]:
        """收集 Q-drift 维度指标：step_hit_rate, sensitivity_distribution, activation_delta."""
        sensitivity = float(qdrift_data.get("sensitivity_score", 0.0))
        drift = float(qdrift_data.get("drift_score", 0.0))
        dispatch = qdrift_data.get("dispatch", {})

        # step_hit_rate: dispatch 成功选择了策略即视为 hit
        hit_rate = 1.0 if isinstance(dispatch, dict) and dispatch.get("strategy") else 0.0

        # sensitivity_distribution histogram
        sensitivity_hist = {"low": 0, "mid": 0, "high": 0}
        if sensitivity < 0.3:
            sensitivity_hist["low"] = 1
        elif sensitivity < 0.7:
            sensitivity_hist["mid"] = 1
        else:
            sensitivity_hist["high"] = 1

        # activation_delta
        activation_delta = {
            "mean": drift * 0.5,
            "max": drift,
            "std": drift * 0.2,
            "p95": drift * 0.9,
        }

        step_id = self._step_counter
        return {
            "step_hit_rate": {
                "per_step": {step_id: hit_rate},
                "overall": hit_rate,
            },
            "sensitivity_distribution": {
                "per_step": {step_id: sensitivity},
                "histogram": sensitivity_hist,
            },
            "activation_delta": {
                "per_step": {step_id: activation_delta},
            },
        }

    def _collect_ffn_metrics(self, ffn_data: Dict[str, Any]) -> Dict[str, Any]:
        """收集 FFN 维度指标：compute_load, sparse_update_ratio, mixed_precision."""
        compute_path = ffn_data.get("compute_path", "full")
        quantization = ffn_data.get("quantization", {})
        sparse_update = ffn_data.get("sparse_update", {})
        compute_stats = ffn_data.get("compute_stats", {})

        # compute_load per_layer
        per_layer_load: Dict[int, Dict[str, float]] = {}
        if isinstance(compute_stats, dict):
            for layer_id, stats in compute_stats.items():
                lid = int(layer_id) if isinstance(layer_id, str) else layer_id
                per_layer_load[lid] = {
                    "flops": float(stats.get("flops", 0.0)),
                    "bandwidth_gb": float(stats.get("bandwidth_gb", 0.0)),
                    "compute_time_ms": float(stats.get("compute_time_ms", 0.0)),
                }

        # sparse_update_ratio
        overall_sparse = 0.0
        per_layer_sparse: Dict[int, float] = {}
        if isinstance(sparse_update, dict):
            per_layer_sparse_raw = sparse_update.get("per_layer", {})
            for k, v in per_layer_sparse_raw.items():
                lid = int(k) if isinstance(k, str) else k
                per_layer_sparse[lid] = float(v)
            overall_sparse = float(sparse_update.get("overall", 0.0))

        # mixed_precision
        mixed_precision = {
            "fp32_channels": 0,
            "fp16_channels": 0,
            "int8_channels": 0,
            "int4_channels": 0,
        }
        if isinstance(quantization, dict):
            for precision, count in quantization.items():
                p = str(precision).lower()
                key = f"{p}_channels"
                if key in mixed_precision:
                    mixed_precision[key] += int(count)

        return {
            "compute_load": {
                "per_layer": per_layer_load,
            },
            "sparse_update_ratio": {
                "overall": overall_sparse,
                "per_layer": per_layer_sparse,
            },
            "mixed_precision": mixed_precision,
            "compute_path": str(compute_path),
        }

    def _collect_performance_metrics(self, perf_data: Dict[str, Any]) -> Dict[str, Any]:
        """收集性能维度指标：latency, throughput, gpu_utilization."""
        latency_ms = float(perf_data.get("latency_ms", 0.0))
        memory_mb = float(perf_data.get("memory_mb", 0.0))
        gpu_utilization = float(perf_data.get("gpu_utilization", 0.0))

        self._latency_history.append(latency_ms)

        # latency stats (via Metrics)
        if len(self._latency_history) >= 2:
            latency_stats = Metrics.compute_latency_stats(self._latency_history)
        else:
            latency_stats = {
                "mean": latency_ms,
                "median": latency_ms,
                "p95": latency_ms,
                "p99": latency_ms,
                "min": latency_ms,
                "max": latency_ms,
                "std": 0.0,
            }

        # GPU memory via MemoryTracker
        gpu_mem = MemoryTracker.get_gpu_memory_info()

        tokens_per_sec = float(perf_data.get("tokens_per_sec", 0.0))
        tokens_per_step = float(perf_data.get("tokens_per_step", 1.0))

        per_layer_ms = perf_data.get("per_layer_ms", {})
        if isinstance(per_layer_ms, dict):
            per_layer_ms = {
                int(k) if isinstance(k, str) else k: float(v) for k, v in per_layer_ms.items()
            }

        return {
            "latency": {
                "e2e_ms": latency_ms,
                "per_step_ms": {self._step_counter: latency_ms},
                "per_layer_ms": per_layer_ms,
                "warmup_ms": float(perf_data.get("warmup_ms", 0.0)),
                "stats": latency_stats,
            },
            "throughput": {
                "tokens_per_sec": tokens_per_sec,
                "tokens_per_step": tokens_per_step,
            },
            "gpu_utilization": {
                "compute_util": gpu_utilization,
                "memory_util": (
                    gpu_mem.get("allocated", 0.0) / gpu_mem.get("total", 1.0)
                    if gpu_mem.get("total", 0.0) > 0
                    else 0.0
                ),
            },
            "memory": {
                "total_mb": gpu_mem.get("total", 0.0),
                "allocated_mb": gpu_mem.get("allocated", 0.0),
                "free_mb": gpu_mem.get("free", 0.0),
                "input_mb": memory_mb,
            },
        }

    def _collect_accuracy_metrics(self, accuracy_data: Dict[str, Any]) -> Dict[str, Any]:
        """收集精度维度指标：perplexity, bleu, generation_quality."""
        baseline_ppl = float(accuracy_data.get("baseline_perplexity", 0.0))
        optimized_ppl = float(accuracy_data.get("optimized_perplexity", 0.0))

        # perplexity drop via Metrics
        ppl_drop = Metrics.compute_accuracy_drop(baseline_ppl, optimized_ppl)
        delta_percent = 0.0
        if baseline_ppl != 0:
            delta_percent = ppl_drop / baseline_ppl

        baseline_bleu = float(accuracy_data.get("baseline_bleu", 0.0))
        optimized_bleu = float(accuracy_data.get("optimized_bleu", 0.0))
        bleu_drop = Metrics.compute_accuracy_drop(baseline_bleu, optimized_bleu)

        # relative error via Metrics (tensor-level)
        baseline_output = accuracy_data.get("baseline_output")
        optimized_output = accuracy_data.get("optimized_output")
        relative_error = 0.0
        if baseline_output is not None and optimized_output is not None:
            try:
                if isinstance(baseline_output, torch.Tensor) and isinstance(
                    optimized_output, torch.Tensor
                ):
                    relative_error = Metrics.compute_relative_error(
                        baseline_output, optimized_output
                    )
            except Exception:
                pass

        quality = {
            "coherence_score": float(accuracy_data.get("coherence_score", 0.0)),
            "relevance_score": float(accuracy_data.get("relevance_score", 0.0)),
            "fluency_score": float(accuracy_data.get("fluency_score", 0.0)),
        }

        self._accuracy_history.append(
            {
                "step_id": self._step_counter,
                "delta_percent": delta_percent,
                "bleu_drop": bleu_drop,
                "relative_error": relative_error,
            }
        )

        return {
            "perplexity": {
                "baseline": baseline_ppl,
                "optimized": optimized_ppl,
                "delta": ppl_drop,
                "delta_percent": delta_percent,
            },
            "bleu": {
                "baseline": baseline_bleu,
                "optimized": optimized_bleu,
                "drop": bleu_drop,
            },
            "generation_quality": quality,
            "relative_error": relative_error,
        }

    # ------------------------------------------------------------------
    # 告警检查
    # ------------------------------------------------------------------

    def _check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查告警条件。对应 PROFILER_AGENT.md 中告警规则。

        检查项：
        1. accuracy_warning (>0.5%)
        2. accuracy_critical (>1.0%)
        3. latency_warning (>预算 150%)
        4. memory_warning (>预算 90%)
        5. step_anomaly（accuracy 异常抖动）
        6. 连续 3 个 step 的 latency 递增（INFO）
        """
        alerts: List[Dict[str, Any]] = []
        step_id = metrics.get("step_id", -1)
        accuracy_metrics = metrics.get("accuracy_metrics", {})
        perf_metrics = metrics.get("performance_metrics", {})

        # --- accuracy drop ---
        ppl_delta_pct = abs(accuracy_metrics.get("perplexity", {}).get("delta_percent", 0.0))
        rel_err = abs(accuracy_metrics.get("relative_error", 0.0))
        accuracy_drop = max(ppl_delta_pct, rel_err)

        if accuracy_drop > self.alert_thresholds["accuracy_critical"]:
            msg = f"Accuracy drop exceeds 1.0% at step {step_id}: {accuracy_drop * 100:.2f}%"
            alerts.append(
                {
                    "level": "CRITICAL",
                    "message": msg,
                    "recommendation": "Immediately rollback to full precision mode.",
                    "metric": "accuracy_drop",
                    "value": accuracy_drop,
                    "step_id": step_id,
                }
            )
            self.logger.log_alert(
                "critical",
                msg,
                recommendation="Rollback to full precision.",
                step_id=step_id,
            )
        elif accuracy_drop > self.alert_thresholds["accuracy_warning"]:
            msg = f"Accuracy drop approaching 0.5% at step {step_id}: {accuracy_drop * 100:.2f}%"
            alerts.append(
                {
                    "level": "WARNING",
                    "message": msg,
                    "recommendation": "Consider reducing ShadowKV compression ratio.",
                    "metric": "accuracy_drop",
                    "value": accuracy_drop,
                    "step_id": step_id,
                }
            )
            self.logger.log_alert(
                "warning",
                msg,
                recommendation="Reduce compression ratio.",
                step_id=step_id,
            )

        # --- latency warning (>150% budget) ---
        latency_ms = perf_metrics.get("latency", {}).get("e2e_ms", 0.0)
        latency_budget = self.alert_thresholds["latency_budget_ms"]
        if (
            latency_budget > 0
            and latency_ms > latency_budget * self.alert_thresholds["latency_warning_ratio"]
        ):
            msg = (
                f"Latency exceeds 150% budget at step {step_id}: "
                f"{latency_ms:.1f}ms (budget: {latency_budget:.1f}ms)"
            )
            alerts.append(
                {
                    "level": "WARNING",
                    "message": msg,
                    "recommendation": (
                        "Notify optimization agent to adopt more aggressive strategy."
                    ),
                    "metric": "latency",
                    "value": latency_ms,
                    "step_id": step_id,
                }
            )
            self.logger.log_alert(
                "warning",
                msg,
                recommendation="Adopt more aggressive optimization.",
                step_id=step_id,
            )

        # --- memory warning (>90% budget) ---
        memory_mb = perf_metrics.get("memory", {}).get("allocated_mb", 0.0)
        memory_budget = self.alert_thresholds["memory_budget_mb"]
        if (
            memory_budget > 0
            and memory_mb > memory_budget * self.alert_thresholds["memory_warning_ratio"]
        ):
            msg = (
                f"Memory exceeds 90% budget at step {step_id}: "
                f"{memory_mb:.1f}MB (budget: {memory_budget:.1f}MB)"
            )
            alerts.append(
                {
                    "level": "WARNING",
                    "message": msg,
                    "recommendation": "Notify ShadowKV agent to increase compression ratio.",
                    "metric": "memory",
                    "value": memory_mb,
                    "step_id": step_id,
                }
            )
            self.logger.log_alert(
                "warning",
                msg,
                recommendation="Increase compression ratio.",
                step_id=step_id,
            )

        # --- accuracy anomaly jitter (step_anomaly) ---
        if len(self._accuracy_history) >= 2:
            prev = self._accuracy_history[-2]
            curr = self._accuracy_history[-1]
            jitter = abs(curr["delta_percent"] - prev["delta_percent"])
            if jitter > self.alert_thresholds["accuracy_jitter_threshold"]:
                msg = f"Accuracy anomaly jitter detected at step {step_id}: {jitter * 100:.3f}%"
                alerts.append(
                    {
                        "level": "WARNING",
                        "message": msg,
                        "recommendation": "Record anomalous step and recommend separate analysis.",
                        "metric": "accuracy_jitter",
                        "value": jitter,
                        "step_id": step_id,
                    }
                )
                self.logger.log_alert(
                    "warning",
                    msg,
                    recommendation="Analyze step separately.",
                    step_id=step_id,
                )

        # --- 连续 3 个 step latency 递增 ---
        if len(self._latency_history) >= 3:
            last_three = self._latency_history[-3:]
            if last_three[0] < last_three[1] < last_three[2]:
                msg = (
                    f"Latency increasing for 3 consecutive steps: "
                    f"{last_three[0]:.1f} -> {last_three[1]:.1f} -> {last_three[2]:.1f} ms"
                )
                alerts.append(
                    {
                        "level": "INFO",
                        "message": msg,
                        "recommendation": "Possible performance degradation trend.",
                        "metric": "latency_trend",
                        "value": last_three[2],
                        "step_id": step_id,
                    }
                )
                self.logger.log_alert(
                    "info",
                    msg,
                    recommendation="Monitor performance trend.",
                    step_id=step_id,
                )

        return alerts

    # ------------------------------------------------------------------
    # 报告生成与导出
    # ------------------------------------------------------------------

    def _generate_profile_result(self) -> ProfileResult:
        """生成 ProfileResult 对象，与 structs.py 字段一一对应。"""
        # latency 聚合
        latency_per_step: Dict[int, float] = {}
        total_latency = 0.0
        for sid, perf in self._perf_metrics_per_step.items():
            lat = perf.get("latency", {}).get("e2e_ms", 0.0)
            latency_per_step[sid] = lat
            total_latency += lat

        # throughput
        total_tokens = sum(
            p.get("throughput", {}).get("tokens_per_step", 0.0)
            for p in self._perf_metrics_per_step.values()
        )
        throughput = (total_tokens / total_latency * 1000.0) if total_latency > 0 else 0.0

        # accuracy 聚合
        avg_ppl_delta = 0.0
        avg_bleu_drop = 0.0
        if self._accuracy_metrics_per_step:
            ppl_deltas = [
                m.get("perplexity", {}).get("delta", 0.0)
                for m in self._accuracy_metrics_per_step.values()
            ]
            bleu_drops = [
                m.get("bleu", {}).get("drop", 0.0) for m in self._accuracy_metrics_per_step.values()
            ]
            avg_ppl_delta = sum(ppl_deltas) / len(ppl_deltas)
            avg_bleu_drop = sum(bleu_drops) / len(bleu_drops)

        # KV 聚合（取最后一步作为 snapshot，类型与 ProfileResult 匹配）
        kv_precision_distribution: Dict[int, Dict[int, str]] = {}
        kv_reuse_rate: Dict[int, float] = {}
        kv_memory_mb: Dict[int, float] = {}
        if self._kv_metrics_per_step:
            last_kv = list(self._kv_metrics_per_step.values())[-1]
            kv_precision_distribution = last_kv.get("precision_distribution", {}).get(
                "per_token_head", {}
            )
            kv_reuse_rate = last_kv.get("reuse_rate", {}).get("per_layer", {})
            kv_memory_mb = last_kv.get("memory_mb", {}).get("per_layer", {})

        # Q-drift 聚合
        q_drift_hit_rate: Dict[int, float] = {}
        activation_delta: Dict[int, Dict[str, float]] = {}
        for sid, qm in self._qdrift_metrics_per_step.items():
            q_drift_hit_rate[sid] = qm.get("step_hit_rate", {}).get("overall", 0.0)
            per_step_delta = qm.get("activation_delta", {}).get("per_step", {})
            for step_key, delta in per_step_delta.items():
                activation_delta[step_key] = delta

        # FFN 聚合
        ffn_compute_load: Dict[int, Dict[str, float]] = {}
        ffn_sparse_ratio = 0.0
        if self._ffn_metrics_per_step:
            all_ratios = []
            last_ffn = list(self._ffn_metrics_per_step.values())[-1]
            ffn_compute_load = last_ffn.get("compute_load", {}).get("per_layer", {})
            for fm in self._ffn_metrics_per_step.values():
                all_ratios.append(fm.get("sparse_update_ratio", {}).get("overall", 0.0))
            ffn_sparse_ratio = sum(all_ratios) / len(all_ratios) if all_ratios else 0.0

        return ProfileResult(
            model_name=self._model_name,
            run_id=self._run_id,
            kv_precision_distribution=kv_precision_distribution,
            kv_reuse_rate=kv_reuse_rate,
            kv_memory_mb=kv_memory_mb,
            q_drift_hit_rate=q_drift_hit_rate,
            activation_delta=activation_delta,
            ffn_compute_load=ffn_compute_load,
            ffn_sparse_update_ratio=ffn_sparse_ratio,
            latency_e2e_ms=total_latency,
            latency_per_step_ms=latency_per_step,
            throughput_tokens_per_sec=throughput,
            perplexity_delta=avg_ppl_delta,
            bleu_drop=avg_bleu_drop,
            accuracy_metrics={
                "avg_perplexity_delta": avg_ppl_delta,
                "avg_bleu_drop": avg_bleu_drop,
            },
        )

    def _export_baseline(self, filepath: str) -> None:
        """导出基线数据到 JSON。输出 profile_baseline.json。"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.baseline_data, f, ensure_ascii=False, indent=2, default=str)
        self.logger.log_event("export", f"Baseline exported to {filepath}")

    def _export_optimized(self, filepath: str) -> None:
        """导出优化后数据到 JSON。输出 profile_optimized.json。"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.optimized_data, f, ensure_ascii=False, indent=2, default=str)
        self.logger.log_event("export", f"Optimized data exported to {filepath}")

    def _generate_comparison_html(
        self, baseline: Dict[str, Any], optimized: Dict[str, Any], output_path: str
    ) -> None:
        """生成 HTML 可视化对比报告。输出 profile_comparison.html。"""
        reporter = HTMLReporter()
        reporter.generate(baseline, optimized, output_path)
        self.logger.log_event("export", f"Comparison HTML generated at {output_path}")

    def _generate_summary(
        self, baseline: Dict[str, Any], optimized: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成关键指标摘要。"""
        baseline_latency = baseline.get("latency", {}).get("e2e_ms", 0.0)
        optimized_latency = optimized.get("latency", {}).get("e2e_ms", 0.0)
        speedup = baseline_latency / optimized_latency if optimized_latency > 0 else 0.0

        baseline_kv = baseline.get("kv_cache", {}).get("memory_mb", {})
        optimized_kv = optimized.get("kv_cache", {}).get("memory_mb", {})
        baseline_kv_mem = sum(baseline_kv.values()) if isinstance(baseline_kv, dict) else 0.0
        optimized_kv_mem = sum(optimized_kv.values()) if isinstance(optimized_kv, dict) else 0.0
        memory_savings = 1.0 - (optimized_kv_mem / baseline_kv_mem) if baseline_kv_mem > 0 else 0.0

        ppl_delta = optimized.get("accuracy", {}).get("perplexity_delta", 0.0)
        bleu_drop = optimized.get("accuracy", {}).get("bleu_drop", 0.0)

        alerts = optimized.get("alerts", [])
        critical_count = sum(1 for a in alerts if a.get("level") == "CRITICAL")
        warning_count = sum(1 for a in alerts if a.get("level") == "WARNING")
        info_count = sum(1 for a in alerts if a.get("level") == "INFO")

        return {
            "model": self._model_name,
            "run_id": self._run_id,
            "total_steps": optimized.get("total_steps", 0),
            "speedup_ratio": speedup,
            "memory_savings_ratio": memory_savings,
            "perplexity_delta": ppl_delta,
            "bleu_drop": bleu_drop,
            "alerts": {
                "critical": critical_count,
                "warning": warning_count,
                "info": info_count,
                "total": len(alerts),
            },
            "recommendations": self._generate_recommendations(alerts),
        }

    def _generate_recommendations(self, alerts: List[Dict[str, Any]]) -> List[str]:
        """基于告警生成建议。"""
        recommendations = []
        has_critical = any(a.get("level") == "CRITICAL" for a in alerts)
        has_acc_warn = any(
            a.get("metric") == "accuracy_drop" and a.get("level") == "WARNING" for a in alerts
        )
        has_lat_warn = any(a.get("metric") == "latency" for a in alerts)
        has_mem_warn = any(a.get("metric") == "memory" for a in alerts)

        if has_critical:
            recommendations.append("Immediately rollback to full precision mode.")
        if has_acc_warn:
            recommendations.append(
                "Reduce ShadowKV compression ratio or increase precision for critical layers."
            )
        if has_lat_warn:
            recommendations.append(
                "Adopt more aggressive optimization strategies "
                "(e.g., sparse FFN, aggressive KV reuse)."
            )
        if has_mem_warn:
            recommendations.append(
                "Increase KV cache compression ratio or enable more aggressive pruning."
            )
        if not recommendations:
            recommendations.append("Optimization is within acceptable bounds. Continue monitoring.")

        return recommendations

    # ------------------------------------------------------------------
    # 数据聚合辅助
    # ------------------------------------------------------------------

    def _aggregate_step_data(self) -> Dict[str, Any]:
        """聚合所有 step 数据为优化数据字典（匹配 ARCHITECTURE.md §2.1 输出格式）。"""
        all_alerts = []
        for alerts in self._alerts_per_step.values():
            all_alerts.extend(alerts)

        return {
            "model": self._model_name,
            "run_id": self._run_id,
            "total_steps": self._step_counter,
            "kv_cache": {
                "precision_distribution": self._aggregate_kv_precision(),
                "reuse_rate": self._aggregate_kv_reuse(),
                "memory_mb": self._aggregate_kv_memory(),
            },
            "q_drift": {
                "step_hit_rate": self._aggregate_qdrift_hit_rate(),
                "activation_delta": self._aggregate_qdrift_delta(),
            },
            "ffn": {
                "compute_load": self._aggregate_ffn_compute(),
                "sparse_update_ratio": self._aggregate_ffn_sparse(),
            },
            "latency": {
                "e2e_ms": sum(
                    p.get("latency", {}).get("e2e_ms", 0.0)
                    for p in self._perf_metrics_per_step.values()
                ),
                "per_step_ms": {
                    k: v.get("latency", {}).get("e2e_ms", 0.0)
                    for k, v in self._perf_metrics_per_step.items()
                },
            },
            "accuracy": {
                "perplexity_delta": self._aggregate_accuracy_ppl_delta(),
                "bleu_drop": self._aggregate_accuracy_bleu_drop(),
            },
            "alerts": all_alerts,
        }

    def _aggregate_kv_precision(self) -> Dict[int, Dict[int, str]]:
        if not self._kv_metrics_per_step:
            return {}
        last = list(self._kv_metrics_per_step.values())[-1]
        return last.get("precision_distribution", {}).get("per_token_head", {})

    def _aggregate_kv_reuse(self) -> Dict[int, float]:
        if not self._kv_metrics_per_step:
            return {}
        last = list(self._kv_metrics_per_step.values())[-1]
        return last.get("reuse_rate", {}).get("per_layer", {})

    def _aggregate_kv_memory(self) -> Dict[int, float]:
        if not self._kv_metrics_per_step:
            return {}
        last = list(self._kv_metrics_per_step.values())[-1]
        return last.get("memory_mb", {}).get("per_layer", {})

    def _aggregate_qdrift_hit_rate(self) -> Dict[int, float]:
        result = {}
        for sid, m in self._qdrift_metrics_per_step.items():
            result[sid] = m.get("step_hit_rate", {}).get("overall", 0.0)
        return result

    def _aggregate_qdrift_delta(self) -> Dict[int, Dict[str, float]]:
        result = {}
        for sid, m in self._qdrift_metrics_per_step.items():
            per_step = m.get("activation_delta", {}).get("per_step", {})
            for step_key, delta in per_step.items():
                result[step_key] = delta
        return result

    def _aggregate_ffn_compute(self) -> Dict[int, Dict[str, float]]:
        if not self._ffn_metrics_per_step:
            return {}
        last = list(self._ffn_metrics_per_step.values())[-1]
        return last.get("compute_load", {}).get("per_layer", {})

    def _aggregate_ffn_sparse(self) -> float:
        if not self._ffn_metrics_per_step:
            return 0.0
        ratios = [
            m.get("sparse_update_ratio", {}).get("overall", 0.0)
            for m in self._ffn_metrics_per_step.values()
        ]
        return sum(ratios) / len(ratios) if ratios else 0.0

    def _aggregate_accuracy_ppl_delta(self) -> float:
        if not self._accuracy_metrics_per_step:
            return 0.0
        deltas = [
            m.get("perplexity", {}).get("delta", 0.0)
            for m in self._accuracy_metrics_per_step.values()
        ]
        return sum(deltas) / len(deltas) if deltas else 0.0

    def _aggregate_accuracy_bleu_drop(self) -> float:
        if not self._accuracy_metrics_per_step:
            return 0.0
        drops = [
            m.get("bleu", {}).get("drop", 0.0) for m in self._accuracy_metrics_per_step.values()
        ]
        return sum(drops) / len(drops) if drops else 0.0

    # ------------------------------------------------------------------
    # 外部注入接口
    # ------------------------------------------------------------------

    def attach_bus(self, bus: ProfilingBus) -> None:
        """注入 ProfilingBus 实例，用于 step 内消息发布。"""
        self._bus = bus
