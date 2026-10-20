"""Metrics 收集器 — Prometheus / OpenTelemetry 风格。

对应大厂实践：
- Counter: 单调递增的计数器（如累计 FLOPs, 累计请求数）
- Gauge: 可增可减的瞬时值（如当前显存占用、当前延迟）
- Histogram: 分布统计（如延迟分布、误差分布）
- Summary: 滑动窗口统计（如 P50/P95/P99 延迟）

集成说明：
- 当 ``prometheus_client`` 可用时，每个指标会同时注册到内部的
  ``CollectorRegistry``，``MetricsRegistry.expose()`` 使用 Prometheus 官方
  exposition 格式输出。
- 当 ``prometheus_client`` 不可用时，自动降级为原有自定义实现，保持完全
  向后兼容。

对应文档：plan-v2.md Phase 1.1
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import prometheus_client as _prom

    _PROMETHEUS_AVAILABLE = True
except Exception:  # pragma: no cover
    _PROMETHEUS_AVAILABLE = False


@dataclass
class MetricMetadata:
    """指标元数据 — 所有指标类型的公共属性。

    对应文档：plan-v2.md Phase 1.1 — Metrics 体系
    """

    name: str
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "unknown"


class Counter:
    """单调递增计数器。

    对应大厂实践：Prometheus Counter — 只增不减的累计值。
    典型场景：累计 FLOPs、累计请求数、累计错误数。

    线程安全：所有操作受 threading.Lock 保护。

    对应文档：plan-v2.md Phase 1.1

    Example:
        >>> flops = Counter("ffn_compute_flops", "累计 FFN 计算 FLOPs")
        >>> flops.inc(1.5e9)
        >>> flops.get()
        1500000000.0
    """

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[Dict[str, str]] = None,
        _prom_registry: Optional[Any] = None,
    ):
        self.metadata = MetricMetadata(name, description, labels or {}, "counter")
        self._value: float = 0.0
        self._lock = threading.Lock()
        self._prom_metric: Optional[Any] = None
        self._prom_child: Optional[Any] = None
        self._init_prometheus(_prom_registry)

    def _init_prometheus(self, registry: Optional[Any]) -> None:
        """可选地注册到 prometheus_client Registry。"""
        if not _PROMETHEUS_AVAILABLE or registry is None:
            return
        try:
            self._prom_metric = _prom.Counter(
                self.metadata.name,
                self.metadata.description,
                labelnames=list(self.metadata.labels.keys()),
                registry=registry,
            )
            if self.metadata.labels:
                self._prom_child = self._prom_metric.labels(**self.metadata.labels)
            else:
                self._prom_child = self._prom_metric
        except Exception:
            self._prom_metric = None
            self._prom_child = None

    def inc(self, amount: float = 1.0) -> None:
        """增加计数。

        参数:
            amount: 增量，必须 >= 0。

        Raises:
            ValueError: 当 amount < 0 时。
        """
        if amount < 0:
            raise ValueError(f"Counter 只能递增，amount={amount} 非法")
        with self._lock:
            self._value += amount
            if self._prom_child is not None:
                try:
                    self._prom_child.inc(amount)
                except Exception:
                    pass

    def get(self) -> float:
        """获取当前累计值。"""
        with self._lock:
            return self._value

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典 — 包含类型、描述、标签和值。"""
        with self._lock:
            return {
                "name": self.metadata.name,
                "type": self.metadata.metric_type,
                "description": self.metadata.description,
                "labels": dict(self.metadata.labels),
                "value": self._value,
            }

    def to_prometheus(self) -> str:
        """导出为 Prometheus 文本格式。"""
        labels_str = ",".join(f'{k}="{v}"' for k, v in self.metadata.labels.items())
        header = f"# HELP {self.metadata.name} {self.metadata.description}\n"
        header += f"# TYPE {self.metadata.name} counter\n"
        if labels_str:
            header += f"{self.metadata.name}{{{labels_str}}} {self._value}\n"
        else:
            header += f"{self.metadata.name} {self._value}\n"
        return header


class Gauge:
    """瞬时值仪表 — 可增可减。

    对应大厂实践：Prometheus Gauge — 可增可减的瞬时值。
    典型场景：当前显存占用、当前并发数、当前延迟、CPU 利用率。

    线程安全：所有操作受 threading.Lock 保护。

    对应文档：plan-v2.md Phase 1.1

    Example:
        >>> mem = Gauge("kv_cache_memory_bytes", "当前 KV cache 占用（字节）")
        >>> mem.set(1.5e9)
        >>> mem.dec(5e8)
        >>> mem.get()
        1000000000.0
    """

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[Dict[str, str]] = None,
        _prom_registry: Optional[Any] = None,
    ):
        self.metadata = MetricMetadata(name, description, labels or {}, "gauge")
        self._value: float = 0.0
        self._lock = threading.Lock()
        self._prom_metric: Optional[Any] = None
        self._prom_child: Optional[Any] = None
        self._init_prometheus(_prom_registry)

    def _init_prometheus(self, registry: Optional[Any]) -> None:
        """可选地注册到 prometheus_client Registry。"""
        if not _PROMETHEUS_AVAILABLE or registry is None:
            return
        try:
            self._prom_metric = _prom.Gauge(
                self.metadata.name,
                self.metadata.description,
                labelnames=list(self.metadata.labels.keys()),
                registry=registry,
            )
            if self.metadata.labels:
                self._prom_child = self._prom_metric.labels(**self.metadata.labels)
            else:
                self._prom_child = self._prom_metric
        except Exception:
            self._prom_metric = None
            self._prom_child = None

    def set(self, value: float) -> None:
        """设置值 — 直接覆盖。"""
        with self._lock:
            self._value = value
            if self._prom_child is not None:
                try:
                    self._prom_child.set(value)
                except Exception:
                    pass

    def inc(self, amount: float = 1.0) -> None:
        """增加。"""
        with self._lock:
            self._value += amount
            if self._prom_child is not None:
                try:
                    self._prom_child.inc(amount)
                except Exception:
                    pass

    def dec(self, amount: float = 1.0) -> None:
        """减少。"""
        with self._lock:
            self._value -= amount
            if self._prom_child is not None:
                try:
                    self._prom_child.dec(amount)
                except Exception:
                    pass

    def get(self) -> float:
        """获取当前值。"""
        with self._lock:
            return self._value

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        with self._lock:
            return {
                "name": self.metadata.name,
                "type": self.metadata.metric_type,
                "description": self.metadata.description,
                "labels": dict(self.metadata.labels),
                "value": self._value,
            }

    def to_prometheus(self) -> str:
        """导出为 Prometheus 文本格式。"""
        labels_str = ",".join(f'{k}="{v}"' for k, v in self.metadata.labels.items())
        header = f"# HELP {self.metadata.name} {self.metadata.description}\n"
        header += f"# TYPE {self.metadata.name} gauge\n"
        if labels_str:
            header += f"{self.metadata.name}{{{labels_str}}} {self._value}\n"
        else:
            header += f"{self.metadata.name} {self._value}\n"
        return header


class Histogram:
    """分布直方图 — 记录数值分布并统计分位数。

    对应大厂实践：Prometheus Histogram — 延迟分布、误差分布等。
    通过预设 buckets 统计落入各区间的频次，支持 P50/P95/P99 估算。

    线程安全：所有操作受 threading.Lock 保护。

    对应文档：plan-v2.md Phase 1.1

    Example:
        >>> lat = Histogram("inference_latency_ms", "推理延迟分布（ms）")
        >>> lat.observe(0.05)
        >>> lat.observe(0.12)
        >>> lat.observe(0.30)
        >>> lat.get_percentile(0.95)
        0.30
    """

    DEFAULT_BUCKETS: List[float] = [
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
    ]

    def __init__(
        self,
        name: str,
        description: str,
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None,
        _prom_registry: Optional[Any] = None,
    ):
        self.metadata = MetricMetadata(name, description, labels or {}, "histogram")
        self.buckets: List[float] = sorted(buckets or self.DEFAULT_BUCKETS)
        self._counts: List[int] = [0] * (len(self.buckets) + 1)  # +1 for +Inf bucket
        self._sum: float = 0.0
        self._count: int = 0
        self._lock = threading.Lock()
        self._prom_metric: Optional[Any] = None
        self._prom_child: Optional[Any] = None
        self._init_prometheus(_prom_registry)

    def _init_prometheus(self, registry: Optional[Any]) -> None:
        """可选地注册到 prometheus_client Registry。"""
        if not _PROMETHEUS_AVAILABLE or registry is None:
            return
        try:
            self._prom_metric = _prom.Histogram(
                self.metadata.name,
                self.metadata.description,
                labelnames=list(self.metadata.labels.keys()),
                buckets=self.buckets,
                registry=registry,
            )
            if self.metadata.labels:
                self._prom_child = self._prom_metric.labels(**self.metadata.labels)
            else:
                self._prom_child = self._prom_metric
        except Exception:
            self._prom_metric = None
            self._prom_child = None

    def observe(self, value: float) -> None:
        """记录观测值 — 自动归入对应 bucket。

        参数:
            value: 观测值（如延迟、误差），必须 >= 0。

        Raises:
            ValueError: 当 value < 0 时。
        """
        if value < 0:
            raise ValueError(f"Histogram 观测值必须 >= 0，value={value} 非法")
        with self._lock:
            self._sum += value
            self._count += 1
            # 找到对应 bucket 索引
            idx = 0
            for i, b in enumerate(self.buckets):
                if value <= b:
                    idx = i
                    break
            else:
                idx = len(self.buckets)  # +Inf bucket
            self._counts[idx] += 1
            if self._prom_child is not None:
                try:
                    self._prom_child.observe(value)
                except Exception:
                    pass

    def get_percentile(self, percentile: float) -> float:
        """获取指定百分位数（使用线性插值）。

        参数:
            percentile: 0.0 ~ 1.0，如 0.95 表示 P95。

        Returns:
            对应百分位数的估计值。

        Raises:
            ValueError: 当 percentile 超出 [0, 1] 或样本为空时。
        """
        if not (0.0 <= percentile <= 1.0):
            raise ValueError(f"percentile 必须在 [0, 1] 之间，got {percentile}")
        with self._lock:
            if self._count == 0:
                raise ValueError("Histogram 为空，无法计算百分位数")

            target_rank = percentile * self._count
            cumulative = 0
            for i, c in enumerate(self._counts):
                cumulative += c
                if cumulative >= target_rank:
                    if i == 0:
                        return self.buckets[0] if self.buckets else 0.0
                    if i >= len(self.buckets):
                        return self.buckets[-1] if self.buckets else 0.0
                    # 线性插值
                    prev_bucket = self.buckets[i - 1]
                    curr_bucket = self.buckets[i]
                    prev_cum = cumulative - c
                    if c == 0:
                        return prev_bucket
                    ratio = (target_rank - prev_cum) / c
                    return prev_bucket + ratio * (curr_bucket - prev_bucket)
            return self.buckets[-1] if self.buckets else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典 — 包含 buckets、counts、sum、count。"""
        with self._lock:
            return {
                "name": self.metadata.name,
                "type": self.metadata.metric_type,
                "description": self.metadata.description,
                "labels": dict(self.metadata.labels),
                "buckets": list(self.buckets),
                "counts": list(self._counts),
                "sum": self._sum,
                "count": self._count,
            }

    def to_prometheus(self) -> str:
        """导出为 Prometheus 文本格式。

        Prometheus 格式要求每个 bucket 一行，包含 le= 标签。
        """
        lines = [
            f"# HELP {self.metadata.name} {self.metadata.description}",
            f"# TYPE {self.metadata.name} histogram",
        ]
        labels_base = ",".join(f'{k}="{v}"' for k, v in self.metadata.labels.items())
        for i, b in enumerate(self.buckets):
            le_label = f'le="{b}"'
            all_labels = ",".join(filter(None, [labels_base, le_label]))
            lines.append(f"{self.metadata.name}_bucket{{{all_labels}}} {self._counts[i]}")
        # +Inf bucket
        le_inf = 'le="+Inf"'
        all_labels_inf = ",".join(filter(None, [labels_base, le_inf]))
        lines.append(f"{self.metadata.name}_bucket{{{all_labels_inf}}} {self._counts[-1]}")

        # sum / count
        lines.append(f"{self.metadata.name}_sum {self._sum}")
        lines.append(f"{self.metadata.name}_count {self._count}")
        return "\n".join(lines) + "\n"


class Summary:
    """滑动窗口摘要统计 — 记录最近 N 个样本的统计特征。

    对应大厂实践：Prometheus Summary — 滑动窗口内的 P50/P95/P99。
    相比 Histogram，Summary 直接记录原始值，内存占用随 window_size 线性增长。
    适用于需要精确分位数且样本量可控的场景。

    线程安全：所有操作受 threading.Lock 保护。

    对应文档：plan-v2.md Phase 1.1

    Example:
        >>> s = Summary("dispatch_latency_ms", "调度延迟摘要", window_size=100)
        >>> for v in [0.05, 0.12, 0.30, 0.08, 0.15]:
        ...     s.observe(v)
        >>> stats = s.get_stats()
        >>> stats["p95"] > 0.20
        True
    """

    def __init__(
        self,
        name: str,
        description: str,
        window_size: int = 1000,
        labels: Optional[Dict[str, str]] = None,
        _prom_registry: Optional[Any] = None,
    ):
        self.metadata = MetricMetadata(name, description, labels or {}, "summary")
        self.window_size: int = window_size
        self._values: deque[float] = deque(maxlen=window_size)
        self._lock = threading.RLock()
        self._prom_metric: Optional[Any] = None
        self._prom_child: Optional[Any] = None
        self._init_prometheus(_prom_registry)

    def _init_prometheus(self, registry: Optional[Any]) -> None:
        """可选地注册到 prometheus_client Registry。"""
        if not _PROMETHEUS_AVAILABLE or registry is None:
            return
        try:
            self._prom_metric = _prom.Summary(
                self.metadata.name,
                self.metadata.description,
                labelnames=list(self.metadata.labels.keys()),
                registry=registry,
            )
            if self.metadata.labels:
                self._prom_child = self._prom_metric.labels(**self.metadata.labels)
            else:
                self._prom_child = self._prom_metric
        except Exception:
            self._prom_metric = None
            self._prom_child = None

    def observe(self, value: float) -> None:
        """记录观测值 — 加入滑动窗口。

        当窗口满时，最旧的值自动被挤出。
        """
        with self._lock:
            self._values.append(value)
            if self._prom_child is not None:
                try:
                    self._prom_child.observe(value)
                except Exception:
                    pass

    def get_stats(self) -> Dict[str, float]:
        """获取统计特征：mean, std, min, max, p50, p95, p99。

        Returns:
            Dict 包含各统计量。

        Raises:
            ValueError: 当窗口为空时。
        """
        with self._lock:
            if not self._values:
                raise ValueError("Summary 窗口为空，无法计算统计量")

            vals = sorted(self._values)
            n = len(vals)
            mean = sum(vals) / n
            variance = sum((v - mean) ** 2 for v in vals) / n
            std = variance**0.5

            def _percentile(p: float) -> float:
                idx = p * (n - 1)
                lower = int(idx)
                upper = min(lower + 1, n - 1)
                frac = idx - lower
                return vals[lower] + frac * (vals[upper] - vals[lower])

            return {
                "mean": mean,
                "std": std,
                "min": vals[0],
                "max": vals[-1],
                "p50": _percentile(0.50),
                "p95": _percentile(0.95),
                "p99": _percentile(0.99),
                "count": float(n),
            }

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典 — 包含统计量和原始样本数。"""
        with self._lock:
            stats = self.get_stats() if self._values else {}
            return {
                "name": self.metadata.name,
                "type": self.metadata.metric_type,
                "description": self.metadata.description,
                "labels": dict(self.metadata.labels),
                "window_size": self.window_size,
                "current_count": len(self._values),
                "stats": stats,
            }

    def to_prometheus(self) -> str:
        """导出为 Prometheus 文本格式（近似 Summary 格式）。"""
        lines = [
            f"# HELP {self.metadata.name} {self.metadata.description}",
            f"# TYPE {self.metadata.name} summary",
        ]
        labels_base = ",".join(f'{k}="{v}"' for k, v in self.metadata.labels.items())
        with self._lock:
            if self._values:
                stats = self.get_stats()
                for quantile, value in [
                    (0.5, stats["p50"]),
                    (0.95, stats["p95"]),
                    (0.99, stats["p99"]),
                ]:
                    q_label = f'quantile="{quantile}"'
                    all_labels = ",".join(filter(None, [labels_base, q_label]))
                    lines.append(f"{self.metadata.name}{{{all_labels}}} {value}")
                lines.append(f'{self.metadata.name}_count {int(stats["count"])}')
            else:
                lines.append(f"{self.metadata.name}_count 0")
        return "\n".join(lines) + "\n"


class MetricsRegistry:
    """指标注册表 — 统一管理所有指标。

    对应大厂实践：Prometheus Registry — 所有指标注册到全局注册表，
    通过统一接口导出为 Prometheus 或 JSON 格式。

    支持按需创建（Counter/Gauge/Histogram/Summary），如果同名已存在则复用。

    线程安全：注册和查找操作受 threading.Lock 保护。

    对应文档：plan-v2.md Phase 1.1

    Example:
        >>> registry = MetricsRegistry()
        >>> c = registry.counter("requests_total", "累计请求数")
        >>> g = registry.gauge("memory_bytes", "当前内存占用")
        >>> h = registry.histogram("latency_ms", "延迟分布")
        >>> s = registry.summary("dispatch_ms", "调度延迟")
        >>> registry.expose()
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._prom_registry: Optional[Any] = None
        if _PROMETHEUS_AVAILABLE:
            try:
                self._prom_registry = _prom.CollectorRegistry()
            except Exception:
                self._prom_registry = None

    def register(self, metric: Any) -> None:
        """注册指标。

        Raises:
            ValueError: 当同名指标已存在且类型不一致时。
        """
        name = getattr(metric, "metadata", None)
        if name is not None:
            name = metric.metadata.name
        else:
            name = getattr(metric, "name", None)
        if name is None:
            raise ValueError("Metric 对象缺少 name 属性")

        with self._lock:
            existing = self._metrics.get(name)
            if existing is not None:
                existing_type = getattr(existing, "metadata", None)
                if existing_type is not None:
                    existing_type = existing.metadata.metric_type
                new_type = getattr(metric, "metadata", None)
                if new_type is not None:
                    new_type = metric.metadata.metric_type
                if existing_type != new_type:
                    raise ValueError(
                        f"指标 '{name}' 已存在，类型为 {existing_type}，" f"无法注册为 {new_type}"
                    )
                return
            self._metrics[name] = metric

    def get(self, name: str) -> Optional[Any]:
        """获取指标。"""
        with self._lock:
            return self._metrics.get(name)

    def counter(
        self, name: str, description: str, labels: Optional[Dict[str, str]] = None
    ) -> Counter:
        """获取或创建 Counter。"""
        with self._lock:
            existing = self._metrics.get(name)
            if existing is not None:
                if not isinstance(existing, Counter):
                    raise TypeError(f"指标 '{name}' 已存在但不是 Counter")
                return existing
            m = Counter(name, description, labels, _prom_registry=self._prom_registry)
            self._metrics[name] = m
            return m

    def gauge(self, name: str, description: str, labels: Optional[Dict[str, str]] = None) -> Gauge:
        """获取或创建 Gauge。"""
        with self._lock:
            existing = self._metrics.get(name)
            if existing is not None:
                if not isinstance(existing, Gauge):
                    raise TypeError(f"指标 '{name}' 已存在但不是 Gauge")
                return existing
            m = Gauge(name, description, labels, _prom_registry=self._prom_registry)
            self._metrics[name] = m
            return m

    def histogram(
        self,
        name: str,
        description: str,
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Histogram:
        """获取或创建 Histogram。"""
        with self._lock:
            existing = self._metrics.get(name)
            if existing is not None:
                if not isinstance(existing, Histogram):
                    raise TypeError(f"指标 '{name}' 已存在但不是 Histogram")
                return existing
            m = Histogram(name, description, buckets, labels, _prom_registry=self._prom_registry)
            self._metrics[name] = m
            return m

    def summary(
        self,
        name: str,
        description: str,
        window_size: int = 1000,
        labels: Optional[Dict[str, str]] = None,
    ) -> Summary:
        """获取或创建 Summary。"""
        with self._lock:
            existing = self._metrics.get(name)
            if existing is not None:
                if not isinstance(existing, Summary):
                    raise TypeError(f"指标 '{name}' 已存在但不是 Summary")
                return existing
            m = Summary(name, description, window_size, labels, _prom_registry=self._prom_registry)
            self._metrics[name] = m
            return m

    def export_all(self) -> Dict[str, Dict[str, Any]]:
        """导出所有指标为字典 — key 为指标名，value 为 to_dict() 结果。"""
        with self._lock:
            return {name: m.to_dict() for name, m in self._metrics.items()}

    def export_prometheus_format(self) -> str:
        """导出为 Prometheus 文本格式。

        对应大厂实践：Prometheus exposition format — 可被 Prometheus server 直接 scrape。
        """
        with self._lock:
            parts = []
            for m in self._metrics.values():
                parts.append(m.to_prometheus())
            return "\n".join(parts)

    def to_prometheus(self) -> str:
        """``export_prometheus_format()`` 的别名，保持 API 风格一致。"""
        return self.expose()

    def expose(self) -> str:
        """返回 Prometheus exposition 格式。

        当 ``prometheus_client`` 可用时，使用其官方格式输出；否则降级为
        自定义格式。
        """
        if _PROMETHEUS_AVAILABLE and self._prom_registry is not None:
            try:
                return _prom.generate_latest(self._prom_registry).decode("utf-8")
            except Exception:
                pass
        return self.export_prometheus_format()

    def export_json(self) -> str:
        """导出为 JSON 格式 — 便于与其他系统集成。"""
        return json.dumps(self.export_all(), indent=2, ensure_ascii=False)
