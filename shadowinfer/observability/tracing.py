"""Tracing 系统 — 记录推理调用链。

对应大厂实践：OpenTelemetry / Jaeger 风格分布式追踪。
每个 denoising step 是一个 Trace，包含多个 Span：
- Span: qdrift.evaluate (耗时, 输入/输出大小)
- Span: shadowkv.compress (耗时, 压缩前后大小)
- Span: ffn.compute (耗时, 计算路径, FLOPs)
- Span: profiler.record (耗时, 指标数量)

集成说明：
- 当 ``opentelemetry-api`` / ``opentelemetry-sdk`` 可用时，``Tracer`` 可以
  同时创建真实的 OTel Span 并导出到 OTLP 或 Console。
- 当 OTel 不可用时，自动降级为原有的内存 Tracer，保持完全向后兼容。

对应文档：plan-v2.md Phase 1.2
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.sdk.resources import Resource as _OTelResource
    from opentelemetry.sdk.trace import TracerProvider as _OTelTracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor as _BatchSpanProcessor
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter as _ConsoleSpanExporter

    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover
    _OTEL_AVAILABLE = False

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as _OTLPGrpcSpanExporter,
    )

    _OTEL_GRPC_AVAILABLE = True
except Exception:  # pragma: no cover
    _OTEL_GRPC_AVAILABLE = False

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as _OTLPHttpSpanExporter,
    )

    _OTEL_HTTP_AVAILABLE = True
except Exception:  # pragma: no cover
    _OTEL_HTTP_AVAILABLE = False


@dataclass
class Span:
    """追踪中的单个操作 — 对应 OpenTelemetry Span 概念。

    属性:
        trace_id: Trace 唯一标识
        span_id: Span 唯一标识
        parent_id: 父 Span ID（None 表示 root span）
        name: Span 名称（如 "shadowkv.compress"）
        start_time: 开始时间（time.time()）
        end_time: 结束时间（time.time()），None 表示未结束
        attributes: 附加属性（如 {"input_size": 1024, "output_size": 512}）
        status: 状态 — "OK", "ERROR", "WARNING"
        status_message: 状态消息（如错误描述）

    对应文档：plan-v2.md Phase 1.2

    Example:
        >>> span = Span(
        ...     trace_id="trace-001",
        ...     span_id="span-001",
        ...     parent_id=None,
        ...     name="qdrift.evaluate",
        ...     start_time=time.time(),
        ... )
        >>> span.set_attribute("sensitivity_score", 0.85)
        >>> span.finish()
        >>> span.duration_ms is not None
        True
    """

    trace_id: str
    span_id: str
    parent_id: Optional[str]
    name: str
    start_time: float
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "OK"
    status_message: Optional[str] = None

    def set_attribute(self, key: str, value: Any) -> None:
        """设置属性 — 记录与 Span 相关的任意键值对。

        典型属性：
        - input_size / output_size: 数据大小
        - flops: 计算量
        - layer_id: 层 ID
        - step_id: denoising step ID
        """
        self.attributes[key] = value

    def set_status(self, status: str, message: Optional[str] = None) -> None:
        """设置状态 — 标记 Span 执行结果。

        参数:
            status: "OK" | "ERROR" | "WARNING"
            message: 可选的状态消息

        Raises:
            ValueError: 当 status 不是合法值时。
        """
        if status not in ("OK", "ERROR", "WARNING"):
            raise ValueError(f"status 必须是 OK/ERROR/WARNING，got {status}")
        self.status = status
        self.status_message = message

    def finish(self, end_time: Optional[float] = None) -> None:
        """结束 Span — 记录结束时间。

        如果 end_time 未提供，使用当前时间。
        """
        self.end_time = end_time if end_time is not None else time.time()

    @property
    def duration_ms(self) -> Optional[float]:
        """持续时间（ms）— 从开始到结束的毫秒数。

        Returns:
            毫秒数，如果 Span 未结束则返回 None。
        """
        if self.end_time is None or self.start_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典 — 包含完整 Span 信息。"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": dict(self.attributes),
            "status": self.status,
            "status_message": self.status_message,
        }


class TracerProvider:
    """OpenTelemetry TracerProvider 包装器。

    当 OTel 可用时，负责构建 Provider、注册 exporter，并将其设置为全局
    Provider。当 OTel 不可用时，该包装器为空操作，不影响内存 Tracer。

    Example:
        >>> provider = TracerProvider(exporter=ConsoleSpanExporter())
        >>> tracer = Tracer("shadowinfer")
        >>> tracer.set_otel_exporter(provider=provider)
    """

    def __init__(
        self,
        exporter: Optional[Any] = None,
        service_name: str = "shadowinfer",
    ) -> None:
        self.service_name = service_name
        self._provider: Optional[Any] = None
        self._exporter = exporter
        if _OTEL_AVAILABLE:
            try:
                resource = _OTelResource.create({"service.name": service_name})
                self._provider = _OTelTracerProvider(resource=resource)
                exporter = exporter or _ConsoleSpanExporter()
                self._provider.add_span_processor(_BatchSpanProcessor(exporter))
                _otel_trace.set_tracer_provider(self._provider)
            except Exception:
                self._provider = None

    def get_tracer(self, name: Optional[str] = None) -> Optional[Any]:
        """获取 OTel Tracer 实例。"""
        if not _OTEL_AVAILABLE or self._provider is None:
            return None
        try:
            return _otel_trace.get_tracer(name or self.service_name)
        except Exception:
            return None

    def shutdown(self) -> None:
        """关闭 Provider 并刷新 exporter。"""
        if self._provider is not None and hasattr(self._provider, "shutdown"):
            try:
                self._provider.shutdown()
            except Exception:
                pass


def _make_otlp_exporter(endpoint: Optional[str] = None, protocol: str = "grpc") -> Optional[Any]:
    """便捷工厂函数：创建 OTLP exporter（grpc 或 http）。"""
    if not _OTEL_AVAILABLE:
        return None
    if protocol == "grpc" and _OTEL_GRPC_AVAILABLE:
        try:
            kwargs: Dict[str, Any] = {}
            if endpoint is not None:
                kwargs["endpoint"] = endpoint
            return _OTLPGrpcSpanExporter(**kwargs)
        except Exception:
            pass
    if _OTEL_HTTP_AVAILABLE:
        try:
            kwargs = {}
            if endpoint is not None:
                kwargs["endpoint"] = endpoint
            return _OTLPHttpSpanExporter(**kwargs)
        except Exception:
            pass
    return None


class Tracer:
    """追踪器 — 创建和管理 Traces 和 Spans。

    对应大厂实践：OpenTelemetry Tracer — 生成 Trace ID 和 Span ID，
    构建 Span 树，导出为 Jaeger / Zipkin 格式。

    每个 denoising step 对应一个 Trace，包含多个嵌套 Span：
    - root: "step.0" (整个 step)
      - "qdrift.evaluate"
      - "shadowkv.compress"
      - "ffn.compute"
      - "profiler.record"

    对应文档：plan-v2.md Phase 1.2

    Example:
        >>> tracer = Tracer("shadowinfer")
        >>> trace_id = tracer.start_trace()
        >>> span1 = tracer.start_span(trace_id, "qdrift.evaluate")
        >>> span1.set_attribute("sensitivity_score", 0.85)
        >>> tracer.finish_span(trace_id, span1.span_id)
        >>> tracer.export_trace(trace_id)
    """

    def __init__(self, service_name: str = "shadowinfer") -> None:
        self.service_name = service_name
        self._traces: Dict[str, List[Span]] = {}
        self._otel_tracer: Optional[Any] = None
        self._otel_spans: Dict[str, Any] = {}

    def _generate_id(self) -> str:
        """生成唯一 ID — 基于 UUID4。"""
        return uuid.uuid4().hex[:16]

    def start_trace(self, trace_id: Optional[str] = None) -> str:
        """开始新的 Trace。

        参数:
            trace_id: 可选的自定义 Trace ID，未提供则自动生成。

        Returns:
            Trace ID。
        """
        tid = trace_id if trace_id is not None else self._generate_id()
        self._traces[tid] = []
        return tid

    def set_otel_exporter(
        self,
        exporter: Optional[Any] = None,
        provider: Optional[TracerProvider] = None,
        endpoint: Optional[str] = None,
        protocol: str = "grpc",
    ) -> None:
        """配置 OTel exporter。

        参数:
            exporter: 可选的 OTel SpanExporter 实例。为 None 时，若指定了
                endpoint 则创建 OTLP exporter，否则使用 Console exporter。
            provider: 可选的 ``TracerProvider`` 包装器。提供时优先使用。
            endpoint: OTLP endpoint，例如 ``http://localhost:4317``。
            protocol: OTLP 协议，``grpc`` 或 ``http``。

        当 OTel 不可用时，本方法为空操作。
        """
        if not _OTEL_AVAILABLE:
            return
        try:
            if provider is not None:
                self._otel_tracer = provider.get_tracer(self.service_name)
                return
            if exporter is None and endpoint is not None:
                exporter = _make_otlp_exporter(endpoint, protocol)
            provider = TracerProvider(exporter=exporter, service_name=self.service_name)
            self._otel_tracer = provider.get_tracer(self.service_name)
        except Exception:
            self._otel_tracer = None

    def start_span(
        self,
        trace_id: str,
        name: str,
        parent_id: Optional[str] = None,
    ) -> Span:
        """开始新的 Span。

        参数:
            trace_id: 所属 Trace ID
            name: Span 名称
            parent_id: 父 Span ID（None 表示 root span）

        Returns:
            新创建的 Span 对象。

        Raises:
            KeyError: 当 trace_id 不存在时。
        """
        if trace_id not in self._traces:
            raise KeyError(f"Trace '{trace_id}' 不存在，请先调用 start_trace()")

        span = Span(
            trace_id=trace_id,
            span_id=self._generate_id(),
            parent_id=parent_id,
            name=name,
            start_time=time.time(),
        )
        self._traces[trace_id].append(span)

        if self._otel_tracer is not None:
            try:
                context: Optional[Any] = None
                if parent_id is not None and parent_id in self._otel_spans:
                    context = _otel_trace.set_span_in_context(self._otel_spans[parent_id])
                otel_span = self._otel_tracer.start_span(name, context=context)
                otel_span.set_attribute("shadowinfer.trace_id", trace_id)
                otel_span.set_attribute("shadowinfer.span_id", span.span_id)
                if parent_id is not None:
                    otel_span.set_attribute("shadowinfer.parent_id", parent_id)
                self._otel_spans[span.span_id] = otel_span
            except Exception:
                pass

        return span

    def finish_span(self, trace_id: str, span_id: str) -> None:
        """结束 Span。

        参数:
            trace_id: Trace ID
            span_id: 要结束的 Span ID

        Raises:
            KeyError: 当 trace_id 或 span_id 不存在时。
        """
        if trace_id not in self._traces:
            raise KeyError(f"Trace '{trace_id}' 不存在")

        for span in self._traces[trace_id]:
            if span.span_id == span_id:
                span.finish()
                otel_span = self._otel_spans.pop(span_id, None)
                if otel_span is not None:
                    try:
                        for key, value in span.attributes.items():
                            otel_span.set_attribute(key, value)
                        if span.status == "ERROR":
                            otel_span.set_status(
                                _otel_trace.StatusCode.ERROR,
                                span.status_message or "",
                            )
                        elif span.status == "WARNING":
                            otel_span.set_attribute("status", "WARNING")
                        otel_span.end()
                    except Exception:
                        pass
                return
        raise KeyError(f"Span '{span_id}' 在 Trace '{trace_id}' 中不存在")

    def get_trace(self, trace_id: str) -> List[Span]:
        """获取 Trace 的所有 Spans（按 start_time 排序）。

        参数:
            trace_id: Trace ID

        Returns:
            Span 列表。

        Raises:
            KeyError: 当 trace_id 不存在时。
        """
        if trace_id not in self._traces:
            raise KeyError(f"Trace '{trace_id}' 不存在")
        return sorted(self._traces[trace_id], key=lambda s: s.start_time)

    def get_trace_duration_ms(self, trace_id: str) -> Optional[float]:
        """获取 Trace 总持续时间。

        计算方式：最后一个结束 Span 的 end_time - 第一个开始 Span 的 start_time。

        Returns:
            毫秒数，如果 Trace 为空或包含未结束 Span 则返回 None。
        """
        spans = self.get_trace(trace_id)
        if not spans:
            return None
        if any(s.end_time is None for s in spans):
            return None
        start = min(s.start_time for s in spans)
        end = max(s.end_time for s in spans)  # type: ignore[arg-type]
        return (end - start) * 1000.0

    def export_trace(self, trace_id: str) -> Dict[str, Any]:
        """导出 Trace 为字典。

        包含：
        - trace_id
        - service_name
        - total_duration_ms
        - spans: Span 列表（字典形式）
        """
        spans = self.get_trace(trace_id)
        return {
            "trace_id": trace_id,
            "service_name": self.service_name,
            "total_duration_ms": self.get_trace_duration_ms(trace_id),
            "spans": [s.to_dict() for s in spans],
        }

    def export_all_traces(self) -> Dict[str, List[Dict[str, Any]]]:
        """导出所有 Traces。"""
        return {tid: [s.to_dict() for s in spans] for tid, spans in self._traces.items()}

    def get_critical_path(self, trace_id: str) -> List[Span]:
        """获取 Trace 的关键路径 — 耗时最长的调用链。

        关键路径定义为从 root span 到某个叶子 span 的完整链路，
        其路径上所有 span 的 duration_ms 之和最大。

        算法：
        1. 构建 parent → children 映射
        2. 对每个 leaf span，回溯到 root，累加 duration
        3. 返回 duration 最大的路径

        对应文档：plan-v2.md Phase 1.2

        参数:
            trace_id: Trace ID

        Returns:
            按时间顺序排列的关键路径 Span 列表。
        """
        spans = self.get_trace(trace_id)
        if not spans:
            return []

        # 构建 parent -> children 映射
        children_map: Dict[Optional[str], List[Span]] = {}
        for span in spans:
            children_map.setdefault(span.parent_id, []).append(span)

        # 找到所有 leaf spans（没有 children 的 spans）
        leaf_spans = [
            s
            for s in spans
            if s.span_id not in {c.parent_id for c in spans if c.parent_id is not None}
        ]

        if not leaf_spans:
            # 所有 span 都是 root 级别（无 parent），按 duration 排序
            return sorted(spans, key=lambda s: (s.duration_ms or 0), reverse=True)[:1]

        # 对每条 leaf 回溯路径，计算总 duration
        max_duration = -1.0
        max_path: List[Span] = []

        for leaf in leaf_spans:
            path: List[Span] = []
            current: Optional[str] = leaf.span_id
            while current is not None:
                for span in spans:
                    if span.span_id == current:
                        path.append(span)
                        current = span.parent_id
                        break
                else:
                    break
            path.reverse()

            total_duration = sum((s.duration_ms or 0) for s in path)
            if total_duration > max_duration:
                max_duration = total_duration
                max_path = path

        return max_path


class CriticalPath:
    """关键路径分析工具 — 从 Tracer 中提取并分析关键路径。

    该类作为 ``Tracer.get_critical_path()`` 的补充，提供对关键路径的结构化
    访问，与 OTel 概念保持一致。

    Example:
        >>> tracer = Tracer()
        >>> trace_id = tracer.start_trace()
        >>> root = tracer.start_span(trace_id, "root")
        >>> child = tracer.start_span(trace_id, "child", parent_id=root.span_id)
        >>> tracer.finish_span(trace_id, child.span_id)
        >>> tracer.finish_span(trace_id, root.span_id)
        >>> cp = CriticalPath.from_tracer(tracer, trace_id)
        >>> len(cp.spans) >= 1
        True
    """

    def __init__(self, spans: List[Span]) -> None:
        self.spans = spans

    @classmethod
    def from_tracer(cls, tracer: Tracer, trace_id: str) -> "CriticalPath":
        """从 Tracer 实例构建 CriticalPath。"""
        return cls(tracer.get_critical_path(trace_id))

    @property
    def duration_ms(self) -> float:
        """关键路径总耗时（ms）。"""
        return sum((s.duration_ms or 0.0) for s in self.spans)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "duration_ms": self.duration_ms,
            "spans": [s.to_dict() for s in self.spans],
        }
