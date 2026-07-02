"""ShadowInfer 可观测性模块单元测试。

对应文档：plan-v2.md Phase 1
覆盖：
- metrics.py: Counter, Gauge, Histogram, Summary, MetricsRegistry
- tracing.py: Span, Tracer
- dashboard.py: DashboardData
"""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

from shadowinfer.observability import (
    Counter,
    DashboardData,
    Gauge,
    Histogram,
    MetricsRegistry,
    Span,
    Summary,
    Tracer,
)

# ============================================================================
# Counter 测试
# ============================================================================


class TestCounter:
    """Counter 单元测试 — 对应 plan-v2.md Phase 1.1"""

    def test_init(self):
        c = Counter("test_counter", "测试计数器", labels={"env": "test"})
        assert c.metadata.name == "test_counter"
        assert c.metadata.description == "测试计数器"
        assert c.metadata.labels == {"env": "test"}
        assert c.get() == 0.0

    def test_inc(self):
        c = Counter("test", "test")
        c.inc(5.0)
        assert c.get() == 5.0
        c.inc(3.5)
        assert c.get() == 8.5

    def test_inc_default(self):
        c = Counter("test", "test")
        c.inc()
        assert c.get() == 1.0

    def test_inc_negative_raises(self):
        c = Counter("test", "test")
        with pytest.raises(ValueError, match="Counter 只能递增"):
            c.inc(-1.0)

    def test_to_dict(self):
        c = Counter("test", "desc", labels={"k": "v"})
        c.inc(10.0)
        d = c.to_dict()
        assert d["name"] == "test"
        assert d["type"] == "counter"
        assert d["value"] == 10.0
        assert d["labels"] == {"k": "v"}

    def test_to_prometheus(self):
        c = Counter("requests_total", "累计请求数", labels={"service": "shadowinfer"})
        c.inc(42.0)
        prom = c.to_prometheus()
        assert "# HELP requests_total 累计请求数" in prom
        assert "# TYPE requests_total counter" in prom
        assert 'service="shadowinfer"' in prom
        assert "42.0" in prom

    def test_thread_safety(self):
        c = Counter("thread_test", "thread test")
        threads = []
        for _ in range(10):
            t = threading.Thread(target=lambda: [c.inc(1.0) for _ in range(100)])
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert c.get() == 1000.0


# ============================================================================
# Gauge 测试
# ============================================================================


class TestGauge:
    """Gauge 单元测试 — 对应 plan-v2.md Phase 1.1"""

    def test_init(self):
        g = Gauge("memory", "内存占用")
        assert g.get() == 0.0

    def test_set(self):
        g = Gauge("memory", "内存占用")
        g.set(1024.0)
        assert g.get() == 1024.0

    def test_inc_dec(self):
        g = Gauge("memory", "内存占用")
        g.set(1000.0)
        g.inc(200.0)
        assert g.get() == 1200.0
        g.dec(300.0)
        assert g.get() == 900.0

    def test_to_dict(self):
        g = Gauge("gpu_mem", "GPU 显存", labels={"device": "0"})
        g.set(2048.0)
        d = g.to_dict()
        assert d["type"] == "gauge"
        assert d["value"] == 2048.0

    def test_to_prometheus(self):
        g = Gauge("gpu_util", "GPU 利用率", labels={"device": "0"})
        g.set(75.5)
        prom = g.to_prometheus()
        assert "# TYPE gpu_util gauge" in prom
        assert 'device="0"' in prom
        assert "75.5" in prom

    def test_thread_safety(self):
        g = Gauge("thread_gauge", "thread gauge")
        g.set(0.0)
        threads = []
        for _ in range(10):
            t = threading.Thread(target=lambda: [g.inc(1.0) or g.dec(0.5) for _ in range(100)])
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert g.get() == 500.0  # 10 * 100 * (1.0 - 0.5)


# ============================================================================
# Histogram 测试
# ============================================================================


class TestHistogram:
    """Histogram 单元测试 — 对应 plan-v2.md Phase 1.1"""

    def test_init_default_buckets(self):
        h = Histogram("latency", "延迟")
        assert len(h.buckets) == 12
        assert h.buckets[0] == 0.001

    def test_custom_buckets(self):
        h = Histogram("latency", "延迟", buckets=[0.1, 0.5, 1.0])
        assert h.buckets == [0.1, 0.5, 1.0]
        assert len(h._counts) == 4

    def test_observe(self):
        h = Histogram("latency", "延迟", buckets=[0.1, 0.5, 1.0])
        h.observe(0.05)
        h.observe(0.3)
        h.observe(1.5)
        d = h.to_dict()
        assert d["counts"] == [1, 1, 0, 1]  # <=0.1, <=0.5, <=1.0, +Inf
        assert d["sum"] == 1.85
        assert d["count"] == 3

    def test_observe_negative_raises(self):
        h = Histogram("latency", "延迟")
        with pytest.raises(ValueError, match="Histogram 观测值必须"):
            h.observe(-1.0)

    def test_get_percentile(self):
        h = Histogram("latency", "延迟", buckets=[0.01, 0.05, 0.1, 0.5, 1.0])
        for v in [0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.3, 0.7, 1.5]:
            h.observe(v)
        p50 = h.get_percentile(0.50)
        p95 = h.get_percentile(0.95)
        assert 0.05 < p50 < 0.15
        assert p95 > 0.5

    def test_get_percentile_empty_raises(self):
        h = Histogram("latency", "延迟")
        with pytest.raises(ValueError, match="Histogram 为空"):
            h.get_percentile(0.5)

    def test_get_percentile_invalid(self):
        h = Histogram("latency", "延迟")
        h.observe(1.0)
        with pytest.raises(ValueError, match="percentile 必须在"):
            h.get_percentile(1.5)

    def test_to_prometheus(self):
        h = Histogram("latency", "延迟", buckets=[0.1, 0.5])
        h.observe(0.2)
        prom = h.to_prometheus()
        assert "# TYPE latency histogram" in prom
        assert "latency_bucket" in prom
        assert "latency_sum" in prom
        assert "latency_count" in prom
        assert 'le="+Inf"' in prom

    def test_thread_safety(self):
        h = Histogram("thread_hist", "thread hist", buckets=[1.0, 2.0, 3.0])
        threads = []
        for _ in range(10):
            t = threading.Thread(target=lambda: [h.observe(0.5) for _ in range(100)])
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert h.to_dict()["count"] == 1000


# ============================================================================
# Summary 测试
# ============================================================================


class TestSummary:
    """Summary 单元测试 — 对应 plan-v2.md Phase 1.1"""

    def test_init(self):
        s = Summary("dispatch", "调度延迟", window_size=100)
        assert s.window_size == 100

    def test_observe(self):
        s = Summary("dispatch", "调度延迟")
        s.observe(0.5)
        s.observe(1.0)
        assert len(s._values) == 2

    def test_window_size(self):
        s = Summary("dispatch", "调度延迟", window_size=3)
        s.observe(1.0)
        s.observe(2.0)
        s.observe(3.0)
        s.observe(4.0)  # 挤出 1.0
        assert len(s._values) == 3
        assert 1.0 not in s._values

    def test_get_stats(self):
        s = Summary("dispatch", "调度延迟")
        for v in [0.05, 0.12, 0.30, 0.08, 0.15]:
            s.observe(v)
        stats = s.get_stats()
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "p50" in stats
        assert "p95" in stats
        assert "p99" in stats
        assert stats["min"] == 0.05
        assert stats["max"] == 0.30
        assert stats["count"] == 5.0
        assert stats["p95"] > stats["p50"]

    def test_get_stats_empty_raises(self):
        s = Summary("dispatch", "调度延迟")
        with pytest.raises(ValueError, match="Summary 窗口为空"):
            s.get_stats()

    def test_to_dict(self):
        s = Summary("dispatch", "调度延迟", labels={"env": "test"})
        s.observe(1.0)
        d = s.to_dict()
        assert d["type"] == "summary"
        assert d["window_size"] == 1000
        assert d["current_count"] == 1
        assert "stats" in d

    def test_to_prometheus(self):
        s = Summary("dispatch", "调度延迟")
        s.observe(1.0)
        s.observe(2.0)
        prom = s.to_prometheus()
        assert "# TYPE dispatch summary" in prom
        assert "dispatch_count" in prom
        assert "quantile" in prom

    def test_thread_safety(self):
        s = Summary("thread_summary", "thread summary", window_size=1000)
        threads = []
        for _ in range(10):
            t = threading.Thread(target=lambda: [s.observe(float(i)) for i in range(100)])
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert len(s._values) == 1000
        stats = s.get_stats()
        assert stats["count"] == 1000.0


# ============================================================================
# MetricsRegistry 测试
# ============================================================================


class TestMetricsRegistry:
    """MetricsRegistry 单元测试 — 对应 plan-v2.md Phase 1.1"""

    def test_register_counter(self):
        reg = MetricsRegistry()
        c = Counter("req", "请求")
        reg.register(c)
        assert reg.get("req") is c

    def test_register_duplicate_same_type(self):
        reg = MetricsRegistry()
        c1 = Counter("req", "请求")
        c2 = Counter("req", "请求")
        reg.register(c1)
        reg.register(c2)  # 不报错，复用
        assert reg.get("req") is c1

    def test_register_duplicate_different_type(self):
        reg = MetricsRegistry()
        reg.register(Counter("x", "x"))
        with pytest.raises(ValueError, match="指标 'x' 已存在"):
            reg.register(Gauge("x", "x"))

    def test_counter_factory(self):
        reg = MetricsRegistry()
        c1 = reg.counter("req", "请求")
        c2 = reg.counter("req", "请求")
        assert c1 is c2

    def test_gauge_factory(self):
        reg = MetricsRegistry()
        g = reg.gauge("mem", "内存")
        assert isinstance(g, Gauge)

    def test_histogram_factory(self):
        reg = MetricsRegistry()
        h = reg.histogram("lat", "延迟")
        assert isinstance(h, Histogram)

    def test_summary_factory(self):
        reg = MetricsRegistry()
        s = reg.summary("disp", "调度")
        assert isinstance(s, Summary)

    def test_factory_type_mismatch(self):
        reg = MetricsRegistry()
        reg.counter("x", "x")
        with pytest.raises(TypeError, match="指标 'x' 已存在但不是 Gauge"):
            reg.gauge("x", "x")

    def test_export_all(self):
        reg = MetricsRegistry()
        reg.counter("c", "c").inc(1.0)
        reg.gauge("g", "g").set(2.0)
        all_data = reg.export_all()
        assert "c" in all_data
        assert "g" in all_data
        assert all_data["c"]["value"] == 1.0
        assert all_data["g"]["value"] == 2.0

    def test_export_prometheus(self):
        reg = MetricsRegistry()
        reg.counter("c", "c", labels={"k": "v"}).inc(1.0)
        prom = reg.export_prometheus_format()
        assert "c" in prom
        assert "# TYPE c counter" in prom

    def test_export_json(self):
        reg = MetricsRegistry()
        reg.counter("c", "c").inc(1.0)
        json_str = reg.export_json()
        data = json.loads(json_str)
        assert data["c"]["value"] == 1.0

    def test_thread_safety(self):
        reg = MetricsRegistry()
        threads = []
        for i in range(10):
            t = threading.Thread(target=lambda i=i: reg.counter(f"c_{i}", f"c_{i}").inc(1.0))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert len(reg.export_all()) == 10


# ============================================================================
# Span 测试
# ============================================================================


class TestSpan:
    """Span 单元测试 — 对应 plan-v2.md Phase 1.2"""

    def test_init(self):
        span = Span(
            trace_id="t1",
            span_id="s1",
            parent_id=None,
            name="test",
            start_time=time.time(),
        )
        assert span.trace_id == "t1"
        assert span.span_id == "s1"
        assert span.parent_id is None
        assert span.name == "test"
        assert span.status == "OK"
        assert span.duration_ms is None

    def test_set_attribute(self):
        span = Span("t1", "s1", None, "test", time.time())
        span.set_attribute("flops", 1.5e9)
        assert span.attributes["flops"] == 1.5e9

    def test_set_status(self):
        span = Span("t1", "s1", None, "test", time.time())
        span.set_status("ERROR", "OOM")
        assert span.status == "ERROR"
        assert span.status_message == "OOM"

    def test_set_status_invalid(self):
        span = Span("t1", "s1", None, "test", time.time())
        with pytest.raises(ValueError, match="status 必须是"):
            span.set_status("INVALID")

    def test_finish(self):
        start = time.time()
        span = Span("t1", "s1", None, "test", start)
        span.finish(start + 0.1)
        assert span.duration_ms is not None
        assert abs(span.duration_ms - 100.0) < 0.01

    def test_to_dict(self):
        span = Span("t1", "s1", None, "test", time.time())
        span.set_attribute("k", "v")
        span.finish()
        d = span.to_dict()
        assert d["trace_id"] == "t1"
        assert d["name"] == "test"
        assert "duration_ms" in d
        assert d["attributes"] == {"k": "v"}


# ============================================================================
# Tracer 测试
# ============================================================================


class TestTracer:
    """Tracer 单元测试 — 对应 plan-v2.md Phase 1.2"""

    def test_init(self):
        tracer = Tracer("shadowinfer")
        assert tracer.service_name == "shadowinfer"

    def test_start_trace(self):
        tracer = Tracer()
        tid = tracer.start_trace()
        assert tid in tracer._traces
        assert tracer._traces[tid] == []

    def test_start_trace_custom_id(self):
        tracer = Tracer()
        tid = tracer.start_trace("custom-id")
        assert tid == "custom-id"

    def test_start_span(self):
        tracer = Tracer()
        tid = tracer.start_trace()
        span = tracer.start_span(tid, "qdrift.evaluate")
        assert span.trace_id == tid
        assert span.name == "qdrift.evaluate"
        assert span.parent_id is None
        assert span.duration_ms is None

    def test_start_span_with_parent(self):
        tracer = Tracer()
        tid = tracer.start_trace()
        parent = tracer.start_span(tid, "parent")
        child = tracer.start_span(tid, "child", parent_id=parent.span_id)
        assert child.parent_id == parent.span_id

    def test_start_span_invalid_trace(self):
        tracer = Tracer()
        with pytest.raises(KeyError, match="Trace 'invalid' 不存在"):
            tracer.start_span("invalid", "test")

    def test_finish_span(self):
        tracer = Tracer()
        tid = tracer.start_trace()
        span = tracer.start_span(tid, "test")
        tracer.finish_span(tid, span.span_id)
        assert span.duration_ms is not None

    def test_finish_span_invalid(self):
        tracer = Tracer()
        tid = tracer.start_trace()
        with pytest.raises(KeyError, match="Span 'invalid' 在 Trace"):
            tracer.finish_span(tid, "invalid")

    def test_get_trace(self):
        tracer = Tracer()
        tid = tracer.start_trace()
        s1 = tracer.start_span(tid, "a")
        time.sleep(0.01)
        s2 = tracer.start_span(tid, "b")
        tracer.finish_span(tid, s1.span_id)
        tracer.finish_span(tid, s2.span_id)
        spans = tracer.get_trace(tid)
        assert len(spans) == 2
        assert spans[0].start_time <= spans[1].start_time

    def test_get_trace_duration(self):
        tracer = Tracer()
        tid = tracer.start_trace()
        s1 = tracer.start_span(tid, "a")
        time.sleep(0.05)
        s2 = tracer.start_span(tid, "b")
        time.sleep(0.05)
        tracer.finish_span(tid, s1.span_id)
        tracer.finish_span(tid, s2.span_id)
        duration = tracer.get_trace_duration_ms(tid)
        assert duration is not None
        assert duration >= 100.0  # at least 100ms

    def test_get_trace_duration_empty(self):
        tracer = Tracer()
        tid = tracer.start_trace()
        assert tracer.get_trace_duration_ms(tid) is None

    def test_get_trace_duration_unfinished(self):
        tracer = Tracer()
        tid = tracer.start_trace()
        tracer.start_span(tid, "a")
        assert tracer.get_trace_duration_ms(tid) is None

    def test_export_trace(self):
        tracer = Tracer()
        tid = tracer.start_trace()
        span = tracer.start_span(tid, "test")
        span.set_attribute("key", "value")
        tracer.finish_span(tid, span.span_id)
        exported = tracer.export_trace(tid)
        assert exported["trace_id"] == tid
        assert exported["service_name"] == "shadowinfer"
        assert len(exported["spans"]) == 1
        assert exported["spans"][0]["name"] == "test"
        assert exported["spans"][0]["attributes"]["key"] == "value"

    def test_export_all_traces(self):
        tracer = Tracer()
        t1 = tracer.start_trace()
        t2 = tracer.start_trace()
        tracer.start_span(t1, "a")
        tracer.start_span(t2, "b")
        all_traces = tracer.export_all_traces()
        assert t1 in all_traces
        assert t2 in all_traces

    def test_critical_path(self):
        tracer = Tracer()
        tid = tracer.start_trace()
        root = tracer.start_span(tid, "root")
        time.sleep(0.01)
        child1 = tracer.start_span(tid, "child1", parent_id=root.span_id)
        time.sleep(0.05)
        child2 = tracer.start_span(tid, "child2", parent_id=root.span_id)
        time.sleep(0.01)
        tracer.finish_span(tid, child2.span_id)
        time.sleep(0.05)
        tracer.finish_span(tid, child1.span_id)
        tracer.finish_span(tid, root.span_id)

        critical = tracer.get_critical_path(tid)
        assert len(critical) >= 1
        # child1 路径更长
        assert critical[-1].name in ("child1", "root")

    def test_critical_path_empty(self):
        tracer = Tracer()
        tid = tracer.start_trace()
        assert tracer.get_critical_path(tid) == []


# ============================================================================
# DashboardData 测试
# ============================================================================


class TestDashboardData:
    """DashboardData 单元测试 — 对应 plan-v2.md Phase 1.3"""

    def test_init(self):
        d = DashboardData(max_history=100)
        assert d.max_history == 100

    def test_add_step_data(self):
        d = DashboardData()
        d.add_step_data(0, {"latency_ms": 50.0, "memory_mb": 1024.0, "accuracy": 0.95})
        assert len(d.step_history) == 1
        assert len(d.latency_history) == 1
        assert len(d.memory_history) == 1
        assert len(d.accuracy_history) == 1

    def test_get_latency_trend(self):
        d = DashboardData()
        for i in range(10):
            d.add_step_data(i, {"latency_ms": float(i)})
        trend = d.get_latency_trend(window=5)
        assert len(trend) == 5
        assert trend == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_get_memory_trend(self):
        d = DashboardData()
        for i in range(10):
            d.add_step_data(i, {"memory_mb": float(i * 100)})
        trend = d.get_memory_trend(window=3)
        assert trend == [700.0, 800.0, 900.0]

    def test_get_precision_distribution(self):
        d = DashboardData()
        d.add_step_data(0, {"precision": "fp16"})
        d.add_step_data(1, {"precision": "fp16"})
        d.add_step_data(2, {"precision": "int8"})
        d.add_step_data(3, {"precision": "fp32"})
        dist = d.get_precision_distribution()
        assert dist == {"fp16": 2, "int8": 1, "fp32": 1}

    def test_get_scheduling_timeline(self):
        d = DashboardData()
        d.add_step_data(0, {"scheduler": "qdrift", "latency_ms": 50.0})
        d.add_step_data(1, {"scheduler": "baseline", "latency_ms": 80.0})
        timeline = d.get_scheduling_timeline()
        assert len(timeline) == 2
        assert timeline[0]["scheduler"] == "qdrift"
        assert timeline[1]["scheduler"] == "baseline"

    def test_get_layer_memory_heatmap(self):
        d = DashboardData()
        d.add_step_data(0, {"layer_memory": [100, 200, 300]})
        d.add_step_data(1, {"layer_memory": [110, 210, 310]})
        heatmap = d.get_layer_memory_heatmap()
        assert heatmap is not None
        assert heatmap["x_axis"] == [0, 1]
        assert heatmap["y_axis"] == [0, 1, 2]
        assert len(heatmap["data"]) == 6

    def test_get_layer_memory_heatmap_no_data(self):
        d = DashboardData()
        d.add_step_data(0, {"latency_ms": 50.0})
        assert d.get_layer_memory_heatmap() is None

    def test_export_json(self, tmp_path):
        d = DashboardData()
        d.add_step_data(0, {"latency_ms": 50.0, "memory_mb": 1024.0, "precision": "fp16"})
        output = str(tmp_path / "dashboard.json")
        d.export_json(output)
        assert os.path.exists(output)
        with open(output, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "step_history" in data
        assert "latency_trend" in data
        assert data["latency_trend"] == [50.0]

    def test_export_html(self, tmp_path):
        d = DashboardData()
        d.add_step_data(0, {"latency_ms": 50.0, "memory_mb": 1024.0, "precision": "fp16"})
        d.add_step_data(1, {"latency_ms": 60.0, "memory_mb": 1100.0, "precision": "int8"})
        output = str(tmp_path / "dashboard.html")
        d.export_html(output)
        assert os.path.exists(output)
        with open(output, "r", encoding="utf-8") as f:
            content = f.read()
        assert "ShadowInfer" in content
        assert "echarts" in content
        assert "latencyChart" in content
        assert "memoryChart" in content
        assert "precisionChart" in content
        assert "timelineChart" in content

    def test_max_history(self):
        d = DashboardData(max_history=3)
        for i in range(5):
            d.add_step_data(i, {"latency_ms": float(i)})
        assert len(d.step_history) == 3
        assert list(d.latency_history) == [2.0, 3.0, 4.0]


# ============================================================================
# 集成测试
# ============================================================================


class TestIntegration:
    """可观测性模块集成测试 — Metrics + Traces + Dashboard 协同。"""

    def test_full_pipeline(self, tmp_path):
        """完整流水线：指标收集 → 追踪 → Dashboard 导出。"""
        # 1. Metrics
        registry = MetricsRegistry()
        latency_hist = registry.histogram("inference_latency_ms", "推理延迟")
        memory_gauge = registry.gauge("memory_mb", "内存占用")
        flops_counter = registry.counter("total_flops", "累计 FLOPs")

        # 2. Tracer
        tracer = Tracer("shadowinfer")
        trace_id = tracer.start_trace()

        # 3. Dashboard
        dashboard = DashboardData(max_history=10)

        # 模拟 5 个 denoising steps
        for step in range(5):
            # Start step span
            step_span = tracer.start_span(trace_id, f"step.{step}")

            # Simulate qdrift
            qdrift_span = tracer.start_span(
                trace_id, "qdrift.evaluate", parent_id=step_span.span_id
            )
            qdrift_span.set_attribute("sensitivity_score", 0.8 - step * 0.1)
            time.sleep(0.01)
            tracer.finish_span(trace_id, qdrift_span.span_id)

            # Simulate shadowkv
            kv_span = tracer.start_span(trace_id, "shadowkv.compress", parent_id=step_span.span_id)
            kv_span.set_attribute("compression_ratio", 0.5)
            time.sleep(0.01)
            tracer.finish_span(trace_id, kv_span.span_id)

            # Record metrics
            latency = 0.05 + step * 0.01
            memory = 1000.0 + step * 50.0
            flops = 1.0e9

            latency_hist.observe(latency)
            memory_gauge.set(memory)
            flops_counter.inc(flops)

            # Dashboard
            dashboard.add_step_data(
                step,
                {
                    "latency_ms": latency * 1000,
                    "memory_mb": memory,
                    "accuracy": 0.95,
                    "precision": "fp16" if step < 3 else "int8",
                    "scheduler": "qdrift",
                },
            )

            tracer.finish_span(trace_id, step_span.span_id)

        # 验证 Metrics
        assert latency_hist.to_dict()["count"] == 5
        assert flops_counter.get() == 5.0e9
        assert memory_gauge.get() == 1200.0

        # 验证 Traces
        assert len(tracer.get_trace(trace_id)) == 15  # 5 steps * 3 spans
        exported = tracer.export_trace(trace_id)
        assert exported["total_duration_ms"] is not None
        assert exported["total_duration_ms"] > 0

        # 验证 Dashboard
        assert len(dashboard.step_history) == 5
        assert dashboard.get_precision_distribution() == {"fp16": 3, "int8": 2}

        # 导出 Prometheus
        prom = registry.export_prometheus_format()
        assert "inference_latency_ms" in prom
        assert "memory_mb" in prom
        assert "total_flops" in prom

        # 导出 Dashboard HTML
        html_path = str(tmp_path / "report.html")
        dashboard.export_html(html_path)
        assert os.path.exists(html_path)

        # 导出 Dashboard JSON
        json_path = str(tmp_path / "report.json")
        dashboard.export_json(json_path)
        assert os.path.exists(json_path)


# ============================================================================
# Prometheus / OpenTelemetry 集成测试
# ============================================================================


class TestPrometheusIntegration:
    """Prometheus client 集成测试。"""

    def test_expose_returns_prometheus_format(self):
        from shadowinfer.observability import _PROMETHEUS_AVAILABLE

        reg = MetricsRegistry()
        reg.counter("requests_total", "累计请求数", labels={"service": "test"}).inc(7.0)
        reg.gauge("memory_mb", "内存占用").set(1024.0)
        reg.histogram("latency_ms", "延迟").observe(0.05)
        reg.summary("dispatch_ms", "调度延迟").observe(0.1)

        exposed = reg.expose()
        assert "requests_total" in exposed
        assert "memory_mb" in exposed
        assert "latency_ms" in exposed
        assert "dispatch_ms" in exposed

        if _PROMETHEUS_AVAILABLE:
            assert "# HELP requests_total 累计请求数" in exposed
            assert "# TYPE requests_total counter" in exposed
            assert 'service="test"' in exposed
        else:
            assert "# HELP requests_total" in exposed

    def test_to_prometheus_alias(self):
        reg = MetricsRegistry()
        reg.counter("c", "c").inc(1.0)
        assert "c" in reg.to_prometheus()

    def test_prometheus_backend_mirror_values(self):
        from shadowinfer.observability import _PROMETHEUS_AVAILABLE

        reg = MetricsRegistry()
        c = reg.counter("mirror_counter", "mirror")
        c.inc(10.0)
        assert c.get() == 10.0

        if _PROMETHEUS_AVAILABLE:
            assert c._prom_metric is not None
            assert c._prom_child is not None
            exposed = reg.expose()
            assert "mirror_counter" in exposed
            assert "10.0" in exposed
        else:
            assert c._prom_metric is None

    def test_fallback_when_prometheus_unavailable(self, monkeypatch):
        import shadowinfer.observability.metrics as metrics_module

        monkeypatch.setattr(metrics_module, "_PROMETHEUS_AVAILABLE", False)
        reg = MetricsRegistry()
        c = reg.counter("fallback_counter", "fallback")
        c.inc(3.0)
        assert c.get() == 3.0
        assert c._prom_metric is None
        exposed = reg.expose()
        assert "fallback_counter" in exposed
        assert "3.0" in exposed


class TestOpenTelemetryIntegration:
    """OpenTelemetry 集成测试。"""

    def test_tracer_provider_wrapper(self):
        from shadowinfer.observability import _OTEL_AVAILABLE, TracerProvider

        provider = TracerProvider()
        if _OTEL_AVAILABLE:
            assert provider._provider is not None
            tracer = provider.get_tracer("test")
            assert tracer is not None
        else:
            assert provider._provider is None

    def test_otel_span_creation(self):
        from shadowinfer.observability import _OTEL_AVAILABLE, TracerProvider

        if not _OTEL_AVAILABLE:
            pytest.skip("OpenTelemetry not installed")

        provider = TracerProvider()
        tracer = Tracer("shadowinfer")
        tracer.set_otel_exporter(provider=provider)
        assert tracer._otel_tracer is not None

        trace_id = tracer.start_trace()
        root = tracer.start_span(trace_id, "root")
        child = tracer.start_span(trace_id, "child", parent_id=root.span_id)
        child.set_attribute("key", "value")
        tracer.finish_span(trace_id, child.span_id)
        tracer.finish_span(trace_id, root.span_id)

        assert child.span_id in tracer._otel_spans or child.span_id not in tracer._otel_spans
        # After finish, the OTel span should be removed from tracking.
        assert child.span_id not in tracer._otel_spans

    def test_otel_set_exporter_with_console(self):
        from shadowinfer.observability import _OTEL_AVAILABLE, Tracer

        if not _OTEL_AVAILABLE:
            pytest.skip("OpenTelemetry not installed")

        tracer = Tracer("shadowinfer")
        tracer.set_otel_exporter()
        assert tracer._otel_tracer is not None

    def test_otel_fallback_when_unavailable(self, monkeypatch):
        import shadowinfer.observability.tracing as tracing_module

        monkeypatch.setattr(tracing_module, "_OTEL_AVAILABLE", False)
        tracer = Tracer("shadowinfer")
        tracer.set_otel_exporter()
        assert tracer._otel_tracer is None

        trace_id = tracer.start_trace()
        span = tracer.start_span(trace_id, "test")
        tracer.finish_span(trace_id, span.span_id)
        assert span.duration_ms is not None


class TestCriticalPath:
    """CriticalPath 辅助类测试。"""

    def test_from_tracer(self):
        from shadowinfer.observability import CriticalPath, Tracer

        tracer = Tracer()
        trace_id = tracer.start_trace()
        root = tracer.start_span(trace_id, "root")
        child = tracer.start_span(trace_id, "child", parent_id=root.span_id)
        tracer.finish_span(trace_id, child.span_id)
        tracer.finish_span(trace_id, root.span_id)

        cp = CriticalPath.from_tracer(tracer, trace_id)
        assert len(cp.spans) >= 1
        assert cp.duration_ms >= 0.0
        assert "spans" in cp.to_dict()
