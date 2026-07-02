"""Dashboard 可视化 — 实时展示推理指标。

基于纯 HTML + ECharts CDN 构建交互式 Dashboard：
- 实时延迟曲线
- KV cache 内存占用热力图
- 精度分布饼图
- 调度决策时间线
- 对比视图

对应文档：plan-v2.md Phase 1.3
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Dict, List, Optional


class DashboardData:
    """Dashboard 数据聚合器 — 收集并聚合推理过程中的实时数据。

    维护滑动窗口的历史数据，支持导出为 HTML（含 ECharts）和 JSON。

    对应文档：plan-v2.md Phase 1.3

    Example:
        >>> dashboard = DashboardData(max_history=100)
        >>> dashboard.add_step_data(0, {
        ...     "latency_ms": 50.0,
        ...     "memory_mb": 1024.0,
        ...     "accuracy": 0.95,
        ...     "precision": "fp16",
        ...     "scheduler": "qdrift",
        ... })
        >>> dashboard.export_json("report.json")
    """

    def __init__(self, max_history: int = 1000) -> None:
        self.max_history = max_history
        self.step_history: deque[Dict[str, Any]] = deque(maxlen=max_history)
        self.latency_history: deque[float] = deque(maxlen=max_history)
        self.memory_history: deque[float] = deque(maxlen=max_history)
        self.accuracy_history: deque[float] = deque(maxlen=max_history)

    def add_step_data(self, step_id: int, data: Dict[str, Any]) -> None:
        """添加单 step 数据。

        参数:
            step_id: Step 编号
            data: 包含以下字段的字典：
                - latency_ms: 延迟（毫秒）
                - memory_mb: 内存占用（MB）
                - accuracy: 精度（0~1）
                - precision: 精度类型（fp32/fp16/int8/int4）
                - scheduler: 调度器名称（qdrift/baseline/etc）
                - layer_precisions: 各层精度列表（可选，用于热力图）
                - layer_memory: 各层内存列表（可选，用于热力图）
        """
        record = {"step_id": step_id, **data}
        self.step_history.append(record)

        if "latency_ms" in data:
            self.latency_history.append(float(data["latency_ms"]))
        if "memory_mb" in data:
            self.memory_history.append(float(data["memory_mb"]))
        if "accuracy" in data:
            self.accuracy_history.append(float(data["accuracy"]))

    def get_latency_trend(self, window: int = 50) -> List[float]:
        """获取延迟趋势 — 最近 N 个 step 的延迟。

        参数:
            window: 窗口大小

        Returns:
            延迟列表（毫秒）
        """
        return list(self.latency_history)[-window:]

    def get_memory_trend(self, window: int = 50) -> List[float]:
        """获取内存趋势 — 最近 N 个 step 的内存占用。

        参数:
            window: 窗口大小

        Returns:
            内存列表（MB）
        """
        return list(self.memory_history)[-window:]

    def get_precision_distribution(self) -> Dict[str, int]:
        """获取精度分布 — 统计各精度类型出现次数。

        Returns:
            Dict[precision_str, count]
        """
        distribution: Dict[str, int] = {}
        for record in self.step_history:
            precision = record.get("precision")
            if precision:
                distribution[precision] = distribution.get(precision, 0) + 1
        return distribution

    def get_scheduling_timeline(self) -> List[Dict[str, Any]]:
        """获取调度时间线 — 按 step 顺序的调度决策记录。

        Returns:
            List[Dict]，每个元素包含 step_id 和 scheduler 决策。
        """
        timeline: List[Dict[str, Any]] = []
        for record in self.step_history:
            scheduler = record.get("scheduler")
            if scheduler:
                timeline.append(
                    {
                        "step_id": record.get("step_id"),
                        "scheduler": scheduler,
                        "latency_ms": record.get("latency_ms"),
                        "memory_mb": record.get("memory_mb"),
                    }
                )
        return timeline

    def get_layer_memory_heatmap(self) -> Optional[Dict[str, Any]]:
        """获取层内存热力图数据。

        需要 step_data 中包含 layer_memory 字段（List[float]）。

        Returns:
            ECharts heatmap 数据格式，或 None（如果无热力图数据）。
        """
        heatmap_data = []
        x_axis: List[int] = []
        y_axis: List[int] = []

        for record in self.step_history:
            layer_memory = record.get("layer_memory")
            step_id = record.get("step_id")
            if not isinstance(layer_memory, list) or step_id is None:
                continue
            x_axis.append(step_id)
            for layer_idx, mem in enumerate(layer_memory):
                if layer_idx not in y_axis:
                    y_axis.append(layer_idx)
                heatmap_data.append([step_id, layer_idx, round(float(mem), 2)])

        if not heatmap_data:
            return None

        y_axis.sort()
        return {
            "x_axis": x_axis,
            "y_axis": y_axis,
            "data": heatmap_data,
        }

    def export_html(self, output_path: str) -> None:
        """导出为 HTML 报告（使用 ECharts CDN）。

        包含以下图表：
        1. 延迟曲线（Line Chart）
        2. 内存热力图（Heatmap）
        3. 精度分布饼图（Pie Chart）
        4. 调度时间线（Timeline / Bar Chart）
        5. 对比视图（Baseline vs Optimized Bar Chart）

        对应文档：plan-v2.md Phase 1.3
        """
        latency_data = self.get_latency_trend(window=len(self.step_history))
        memory_data = self.get_memory_trend(window=len(self.step_history))
        precision_data = self.get_precision_distribution()
        timeline_data = self.get_scheduling_timeline()
        heatmap_data = self.get_layer_memory_heatmap()

        # 构建 ECharts 数据
        steps = list(range(len(latency_data)))
        latency_series = latency_data
        memory_series = memory_data

        precision_pie = [{"value": count, "name": prec} for prec, count in precision_data.items()]

        timeline_categories = list(set(t["scheduler"] for t in timeline_data))
        timeline_series = {
            cat: [t["latency_ms"] for t in timeline_data if t["scheduler"] == cat]
            for cat in timeline_categories
        }
        timeline_steps = [t["step_id"] for t in timeline_data]

        avg_latency = round(sum(latency_data) / len(latency_data), 2) if latency_data else 0
        avg_memory = round(sum(memory_data) / len(memory_data), 2) if memory_data else 0
        avg_accuracy = (
            round(sum(self.accuracy_history) / len(self.accuracy_history), 4)
            if self.accuracy_history
            else 0
        )
        heatmap_chart_html = (
            '<div class="chart">'
            '<div class="chart-title">层内存热力图 (Layer Memory Heatmap)</div>'
            '<div id="heatmapChart" class="chart-container"></div></div>'
            if heatmap_data
            else ""
        )
        max_heatmap_value = 100
        if heatmap_data and heatmap_data["data"]:
            max_heatmap_value = max(d[2] for d in heatmap_data["data"])
        heatmap_option = json.dumps(
            {
                "tooltip": {"position": "top"},
                "xAxis": {
                    "type": "category",
                    "data": heatmap_data["x_axis"] if heatmap_data else [],
                    "splitArea": {"show": True},
                },
                "yAxis": {
                    "type": "category",
                    "data": heatmap_data["y_axis"] if heatmap_data else [],
                    "splitArea": {"show": True},
                },
                "visualMap": {
                    "min": 0,
                    "max": max_heatmap_value,
                    "calculable": True,
                    "orient": "horizontal",
                    "left": "center",
                    "bottom": "0%",
                },
                "series": [
                    {
                        "name": "Memory MB",
                        "type": "heatmap",
                        "data": heatmap_data["data"] if heatmap_data else [],
                        "label": {"show": True},
                    }
                ],
            }
        )
        heatmap_script = ""
        if heatmap_data:
            heatmap_script = (
                "var heatmapChart = echarts.init("
                "document.getElementById('heatmapChart'));\n"
                f"        heatmapChart.setOption({heatmap_option});"
            )

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ShadowInfer Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5;
        }}
        .header {{
            text-align: center; padding: 20px; background: #1a1a2e; color: white;
            border-radius: 8px; margin-bottom: 20px;
        }}
        .grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .chart {{
            background: white; border-radius: 8px; padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #333;
        }}
        .chart-container {{
            width: 100%; height: 300px;
        }}
        .stats {{
            display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white; border-radius: 8px; padding: 15px;
            text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 28px; font-weight: bold; color: #1a1a2e;
        }}
        .stat-label {{
            font-size: 12px; color: #666; margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ShadowInfer 推理可观测性 Dashboard</h1>
        <p>Metrics / Traces / Logs 三位一体 — 对应 plan-v2.md Phase 1.3</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{len(self.step_history)}</div>
            <div class="stat-label">总 Steps</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_latency:.2f}</div>
            <div class="stat-label">平均延迟 (ms)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_memory:.2f}</div>
            <div class="stat-label">平均内存 (MB)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_accuracy:.4f}</div>
            <div class="stat-label">平均精度</div>
        </div>
    </div>

    <div class="grid">
        <div class="chart">
            <div class="chart-title">延迟趋势 (Latency Trend)</div>
            <div id="latencyChart" class="chart-container"></div>
        </div>
        <div class="chart">
            <div class="chart-title">内存趋势 (Memory Trend)</div>
            <div id="memoryChart" class="chart-container"></div>
        </div>
        <div class="chart">
            <div class="chart-title">精度分布 (Precision Distribution)</div>
            <div id="precisionChart" class="chart-container"></div>
        </div>
        <div class="chart">
            <div class="chart-title">调度时间线 (Scheduling Timeline)</div>
            <div id="timelineChart" class="chart-container"></div>
        </div>
        {heatmap_chart_html}
    </div>

    <script>
        // 延迟曲线
        var latencyChart = echarts.init(document.getElementById('latencyChart'));
        latencyChart.setOption({{
            tooltip: {{ trigger: 'axis' }},
            xAxis: {{ type: 'category', data: {json.dumps(steps)} }},
            yAxis: {{ type: 'value', name: 'ms' }},
            series: [{{
                data: {json.dumps(latency_series)},
                type: 'line', smooth: true,
                areaStyle: {{ opacity: 0.3 }},
                itemStyle: {{ color: '#5470c6' }}
            }}]
        }});

        // 内存趋势
        var memoryChart = echarts.init(document.getElementById('memoryChart'));
        memoryChart.setOption({{
            tooltip: {{ trigger: 'axis' }},
            xAxis: {{ type: 'category', data: {json.dumps(steps)} }},
            yAxis: {{ type: 'value', name: 'MB' }},
            series: [{{
                data: {json.dumps(memory_series)},
                type: 'line', smooth: true,
                areaStyle: {{ opacity: 0.3 }},
                itemStyle: {{ color: '#91cc75' }}
            }}]
        }});

        // 精度分布饼图
        var precisionChart = echarts.init(document.getElementById('precisionChart'));
        precisionChart.setOption({{
            tooltip: {{ trigger: 'item' }},
            legend: {{ orient: 'vertical', left: 'left' }},
            series: [{{
                type: 'pie',
                radius: ['40%', '70%'],
                data: {json.dumps(precision_pie)},
                itemStyle: {{
                    borderRadius: 5,
                    borderColor: '#fff',
                    borderWidth: 2
                }}
            }}]
        }});

        // 调度时间线
        var timelineChart = echarts.init(document.getElementById('timelineChart'));
        timelineChart.setOption({{
            tooltip: {{ trigger: 'axis' }},
            legend: {{ data: {json.dumps(list(timeline_categories))} }},
            xAxis: {{ type: 'category', data: {json.dumps(timeline_steps)} }},
            yAxis: {{ type: 'value', name: 'ms' }},
            series: {json.dumps([
                {"name": cat, "type": "bar", "stack": "total", "data": timeline_series[cat]}
                for cat in timeline_categories
            ])}
        }});

        {heatmap_script}

        window.addEventListener('resize', function() {{
            latencyChart.resize();
            memoryChart.resize();
            precisionChart.resize();
            timelineChart.resize();
            {'heatmapChart.resize();' if heatmap_data else ''}
        }});
    </script>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def export_json(self, output_path: str) -> None:
        """导出为 JSON 数据 — 便于下游系统消费。"""
        data = {
            "step_history": list(self.step_history),
            "latency_trend": self.get_latency_trend(window=len(self.step_history)),
            "memory_trend": self.get_memory_trend(window=len(self.step_history)),
            "precision_distribution": self.get_precision_distribution(),
            "scheduling_timeline": self.get_scheduling_timeline(),
            "layer_memory_heatmap": self.get_layer_memory_heatmap(),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
