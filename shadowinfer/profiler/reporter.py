"""HTML 报告生成器。

对应 PROFILER_AGENT.md 输出格式要求，生成包含图表和表格的可视化对比报告。
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List


class HTMLReporter:
    """生成 HTML 可视化对比报告。"""

    def generate(
        self, baseline: Dict[str, Any], optimized: Dict[str, Any], output_path: str
    ) -> None:
        """生成包含图表和表格的 HTML 报告。

        Args:
            baseline: 基线性能数据字典。
            optimized: 优化后性能数据字典。
            output_path: 输出 HTML 文件路径。
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        html = self._build_html(baseline, optimized)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def _build_html(self, baseline: Dict[str, Any], optimized: Dict[str, Any]) -> str:
        model = optimized.get("model", baseline.get("model", "Unknown"))
        run_id = optimized.get("run_id", baseline.get("run_id", "N/A"))
        total_steps = optimized.get("total_steps", 0)

        # 汇总数值
        baseline_latency = baseline.get("latency", {}).get("e2e_ms", 0.0)
        optimized_latency = optimized.get("latency", {}).get("e2e_ms", 0.0)
        speedup = baseline_latency / optimized_latency if optimized_latency > 0 else 0.0

        baseline_kv_mem = (
            sum(baseline.get("kv_cache", {}).get("memory_mb", {}).values())
            if baseline.get("kv_cache", {}).get("memory_mb")
            else 0.0
        )
        optimized_kv_mem = (
            sum(optimized.get("kv_cache", {}).get("memory_mb", {}).values())
            if optimized.get("kv_cache", {}).get("memory_mb")
            else 0.0
        )

        ppl_delta = optimized.get("accuracy", {}).get("perplexity_delta", 0.0)
        bleu_drop = optimized.get("accuracy", {}).get("bleu_drop", 0.0)

        alerts: List[Dict[str, Any]] = optimized.get("alerts", [])
        alert_counts = {"CRITICAL": 0, "WARNING": 0, "INFO": 0}
        for a in alerts:
            alert_counts[a.get("level", "INFO")] += 1

        baseline_per_step = baseline.get("latency", {}).get("per_step_ms", {})
        optimized_per_step = optimized.get("latency", {}).get("per_step_ms", {})

        baseline_latency_json = json.dumps(baseline_per_step, default=str)
        optimized_latency_json = json.dumps(optimized_per_step, default=str)
        alert_counts_json = json.dumps(alert_counts)

        latency_pct = (
            ((optimized_latency - baseline_latency) / baseline_latency * 100)
            if baseline_latency > 0
            else 0
        )
        latency_change = (
            f"{'+' if optimized_latency > baseline_latency else ''}"
            f"{optimized_latency - baseline_latency:.2f} ms ({latency_pct:.1f}%)"
        )
        kv_pct = (
            ((optimized_kv_mem - baseline_kv_mem) / baseline_kv_mem * 100)
            if baseline_kv_mem > 0
            else 0
        )
        kv_change = (
            f"{'+' if optimized_kv_mem > baseline_kv_mem else ''}"
            f"{optimized_kv_mem - baseline_kv_mem:.1f} MB ({kv_pct:.1f}%)"
        )

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ShadowInfer Profile Report — {model}</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f7fa;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            padding: 40px;
        }}
        h1 {{
            font-size: 28px;
            margin-bottom: 8px;
            color: #1a1a2e;
        }}
        .meta {{
            color: #666;
            font-size: 14px;
            margin-bottom: 32px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #4a90d9;
        }}
        .card.warning {{
            border-left-color: #e6a23c;
        }}
        .card.critical {{
            border-left-color: #f56c6c;
        }}
        .card.success {{
            border-left-color: #67c23a;
        }}
        .card h3 {{
            font-size: 12px;
            color: #666;
            margin: 0 0 8px 0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .card .value {{
            font-size: 28px;
            font-weight: 700;
            color: #1a1a2e;
        }}
        .card .unit {{
            font-size: 14px;
            color: #888;
            margin-left: 4px;
        }}
        .section {{
            margin-top: 40px;
        }}
        .section h2 {{
            font-size: 20px;
            color: #1a1a2e;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid #eee;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #555;
        }}
        tr:hover {{
            background: #fafbfc;
        }}
        .chart {{
            width: 100%;
            height: 400px;
            margin-top: 20px;
        }}
        .alert-item {{
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 8px;
            font-size: 14px;
        }}
        .alert-critical {{
            background: #fef0f0;
            color: #c45656;
            border: 1px solid #fbc4c4;
        }}
        .alert-warning {{
            background: #fdf6ec;
            color: #a16215;
            border: 1px solid #f5dab1;
        }}
        .alert-info {{
            background: #f0f9ff;
            color: #2c6cb3;
            border: 1px solid #b3d8ff;
        }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            color: #999;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ShadowInfer Profile Report</h1>
        <div class="meta">
            Model: <strong>{model}</strong> &nbsp;|&nbsp; Run ID: <strong>{run_id}</strong> "
            f"&nbsp;|&nbsp; Total Steps: <strong>{total_steps}</strong> &nbsp;|&nbsp; "
            f"Generated: <strong>{datetime.now(timezone.utc).isoformat()}</strong>
        </div>

        <div class="summary-grid">
            <div class="card success">
                <h3>Speedup Ratio</h3>
                <div class="value">{speedup:.2f}<span class="unit">x</span></div>
            </div>
            <div class="card {'critical' if alert_counts['CRITICAL'] > 0 else 'success'}">
                <h3>Critical Alerts</h3>
                <div class="value">{alert_counts['CRITICAL']}</div>
            </div>
            <div class="card {'warning' if alert_counts['WARNING'] > 0 else 'success'}">
                <h3>Warning Alerts</h3>
                <div class="value">{alert_counts['WARNING']}</div>
            </div>
            <div class="card">
                <h3>Perplexity Delta</h3>
                <div class="value">{ppl_delta:.4f}</div>
            </div>
            <div class="card">
                <h3>BLEU Drop</h3>
                <div class="value">{bleu_drop:.4f}</div>
            </div>
            <div class="card">
                <h3>KV Memory (Opt)</h3>
                <div class="value">{optimized_kv_mem:.1f}<span class="unit">MB</span></div>
            </div>
        </div>

        <div class="section">
            <h2>Latency Comparison (Per Step)</h2>
            <div id="latencyChart" class="chart"></div>
        </div>

        <div class="section">
            <h2>Metrics Comparison Table</h2>
            <table>
                <thead>
                    <tr><th>Metric</th><th>Baseline</th><th>Optimized</th><th>Change</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td>End-to-End Latency</td>
                        <td>{baseline_latency:.2f} ms</td>
                        <td>{optimized_latency:.2f} ms</td>
                        <td>{latency_change}</td>
                    </tr>
                    <tr>
                        <td>KV Cache Memory</td>
                        <td>{baseline_kv_mem:.1f} MB</td>
                        <td>{optimized_kv_mem:.1f} MB</td>
                        <td>{kv_change}</td>
                    </tr>
                    <tr>
                        <td>Perplexity Delta</td>
                        <td>—</td>
                        <td>{ppl_delta:.4f}</td>
                        <td>—</td>
                    </tr>
                    <tr>
                        <td>BLEU Drop</td>
                        <td>—</td>
                        <td>{bleu_drop:.4f}</td>
                        <td>—</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Alerts</h2>
            {self._render_alerts(alerts)}
        </div>

        <div class="footer">
            Generated by ShadowInfer Profiler Agent v1.0
        </div>
    </div>

    <script>
        const baselineLatency = {baseline_latency_json};
        const optimizedLatency = {optimized_latency_json};
        const alertCounts = {alert_counts_json};

        const steps = Object.keys(baselineLatency).map(Number).sort((a, b) => a - b);
        const baselineData = steps.map(s => baselineLatency[s] || 0);
        const optimizedData = steps.map(s => optimizedLatency[s] || 0);

        const latencyChart = echarts.init(document.getElementById('latencyChart'));
        const option = {{
            tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'cross' }} }},
            legend: {{ data: ['Baseline', 'Optimized'] }},
            grid: {{ left: '3%', right: '4%', bottom: '3%', containLabel: true }},
            xAxis: {{
                type: 'category',
                boundaryGap: false,
                data: steps.map(s => 'Step ' + s)
            }},
            yAxis: {{ type: 'value', name: 'Latency (ms)' }},
            series: [
                {{
                    name: 'Baseline',
                    type: 'line',
                    data: baselineData,
                    smooth: true,
                    lineStyle: {{ color: '#909399' }},
                    itemStyle: {{ color: '#909399' }}
                }},
                {{
                    name: 'Optimized',
                    type: 'line',
                    data: optimizedData,
                    smooth: true,
                    lineStyle: {{ color: '#4a90d9' }},
                    itemStyle: {{ color: '#4a90d9' }}
                }}
            ]
        }};
        latencyChart.setOption(option);
        window.addEventListener('resize', function() {{ latencyChart.resize(); }});
    </script>
</body>
</html>"""
        return html

    def _render_alerts(self, alerts: List[Dict[str, Any]]) -> str:
        if not alerts:
            return (
                '<div class="alert-item alert-info">'
                "No alerts detected. System operating within normal parameters.</div>"
            )

        html_parts = []
        for alert in alerts:
            level = alert.get("level", "INFO").lower()
            msg = alert.get("message", "")
            rec = alert.get("recommendation", "")
            html_parts.append(
                f'<div class="alert-item alert-{level}">'
                f"<strong>[{level.upper()}]</strong> {msg}"
                f"<br><small>Recommendation: {rec}</small>"
                f"</div>"
            )
        return "\n".join(html_parts)
