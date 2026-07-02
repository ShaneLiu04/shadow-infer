"""ShadowInfer Web Profiler — Streamlit app for visualizing profiling results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st


def load_json(path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON profiling result file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        st.error(f"Failed to load {path}: {exc}")
        return None


def extract_latency_per_step(data: Dict[str, Any]) -> pd.DataFrame:
    """Extract per-step latency into a DataFrame."""
    per_step = data.get("latency", {}).get("per_step_ms", {})
    rows = []
    for step_id, latency in per_step.items():
        rows.append({"step": int(step_id), "latency_ms": float(latency)})
    return pd.DataFrame(rows).sort_values("step")


def extract_kv_memory(data: Dict[str, Any]) -> pd.DataFrame:
    """Extract per-layer KV memory into a DataFrame."""
    memory = data.get("kv_cache", {}).get("memory_mb", {})
    rows = []
    for layer_id, mem in memory.items():
        rows.append({"layer": int(layer_id), "memory_mb": float(mem)})
    return pd.DataFrame(rows).sort_values("layer")


def extract_alerts(data: Dict[str, Any]) -> pd.DataFrame:
    """Extract alerts into a DataFrame."""
    alerts = data.get("alerts", [])
    if not alerts:
        return pd.DataFrame()
    rows = []
    for alert in alerts:
        rows.append(
            {
                "level": alert.get("level", "INFO"),
                "metric": alert.get("metric", "unknown"),
                "message": alert.get("message", ""),
                "value": alert.get("value", 0.0),
                "step": alert.get("step_id", -1),
            }
        )
    return pd.DataFrame(rows)


def render_summary(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> None:
    """Render summary metrics."""
    base_latency = baseline.get("latency", {}).get("e2e_ms", 0.0)
    opt_latency = optimized.get("latency", {}).get("e2e_ms", 0.0)
    speedup = base_latency / opt_latency if opt_latency > 0 else 0.0

    base_kv = sum(baseline.get("kv_cache", {}).get("memory_mb", {}).values())
    opt_kv = sum(optimized.get("kv_cache", {}).get("memory_mb", {}).values())
    memory_savings = 1.0 - (opt_kv / base_kv) if base_kv > 0 else 0.0

    acc_drop = optimized.get("accuracy", {}).get("perplexity_delta", 0.0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Baseline Latency", f"{base_latency:.2f} ms")
    col2.metric("Optimized Latency", f"{opt_latency:.2f} ms", f"{speedup:.2f}x speedup")
    col3.metric("Memory Savings", f"{memory_savings * 100:.1f}%")
    col4.metric("Accuracy Drop", f"{acc_drop:.4f}")


def render_latency_comparison(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> None:
    """Render latency comparison chart."""
    st.subheader("Per-Step Latency")
    base_df = extract_latency_per_step(baseline)
    opt_df = extract_latency_per_step(optimized)

    if base_df.empty and opt_df.empty:
        st.info("No per-step latency data available.")
        return

    chart_data = pd.DataFrame(
        {
            "step": base_df["step"] if not base_df.empty else opt_df["step"],
            "baseline_ms": base_df["latency_ms"] if not base_df.empty else None,
            "optimized_ms": opt_df["latency_ms"] if not opt_df.empty else None,
        }
    )
    st.line_chart(chart_data.set_index("step"))


def render_kv_memory(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> None:
    """Render KV memory comparison chart."""
    st.subheader("KV Cache Memory per Layer")
    base_df = extract_kv_memory(baseline)
    opt_df = extract_kv_memory(optimized)

    if base_df.empty and opt_df.empty:
        st.info("No KV memory data available.")
        return

    chart_data = pd.DataFrame(
        {
            "layer": base_df["layer"] if not base_df.empty else opt_df["layer"],
            "baseline_mb": base_df["memory_mb"] if not base_df.empty else None,
            "optimized_mb": opt_df["memory_mb"] if not opt_df.empty else None,
        }
    )
    st.bar_chart(chart_data.set_index("layer"))


def render_alerts(data: Dict[str, Any], title: str) -> None:
    """Render alerts table."""
    st.subheader(title)
    alerts_df = extract_alerts(data)
    if alerts_df.empty:
        st.success("No alerts.")
        return
    st.dataframe(alerts_df, use_container_width=True)


def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(page_title="ShadowInfer Web Profiler", layout="wide")
    st.title("ShadowInfer Web Profiler")
    st.markdown("Upload `profile_baseline.json` and `profile_optimized.json" " to compare results.")

    col1, col2 = st.columns(2)
    with col1:
        baseline_file = st.file_uploader("Baseline JSON", type=["json"], key="baseline")
    with col2:
        optimized_file = st.file_uploader("Optimized JSON", type=["json"], key="optimized")

    if baseline_file is None or optimized_file is None:
        st.info("Please upload both baseline and optimized profiling results.")

        # Allow demo with default output files if present
        default_baseline = Path("outputs/profile_baseline.json")
        default_optimized = Path("outputs/profile_optimized.json")
        if default_baseline.exists() and default_optimized.exists():
            if st.button("Load default outputs/profile_*.json"):
                baseline = load_json(str(default_baseline))
                optimized = load_json(str(default_optimized))
                _render_dashboard(baseline, optimized)
        return

    baseline = json.loads(baseline_file.getvalue().decode("utf-8"))
    optimized = json.loads(optimized_file.getvalue().decode("utf-8"))
    _render_dashboard(baseline, optimized)


def _render_dashboard(
    baseline: Optional[Dict[str, Any]], optimized: Optional[Dict[str, Any]]
) -> None:
    """Render the full dashboard."""
    if baseline is None or optimized is None:
        return

    render_summary(baseline, optimized)

    tab1, tab2, tab3 = st.tabs(["Latency", "KV Memory", "Alerts"])
    with tab1:
        render_latency_comparison(baseline, optimized)
    with tab2:
        render_kv_memory(baseline, optimized)
    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            render_alerts(baseline, "Baseline Alerts")
        with col_b:
            render_alerts(optimized, "Optimized Alerts")


if __name__ == "__main__":
    main()
