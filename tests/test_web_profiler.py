"""Tests for ShadowInfer Web Profiler."""

import json

import pandas as pd
import pytest

from shadowinfer.web_profiler.app import (
    extract_alerts,
    extract_kv_memory,
    extract_latency_per_step,
    load_json,
    render_summary,
)


@pytest.fixture
def sample_baseline():
    return {
        "model": "test",
        "latency": {
            "e2e_ms": 100.0,
            "per_step_ms": {0: 2.0, 1: 2.5, 2: 2.2},
        },
        "kv_cache": {"memory_mb": {0: 10.0, 1: 12.0}},
        "accuracy": {"perplexity_delta": 0.005},
        "alerts": [],
    }


@pytest.fixture
def sample_optimized():
    return {
        "model": "test",
        "latency": {
            "e2e_ms": 70.0,
            "per_step_ms": {0: 1.5, 1: 1.6, 2: 1.4},
        },
        "kv_cache": {"memory_mb": {0: 6.0, 1: 7.0}},
        "accuracy": {"perplexity_delta": 0.008},
        "alerts": [
            {
                "level": "WARNING",
                "metric": "latency",
                "message": "Latency high",
                "value": 80.0,
                "step_id": 1,
            }
        ],
    }


def test_extract_latency_per_step(sample_baseline):
    df = extract_latency_per_step(sample_baseline)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["step", "latency_ms"]
    assert df["latency_ms"].iloc[0] == 2.0


def test_extract_kv_memory(sample_baseline):
    df = extract_kv_memory(sample_baseline)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df["memory_mb"].sum() == 22.0


def test_extract_alerts(sample_optimized):
    df = extract_alerts(sample_optimized)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["level"] == "WARNING"


def test_load_json_valid(sample_baseline, tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(sample_baseline), encoding="utf-8")
    loaded = load_json(str(path))
    assert loaded is not None
    assert loaded["model"] == sample_baseline["model"]
    assert loaded["latency"]["e2e_ms"] == sample_baseline["latency"]["e2e_ms"]


def test_load_json_missing():
    loaded = load_json("/nonexistent/path.json")
    assert loaded is None


def test_render_summary_no_error(sample_baseline, sample_optimized):
    # streamlit calls cannot be fully tested without a streamlit runtime,
    # so we just verify the function accepts the expected inputs without raising.
    # The function itself only calls st.metric, which is safe to import.
    try:
        render_summary(sample_baseline, sample_optimized)
    except Exception as exc:
        pytest.fail(f"render_summary raised unexpectedly: {exc}")


def test_app_module_importable():
    import shadowinfer.web_profiler.app as app_module

    assert hasattr(app_module, "main")
