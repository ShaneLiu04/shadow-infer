"""Tests for performance regression tracking."""

import pytest

from shadowinfer.benchmarking.regression import (
    RegressionResult,
    RegressionTracker,
    record_from_pipeline_summary,
)


@pytest.fixture
def sample_result():
    return RegressionResult(
        timestamp=1.0,
        run_id="run-1",
        model="SimpleDiffusionLLM",
        backend="pytorch",
        num_steps=20,
        latency_ms=100.0,
        memory_mb=50.0,
        accuracy_drop=0.005,
        speedup=1.5,
        metadata={},
    )


@pytest.fixture
def tracker(tmp_path):
    return RegressionTracker(history_path=str(tmp_path / "history.jsonl"))


def test_record_and_load(tracker, sample_result):
    tracker.record(sample_result)
    loaded = tracker.get_history()
    assert len(loaded) == 1
    assert loaded[0].run_id == "run-1"


def test_get_baseline(tracker, sample_result):
    tracker.record(sample_result)
    baseline = tracker.get_baseline(model="SimpleDiffusionLLM", backend="pytorch")
    assert baseline is not None
    assert baseline.run_id == "run-1"


def test_detect_regression_no_baseline(tracker, sample_result):
    report = tracker.detect_regression(sample_result)
    assert not report["has_regression"]
    assert report["baseline"] is None
    assert report["checks"]["latency"]["status"] == "no_baseline"


def test_detect_latency_regression(tracker, sample_result):
    tracker.record(sample_result)
    regressed = RegressionResult(
        timestamp=2.0,
        run_id="run-2",
        model="SimpleDiffusionLLM",
        backend="pytorch",
        num_steps=20,
        latency_ms=120.0,
        memory_mb=50.0,
        accuracy_drop=0.005,
        speedup=1.5,
        metadata={},
    )
    report = tracker.detect_regression(regressed, latency_threshold=0.10)
    assert report["has_regression"]
    assert report["checks"]["latency"]["status"] == "regression"
    assert report["checks"]["memory"]["status"] == "ok"


def test_detect_memory_regression(tracker, sample_result):
    tracker.record(sample_result)
    regressed = RegressionResult(
        timestamp=2.0,
        run_id="run-2",
        model="SimpleDiffusionLLM",
        backend="pytorch",
        num_steps=20,
        latency_ms=100.0,
        memory_mb=60.0,
        accuracy_drop=0.005,
        speedup=1.5,
        metadata={},
    )
    report = tracker.detect_regression(regressed, memory_threshold=0.10)
    assert report["has_regression"]
    assert report["checks"]["memory"]["status"] == "regression"


def test_detect_accuracy_regression(tracker, sample_result):
    tracker.record(sample_result)
    regressed = RegressionResult(
        timestamp=2.0,
        run_id="run-2",
        model="SimpleDiffusionLLM",
        backend="pytorch",
        num_steps=20,
        latency_ms=100.0,
        memory_mb=50.0,
        accuracy_drop=0.02,
        speedup=1.5,
        metadata={},
    )
    report = tracker.detect_regression(regressed, accuracy_threshold=0.01)
    assert report["has_regression"]
    assert report["checks"]["accuracy"]["status"] == "regression"


def test_summary(tracker, sample_result):
    tracker.record(sample_result)
    summary = tracker.summary(model="SimpleDiffusionLLM", backend="pytorch")
    assert summary["count"] == 1
    assert len(summary["latency_trend"]) == 1


def test_record_from_pipeline_summary(tmp_path):
    summary = {
        "run_id": "run-x",
        "model": "SimpleDiffusionLLM",
        "total_steps": 20,
        "optimized_latency_ms": 80.0,
        "memory_savings_ratio": 0.3,
        "accuracy_drop": 0.004,
        "speedup": 1.6,
    }
    history_path = tmp_path / "history.jsonl"
    result = record_from_pipeline_summary(
        summary, history_path=str(history_path), backend="pytorch"
    )
    assert result.model == "SimpleDiffusionLLM"
    assert result.latency_ms == 80.0
    assert history_path.exists()


def test_history_persistence(tmp_path, sample_result):
    history_path = tmp_path / "history.jsonl"
    tracker = RegressionTracker(history_path=str(history_path))
    tracker.record(sample_result)

    tracker2 = RegressionTracker(history_path=str(history_path))
    assert len(tracker2.get_history()) == 1
