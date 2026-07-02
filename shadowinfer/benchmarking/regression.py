"""Performance regression tracking for ShadowInfer.

This module records benchmark results over time and detects regressions
in latency, memory, and accuracy.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RegressionResult:
    """Single benchmark snapshot used for regression tracking."""

    timestamp: float
    run_id: str
    model: str
    backend: str
    num_steps: int
    latency_ms: float
    memory_mb: float
    accuracy_drop: float
    speedup: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegressionResult":
        return cls(**data)


class RegressionTracker:
    """Track benchmark results and detect performance regressions.

    The tracker stores results in a JSONL file and compares the latest result
    against a baseline (e.g., the previous main-branch result or a named
    reference run).
    """

    def __init__(self, history_path: Optional[str] = None) -> None:
        """Initialize the tracker.

        Args:
            history_path: Path to the JSONL history file. If None, uses
                ``benchmarks/results/regression_history.jsonl``.
        """
        if history_path is None:
            history_path = "benchmarks/results/regression_history.jsonl"
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._history: List[RegressionResult] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load existing history from disk."""
        if not self.history_path.exists():
            return
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self._history.append(RegressionResult.from_dict(data))
        except Exception:
            self._history = []

    def record(self, result: RegressionResult) -> None:
        """Append a new result to the history."""
        self._history.append(result)
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False, default=str) + "\n")

    def get_history(
        self,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[RegressionResult]:
        """Return filtered history.

        Args:
            model: Filter by model name.
            backend: Filter by backend name.
            limit: Maximum number of results to return (most recent first).

        Returns:
            List of matching RegressionResult entries.
        """
        results = self._history[:]
        if model is not None:
            results = [r for r in results if r.model == model]
        if backend is not None:
            results = [r for r in results if r.backend == backend]
        results = sorted(results, key=lambda r: r.timestamp, reverse=True)
        if limit is not None:
            results = results[:limit]
        return results

    def get_baseline(
        self, model: Optional[str] = None, backend: Optional[str] = None
    ) -> Optional[RegressionResult]:
        """Return the most recent historical result for the given model/backend."""
        history = self.get_history(model=model, backend=backend, limit=1)
        return history[0] if history else None

    def detect_regression(
        self,
        current: RegressionResult,
        latency_threshold: float = 0.10,
        memory_threshold: float = 0.10,
        accuracy_threshold: float = 0.005,
    ) -> Dict[str, Any]:
        """Compare ``current`` against the stored baseline and report regressions.

        Args:
            current: The latest benchmark result.
            latency_threshold: Relative latency increase that triggers a regression.
            memory_threshold: Relative memory increase that triggers a regression.
            accuracy_threshold: Absolute accuracy drop increase that triggers a regression.

        Returns:
            Dictionary with regression flags, deltas, and summary.
        """
        baseline = self.get_baseline(model=current.model, backend=current.backend)
        report: Dict[str, Any] = {
            "has_regression": False,
            "baseline": baseline.to_dict() if baseline else None,
            "current": current.to_dict(),
            "checks": {},
        }

        if baseline is None:
            report["checks"]["latency"] = {
                "status": "no_baseline",
                "delta_relative": 0.0,
                "threshold": latency_threshold,
            }
            report["checks"]["memory"] = {
                "status": "no_baseline",
                "delta_relative": 0.0,
                "threshold": memory_threshold,
            }
            report["checks"]["accuracy"] = {
                "status": "no_baseline",
                "delta_absolute": 0.0,
                "threshold": accuracy_threshold,
            }
            return report

        latency_delta = _relative_delta(baseline.latency_ms, current.latency_ms)
        memory_delta = _relative_delta(baseline.memory_mb, current.memory_mb)
        accuracy_delta = current.accuracy_drop - baseline.accuracy_drop

        latency_regress = latency_delta > latency_threshold
        memory_regress = memory_delta > memory_threshold
        accuracy_regress = accuracy_delta > accuracy_threshold

        report["has_regression"] = latency_regress or memory_regress or accuracy_regress
        report["checks"]["latency"] = {
            "status": "regression" if latency_regress else "ok",
            "delta_relative": latency_delta,
            "threshold": latency_threshold,
        }
        report["checks"]["memory"] = {
            "status": "regression" if memory_regress else "ok",
            "delta_relative": memory_delta,
            "threshold": memory_threshold,
        }
        report["checks"]["accuracy"] = {
            "status": "regression" if accuracy_regress else "ok",
            "delta_absolute": accuracy_delta,
            "threshold": accuracy_threshold,
        }
        return report

    def summary(self, model: Optional[str] = None, backend: Optional[str] = None) -> Dict[str, Any]:
        """Return a summary of historical trends."""
        history = self.get_history(model=model, backend=backend)
        if not history:
            return {"count": 0, "latency_trend": [], "memory_trend": [], "accuracy_trend": []}

        return {
            "count": len(history),
            "latency_trend": [(r.timestamp, r.latency_ms) for r in history],
            "memory_trend": [(r.timestamp, r.memory_mb) for r in history],
            "accuracy_trend": [(r.timestamp, r.accuracy_drop) for r in history],
            "latest": history[0].to_dict(),
        }


def _relative_delta(baseline: float, current: float) -> float:
    """Compute relative delta. Lower is better for latency/memory."""
    if baseline == 0:
        return 0.0 if current == 0 else float("inf")
    return (current - baseline) / baseline


def record_from_pipeline_summary(
    summary: Dict[str, Any],
    history_path: Optional[str] = None,
    backend: str = "mock",
    metadata: Optional[Dict[str, Any]] = None,
) -> RegressionResult:
    """Convenience helper to record a result from an Orchestrator pipeline summary."""
    result = RegressionResult(
        timestamp=time.time(),
        run_id=summary.get("run_id", "unknown"),
        model=summary.get("model", "unknown"),
        backend=backend,
        num_steps=summary.get("total_steps", 0),
        latency_ms=summary.get("optimized_latency_ms", 0.0),
        memory_mb=summary.get("memory_savings_ratio", 0.0) * 1000.0,
        accuracy_drop=summary.get("accuracy_drop", 0.0),
        speedup=summary.get("speedup", 0.0),
        metadata=metadata or {},
    )
    tracker = RegressionTracker(history_path=history_path)
    tracker.record(result)
    return result
