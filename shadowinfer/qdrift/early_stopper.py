"""Uncertainty-aware early stopping for Diffusion LLM denoising.

Diffusion LLM spends a fixed budget of denoising steps.  In practice, the
output often stabilizes before the budget is exhausted.  This module tracks
the similarity between consecutive step outputs and signals the Orchestrator
to stop early once the output has been stable for a configurable window.

Version: 3.2.2
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class EarlyStopState:
    """Snapshot returned by ``UncertaintyEarlyStopper.observe``."""

    step_id: int
    similarity: float
    should_stop: bool
    stable_steps: int
    total_steps: int


class UncertaintyEarlyStopper:
    """Stop denoising early when consecutive outputs become stable.

    Args:
        min_steps: Minimum number of steps to run before early stopping is
            allowed.
        max_steps: Hard upper bound on the number of steps.
        stability_window: Number of consecutive stable steps required to
            trigger early stopping.
        similarity_threshold: Similarity value above which two consecutive
            outputs are considered "stable".
        metric: Similarity metric.  Supported: ``relative_l2`` (higher means
            more similar), ``cosine`` (higher means more similar),
            ``mse`` (lower means more similar).
    """

    def __init__(
        self,
        min_steps: int = 5,
        max_steps: int = 50,
        stability_window: int = 3,
        similarity_threshold: float = 0.995,
        metric: str = "relative_l2",
    ) -> None:
        if min_steps < 1:
            raise ValueError("min_steps must be >= 1")
        if stability_window < 1:
            raise ValueError("stability_window must be >= 1")
        if metric not in {"relative_l2", "cosine", "mse"}:
            raise ValueError(f"Unsupported metric: {metric}")

        self.min_steps = min_steps
        self.max_steps = max_steps
        self.stability_window = stability_window
        self.similarity_threshold = similarity_threshold
        self.metric = metric
        self._previous: Optional[torch.Tensor] = None
        self._stable_count = 0
        self._step_id = -1
        self._history: List[Dict[str, Any]] = []

    def observe(self, step_id: int, output: torch.Tensor) -> EarlyStopState:
        """Observe a step output and decide whether to stop early.

        Args:
            step_id: Current denoising step index (0-based).
            output: Model output tensor for the current step.  Any shape is
                accepted as long as it is comparable across steps.

        Returns:
            ``EarlyStopState`` containing the similarity to the previous step,
            whether early stopping should trigger, and diagnostic counters.
        """
        self._step_id = step_id
        output = output.detach().float()

        if self._previous is None:
            similarity = 0.0
            self._stable_count = 0
        else:
            similarity = _compute_similarity(self._previous, output, metric=self.metric)
            if _is_stable(similarity, self.similarity_threshold, self.metric):
                self._stable_count += 1
            else:
                self._stable_count = 0

        self._previous = output

        should_stop = False
        if step_id + 1 >= self.max_steps:
            should_stop = True
        elif step_id + 1 >= self.min_steps and self._stable_count >= self.stability_window:
            should_stop = True
            logger.info(
                "Early stopping triggered at step %d (similarity=%.4f, " "stable_count=%d)",
                step_id,
                similarity,
                self._stable_count,
            )

        record = {
            "step_id": step_id,
            "similarity": similarity,
            "stable_count": self._stable_count,
            "should_stop": should_stop,
        }
        self._history.append(record)

        return EarlyStopState(
            step_id=step_id,
            similarity=similarity,
            should_stop=should_stop,
            stable_steps=self._stable_count,
            total_steps=step_id + 1,
        )

    def should_stop(self) -> bool:
        """Return True if the most recent observation requested a stop."""
        if not self._history:
            return False
        return self._history[-1]["should_stop"]

    def reset(self) -> None:
        """Clear internal state for a new inference run."""
        self._previous = None
        self._stable_count = 0
        self._step_id = -1
        self._history.clear()

    def history(self) -> List[Dict[str, Any]]:
        """Return the full observation history."""
        return list(self._history)


def _compute_similarity(
    prev: torch.Tensor,
    curr: torch.Tensor,
    metric: str,
) -> float:
    """Compute similarity between two step outputs."""
    prev = prev.flatten()
    curr = curr.flatten()

    if metric == "relative_l2":
        denom = torch.norm(curr) + 1e-8
        sim = 1.0 - float(torch.norm(prev - curr) / denom)
    elif metric == "cosine":
        sim = float(F.cosine_similarity(prev.unsqueeze(0), curr.unsqueeze(0)).item())
    elif metric == "mse":
        sim = float(F.mse_loss(prev, curr).item())
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return sim


def _is_stable(similarity: float, threshold: float, metric: str) -> bool:
    """Return True if the similarity value indicates stability."""
    if metric == "mse":
        return similarity <= threshold
    return similarity >= threshold


@dataclass
class EarlyStopConfig:
    """Configuration dataclass for the early stopper."""

    enabled: bool = False
    min_steps: int = 5
    max_steps: int = 50
    stability_window: int = 3
    similarity_threshold: float = 0.995
    metric: str = "relative_l2"

    def build(self) -> UncertaintyEarlyStopper:
        """Build an ``UncertaintyEarlyStopper`` from this config."""
        if not self.enabled:
            raise RuntimeError("Early stopping is disabled")
        return UncertaintyEarlyStopper(
            min_steps=self.min_steps,
            max_steps=self.max_steps,
            stability_window=self.stability_window,
            similarity_threshold=self.similarity_threshold,
            metric=self.metric,
        )
