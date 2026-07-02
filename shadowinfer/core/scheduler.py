"""Learned Multi-Agent Scheduler for ShadowInfer.

Trains a lightweight surrogate model offline from recorded step results and
predicts the best (shadowkv_mode, ffn_mode) action for each step.

对应文档：ROADMAP.md §4.2
版本：v3.2
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Ordered action space used by the learned scheduler.
_ACTIONS: List[Tuple[str, str]] = [
    ("conservative", "full"),
    ("balanced", "full"),
    ("balanced", "mixed"),
    ("balanced", "sparse"),
    ("aggressive", "mixed"),
    ("aggressive", "sparse"),
]


def _action_index(shadowkv_mode: str, ffn_mode: str) -> int:
    try:
        return _ACTIONS.index((shadowkv_mode, ffn_mode))
    except ValueError:
        return 0


@dataclass
class StepExperience:
    """One recorded step experience used for training the scheduler."""

    step_id: int
    total_steps: int
    noise_level: float
    sensitivity_score: float
    drift_score: float
    prev_latency_ms: float
    prev_memory_mb: float
    prev_accuracy_drop: float
    shadowkv_mode: str
    ffn_mode: str
    latency_ms: float
    memory_mb: float
    accuracy_drop: float

    def features(self) -> torch.Tensor:
        return torch.tensor(
            [
                self.step_id / max(self.total_steps, 1),
                self.noise_level,
                self.sensitivity_score,
                self.drift_score,
                self.prev_latency_ms,
                self.prev_memory_mb,
                self.prev_accuracy_drop,
            ],
            dtype=torch.float32,
        )

    def reward(self, latency_weight: float = 1.0, memory_weight: float = 0.5) -> float:
        """Reward is higher when latency and memory are lower and accuracy is preserved."""
        return -latency_weight * self.latency_ms - memory_weight * self.memory_mb


class _SchedulerNet(nn.Module):
    """Small MLP that predicts per-action value."""

    def __init__(self, input_dim: int = 7, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(_ACTIONS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedScheduler:
    """Learned multi-agent scheduler.

    The scheduler maintains a small neural network that maps step features to
    expected cumulative reward for each (shadowkv_mode, ffn_mode) action.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model = _SchedulerNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.experiences: List[StepExperience] = []
        self.model_path = Path(model_path) if model_path else None
        if self.model_path and self.model_path.exists():
            self.load(self.model_path)

    def add_experience(self, experience: StepExperience) -> None:
        """Record one step experience."""
        self.experiences.append(experience)

    def predict(
        self,
        step_id: int,
        total_steps: int,
        noise_level: float,
        sensitivity_score: float,
        drift_score: float,
        prev_latency_ms: float = 0.0,
        prev_memory_mb: float = 0.0,
        prev_accuracy_drop: float = 0.0,
    ) -> Tuple[str, str]:
        """Predict the best action for the current step."""
        exp = StepExperience(
            step_id=step_id,
            total_steps=total_steps,
            noise_level=noise_level,
            sensitivity_score=sensitivity_score,
            drift_score=drift_score,
            prev_latency_ms=prev_latency_ms,
            prev_memory_mb=prev_memory_mb,
            prev_accuracy_drop=prev_accuracy_drop,
            shadowkv_mode="balanced",
            ffn_mode="mixed",
            latency_ms=0.0,
            memory_mb=0.0,
            accuracy_drop=0.0,
        )
        with torch.no_grad():
            values = self.model(exp.features().unsqueeze(0))
            action_idx = int(values.argmax(dim=-1).item())
        return _ACTIONS[action_idx]

    def train(self, epochs: int = 10) -> List[float]:
        """Train the scheduler on recorded experiences.

        Returns:
            List of average loss per epoch.
        """
        if len(self.experiences) < 2:
            return []

        losses: List[float] = []
        criterion = nn.MSELoss()

        X = torch.stack([e.features() for e in self.experiences])
        rewards = torch.tensor([e.reward() for e in self.experiences], dtype=torch.float32)
        actions = torch.tensor(
            [_action_index(e.shadowkv_mode, e.ffn_mode) for e in self.experiences],
            dtype=torch.long,
        )

        # Normalize rewards for stable training.
        reward_mean = rewards.mean()
        reward_std = rewards.std() + 1e-8
        normalized_rewards = (rewards - reward_mean) / reward_std

        for _ in range(epochs):
            values = self.model(X)  # [N, num_actions]
            target = values.detach().clone()
            target[range(len(actions)), actions] = normalized_rewards
            loss = criterion(values, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.item()))

        return losses

    def save(self, path: Optional[str] = None) -> None:
        """Save model weights and experiences."""
        save_path = Path(path) if path else self.model_path
        if save_path is None:
            raise ValueError("No save path provided.")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "experiences": [self._exp_to_dict(e) for e in self.experiences],
            },
            save_path,
        )

    def load(self, path: Optional[str] = None) -> None:
        """Load model weights and experiences."""
        load_path = Path(path) if path else self.model_path
        if load_path is None or not load_path.exists():
            return
        checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.experiences = [self._exp_from_dict(d) for d in checkpoint.get("experiences", [])]

    @staticmethod
    def _exp_to_dict(exp: StepExperience) -> Dict[str, Any]:
        return {
            "step_id": exp.step_id,
            "total_steps": exp.total_steps,
            "noise_level": exp.noise_level,
            "sensitivity_score": exp.sensitivity_score,
            "drift_score": exp.drift_score,
            "prev_latency_ms": exp.prev_latency_ms,
            "prev_memory_mb": exp.prev_memory_mb,
            "prev_accuracy_drop": exp.prev_accuracy_drop,
            "shadowkv_mode": exp.shadowkv_mode,
            "ffn_mode": exp.ffn_mode,
            "latency_ms": exp.latency_ms,
            "memory_mb": exp.memory_mb,
            "accuracy_drop": exp.accuracy_drop,
        }

    @staticmethod
    def _exp_from_dict(data: Dict[str, Any]) -> StepExperience:
        return StepExperience(**data)

    def export_policy_json(self, output_path: str) -> None:
        """Export a sample grid policy for inspection."""
        grid_points = [0.0, 0.25, 0.5, 0.75, 1.0]
        policy = []
        for noise in grid_points:
            for sens in grid_points:
                for drift in grid_points:
                    action = self.predict(
                        step_id=int(noise * 50),
                        total_steps=50,
                        noise_level=noise,
                        sensitivity_score=sens,
                        drift_score=drift,
                    )
                    policy.append(
                        {
                            "noise_level": noise,
                            "sensitivity_score": sens,
                            "drift_score": drift,
                            "shadowkv_mode": action[0],
                            "ffn_mode": action[1],
                        }
                    )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(policy, f, indent=2, ensure_ascii=False)
