"""ShadowInfer serving configuration.

对应文档：ROADMAP.md §3.6 Production & Serving
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore

    _YAML_AVAILABLE = True
except Exception:  # pragma: no cover
    _YAML_AVAILABLE = False


@dataclass
class ServingConfig:
    """Production serving runtime configuration.

    All fields have sensible defaults for CPU-only / single-user demos; they
    should be overridden via a YAML config in real deployments.
    """

    # Concurrency
    max_concurrent_requests: int = 1
    request_timeout: float = 300.0

    # Rate limiting (token bucket)
    rate_limit_rps: float = 10.0
    rate_limit_burst: int = 20

    # A/B testing weights (must sum to 1.0)
    ab_weights: Dict[str, float] = field(
        default_factory=lambda: {"baseline": 0.5, "optimized": 0.5}
    )

    # Observability
    metrics_enabled: bool = True

    # Hot config reload
    hot_config_path: Optional[str] = None
    hot_config_interval: float = 5.0

    # Default generation parameters
    default_num_steps: int = 20
    default_strategy: str = "optimized"

    # Backend selection (CLI --backend takes precedence when explicitly set)
    backend_type: Optional[str] = None
    backend_args: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServingConfig":
        """Build a config from a plain dictionary."""
        if not isinstance(data, dict):
            raise ValueError("ServingConfig must be built from a dictionary")
        kwargs: Dict[str, Any] = {}
        for key in cls.__dataclass_fields__:
            if key in data:
                kwargs[key] = data[key]

        # Support nested ``backend: {type, args}`` section.
        backend = data.get("backend")
        if isinstance(backend, dict):
            if "type" in backend:
                kwargs["backend_type"] = backend["type"]
            if "args" in backend:
                kwargs["backend_args"] = backend["args"]

        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "ServingConfig":
        """Load config from a YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Serving config not found: {path}")
        if not _YAML_AVAILABLE:
            raise RuntimeError("PyYAML is required to load YAML serving config")
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    def validate(self) -> None:
        """Validate the configuration and raise ValueError on invalid input."""
        if self.max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be >= 1")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be > 0")
        if self.rate_limit_rps <= 0:
            raise ValueError("rate_limit_rps must be > 0")
        if self.rate_limit_burst < 1:
            raise ValueError("rate_limit_burst must be >= 1")
        if not self.ab_weights:
            raise ValueError("ab_weights must not be empty")
        total = sum(self.ab_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ab_weights must sum to 1.0, got {total}")
        for key in self.ab_weights:
            if self.ab_weights[key] < 0:
                raise ValueError(f"ab_weights[{key}] must be non-negative")
        if self.hot_config_interval <= 0:
            raise ValueError("hot_config_interval must be > 0")
        if self.default_num_steps < 1:
            raise ValueError("default_num_steps must be >= 1")
        if self.default_strategy not in ("baseline", "optimized", "ab"):
            raise ValueError("default_strategy must be baseline/optimized/ab")
        if self.backend_type is not None and not isinstance(self.backend_type, str):
            raise ValueError("backend_type must be a string or null")
        if not isinstance(self.backend_args, dict):
            raise ValueError("backend_args must be a dictionary")
