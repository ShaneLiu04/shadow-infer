"""Production serving backend with rate limiting, concurrency control,
observability, A/B testing, and hot config reload.

对应文档：ROADMAP.md §3.6 Production & Serving
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Dict, Optional

import torch

from shadowinfer.core.model_backend import ModelBackend
from shadowinfer.orchestrator import Orchestrator
from shadowinfer.serving.config import ServingConfig
from shadowinfer.serving.limiters import ConcurrencyLimiter, TokenBucketRateLimiter
from shadowinfer.serving.metrics import ServingMetrics


class ServingBackend:
    """Wraps a ``ModelBackend`` for production request-time inference.

    Features:
    - Token-bucket rate limiting (HTTP 429).
    - Concurrency limiting with timeout (HTTP 503).
    - Prometheus-style metrics.
    - Weighted A/B traffic splitting.
    - Hot config reload without restart.
    """

    def __init__(
        self,
        model_backend: ModelBackend,
        config: Optional[ServingConfig] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_backend = model_backend
        self.config_path = config_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure the backend is loaded on the target device.
        model_name = self.model_backend.get_model_config().get("name", "shadowinfer_model")
        self.model_backend.load(model_name, device=self.device)

        self._config_lock = threading.RLock()
        self._config = config or ServingConfig()
        self._config.validate()

        self._rate_limiter = TokenBucketRateLimiter(
            self._config.rate_limit_rps,
            self._config.rate_limit_burst,
        )
        self._concurrency_limiter = ConcurrencyLimiter(self._config.max_concurrent_requests)
        self.metrics = ServingMetrics(enabled=self._config.metrics_enabled)

        self._reload_thread: Optional[threading.Thread] = None
        self._reload_stop = threading.Event()
        self._start_hot_reload()

    # ------------------------------------------------------------------
    # Config management
    # ------------------------------------------------------------------

    @property
    def config(self) -> ServingConfig:
        with self._config_lock:
            return self._config

    def _update_config(self, new_config: ServingConfig) -> None:
        """Atomically swap runtime configuration."""
        new_config.validate()
        with self._config_lock:
            old = self._config
            self._config = new_config
            # Rebuild limiters if limits changed.
            if (
                old.rate_limit_rps != new_config.rate_limit_rps
                or old.rate_limit_burst != new_config.rate_limit_burst
            ):
                self._rate_limiter = TokenBucketRateLimiter(
                    new_config.rate_limit_rps,
                    new_config.rate_limit_burst,
                )
            if old.max_concurrent_requests != new_config.max_concurrent_requests:
                self._concurrency_limiter = ConcurrencyLimiter(new_config.max_concurrent_requests)
            self.metrics.enabled = new_config.metrics_enabled

    def reload_config(self, path: Optional[str] = None) -> None:
        """Manually reload serving config from a YAML file."""
        path = path or self.config.hot_config_path
        if path is None:
            return
        new_config = ServingConfig.from_yaml(path)
        self._update_config(new_config)

    def _start_hot_reload(self) -> None:
        """Start a background thread that periodically reloads config."""
        if not self.config.hot_config_path:
            return

        def _watch() -> None:
            while not self._reload_stop.is_set():
                try:
                    self.reload_config()
                except Exception:  # pragma: no cover - defensive
                    pass
                self._reload_stop.wait(self.config.hot_config_interval)

        self._reload_thread = threading.Thread(target=_watch, daemon=True)
        self._reload_thread.start()

    def stop_hot_reload(self) -> None:
        """Stop the hot config reload thread."""
        if self._reload_thread is not None:
            self._reload_stop.set()
            self._reload_thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def pick_ab_strategy(self) -> str:
        """Pick a strategy according to configured A/B weights."""
        cfg = self.config
        weights = cfg.ab_weights
        strategies = list(weights.keys())
        probs = [weights[k] for k in strategies]
        return random.choices(strategies, weights=probs, k=1)[0]

    def generate(
        self,
        prompt: str,
        num_steps: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a single generation request.

        Args:
            prompt: Input prompt text.
            num_steps: Number of denoising steps; defaults to config.
            strategy: One of ``baseline``, ``optimized``, or ``ab``;
                defaults to config.

        Returns:
            Dictionary with request metadata and the Orchestrator result.

        Raises:
            RuntimeError: if rate limit or concurrency limit is exceeded.
        """
        cfg = self.config
        num_steps = num_steps if num_steps is not None else cfg.default_num_steps
        strategy = strategy if strategy is not None else cfg.default_strategy

        if strategy == "ab":
            strategy = self.pick_ab_strategy()

        # Rate limit check.
        if not self._rate_limiter.acquire():
            self.metrics.record_request("429", 0.0, limited_by="rate")
            raise RuntimeError("rate limit exceeded")

        # Concurrency limit check.
        if not self._concurrency_limiter.acquire(timeout=cfg.request_timeout):
            self.metrics.record_request("503", 0.0, limited_by="concurrency")
            raise RuntimeError("concurrency limit exceeded")

        self.metrics.set_concurrency(self._concurrency_limiter.in_flight)

        try:
            with self.metrics.time_request() as timer:
                result = self._run_inference(prompt, num_steps, strategy)
            self.metrics.record_request("200", timer.latency_ms)
            return result
        except Exception:
            self.metrics.record_request("500", 0.0)
            raise
        finally:
            self._concurrency_limiter.release()
            self.metrics.set_concurrency(self._concurrency_limiter.in_flight)

    def _run_inference(
        self,
        prompt: str,
        num_steps: int,
        strategy: str,
    ) -> Dict[str, Any]:
        """Internal inference path."""
        # Reset per-request KV cache state.
        self.model_backend.set_kv_cache({})

        orch = Orchestrator(config_path=self.config_path)
        orch.set_model_backend(self.model_backend)
        orch.initialize(model_config=self.model_backend.get_model_config())

        start = time.perf_counter()
        with orch:
            if strategy == "baseline":
                result = orch.run_baseline(prompt=prompt, num_steps=num_steps)
            else:
                result = orch.run_full_pipeline(prompt=prompt, num_steps=num_steps)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return {
            "prompt": prompt,
            "strategy": strategy,
            "num_steps": num_steps,
            "serving_latency_ms": elapsed_ms,
            "result": result,
        }
