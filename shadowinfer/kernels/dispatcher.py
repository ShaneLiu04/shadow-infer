"""Kernel dispatch utilities with auto-tuning integration.

Provides a thin dispatcher that decides between CUDA and CPU execution paths
and consults the persistent auto-tuning cache for launch configurations.

Version: 3.1.1
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import Tensor

from .auto_tuning import KernelAutoTuner, TuningConfig


class KernelDispatcher:
    """Centralized dispatcher for kernel execution.

    The dispatcher:
    1. Checks whether CUDA is available and the input tensors live on a GPU.
    2. Looks up a tuned launch configuration from ``KernelAutoTuner``.
    3. Falls back to the CPU implementation when CUDA is unavailable or when
       the environment variable ``SHADOWINFER_FORCE_CPU_KERNELS=1`` is set.
    """

    def __init__(
        self,
        tuner: Optional[KernelAutoTuner] = None,
    ) -> None:
        """Initialize the dispatcher.

        Args:
            tuner: ``KernelAutoTuner`` instance. If None, a default tuner is
                created with the standard cache location.
        """
        self.tuner = tuner or KernelAutoTuner()

    # ------------------------------------------------------------------
    # Path selection
    # ------------------------------------------------------------------

    @staticmethod
    def force_cpu() -> bool:
        """Return True when CPU fallback is forced via environment variable."""
        return os.environ.get("SHADOWINFER_FORCE_CPU_KERNELS", "0") == "1"

    @staticmethod
    def _device_name(tensor: Tensor) -> str:
        """Return a human-readable device name for cache keys."""
        if tensor.is_cuda and torch.cuda.is_available():
            try:
                return torch.cuda.get_device_name(tensor.device).strip() or "cuda"
            except Exception:
                return "cuda"
        return str(tensor.device)

    @staticmethod
    def _tensor_shapes(tensors: List[Tensor]) -> List[Tuple[int, ...]]:
        """Extract stable shape tuples from a list of tensors."""
        return [tuple(t.shape) for t in tensors]

    def use_cuda(self, tensor: Tensor, cuda_ops: Optional[Any]) -> bool:
        """Return True if the CUDA path should be used for ``tensor``."""
        if self.force_cpu():
            return False
        if cuda_ops is None:
            return False
        if not tensor.is_cuda:
            return False
        return True

    # ------------------------------------------------------------------
    # Tuning helpers
    # ------------------------------------------------------------------

    def get_config(
        self,
        kernel_name: str,
        tensors: List[Tensor],
        default: Optional[TuningConfig] = None,
    ) -> Optional[TuningConfig]:
        """Look up the best cached config for a kernel invocation.

        Args:
            kernel_name: Logical kernel identifier.
            tensors: Input tensors used to build the shape signature.
            default: Config returned when no cache entry exists.

        Returns:
            A ``TuningConfig`` or ``default``.
        """
        if not tensors:
            return default
        reference = tensors[0]
        device_name = self._device_name(reference)
        dtype = str(reference.dtype)
        shapes = self._tensor_shapes(tensors)
        return self.tuner.get_best_config(kernel_name, shapes, dtype, device_name, default=default)

    def tune(
        self,
        kernel_name: str,
        tensors: List[Tensor],
        benchmark_fn: Callable[[TuningConfig], float],
        candidates: Optional[Tuple[TuningConfig, ...]] = None,
        warmup: int = 2,
        repeats: int = 5,
    ) -> TuningConfig:
        """Benchmark and cache the best config for a kernel invocation.

        Args:
            kernel_name: Logical kernel identifier.
            tensors: Input tensors used to build the shape signature.
            benchmark_fn: Function returning elapsed milliseconds for a config.
            candidates: Candidate configs. Uses tuner defaults if None.
            warmup: Number of warmup iterations per config.
            repeats: Number of timed iterations per config.

        Returns:
            The fastest ``TuningConfig``.
        """
        reference = tensors[0]
        device_name = self._device_name(reference)
        dtype = str(reference.dtype)
        shapes = self._tensor_shapes(tensors)
        return self.tuner.benchmark(
            kernel_name=kernel_name,
            tensor_shapes=shapes,
            dtype=dtype,
            device_name=device_name,
            benchmark_fn=benchmark_fn,
            candidates=candidates,
            warmup=warmup,
            repeats=repeats,
        )

    def dispatch(
        self,
        kernel_name: str,
        reference_tensor: Tensor,
        cuda_ops: Optional[Any],
        cuda_fn: Callable[..., Any],
        cpu_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Dispatch to CUDA or CPU implementation.

        The dispatcher also appends the selected ``TuningConfig`` to ``kwargs``
        as ``launch_config`` when the CUDA path is active. Existing CUDA
        bindings that do not accept ``launch_config`` can simply ignore it;
        future bindings can use it to choose grid/block dimensions.

        Args:
            kernel_name: Logical kernel identifier.
            reference_tensor: Tensor used to decide device and cache key.
            cuda_ops: Loaded CUDA extension module (may be None).
            cuda_fn: Callable invoking the CUDA kernel.
            cpu_fn: Callable invoking the CPU fallback.
            *args: Positional arguments forwarded to the chosen function.
            **kwargs: Keyword arguments forwarded to the chosen function. May
                be augmented with ``launch_config`` on CUDA.

        Returns:
            The result of the chosen kernel implementation.
        """
        if self.use_cuda(reference_tensor, cuda_ops):
            config = self.get_config(kernel_name, [reference_tensor])
            kwargs = dict(kwargs)
            kwargs["launch_config"] = config
            return cuda_fn(*args, **kwargs)
        return cpu_fn(*args, **kwargs)
