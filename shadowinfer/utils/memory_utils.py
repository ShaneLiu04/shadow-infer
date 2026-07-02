"""ShadowInfer GPU 显存追踪工具模块。

对应 PROFILER_AGENT.md 中的 memory 维度。
"""

__version__ = "3.0"

from contextlib import contextmanager
from typing import Dict, Generator

import torch


class MemoryTracker:
    """GPU 显存追踪工具。对应 PROFILER_AGENT.md 中的 memory 维度。"""

    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """
        获取当前 GPU 显存信息。

        返回：
            {
                "allocated": 已分配显存（MB），
                "reserved": 预留显存（MB），
                "free": 空闲显存（MB），
                "total": 总显存（MB），
            }

        Returns:
            显存信息字典（单位 MB）。
        """
        if not torch.cuda.is_available():
            return {
                "allocated": 0.0,
                "reserved": 0.0,
                "free": 0.0,
                "total": 0.0,
            }

        device = torch.cuda.current_device()

        allocated = torch.cuda.memory_allocated(device) / (1024**2)
        reserved = torch.cuda.memory_reserved(device) / (1024**2)
        total = torch.cuda.get_device_properties(device).total_memory / (1024**2)
        free = total - reserved

        return {
            "allocated": float(allocated),
            "reserved": float(reserved),
            "free": float(free),
            "total": float(total),
        }

    @staticmethod
    def get_tensor_memory(tensor: torch.Tensor) -> int:
        """
        计算 tensor 占用的字节数。

        Args:
            tensor: 任意张量。

        Returns:
            字节数（int）。
        """
        if tensor.numel() == 0:
            return 0
        return tensor.numel() * tensor.element_size()

    @staticmethod
    def get_kv_cache_memory(kv_cache: Dict[str, torch.Tensor]) -> int:
        """
        计算 KV cache 总内存。

        Args:
            kv_cache: 字典，键为标识，值为张量。

        Returns:
            总字节数（int）。
        """
        total = 0
        for key, tensor in kv_cache.items():
            total += MemoryTracker.get_tensor_memory(tensor)
        return total

    @staticmethod
    @contextmanager
    def track_peak_memory() -> Generator[Dict[str, float], None, None]:
        """
        上下文管理器，追踪峰值显存使用。

        用法：
            with MemoryTracker.track_peak_memory() as stats:
                # 执行推理
                ...
            print(stats["peak_allocated_mb"])

        Yields:
            统计字典，包含 peak_allocated_mb 和 peak_reserved_mb。
        """
        if not torch.cuda.is_available():
            yield {"peak_allocated_mb": 0.0, "peak_reserved_mb": 0.0}
            return

        device = torch.cuda.current_device()

        # 重置峰值统计
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        stats = {"peak_allocated_mb": 0.0, "peak_reserved_mb": 0.0}

        try:
            yield stats
        finally:
            torch.cuda.synchronize(device)
            peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
            peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
            stats["peak_allocated_mb"] = float(peak_allocated)
            stats["peak_reserved_mb"] = float(peak_reserved)

    @staticmethod
    def reset_peak_memory() -> None:
        """重置峰值显存记录。"""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(device)
