"""ModelBackend — Diffusion LLM 推理后端抽象。

对应文档：ARCHITECTURE.md §5.2, ROADMAP.md §3.2
版本：v3.1
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import torch

from shadowinfer.core.structs import KVCacheEntry, StepConfig


class ModelBackend(abc.ABC):
    """Diffusion LLM 推理后端抽象基类。

    ModelBackend 负责将 ShadowInfer 的优化决策（StepConfig、KV cache 等）
    应用到真实的模型推理中。不同的模型实现（Fast-dLLM-v2、DiffuLLaMA 等）
    可以通过继承此类接入 ShadowInfer。
    """

    @abc.abstractmethod
    def load(
        self, model_name: str, device: Optional[str] = None, **kwargs: Any
    ) -> None:
        """加载模型。

        Args:
            model_name: 模型名称或路径。
            device: 运行设备，如 "cuda" 或 "cpu"。
            **kwargs: 后端特定参数（如 HuggingFace 的 ``local_files_only``）。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_step(
        self,
        x: torch.Tensor,
        step_cfg: StepConfig,
        kv_cache: Optional[Dict[int, KVCacheEntry]] = None,
    ) -> Dict[str, Any]:
        """执行单个 denoising step。

        Args:
            x: 当前 step 的输入张量，形状 [batch, seq_len, hidden_dim]。
            step_cfg: 当前 step 的优化配置。
            kv_cache: 可选的 KV cache，按 layer_id 索引。

        Returns:
            包含以下键的字典：
                - "output": 当前 step 的输出张量。
                - "kv_cache": 更新后的 KV cache。
                - "attention_scores": 注意力分数，用于 ShadowKV 重要性评分。
                - "loss": 可选的训练/校准损失。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_kv_cache(self) -> Dict[int, KVCacheEntry]:
        """获取当前 KV cache 状态。"""
        raise NotImplementedError

    @abc.abstractmethod
    def set_kv_cache(self, kv_cache: Dict[int, KVCacheEntry]) -> None:
        """设置 KV cache 状态。"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置（层数、头数、维度等）。"""
        raise NotImplementedError

    def warmup(self, num_steps: int = 3) -> None:
        """预热模型（可选）。

        Args:
            num_steps: 预热步数。
        """
        pass

    def calibrate(self, calibration_data: Any, num_steps: int = 10) -> Dict[str, Any]:
        """在验证集上校准模型（可选）。

        Args:
            calibration_data: 校准数据。
            num_steps: 校准步数。

        Returns:
            校准统计信息。
        """
        return {}


class MockModelBackend(ModelBackend):
    """模拟模型后端，用于框架测试和无真实模型时的运行。"""

    def __init__(
        self, model_config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None
    ) -> None:
        self.model_config = model_config or {
            "name": "Fast-dLLM-v2-7B",
            "num_layers": 32,
            "num_heads": 32,
            "head_dim": 128,
            "hidden_dim": 4096,
            "intermediate_dim": 11008,
            "batch_size": 1,
            "seq_len": 128,
        }
        self._kv_cache: Dict[int, KVCacheEntry] = {}
        self._device = torch.device("cpu")

    def load(
        self, model_name: str, device: Optional[str] = None, **kwargs: Any
    ) -> None:
        del kwargs  # unused
        self.model_config["name"] = model_name
        self._device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def forward_step(
        self,
        x: torch.Tensor,
        step_cfg: StepConfig,
        kv_cache: Optional[Dict[int, KVCacheEntry]] = None,
    ) -> Dict[str, Any]:
        batch_size, seq_len, hidden_dim = x.shape
        num_heads = self.model_config.get("num_heads", 32)
        head_dim = self.model_config.get("head_dim", 128)

        output = torch.randn_like(x)
        attention_scores = torch.randn(batch_size, num_heads, seq_len, seq_len, device=x.device)

        # 更新模拟 KV cache
        updated_kv: Dict[int, KVCacheEntry] = {}
        num_layers = self.model_config.get("num_layers", 32)
        for lid in range(num_layers):
            k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=x.device)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=x.device)
            updated_kv[lid] = KVCacheEntry(
                k_tensor=k,
                v_tensor=v,
                precision="fp16" if step_cfg.shadowkv_mode != "aggressive" else "int8",
            )

        return {
            "output": output,
            "kv_cache": updated_kv,
            "attention_scores": attention_scores,
        }

    def get_kv_cache(self) -> Dict[int, KVCacheEntry]:
        return self._kv_cache

    def set_kv_cache(self, kv_cache: Dict[int, KVCacheEntry]) -> None:
        self._kv_cache = kv_cache

    def get_model_config(self) -> Dict[str, Any]:
        return dict(self.model_config)
