"""FFN Optimizer Agent — FFN 层计算优化模块。

对应文档：README.md《核心模块》FFN Optimizer Agent
版本：v3.0
"""

from __future__ import annotations

__version__ = "3.0"

from shadowinfer.ffn_optimizer.ffn_optimizer_agent import FFNOptimizerAgent
from shadowinfer.ffn_optimizer.packed_weight import PackedFFNWeight

__all__ = ["FFNOptimizerAgent", "PackedFFNWeight"]
