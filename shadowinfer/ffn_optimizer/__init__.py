"""FFN Optimizer Agent — FFN 层计算优化模块。

对应文档：FFN_OPTIMIZER_AGENT.md, TECHNICAL_SPEC.md §2.3 / §3.2
版本：v3.0
"""

from __future__ import annotations

__version__ = "3.0"

from shadowinfer.ffn_optimizer.ffn_optimizer_agent import FFNOptimizerAgent
from shadowinfer.ffn_optimizer.packed_weight import PackedFFNWeight

__all__ = ["FFNOptimizerAgent", "PackedFFNWeight"]
