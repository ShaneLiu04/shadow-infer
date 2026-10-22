"""Executable inference paths for ShadowInfer (AR013, AR015, AR016)."""

from shadowinfer.execution.block_decoding import (
    BlockDecodingConfig,
    GenerateResult,
    ScheduleDecision,
    block_causal_mask,
    segment_causal_mask,
)
from shadowinfer.execution.macro_mode import MacroBlockEntry, MacroModeConfig, MacroModeState
from shadowinfer.execution.step_reuse import LayerReuseEntry, StepReuseConfig, StepReuseState

__all__ = [
    "BlockDecodingConfig",
    "GenerateResult",
    "LayerReuseEntry",
    "MacroBlockEntry",
    "MacroModeConfig",
    "MacroModeState",
    "ScheduleDecision",
    "StepReuseConfig",
    "StepReuseState",
    "block_causal_mask",
    "segment_causal_mask",
]
