"""Convergence-aware macro-block mode control for diffusion denoising (AR015).

The state stores, per (layer, macro block), the input/output hidden states and
K/V rows from the last full computation. A single per-row relative-drift gate
plus a per-block active-ratio gate classify each macro block as SKIP (zero
active rows), DELTA (minority active rows) or REFRESH (majority active rows).
Hysteresis is built in: any active row immediately releases a frozen block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

MODE_SKIP = "SKIP"
MODE_DELTA = "DELTA"
MODE_REFRESH = "REFRESH"


@dataclass(frozen=True)
class MacroModeConfig:
    """Configuration for macro-block SKIP/DELTA/REFRESH mode control."""

    enabled: bool = False
    macro_block_size: int = 32
    skip_threshold: float = 0.1
    delta_min_active_ratio: float = 0.5
    warmup_steps: int = 1
    force_refresh_every: int = 0

    def validate(self) -> None:
        if self.macro_block_size < 1:
            raise ValueError(f"macro_block_size must be >= 1, got {self.macro_block_size}")
        if self.skip_threshold < 0:
            raise ValueError(f"skip_threshold must be >= 0, got {self.skip_threshold}")
        if not 0.0 <= self.delta_min_active_ratio <= 1.0:
            raise ValueError(
                "delta_min_active_ratio must be within [0, 1], got "
                f"{self.delta_min_active_ratio}"
            )
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.force_refresh_every < 0:
            raise ValueError(
                f"force_refresh_every must be >= 0 (0 disables), got {self.force_refresh_every}"
            )

    def __post_init__(self) -> None:
        self.validate()


@dataclass
class MacroBlockEntry:
    """Cached macro-block state from the last full computation."""

    hidden_in: torch.Tensor
    hidden_out: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    seq_start: int = 0
    seq_end: int = 0


@dataclass
class MacroModeState:
    """Per-(layer, macro block) cache plus per-step classification counters."""

    entries: Dict[Tuple[int, int], MacroBlockEntry] = field(default_factory=dict)
    last_attention: Optional[torch.Tensor] = None
    last_modes: Dict[int, Dict[int, str]] = field(default_factory=dict)
    last_active_rows: Dict[int, List[int]] = field(default_factory=dict)
    _skip_blocks: int = field(default=0, repr=False)
    _delta_blocks: int = field(default=0, repr=False)
    _refresh_blocks: int = field(default=0, repr=False)
    _skipped_rows: int = field(default=0, repr=False)
    _active_rows: int = field(default=0, repr=False)
    _rewritten_rows: int = field(default=0, repr=False)

    def reset(self) -> None:
        self.entries.clear()
        self.last_attention = None
        self.last_modes.clear()
        self.last_active_rows.clear()
        self._skip_blocks = 0
        self._delta_blocks = 0
        self._refresh_blocks = 0
        self._skipped_rows = 0
        self._active_rows = 0
        self._rewritten_rows = 0

    def begin_step(self) -> None:
        self.last_modes.clear()
        self.last_active_rows.clear()
        self._skip_blocks = 0
        self._delta_blocks = 0
        self._refresh_blocks = 0
        self._skipped_rows = 0
        self._active_rows = 0
        self._rewritten_rows = 0

    def record_rewritten(self, rows: int) -> None:
        self._rewritten_rows += int(rows)

    def record_rows(
        self,
        layer_id: int,
        macro_idx: int,
        seq_slice: Tuple[int, int],
        hidden_in: torch.Tensor,
        hidden_out: torch.Tensor,
        k_rows: torch.Tensor,
        v_rows: torch.Tensor,
    ) -> None:
        """Cache one macro block; the recorded content is frozen until drift."""
        start, end = int(seq_slice[0]), int(seq_slice[1])
        self.entries[(layer_id, macro_idx)] = MacroBlockEntry(
            hidden_in=hidden_in.detach().clone(),
            hidden_out=hidden_out.detach().clone(),
            k=k_rows.detach().clone(),
            v=v_rows.detach().clone(),
            seq_start=start,
            seq_end=end,
        )

    def classify(
        self,
        layer_id: int,
        hidden: torch.Tensor,
        step_id: int,
        config: MacroModeConfig,
    ) -> Tuple[Dict[int, str], List[int]]:
        """Classify every macro block of one layer.

        Returns ``(modes, active_rows)`` where ``modes`` maps macro index to
        SKIP/DELTA/REFRESH and ``active_rows`` lists the global sequence rows
        that must be recomputed this step (sorted).
        """
        seq_len = int(hidden.shape[1])
        block_size = config.macro_block_size
        num_blocks = (seq_len + block_size - 1) // block_size

        forced = step_id < config.warmup_steps or (
            config.force_refresh_every > 0 and step_id % config.force_refresh_every == 0
        )

        modes: Dict[int, str] = {}
        active: List[int] = []
        if forced:
            for macro_idx in range(num_blocks):
                start = macro_idx * block_size
                end = min(start + block_size, seq_len)
                modes[macro_idx] = MODE_REFRESH
                active.extend(range(start, end))
        else:
            # Vectorized path: assemble one full-sequence reference so the
            # per-row drift is a single tensor op chain (no per-block .item()
            # or nonzero calls); blocks with missing/invalid entries are
            # REFRESH with all their rows active.
            ref = torch.empty_like(hidden)
            valid = torch.zeros(seq_len, dtype=torch.bool, device=hidden.device)
            invalid_blocks: List[int] = []
            for macro_idx in range(num_blocks):
                start = macro_idx * block_size
                end = min(start + block_size, seq_len)
                entry = self.entries.get((layer_id, macro_idx))
                rows = end - start
                if entry is None or not self._entry_matches(entry, hidden, rows):
                    if entry is not None:
                        del self.entries[(layer_id, macro_idx)]
                    invalid_blocks.append(macro_idx)
                    continue
                ref[:, start:end, :] = entry.hidden_in
                valid[start:end] = True
            denom = ref.norm(dim=-1)
            diff = (hidden - ref).norm(dim=-1)
            safe_denom = denom.clamp_min(torch.finfo(denom.dtype).tiny)
            drift = diff / safe_denom
            row_active = ((denom <= 0) | (drift >= config.skip_threshold)).any(dim=0)
            row_active = row_active | ~valid
            invalid_set = set(invalid_blocks)
            padded = row_active.tolist() + [False] * (num_blocks * block_size - seq_len)
            for macro_idx in range(num_blocks):
                start = macro_idx * block_size
                end = min(start + block_size, seq_len)
                rows = end - start
                if macro_idx in invalid_set:
                    modes[macro_idx] = MODE_REFRESH
                    continue
                active_count = sum(padded[start:end])
                if active_count == 0:
                    modes[macro_idx] = MODE_SKIP
                    self._skipped_rows += rows
                elif active_count / rows >= config.delta_min_active_ratio:
                    modes[macro_idx] = MODE_REFRESH
                else:
                    modes[macro_idx] = MODE_DELTA
            active = [row for row in range(seq_len) if row_active[row]]
            active.sort()

        self._skip_blocks += sum(1 for mode in modes.values() if mode == MODE_SKIP)
        self._delta_blocks += sum(1 for mode in modes.values() if mode == MODE_DELTA)
        self._refresh_blocks += sum(1 for mode in modes.values() if mode == MODE_REFRESH)
        self._active_rows += len(active)
        self.last_modes[layer_id] = dict(modes)
        self.last_active_rows[layer_id] = list(active)
        return modes, list(active)

    @staticmethod
    def _entry_matches(entry: MacroBlockEntry, hidden: torch.Tensor, rows: int) -> bool:
        ref = entry.hidden_in
        return (
            ref.dim() == 3
            and ref.shape[0] == hidden.shape[0]
            and ref.shape[1] == rows
            and ref.shape[2] == hidden.shape[2]
        )

    @staticmethod
    def _active_rows_mask(
        entry: MacroBlockEntry,
        block_hidden: torch.Tensor,
        skip_threshold: float,
    ) -> torch.Tensor:
        """Per-seq-row active mask [rows]: active iff any batch element drifted."""
        ref = entry.hidden_in
        denom = ref.norm(dim=-1)
        diff = (block_hidden - ref).norm(dim=-1)
        safe_denom = denom.clamp_min(torch.finfo(denom.dtype).tiny)
        drift = diff / safe_denom
        active = (denom <= 0) | (drift >= skip_threshold)
        return active.any(dim=0)

    def stats(self) -> Dict[str, Any]:
        return {
            "skip_blocks": self._skip_blocks,
            "delta_blocks": self._delta_blocks,
            "refresh_blocks": self._refresh_blocks,
            "skipped_rows": self._skipped_rows,
            "active_rows": self._active_rows,
            "rewritten_rows": self._rewritten_rows,
        }
