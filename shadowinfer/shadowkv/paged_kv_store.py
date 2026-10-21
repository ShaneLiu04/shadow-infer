"""Paged incremental KV residency (AR014).

FlashInfer-style page-table indirection: KV tokens are stored in fixed-size
pages, appends only touch the open tail page plus freshly allocated pages
(sealed pages are immutable and never re-packed), and ``gather`` rebuilds
the dense ``[batch, heads, seq, head_dim]`` view in a single pass.

Row-level quantization matches ``PackedKVCache`` bit-for-bit: INT8 scales
are computed per (batch, head, token) row over ``head_dim`` and INT4 uses
per-row group scales, both independent of how rows are batched into pages.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from shadowinfer.utils.quantization import Quantizer

_PRECISIONS = ("fp32", "fp16", "int8", "int4")


class _Page:
    """One page holding a single precision's quantized payload."""

    __slots__ = (
        "precision",
        "k",
        "v",
        "scale_k",
        "scale_v",
        "start_token",
        "num_tokens",
        "sealed",
    )

    def __init__(
        self,
        precision: str,
        k: torch.Tensor,
        v: torch.Tensor,
        scale_k: Optional[torch.Tensor],
        scale_v: Optional[torch.Tensor],
        start_token: int,
        num_tokens: int,
        sealed: bool,
    ) -> None:
        self.precision = precision
        self.k = k
        self.v = v
        self.scale_k = scale_k
        self.scale_v = scale_v
        self.start_token = start_token
        self.num_tokens = num_tokens
        self.sealed = sealed

    @property
    def end_token(self) -> int:
        return self.start_token + self.num_tokens


class PagedKVStore:
    """Page-table-indirected KV storage with incremental appends.

    One store instance covers one layer (mirroring the ``PackedKVCache``
    per-layer convention). Pages live in a slot pool; ``_order`` lists slot
    indices in token order and ``_free_slots`` recycles slots released by
    eviction, so appends never re-pack sealed pages.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        page_size: int = 32,
        device: Any = "cpu",
        int4_group_size: int = 16,
    ) -> None:
        if page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {page_size}")
        if int4_group_size < 1:
            raise ValueError(f"int4_group_size must be >= 1, got {int4_group_size}")
        self.batch_size = int(batch_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.device = device
        self.int4_group_size = int(int4_group_size)
        self.pages: List[Optional[_Page]] = []
        self._order: List[int] = []
        self._free_slots: List[int] = []
        self._max_seq = 0
        self._appended_tokens = 0
        self._page_evictions = 0
        self._evicted_pairs: Set[Tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_seq(self) -> int:
        return self._max_seq

    @property
    def page_count(self) -> int:
        return len(self._order)

    @property
    def open_pages(self) -> int:
        count = 0
        for slot in self._order:
            page = self.pages[slot]
            if page is not None and not page.sealed:
                count += 1
        return count

    @property
    def appended_tokens(self) -> int:
        return self._appended_tokens

    @property
    def page_evictions(self) -> int:
        return self._page_evictions

    @property
    def _int4_effective_group(self) -> int:
        if self.head_dim % self.int4_group_size == 0:
            return self.int4_group_size
        return self.head_dim

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_against(self, batch: int, heads: int, head_dim: int) -> None:
        """Raise ValueError when the store shape does not match the model."""
        if batch != self.batch_size or heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError(
                "PagedKVStore shape mismatch: store="
                f"({self.batch_size},{self.num_heads},{self.head_dim}) vs "
                f"model=({batch},{heads},{head_dim})"
            )

    def _validate_append(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        precision: str,
    ) -> None:
        if precision not in _PRECISIONS:
            raise ValueError(f"Unsupported page precision: {precision}")
        if k_new.dim() != 4 or v_new.dim() != 4:
            raise ValueError(
                "k_new/v_new must be 4-D [batch, heads, tokens, head_dim], got "
                f"k.dim={k_new.dim()}, v.dim={v_new.dim()}"
            )
        if k_new.shape != v_new.shape:
            raise ValueError(
                f"k_new/v_new shapes disagree: {tuple(k_new.shape)} vs " f"{tuple(v_new.shape)}"
            )
        if (
            k_new.shape[0] != self.batch_size
            or k_new.shape[1] != self.num_heads
            or k_new.shape[3] != self.head_dim
        ):
            raise ValueError(
                "PagedKVStore append shape mismatch: expected "
                f"({self.batch_size},{self.num_heads},T,{self.head_dim}), got "
                f"{tuple(k_new.shape)}"
            )

    # ------------------------------------------------------------------
    # Quantization helpers (row-local, bitwise-aligned with PackedKVCache)
    # ------------------------------------------------------------------

    def _quantize_chunk(
        self,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        precision: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if precision == "fp32":
            return k_chunk.clone(), v_chunk.clone(), None, None
        if precision == "fp16":
            return k_chunk.half(), v_chunk.half(), None, None
        if precision == "int8":
            qk, scale_k = Quantizer.quantize_int8(k_chunk)
            qv, scale_v = Quantizer.quantize_int8(v_chunk)
            return qk, qv, scale_k, scale_v
        group = self._int4_effective_group
        k_codes, scale_k = Quantizer.quantize_int4(k_chunk, group_size=group)
        v_codes, scale_v = Quantizer.quantize_int4(v_chunk, group_size=group)
        return (
            self._pack_int4_rows(k_codes),
            self._pack_int4_rows(v_codes),
            scale_k,
            scale_v,
        )

    @staticmethod
    def _pack_int4_rows(codes: torch.Tensor) -> torch.Tensor:
        """Pack INT4 code rows ``[..., num_groups, group]`` into half-bytes."""
        flat = codes.reshape(*codes.shape[:-2], -1)
        offset = (flat + 8).to(torch.uint8)
        if offset.shape[-1] % 2 != 0:
            pad = torch.zeros(*offset.shape[:-1], 1, dtype=torch.uint8, device=offset.device)
            offset = torch.cat([offset, pad], dim=-1)
        high = offset[..., 0::2] << 4
        low = offset[..., 1::2] & 0x0F
        return high | low

    @staticmethod
    def _merge_payload(
        page: _Page,
        payload: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        k_new, v_new, scale_k_new, scale_v_new = payload
        k = torch.cat([page.k, k_new], dim=2)
        v = torch.cat([page.v, v_new], dim=2)
        if (
            page.scale_k is None
            or page.scale_v is None
            or scale_k_new is None
            or scale_v_new is None
        ):
            return k, v, None, None
        scale_k = torch.cat([page.scale_k, scale_k_new], dim=2)
        scale_v = torch.cat([page.scale_v, scale_v_new], dim=2)
        return k, v, scale_k, scale_v

    # ------------------------------------------------------------------
    # Append (FR14.2)
    # ------------------------------------------------------------------

    def append(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        precision: str,
    ) -> None:
        """Append new tokens incrementally; sealed pages are never touched."""
        self._validate_append(k_new, v_new, precision)
        tokens = int(k_new.shape[2])
        if tokens == 0:
            return

        cursor = 0
        while cursor < tokens:
            open_slot = self._open_slot(precision)
            if open_slot is None:
                capacity = self.page_size
            else:
                open_page = self.pages[open_slot]
                capacity = self.page_size - open_page.num_tokens
            take = min(capacity, tokens - cursor)
            k_chunk = k_new[:, :, cursor : cursor + take, :]
            v_chunk = v_new[:, :, cursor : cursor + take, :]
            payload = self._quantize_chunk(k_chunk, v_chunk, precision)

            if open_slot is None:
                if self._free_slots:
                    slot = self._free_slots.pop()
                else:
                    slot = len(self.pages)
                    self.pages.append(None)
                self._order.append(slot)
                self.pages[slot] = _Page(
                    precision,
                    payload[0],
                    payload[1],
                    payload[2],
                    payload[3],
                    self._max_seq,
                    take,
                    take == self.page_size,
                )
            else:
                open_page = self.pages[open_slot]
                merged = self._merge_payload(open_page, payload)
                num_tokens = open_page.num_tokens + take
                self.pages[open_slot] = _Page(
                    precision,
                    merged[0],
                    merged[1],
                    merged[2],
                    merged[3],
                    open_page.start_token,
                    num_tokens,
                    num_tokens == self.page_size,
                )
            cursor += take
            self._max_seq += take

        self._appended_tokens += tokens

    def _open_slot(self, precision: str) -> Optional[int]:
        """Return the tail slot that can still absorb ``precision`` tokens."""
        if not self._order:
            return None
        slot = self._order[-1]
        page = self.pages[slot]
        if page is None or page.sealed:
            return None
        if page.precision != precision:
            page.sealed = True
            return None
        return slot

    # ------------------------------------------------------------------
    # Gather (FR14.3)
    # ------------------------------------------------------------------

    def gather(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rebuild dense float32 (k, v) in one pass over the page table."""
        if not self._order:
            empty = torch.zeros(
                self.batch_size,
                self.num_heads,
                0,
                self.head_dim,
                dtype=torch.float32,
                device=self.device,
            )
            return empty, empty.clone()

        pieces: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        expected = 0
        contiguous = True
        for slot in self._order:
            page = self.pages[slot]
            if page is None:
                contiguous = False
                continue
            if page.start_token != expected:
                contiguous = False
            expected = page.end_token
            pieces.append(
                (
                    page.start_token,
                    self._decode(page.k, page.scale_k, page.precision),
                    self._decode(page.v, page.scale_v, page.precision),
                )
            )

        if contiguous:
            k = torch.cat([piece[1] for piece in pieces], dim=2)
            v = torch.cat([piece[2] for piece in pieces], dim=2)
            return k, v

        k_out = torch.zeros(
            self.batch_size,
            self.num_heads,
            self._max_seq,
            self.head_dim,
            dtype=torch.float32,
            device=self.device,
        )
        v_out = torch.zeros_like(k_out)
        for start, k_piece, v_piece in pieces:
            end = start + k_piece.shape[2]
            k_out[:, :, start:end, :] = k_piece
            v_out[:, :, start:end, :] = v_piece
        return k_out, v_out

    def _decode(
        self,
        payload: torch.Tensor,
        scale: Optional[torch.Tensor],
        precision: str,
    ) -> torch.Tensor:
        """Decode one page payload back to float32 ``[B, H, T, D]``."""
        if precision in ("fp32", "fp16"):
            return payload.float()
        if precision == "int8":
            if scale is None:
                raise ValueError("int8 page payload is missing its scale")
            return payload.float() * scale
        if scale is None:
            raise ValueError("int4 page payload is missing its scale")
        high = (payload >> 4) & 0x0F
        low = payload & 0x0F
        inter = torch.stack([high, low], dim=-1).reshape(*payload.shape[:-1], -1)
        codes = (inter[..., : self.head_dim].to(torch.int16) - 8).to(torch.int8)
        group = self._int4_effective_group
        num_groups = self.head_dim // group
        codes = codes.reshape(*codes.shape[:-1], num_groups, group)
        values = codes.float() * scale
        return values.reshape(*values.shape[:-2], -1)

    # ------------------------------------------------------------------
    # Accounting (FR14.1, AR013 dual-ledger conventions)
    # ------------------------------------------------------------------

    def data_plane_bytes(self) -> int:
        """Return page tensor storage bytes (payload + scales)."""
        total = 0
        for slot in self._order:
            page = self.pages[slot]
            if page is None:
                continue
            for tensor in (page.k, page.v, page.scale_k, page.scale_v):
                if tensor is not None:
                    total += tensor.numel() * tensor.element_size()
        return total

    def resident_bytes(self) -> int:
        """Return data-plane bytes plus metadata and page-table overhead."""
        data = self.data_plane_bytes()
        if not self._order:
            return 0
        tokens = 0
        for slot in self._order:
            page = self.pages[slot]
            if page is not None:
                tokens += page.num_tokens
        token_heads = self.batch_size * self.num_heads * tokens
        return data + 48 * token_heads + 16 * len(self.pages)

    # ------------------------------------------------------------------
    # Eviction (FR14.4, (token, head)-granular parity with PackedKVCache)
    # ------------------------------------------------------------------

    def evict_tokens(self, token_head_set: Set[Tuple[int, int]]) -> int:
        """Evict token-heads page-locally; untouched pages are never re-packed.

        A token is dropped (and its bytes released) only when all its heads
        are evicted; partially evicted tokens keep their position with the
        evicted head rows zeroed (page granularity saves no bytes). Returns
        the number of (token, head) pairs actually removed; already-evicted
        or absent positions are never counted and never revived.
        """
        if not token_head_set or not self._order:
            return 0

        removed = 0
        runs_by_slot: Dict[int, List[_Page]] = {}
        retired: Set[int] = set()
        for slot in list(self._order):
            page = self.pages[slot]
            if page is None:
                continue
            page_removed, runs = self._plan_eviction(page, token_head_set)
            if page_removed == 0:
                continue
            removed += page_removed
            self._page_evictions += 1
            if runs:
                runs_by_slot[slot] = runs
            else:
                retired.add(slot)

        if removed == 0:
            return 0

        for slot in retired:
            self.pages[slot] = None
            self._free_slots.append(slot)

        new_order: List[int] = []
        for slot in self._order:
            if slot in retired:
                continue
            runs = runs_by_slot.get(slot)
            if runs is None:
                new_order.append(slot)
                continue
            for run_index, run_page in enumerate(runs):
                if run_index == 0:
                    self.pages[slot] = run_page
                    new_order.append(slot)
                else:
                    new_slot = self._alloc_slot()
                    self.pages[new_slot] = run_page
                    new_order.append(new_slot)
        self._order = new_order
        return removed

    def _alloc_slot(self) -> int:
        if self._free_slots:
            return self._free_slots.pop()
        slot = len(self.pages)
        self.pages.append(None)
        return slot

    def _plan_eviction(
        self,
        page: _Page,
        token_head_set: Set[Tuple[int, int]],
    ) -> Tuple[int, List[_Page]]:
        """Plan one page's eviction: survivor runs as candidate pages."""
        removed = 0
        kept: List[bool] = []
        partial: Dict[int, List[int]] = {}
        for i in range(page.num_tokens):
            token = page.start_token + i
            evicted = [
                head
                for head in range(self.num_heads)
                if (token, head) in token_head_set and (token, head) not in self._evicted_pairs
            ]
            removed += len(evicted)
            for head in evicted:
                self._evicted_pairs.add((token, head))
            if len(evicted) == self.num_heads:
                kept.append(False)
            else:
                kept.append(True)
                if evicted:
                    partial[i] = evicted
        if removed == 0:
            return 0, []

        runs: List[Tuple[int, int]] = []
        i = 0
        total = page.num_tokens
        while i < total:
            if not kept[i]:
                i += 1
                continue
            j = i
            while j < total and kept[j]:
                j += 1
            runs.append((i, j))
            i = j

        pages_out: List[_Page] = []
        num_runs = len(runs)
        for run_index, (run_i, run_j) in enumerate(runs):
            k_run = page.k[:, :, run_i:run_j, :].clone()
            v_run = page.v[:, :, run_i:run_j, :].clone()
            scale_k_run = (
                page.scale_k[:, :, run_i:run_j, :].clone() if page.scale_k is not None else None
            )
            scale_v_run = (
                page.scale_v[:, :, run_i:run_j, :].clone() if page.scale_v is not None else None
            )
            for i, evicted_heads in partial.items():
                if not (run_i <= i < run_j):
                    continue
                rel = i - run_i
                for head in evicted_heads:
                    if page.precision == "int4":
                        k_run[:, head, rel, :].fill_(0x88)
                        v_run[:, head, rel, :].fill_(0x88)
                    else:
                        k_run[:, head, rel, :].zero_()
                        v_run[:, head, rel, :].zero_()
            sealed = page.sealed or run_index < num_runs - 1 or (run_j - run_i) == self.page_size
            pages_out.append(
                _Page(
                    page.precision,
                    k_run,
                    v_run,
                    scale_k_run,
                    scale_v_run,
                    page.start_token + run_i,
                    run_j - run_i,
                    sealed,
                )
            )
        return removed, pages_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Drop all pages and counters (replace lifecycle / reset)."""
        self.pages = []
        self._order = []
        self._free_slots = []
        self._max_seq = 0
        self._appended_tokens = 0
        self._page_evictions = 0
        self._evicted_pairs = set()

    def __repr__(self) -> str:
        return (
            f"PagedKVStore(batch={self.batch_size}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, page_size={self.page_size}, "
            f"max_seq={self._max_seq}, pages={self.page_count})"
        )
