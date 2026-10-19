"""Packed KV Cache storage for ShadowKV.

Stores per-layer KV cache as actual packed byte tensors to achieve real
memory savings. Supports mixed precision per token/head:
- FP32: 4 bytes/value
- FP16: 2 bytes/value
- INT8: 1 byte/value
- INT4: 0.5 byte/value (two int4 values packed into one uint8)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import torch

import shadowinfer.kernels as _kernels
from shadowinfer.utils.quantization import Quantizer


class PackedKVCache:
    """Byte-level packed KV cache storage with per-token-head precision."""

    PRECISION_BITS = {"fp32": 32, "fp16": 16, "int8": 8, "int4": 4}

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        device: Any = "cpu",
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.device = device
        self.packed_data: Dict[str, Dict[str, Any]] = {}
        self._memory_bytes = 0

    @classmethod
    def pack(
        cls,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        precision_map: Dict[int, Dict[int, str]],
        group_size: int = 16,
        plan: Any = None,
        precomputed: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Pack K/V tensors into byte-level storage.

        Args:
            k_tensor: [batch, num_heads, seq_len, head_dim]
            v_tensor: [batch, num_heads, seq_len, head_dim]
            precision_map: token_idx -> head_idx -> precision
            group_size: INT4 quantization group size. Defaults to 16 to keep
                roundtrip error well below the 1e-1 requirement.
            plan: optional ``PrecisionPlan`` (AR008) to reuse grouping indices.
            precomputed: optional per-precision quantized/gathered tensors from
                the compression stage (AR008) to avoid duplicate quantization.

        Returns:
            Dict with keys: packed_kv, memory_bytes, memory_mb
        """
        batch_size, num_heads, seq_len, head_dim = k_tensor.shape
        cache = cls(batch_size, num_heads, seq_len, head_dim, device=k_tensor.device)
        cache._do_pack(
            k_tensor,
            v_tensor,
            precision_map,
            group_size=group_size,
            plan=plan,
            precomputed=precomputed,
        )
        return {
            "packed_kv": cache,
            "memory_bytes": cache.memory_bytes(),
            "memory_mb": cache.memory_mb(),
        }

    def _do_pack(
        self,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        precision_map: Dict[int, Dict[int, str]],
        group_size: int = 64,
        plan: Any = None,
        precomputed: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Vectorized packing core (AR002 M1).

        Element-wise identical to ``_do_pack_reference``: per-precision groups
        are gathered with advanced indexing and quantized in one batched call
        instead of per-entry slicing. The int4 scale layout ``(n*batch, g, 1)``
        is preserved bit-for-bit (including its known batch>1 indexing defect
        in ``unpack``, kept for compatibility).

        AR008: when ``plan`` is supplied, grouping/flat indices are reused and
        ``precomputed`` (if present) supplies the tensors already produced by
        the compression stage.
        """
        if plan is not None:
            self._do_pack_with_plan(k_tensor, v_tensor, plan, group_size, precomputed=precomputed)
            return

        self.packed_data = {}
        total_bytes = 0

        groups: Dict[str, List[Tuple[int, int]]] = {
            "fp32": [],
            "fp16": [],
            "int8": [],
            "int4": [],
        }
        for token_idx in range(self.seq_len):
            token_map = precision_map.get(token_idx, {})
            for head_idx in range(self.num_heads):
                precision = token_map.get(head_idx, "fp16").lower()
                if precision not in groups:
                    precision = "fp16"
                groups[precision].append((token_idx, head_idx))

        device = k_tensor.device
        for precision, indices in groups.items():
            if not indices:
                continue

            h_idx = torch.tensor([h for _, h in indices], dtype=torch.long, device=device)
            t_idx = torch.tensor([t for t, _ in indices], dtype=torch.long, device=device)

            if precision in ("fp32", "fp16"):
                dtype = torch.float32 if precision == "fp32" else torch.float16
                k_stacked = k_tensor[:, h_idx, t_idx, :].permute(1, 0, 2).to(dtype)
                v_stacked = v_tensor[:, h_idx, t_idx, :].permute(1, 0, 2).to(dtype)
                total_bytes += k_stacked.numel() * k_stacked.element_size()
                total_bytes += v_stacked.numel() * v_stacked.element_size()
                self.packed_data[precision] = {
                    "k": k_stacked,
                    "v": v_stacked,
                    "scale_k": None,
                    "scale_v": None,
                    "indices": indices,
                }

            elif precision == "int8":
                k_stacked = k_tensor[:, h_idx, t_idx, :].permute(1, 0, 2)
                v_stacked = v_tensor[:, h_idx, t_idx, :].permute(1, 0, 2)
                qk, scale_k = Quantizer.quantize_int8(k_stacked)
                qv, scale_v = Quantizer.quantize_int8(v_stacked)
                total_bytes += qk.numel() * qk.element_size()
                total_bytes += qv.numel() * qv.element_size()
                total_bytes += scale_k.numel() * scale_k.element_size()
                total_bytes += scale_v.numel() * scale_v.element_size()
                self.packed_data[precision] = {
                    "k": qk,
                    "v": qv,
                    "scale_k": scale_k,
                    "scale_v": scale_v,
                    "indices": indices,
                }

            elif precision == "int4":
                effective_group_size = (
                    group_size if self.head_dim % group_size == 0 else self.head_dim
                )
                num_groups = self.head_dim // effective_group_size

                k_stacked = k_tensor[:, h_idx, t_idx, :].permute(1, 0, 2)
                v_stacked = v_tensor[:, h_idx, t_idx, :].permute(1, 0, 2)
                qk_all, scale_k_all = Quantizer.quantize_int4(
                    k_stacked, group_size=effective_group_size
                )
                qv_all, scale_v_all = Quantizer.quantize_int4(
                    v_stacked, group_size=effective_group_size
                )

                # AR005: batched INT4 byte packing (bit-identical to the
                # per-entry loop in ``_do_pack_reference``). Rows are padded
                # independently when the flattened row length is odd.
                k_rows, k_row_bytes = self._pack_int4_rows(qk_all)
                v_rows, v_row_bytes = self._pack_int4_rows(qv_all)
                k_stacked_packed = k_rows.reshape(-1)
                v_stacked_packed = v_rows.reshape(-1)

                byte_offsets: List[Tuple[int, int, int, int]] = []
                for n in range(len(indices)):
                    byte_offsets.append(
                        (
                            n * k_row_bytes,
                            (n + 1) * k_row_bytes,
                            n * v_row_bytes,
                            (n + 1) * v_row_bytes,
                        )
                    )
                scale_k = scale_k_all.reshape(-1, num_groups, 1)
                scale_v = scale_v_all.reshape(-1, num_groups, 1)

                total_bytes += k_stacked_packed.numel() * k_stacked_packed.element_size()
                total_bytes += v_stacked_packed.numel() * v_stacked_packed.element_size()
                total_bytes += scale_k.numel() * scale_k.element_size()
                total_bytes += scale_v.numel() * scale_v.element_size()

                self.packed_data[precision] = {
                    "k": k_stacked_packed,
                    "v": v_stacked_packed,
                    "scale_k": scale_k,
                    "scale_v": scale_v,
                    "indices": indices,
                    "byte_offsets": byte_offsets,
                    "group_size": effective_group_size,
                    "num_groups": num_groups,
                }

        self._memory_bytes = total_bytes

    def _do_pack_with_plan(
        self,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        plan: Any,
        group_size: int,
        precomputed: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Plan-driven packing (AR008).

        Reuses ``plan.groups`` / ``plan.flat_indices`` instead of rebuilding
        the precision grouping. When ``precomputed`` is supplied, fp32/fp16 and
        int8 groups skip re-quantization (identical inputs/quantizer); int4
        still quantizes with the pack-specific group size because the
        compression working copy may use a different group size.
        """
        batch_size, num_heads, seq_len, head_dim = k_tensor.shape
        if plan.seq_len != seq_len or plan.num_heads != num_heads:
            raise ValueError(
                "PrecisionPlan shape mismatch: plan "
                f"(seq_len={plan.seq_len}, num_heads={plan.num_heads}) vs "
                f"tensor (seq_len={seq_len}, num_heads={num_heads})"
            )
        if v_tensor.shape != k_tensor.shape:
            raise ValueError("k_tensor and v_tensor must have identical shapes")
        if plan.codes.device.type != k_tensor.device.type:
            raise ValueError(
                f"PrecisionPlan device ({plan.codes.device}) does not match "
                f"KV device ({k_tensor.device})"
            )

        self.packed_data = {}
        total_bytes = 0
        k_flat = k_tensor.reshape(batch_size, num_heads * seq_len, head_dim)
        v_flat = v_tensor.reshape(batch_size, num_heads * seq_len, head_dim)

        for precision, entries in plan.groups.items():
            if not entries:
                continue
            indices = list(entries)
            flat = plan.flat_indices[precision]
            pre = precomputed.get(precision) if precomputed else None

            if precision in ("fp32", "fp16"):
                dtype = torch.float32 if precision == "fp32" else torch.float16
                if pre is not None:
                    k_stacked = pre["qk"]
                    v_stacked = pre["qv"]
                else:
                    k_stacked = k_flat.index_select(1, flat).permute(1, 0, 2).to(dtype)
                    v_stacked = v_flat.index_select(1, flat).permute(1, 0, 2).to(dtype)
                total_bytes += k_stacked.numel() * k_stacked.element_size()
                total_bytes += v_stacked.numel() * v_stacked.element_size()
                self.packed_data[precision] = {
                    "k": k_stacked,
                    "v": v_stacked,
                    "scale_k": None,
                    "scale_v": None,
                    "indices": indices,
                }

            elif precision == "int8":
                if pre is not None:
                    qk = pre["qk"]
                    qv = pre["qv"]
                    scale_k = pre["scale_k"]
                    scale_v = pre["scale_v"]
                else:
                    k_stacked = k_flat.index_select(1, flat).permute(1, 0, 2)
                    v_stacked = v_flat.index_select(1, flat).permute(1, 0, 2)
                    qk, scale_k = Quantizer.quantize_int8(k_stacked)
                    qv, scale_v = Quantizer.quantize_int8(v_stacked)
                total_bytes += qk.numel() * qk.element_size()
                total_bytes += qv.numel() * qv.element_size()
                total_bytes += scale_k.numel() * scale_k.element_size()
                total_bytes += scale_v.numel() * scale_v.element_size()
                self.packed_data[precision] = {
                    "k": qk,
                    "v": qv,
                    "scale_k": scale_k,
                    "scale_v": scale_v,
                    "indices": indices,
                }

            else:
                effective_group_size = group_size if head_dim % group_size == 0 else head_dim
                num_groups = head_dim // effective_group_size
                if pre is not None:
                    k_stacked = pre["k_group"]
                    v_stacked = pre["v_group"]
                else:
                    k_stacked = k_flat.index_select(1, flat).permute(1, 0, 2)
                    v_stacked = v_flat.index_select(1, flat).permute(1, 0, 2)
                qk_all, scale_k_all = Quantizer.quantize_int4(
                    k_stacked, group_size=effective_group_size
                )
                qv_all, scale_v_all = Quantizer.quantize_int4(
                    v_stacked, group_size=effective_group_size
                )
                k_rows, k_row_bytes = self._pack_int4_rows(qk_all)
                v_rows, v_row_bytes = self._pack_int4_rows(qv_all)
                k_stacked_packed = k_rows.reshape(-1)
                v_stacked_packed = v_rows.reshape(-1)

                byte_offsets: List[Tuple[int, int, int, int]] = []
                for n in range(len(indices)):
                    byte_offsets.append(
                        (
                            n * k_row_bytes,
                            (n + 1) * k_row_bytes,
                            n * v_row_bytes,
                            (n + 1) * v_row_bytes,
                        )
                    )
                scale_k = scale_k_all.reshape(-1, num_groups, 1)
                scale_v = scale_v_all.reshape(-1, num_groups, 1)

                total_bytes += k_stacked_packed.numel() * k_stacked_packed.element_size()
                total_bytes += v_stacked_packed.numel() * v_stacked_packed.element_size()
                total_bytes += scale_k.numel() * scale_k.element_size()
                total_bytes += scale_v.numel() * scale_v.element_size()

                self.packed_data[precision] = {
                    "k": k_stacked_packed,
                    "v": v_stacked_packed,
                    "scale_k": scale_k,
                    "scale_v": scale_v,
                    "indices": indices,
                    "byte_offsets": byte_offsets,
                    "group_size": effective_group_size,
                    "num_groups": num_groups,
                }

        self._memory_bytes = total_bytes

    @staticmethod
    def _pack_int4_rows(quantized: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pack INT4 values row-wise into uint8 bytes (AR005 vectorized path).

        Args:
            quantized: INT8-encoded INT4 tensor ``[n, batch, groups, group]``.

        Returns:
            ``(packed_rows [n, ceil(L/2)], row_bytes)`` where ``L`` is the
            flattened per-row element count.
        """
        if quantized.shape[0] == 0:
            return torch.empty(0, 0, dtype=torch.uint8, device=quantized.device), 0
        rows = quantized.reshape(quantized.shape[0], -1)
        offset = (rows + 8).to(torch.uint8)
        if offset.shape[1] % 2 != 0:
            pad = torch.zeros(offset.shape[0], 1, dtype=torch.uint8, device=offset.device)
            offset = torch.cat([offset, pad], dim=1)
        high = offset[:, 0::2] << 4
        low = offset[:, 1::2] & 0x0F
        packed = high | low
        return packed, packed.shape[1]

    def _do_pack_reference(
        self,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        precision_map: Dict[int, Dict[int, str]],
        group_size: int = 64,
    ) -> None:
        """Original per-entry loop implementation (golden baseline)."""
        self.packed_data = {}
        total_bytes = 0

        groups: Dict[str, List[Tuple[int, int]]] = {
            "fp32": [],
            "fp16": [],
            "int8": [],
            "int4": [],
        }
        for token_idx in range(self.seq_len):
            token_map = precision_map.get(token_idx, {})
            for head_idx in range(self.num_heads):
                precision = token_map.get(head_idx, "fp16").lower()
                if precision not in groups:
                    precision = "fp16"
                groups[precision].append((token_idx, head_idx))

        for precision, indices in groups.items():
            if not indices:
                continue

            if precision in ("fp32", "fp16"):
                k_slices = [
                    k_tensor[:, head_idx, token_idx, :].unsqueeze(0)
                    for token_idx, head_idx in indices
                ]
                v_slices = [
                    v_tensor[:, head_idx, token_idx, :].unsqueeze(0)
                    for token_idx, head_idx in indices
                ]
                dtype = torch.float32 if precision == "fp32" else torch.float16
                k_stacked = torch.cat(k_slices, dim=0).to(dtype)
                v_stacked = torch.cat(v_slices, dim=0).to(dtype)
                total_bytes += k_stacked.numel() * k_stacked.element_size()
                total_bytes += v_stacked.numel() * v_stacked.element_size()
                self.packed_data[precision] = {
                    "k": k_stacked,
                    "v": v_stacked,
                    "scale_k": None,
                    "scale_v": None,
                    "indices": indices,
                }

            elif precision == "int8":
                k_slices = [
                    k_tensor[:, head_idx, token_idx, :].unsqueeze(0)
                    for token_idx, head_idx in indices
                ]
                v_slices = [
                    v_tensor[:, head_idx, token_idx, :].unsqueeze(0)
                    for token_idx, head_idx in indices
                ]
                k_stacked = torch.cat(k_slices, dim=0)
                v_stacked = torch.cat(v_slices, dim=0)
                qk, scale_k = Quantizer.quantize_int8(k_stacked)
                qv, scale_v = Quantizer.quantize_int8(v_stacked)
                total_bytes += qk.numel() * qk.element_size()
                total_bytes += qv.numel() * qv.element_size()
                total_bytes += scale_k.numel() * scale_k.element_size()
                total_bytes += scale_v.numel() * scale_v.element_size()
                self.packed_data[precision] = {
                    "k": qk,
                    "v": qv,
                    "scale_k": scale_k,
                    "scale_v": scale_v,
                    "indices": indices,
                }

            elif precision == "int4":
                effective_group_size = (
                    group_size if self.head_dim % group_size == 0 else self.head_dim
                )
                num_groups = self.head_dim // effective_group_size

                k_slices_packed: List[torch.Tensor] = []
                v_slices_packed: List[torch.Tensor] = []
                scale_k_list: List[torch.Tensor] = []
                scale_v_list: List[torch.Tensor] = []
                byte_offsets: List[Tuple[int, int, int, int]] = []
                k_byte_cursor = 0
                v_byte_cursor = 0

                for token_idx, head_idx in indices:
                    k_slice = k_tensor[:, head_idx, token_idx, :].reshape(
                        self.batch_size, self.head_dim
                    )
                    v_slice = v_tensor[:, head_idx, token_idx, :].reshape(
                        self.batch_size, self.head_dim
                    )

                    qk, scale_k = Quantizer.quantize_int4(k_slice, group_size=effective_group_size)
                    qv, scale_v = Quantizer.quantize_int4(v_slice, group_size=effective_group_size)

                    qk_packed = _kernels._pack_int4(qk.reshape(-1))
                    qv_packed = _kernels._pack_int4(qv.reshape(-1))

                    byte_offsets.append(
                        (
                            k_byte_cursor,
                            k_byte_cursor + qk_packed.numel(),
                            v_byte_cursor,
                            v_byte_cursor + qv_packed.numel(),
                        )
                    )
                    k_byte_cursor += qk_packed.numel()
                    v_byte_cursor += qv_packed.numel()

                    k_slices_packed.append(qk_packed)
                    v_slices_packed.append(qv_packed)
                    scale_k_list.append(scale_k)
                    scale_v_list.append(scale_v)

                k_stacked = torch.cat(k_slices_packed, dim=0)
                v_stacked = torch.cat(v_slices_packed, dim=0)
                scale_k = torch.cat(scale_k_list, dim=0)
                scale_v = torch.cat(scale_v_list, dim=0)

                total_bytes += k_stacked.numel() * k_stacked.element_size()
                total_bytes += v_stacked.numel() * v_stacked.element_size()
                total_bytes += scale_k.numel() * scale_k.element_size()
                total_bytes += scale_v.numel() * scale_v.element_size()

                self.packed_data[precision] = {
                    "k": k_stacked,
                    "v": v_stacked,
                    "scale_k": scale_k,
                    "scale_v": scale_v,
                    "indices": indices,
                    "byte_offsets": byte_offsets,
                    "group_size": effective_group_size,
                    "num_groups": num_groups,
                }

        self._memory_bytes = total_bytes

    def unpack(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unpack K/V tensors back to float32."""
        k_out = torch.zeros(
            self.batch_size,
            self.num_heads,
            self.seq_len,
            self.head_dim,
            dtype=torch.float32,
            device=self.device,
        )
        v_out = torch.zeros_like(k_out)

        for precision, data in self.packed_data.items():
            indices = data["indices"]
            if not indices:
                continue

            if precision == "fp32":
                k_vals = data["k"].float()
                v_vals = data["v"].float()
                for slice_idx, (token_idx, head_idx) in enumerate(indices):
                    k_out[:, head_idx, token_idx, :] = k_vals[slice_idx]
                    v_out[:, head_idx, token_idx, :] = v_vals[slice_idx]

            elif precision == "fp16":
                k_vals = data["k"].float()
                v_vals = data["v"].float()
                for slice_idx, (token_idx, head_idx) in enumerate(indices):
                    k_out[:, head_idx, token_idx, :] = k_vals[slice_idx]
                    v_out[:, head_idx, token_idx, :] = v_vals[slice_idx]

            elif precision == "int8":
                k_vals = Quantizer.dequantize_int8(data["k"], data["scale_k"])
                v_vals = Quantizer.dequantize_int8(data["v"], data["scale_v"])
                for slice_idx, (token_idx, head_idx) in enumerate(indices):
                    k_out[:, head_idx, token_idx, :] = k_vals[slice_idx]
                    v_out[:, head_idx, token_idx, :] = v_vals[slice_idx]

            elif precision == "int4":
                byte_offsets = data["byte_offsets"]
                group_size = data["group_size"]
                num_groups = data["num_groups"]
                for slice_idx, (token_idx, head_idx) in enumerate(indices):
                    k_start, k_end, v_start, v_end = byte_offsets[slice_idx]
                    k_packed_slice = data["k"][k_start:k_end]
                    v_packed_slice = data["v"][v_start:v_end]

                    k_unpacked = _kernels._unpack_int4(
                        k_packed_slice, self.batch_size * self.head_dim
                    )
                    v_unpacked = _kernels._unpack_int4(
                        v_packed_slice, self.batch_size * self.head_dim
                    )

                    k_unpacked = k_unpacked.reshape(self.batch_size, num_groups, group_size)
                    v_unpacked = v_unpacked.reshape(self.batch_size, num_groups, group_size)

                    k_slice = Quantizer.dequantize_int4(k_unpacked, data["scale_k"][slice_idx])
                    v_slice = Quantizer.dequantize_int4(v_unpacked, data["scale_v"][slice_idx])

                    k_out[:, head_idx, token_idx, :] = k_slice.reshape(
                        self.batch_size, self.head_dim
                    )
                    v_out[:, head_idx, token_idx, :] = v_slice.reshape(
                        self.batch_size, self.head_dim
                    )

        return k_out, v_out

    def memory_bytes(self) -> int:
        """Return actual byte-level memory usage."""
        return self._memory_bytes

    def memory_mb(self) -> float:
        """Return memory usage in megabytes."""
        return self._memory_bytes / (1024.0 * 1024.0)

    def evict(self, token_head_set: Set[Tuple[int, int]]) -> None:
        """Downgrade or zero out a set of token-heads to free memory.

        The current implementation unpacks the cache, zeros the evicted
        token-heads, and re-packs them in INT4 (the smallest precision).
        Non-evicted token-heads keep their original precision.

        Args:
            token_head_set: Set of ``(token_idx, head_idx)`` to evict.
        """
        if not token_head_set or not self.packed_data:
            return

        k_full, v_full = self.unpack()
        # Build a new precision map: evicted entries -> int4, others unchanged.
        new_precision_map: Dict[int, Dict[int, str]] = {}
        for token_idx in range(self.seq_len):
            new_precision_map[token_idx] = {}
            for head_idx in range(self.num_heads):
                if (token_idx, head_idx) in token_head_set:
                    new_precision_map[token_idx][head_idx] = "int4"
                    k_full[:, head_idx, token_idx, :] = 0.0
                    v_full[:, head_idx, token_idx, :] = 0.0
                else:
                    original = self._precision_for(token_idx, head_idx)
                    new_precision_map[token_idx][head_idx] = original

        self.packed_data = {}
        # review I5：重打包沿用 pack 默认分组（group_size=16），避免驱逐后
        # 非驱逐 int4 条目被换组重量化、字节口径与准入估算脱节
        self._do_pack(k_full, v_full, new_precision_map)

    def _precision_for(self, token_idx: int, head_idx: int) -> str:
        """Return the precision string for a token-head from packed_data."""
        for precision, data in self.packed_data.items():
            if (token_idx, head_idx) in data.get("indices", []):
                return precision
        return "fp16"

    def apply_reuse_mask(
        self,
        kv_previous_packed: "PackedKVCache",
        reuse_mask: Dict[Tuple[int, int], bool],
    ) -> None:
        """Copy reused token-heads from previous packed cache into this cache.

        Args:
            kv_previous_packed: PackedKVCache from the previous step.
            reuse_mask: {(token_idx, head_idx): bool} indicating which
                token-heads should be copied from the previous cache.
        """
        if not reuse_mask or not kv_previous_packed.packed_data:
            return

        for precision, data in self.packed_data.items():
            prev_data = kv_previous_packed.packed_data.get(precision)
            if prev_data is None:
                continue

            prev_map = {(t, h): i for i, (t, h) in enumerate(prev_data["indices"])}
            cur_indices = data["indices"]

            for slice_idx, (token_idx, head_idx) in enumerate(cur_indices):
                if not reuse_mask.get((token_idx, head_idx), False):
                    continue

                prev_slice_idx = prev_map.get((token_idx, head_idx))
                if prev_slice_idx is None:
                    continue

                if precision in ("fp32", "fp16", "int8"):
                    data["k"][slice_idx] = prev_data["k"][prev_slice_idx]
                    data["v"][slice_idx] = prev_data["v"][prev_slice_idx]
                    if precision == "int8":
                        data["scale_k"][slice_idx] = prev_data["scale_k"][prev_slice_idx]
                        data["scale_v"][slice_idx] = prev_data["scale_v"][prev_slice_idx]
                elif precision == "int4":
                    k_start, k_end, v_start, v_end = data["byte_offsets"][slice_idx]
                    (
                        prev_k_start,
                        prev_k_end,
                        prev_v_start,
                        prev_v_end,
                    ) = prev_data[
                        "byte_offsets"
                    ][prev_slice_idx]
                    data["k"][k_start:k_end] = prev_data["k"][prev_k_start:prev_k_end]
                    data["v"][v_start:v_end] = prev_data["v"][prev_v_start:prev_v_end]
                    data["scale_k"][slice_idx] = prev_data["scale_k"][prev_slice_idx]
                    data["scale_v"][slice_idx] = prev_data["scale_v"][prev_slice_idx]

    def __repr__(self) -> str:
        return (
            f"PackedKVCache(batch={self.batch_size}, heads={self.num_heads}, "
            f"seq_len={self.seq_len}, head_dim={self.head_dim}, "
            f"memory_mb={self.memory_mb():.2f})"
        )
