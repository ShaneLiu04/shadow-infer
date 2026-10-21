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
from shadowinfer.shadowkv.rd_allocator import EVICTED
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
        group_size: int = 16,
        plan: Any = None,
        precomputed: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Vectorized packing core (AR002 M1).

        Element-wise identical to ``_do_pack_reference``: per-precision groups
        are gathered with advanced indexing and quantized in one batched call
        instead of per-entry slicing. The int4 scale layout ``(n*batch, g, 1)``
        is preserved bit-for-bit and decoded with the corrected AR012 batch
        indexing in ``unpack``.

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
                if precision == EVICTED:
                    continue
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
        the precision grouping. When ``precomputed`` is supplied, fp32/fp16,
        int8 and int4 groups skip re-quantization because compression uses the
        same unified ``int4_group_size`` (AR012); the packed byte layout is
        produced from the precomputed INT4 codes/scales.
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
                if pre is not None and pre.get("qk") is not None and pre.get("scale_k") is not None:
                    qk_all = pre["qk"].reshape(
                        len(indices), batch_size, num_groups, effective_group_size
                    )
                    qv_all = pre["qv"].reshape(
                        len(indices), batch_size, num_groups, effective_group_size
                    )
                    scale_k_all = pre["scale_k"].reshape(len(indices), batch_size, num_groups, 1)
                    scale_v_all = pre["scale_v"].reshape(len(indices), batch_size, num_groups, 1)
                else:
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
                if precision == EVICTED:
                    continue
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
        """Vectorized unpack of all precision buckets to float32 (AR012)."""
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

            heads = torch.tensor([h for _, h in indices], dtype=torch.long, device=self.device)
            tokens = torch.tensor([t for t, _ in indices], dtype=torch.long, device=self.device)

            if precision in ("fp32", "fp16"):
                k_vals = data["k"].float()
                v_vals = data["v"].float()
                k_out[:, heads, tokens, :] = k_vals.permute(1, 0, 2)
                v_out[:, heads, tokens, :] = v_vals.permute(1, 0, 2)

            elif precision == "int8":
                k_vals = Quantizer.dequantize_int8(data["k"], data["scale_k"])
                v_vals = Quantizer.dequantize_int8(data["v"], data["scale_v"])
                k_out[:, heads, tokens, :] = k_vals.permute(1, 0, 2)
                v_out[:, heads, tokens, :] = v_vals.permute(1, 0, 2)

            elif precision == "int4":
                k_vals = self._dequantize_int4_bucket(
                    data["k"], data["scale_k"], data, len(indices)
                )
                v_vals = self._dequantize_int4_bucket(
                    data["v"], data["scale_v"], data, len(indices)
                )
                k_out[:, heads, tokens, :] = k_vals.permute(1, 0, 2)
                v_out[:, heads, tokens, :] = v_vals.permute(1, 0, 2)

        return k_out, v_out

    def unpack_reference(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-entry golden unpack (AR012).

        Semantically identical to :meth:`unpack`; INT4 scales are interpreted
        with the corrected ``[entry, batch, group, 1]`` layout.
        """
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
            group_size = data.get("group_size")
            num_groups = data.get("num_groups")
            scale_k = data["scale_k"]
            scale_v = data["scale_v"]
            if precision == "int4":
                scale_k = scale_k.reshape(len(indices), self.batch_size, num_groups, 1)
                scale_v = scale_v.reshape(len(indices), self.batch_size, num_groups, 1)

            for slice_idx, (token_idx, head_idx) in enumerate(indices):
                if precision in ("fp32", "fp16"):
                    k_slice = data["k"][slice_idx].float()
                    v_slice = data["v"][slice_idx].float()
                elif precision == "int8":
                    k_slice = Quantizer.dequantize_int8(
                        data["k"][slice_idx : slice_idx + 1],
                        data["scale_k"][slice_idx : slice_idx + 1],
                    )[0]
                    v_slice = Quantizer.dequantize_int8(
                        data["v"][slice_idx : slice_idx + 1],
                        data["scale_v"][slice_idx : slice_idx + 1],
                    )[0]
                else:
                    k_start, k_end, v_start, v_end = data["byte_offsets"][slice_idx]
                    k_unpacked = _kernels._unpack_int4(
                        data["k"][k_start:k_end], self.batch_size * self.head_dim
                    ).reshape(self.batch_size, num_groups, group_size)
                    v_unpacked = _kernels._unpack_int4(
                        data["v"][v_start:v_end], self.batch_size * self.head_dim
                    ).reshape(self.batch_size, num_groups, group_size)
                    k_slice = Quantizer.dequantize_int4(k_unpacked, scale_k[slice_idx])
                    v_slice = Quantizer.dequantize_int4(v_unpacked, scale_v[slice_idx])

                k_out[:, head_idx, token_idx, :] = k_slice.reshape(self.batch_size, self.head_dim)
                v_out[:, head_idx, token_idx, :] = v_slice.reshape(self.batch_size, self.head_dim)

        return k_out, v_out

    def _dequantize_int4_bucket(
        self,
        packed: torch.Tensor,
        scale: torch.Tensor,
        data: Dict[str, Any],
        num_entries: int,
    ) -> torch.Tensor:
        """Decode one INT4 bucket into ``[N, batch, head_dim]`` float32."""
        row_bytes = packed.numel() // max(num_entries, 1)
        rows = packed.reshape(num_entries, row_bytes)
        high = (rows >> 4) & 0x0F
        low = rows & 0x0F
        interleaved = torch.stack([high, low], dim=-1).reshape(num_entries, -1)
        num_values = self.batch_size * self.head_dim
        int_codes = (interleaved[:, :num_values].to(torch.int16) - 8).to(torch.int8)
        group_size = data["group_size"]
        num_groups = data["num_groups"]
        codes = int_codes.reshape(num_entries, self.batch_size, num_groups, group_size)
        scales = scale.reshape(num_entries, self.batch_size, num_groups, 1)
        values = codes.float() * scales
        return values.reshape(num_entries, self.batch_size, self.head_dim)

    def memory_bytes(self) -> int:
        """Return actual byte-level memory usage."""
        return self._memory_bytes

    def memory_mb(self) -> float:
        """Return memory usage in megabytes."""
        return self._memory_bytes / (1024.0 * 1024.0)

    def evict(self, token_head_set: Set[Tuple[int, int]]) -> int:
        """Remove token-heads from their precision buckets in place (AR012).

        Unlike the legacy unpack/repack path this truly frees bytes, keeps the
        original INT4 ``group_size`` for untouched entries, and never
        resurrects already-absent entries.

        Returns:
            Number of entries actually removed.
        """
        if not token_head_set or not self.packed_data:
            return 0

        removed = 0
        for precision in list(self.packed_data.keys()):
            data = self.packed_data[precision]
            indices = data["indices"]
            keep_positions: List[int] = []
            keep_indices: List[Tuple[int, int]] = []
            for position, entry in enumerate(indices):
                if entry in token_head_set:
                    removed += 1
                else:
                    keep_positions.append(position)
                    keep_indices.append(entry)

            if len(keep_positions) == len(indices):
                continue
            if not keep_indices:
                del self.packed_data[precision]
                continue

            keep = torch.tensor(keep_positions, dtype=torch.long, device=data["k"].device)
            if precision in ("fp32", "fp16", "int8"):
                data["k"] = data["k"].index_select(0, keep)
                data["v"] = data["v"].index_select(0, keep)
                if precision == "int8":
                    data["scale_k"] = data["scale_k"].index_select(0, keep)
                    data["scale_v"] = data["scale_v"].index_select(0, keep)
            else:
                num_entries = len(indices)
                row_bytes = data["k"].numel() // num_entries
                data["k"] = (
                    data["k"].reshape(num_entries, row_bytes).index_select(0, keep).reshape(-1)
                )
                data["v"] = (
                    data["v"].reshape(num_entries, row_bytes).index_select(0, keep).reshape(-1)
                )
                new_num = len(keep_indices)
                data["scale_k"] = (
                    data["scale_k"]
                    .reshape(num_entries, self.batch_size, data["num_groups"], 1)
                    .index_select(0, keep)
                    .reshape(-1, data["num_groups"], 1)
                )
                data["scale_v"] = (
                    data["scale_v"]
                    .reshape(num_entries, self.batch_size, data["num_groups"], 1)
                    .index_select(0, keep)
                    .reshape(-1, data["num_groups"], 1)
                )
                data["byte_offsets"] = [
                    (
                        i * row_bytes,
                        (i + 1) * row_bytes,
                        i * row_bytes,
                        (i + 1) * row_bytes,
                    )
                    for i in range(new_num)
                ]

            data["indices"] = keep_indices

        self._memory_bytes = self._compute_data_bytes()
        return removed

    def _precision_for(self, token_idx: int, head_idx: int) -> str:
        """Return the precision for a token-head from packed_data.

        Missing entries are treated as evicted (AR009): packing after
        ``evict`` must not resurrect dropped token-heads as fp16.
        """
        for precision, data in self.packed_data.items():
            if (token_idx, head_idx) in data.get("indices", []):
                return precision
        return EVICTED

    def apply_reuse_mask(
        self,
        kv_previous_packed: "PackedKVCache",
        reuse_mask: Dict[Tuple[int, int], bool],
        return_stats: bool = False,
    ) -> Optional[Dict[str, int]]:
        """Copy reused token-heads from a previous packed cache (vectorized).

        Cross-precision entries are not copied (their quantized representation
        differs); they keep the current values and are counted in
        ``skipped_precision``.
        """
        if (
            self.batch_size != kv_previous_packed.batch_size
            or self.num_heads != kv_previous_packed.num_heads
            or self.head_dim != kv_previous_packed.head_dim
        ):
            raise ValueError(
                "packed cache shape mismatch for reuse: "
                f"current=({self.batch_size},{self.num_heads},{self.head_dim}) "
                f"previous=({kv_previous_packed.batch_size},"
                f"{kv_previous_packed.num_heads},{kv_previous_packed.head_dim})"
            )
        for precision in self.packed_data:
            if precision != "int4":
                continue
            current = self.packed_data[precision]
            previous = kv_previous_packed.packed_data.get(precision)
            if previous is not None and current["group_size"] != previous["group_size"]:
                raise ValueError(
                    "INT4 group_size mismatch for reuse: "
                    f"current={current['group_size']} previous={previous['group_size']}"
                )

        stats = {"applied": 0, "skipped_precision": 0}
        if not reuse_mask or not kv_previous_packed.packed_data:
            return stats if return_stats else None

        previous_precision: Dict[Tuple[int, int], str] = {}
        previous_position: Dict[Tuple[int, int], int] = {}
        for precision, prev_data in kv_previous_packed.packed_data.items():
            for position, entry in enumerate(prev_data["indices"]):
                previous_precision[entry] = precision
                previous_position[entry] = position

        for precision, data in self.packed_data.items():
            prev_data = kv_previous_packed.packed_data.get(precision)
            target_positions: List[int] = []
            source_positions: List[int] = []
            for position, entry in enumerate(data["indices"]):
                if not reuse_mask.get(entry, False):
                    continue
                if previous_precision.get(entry) == precision and entry in previous_position:
                    target_positions.append(position)
                    source_positions.append(previous_position[entry])
                elif entry in previous_precision:
                    stats["skipped_precision"] += 1

            if not target_positions:
                continue

            target = torch.tensor(target_positions, dtype=torch.long, device=data["k"].device)
            source = torch.tensor(source_positions, dtype=torch.long, device=data["k"].device)
            if precision in ("fp32", "fp16", "int8"):
                data["k"].index_copy_(0, target, prev_data["k"].index_select(0, source))
                data["v"].index_copy_(0, target, prev_data["v"].index_select(0, source))
                if precision == "int8":
                    data["scale_k"].index_copy_(
                        0, target, prev_data["scale_k"].index_select(0, source)
                    )
                    data["scale_v"].index_copy_(
                        0, target, prev_data["scale_v"].index_select(0, source)
                    )
            else:
                num_entries = len(data["indices"])
                prev_entries = len(prev_data["indices"])
                row_bytes = data["k"].numel() // num_entries
                k_rows = data["k"].reshape(num_entries, row_bytes)
                v_rows = data["v"].reshape(num_entries, row_bytes)
                prev_k_rows = prev_data["k"].reshape(prev_entries, row_bytes)
                prev_v_rows = prev_data["v"].reshape(prev_entries, row_bytes)
                k_rows.index_copy_(0, target, prev_k_rows.index_select(0, source))
                v_rows.index_copy_(0, target, prev_v_rows.index_select(0, source))
                data["k"] = k_rows.reshape(-1)
                data["v"] = v_rows.reshape(-1)
                num_groups = data["num_groups"]
                scale_k = data["scale_k"].reshape(num_entries, self.batch_size, num_groups, 1)
                scale_v = data["scale_v"].reshape(num_entries, self.batch_size, num_groups, 1)
                prev_scale_k = prev_data["scale_k"].reshape(
                    prev_entries, self.batch_size, num_groups, 1
                )
                prev_scale_v = prev_data["scale_v"].reshape(
                    prev_entries, self.batch_size, num_groups, 1
                )
                scale_k.index_copy_(0, target, prev_scale_k.index_select(0, source))
                scale_v.index_copy_(0, target, prev_scale_v.index_select(0, source))
                data["scale_k"] = scale_k.reshape(-1, num_groups, 1)
                data["scale_v"] = scale_v.reshape(-1, num_groups, 1)

            stats["applied"] += len(target_positions)

        return stats if return_stats else None

    def _compute_data_bytes(self) -> int:
        total = 0
        for data in self.packed_data.values():
            for key in ("k", "v", "scale_k", "scale_v"):
                tensor = data.get(key)
                if tensor is not None:
                    total += tensor.numel() * tensor.element_size()
        return total

    def data_plane_bytes(self) -> int:
        """Return tensor storage bytes only (AR013, data-plane accounting)."""
        return self._compute_data_bytes()

    def resident_bytes(self) -> int:
        """Return resident bytes including per-entry metadata (AR012).

        Metadata is estimated at 48 bytes per token-head: a 16-byte Python
        tuple/list slot for ``indices`` plus four int64 offsets (32 bytes).
        """
        if not self.packed_data:
            return 0
        num_entries = sum(len(data["indices"]) for data in self.packed_data.values())
        return self._compute_data_bytes() + 48 * num_entries

    def __repr__(self) -> str:
        return (
            f"PackedKVCache(batch={self.batch_size}, heads={self.num_heads}, "
            f"seq_len={self.seq_len}, head_dim={self.head_dim}, "
            f"memory_mb={self.memory_mb():.2f})"
        )
