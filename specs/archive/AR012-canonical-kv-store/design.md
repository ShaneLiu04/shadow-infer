# [AR012] 设计说明：规范化紧凑 KV Store

## 1. 总体方案

```
packed buckets (precision -> indices, K/V, scales)
        │ flat indices (h*seq+t)
        ├── unpack(): gather bucket rows -> scatter to [B,H*S,D]
        ├── apply_reuse_mask(): flat index_copy_ between same-precision buckets
        ├── evict(): filter indices + tensor rows in place (no unpack/repack)
        └── resident_bytes(): data + scale + index metadata

agent._store_kv: packed only (dense shadow optional)
agent._load_kv: packed.unpack() on demand
```

FlashInfer 的 plan/run 思路：索引/布局一次构建，read 使用批量 gather/flat 索引，不做逐条目
Python 解释。PyramidInfer 的驱逐思想：必须真正释放字节，而不是把条目降成 INT4 继续占空间。

## 2. 数据结构与语义

### 2.1 桶布局（保持不变）

- `packed_data[precision] = {k, v, scale_k, scale_v, indices, ...}`。
- fp32/fp16：`k/v [N,B,D]`。
- int8：`k/v [N,B,D] int8`，`scale [N,B,1]`。
- int4：`k/v` 一维 packed bytes，`scale [N*B,G,1]`，`byte_offsets[(k0,k1,v0,v1)]`，
  `group_size`、`num_groups`。
- `indices` 按 token-major 生成（`float_indices = h*seq+t`）。

### 2.2 向量化 `unpack`

```python
def unpack(self):
    out_k = zeros(B,H,S,D, dtype=float32)
    out_v = ...
    for precision, data in packed_data:
        indices -> h_idx, t_idx (N,)
        if fp32/fp16:
            out[:, h_idx, t_idx, :] = data["k"].permute(1,0,2)
        elif int8:
            k = dequantize_int8(data["k"], data["scale_k"])  # [N,B,D]
            out[:, h_idx, t_idx, :] = k.permute(1,0,2)
        else int4:
            rows = _unpack_int4_rows(data["k"], row_bytes, D)  # [N*B,D]
            rows = rows.reshape(N,B,D)
            scales = data["scale_k"].reshape(N,B,G,1)  # 修正 batch 维度
            k = dequantize_int4_grouped(rows, scales, group_size)
            out[:, h_idx, t_idx, :] = k.permute(1,0,2)
```

- INT4 scale 修正：`scale_k` 的存储顺序是 `entry-major * batch`，索引必须 reshape 为
  `[N,B,G,1]` 后按 `[n,b]` 使用；旧的 `scale[n]` 在 B>1 时错误。
- 缺失/硬驱逐条目保持零。

### 2.3 向量化 `apply_reuse_mask`

- 对每个精度桶，收集 `reuse=True` 且存在于两个 cache 的条目。
- 用 `prev_index` 与 `cur_index` 做一次 `index_copy_`（fp32/fp16/int8），int4 用行字节
  `index_select` 复制。
- 跨精度条目：保留当前量化值，返回 `reuse_stats = {"applied": n, "skipped_precision": m}`；
  `apply_reuse_mask` 签名保持兼容，返回 stats 可选。
- 维度/group_size 不一致抛 `ValueError`。

### 2.4 桶内 `evict`

```python
def evict(self, entries):
    for precision, data in packed_data:
        keep = [i for i, entry in enumerate(data["indices"]) if entry not in entries]
        if len(keep) == len(data["indices"]): continue
        data["indices"] = [data["indices"][i] for i in keep]
        data[k/v/scale] = ...index_select(0, keep)
        int4: rebuild byte_offsets from row_bytes (uniform)
    recompute _memory_bytes
```

- 已不存在于任何桶的条目直接忽略（不复活）。
- `group_size`/`num_groups` 不变。
- `byte_offsets` 可由统一 `row_bytes` 算术重建，避免 O(N) 元数据。

### 2.5 单源存储

- `ShadowKVAgent` 配置 `store_dense_kv`（默认 `False` 当 `use_packed_cache=True`）。
- `_store_kv`: `store_dense_kv=False` 时 `KVCacheEntry.k_tensor=None, v_tensor=None,
  scale_*=None, precision="mixed", packed_kv=...`。
- `_load_kv`: dense 存在则返回；否则 `packed.unpack()`（可缓存最近一次，按 generation 失效）。
- `KVCacheEntry` 字段类型改为 `Optional[Tensor]`；`KVCacheManager` 只统计实际存在的 dense + packed。
- 开启 `store_dense_kv=True` 时仍保留 dense 作为兼容 shadow；`get_memory_usage` 同时计入 packed
  与 dense（真实驻留口径）。内存超预算时先释放 dense shadow（packed 仍在），再按需驱逐
  packed token-head，避免预算被不可回收的 shadow 卡住。
- packed-only 条目的 `precision` 记为 `"mixed"`。

### 2.6 `resident_bytes`

```
sum(tensor.numel()*element_size() for buckets)
+ scale bytes
+ len(indices) * 48  (Python 索引槽 16B + 4 个 int64 offset 32B)
```

`KVCacheManager.get_memory_usage()` 使用 `resident_bytes()`（packed）+ dense（若存在）。
内存超预算时先释放 dense shadow（`_release_dense_shadows`），再执行 packed 驱逐。

## 3. INT4 统一 group size

- 单一来源：`ShadowKVAgent.int4_group_size`（默认 16，可配置），`PackedKVCache.pack`
  与 `_do_pack` 默认取同一值。
- `_compress_kv_with_plan` 使用该 group size 量化 INT4，并把 `qk/qv/scale` 放入 payload；
  `_do_pack_with_plan` 对 INT4 直接使用 `precomputed`（不再二次量化），仅做行打包。
- `rd_allocator.entry_bytes(..., group_size=16)` 默认与 pack 一致。
- `evict` 保留原 `group_size`，不改变未触碰条目的布局。

## 4. 接口

```python
# packed_kv_cache.py
def unpack(self) -> Tuple[Tensor, Tensor]
def unpack_reference(self) -> Tuple[Tensor, Tensor]   # golden
def apply_reuse_mask(self, previous, mask, *, return_stats=False)
def evict(self, entries: Set[Tuple[int,int]]) -> int    # removed count
def resident_bytes(self) -> int

# shadowkv_agent.py
_store_kv(..., packed_kv=...)    # store_dense_kv 控制
_load_kv(layer_id) -> Dict[str, Tensor]
```

## 5. 测试设计

- UT-12.1 unpack 四种精度/B=1,2 与 reference 一致。
- UT-12.2 INT4 B=2 每 batch 相对 MSE <=0.02（修正）。
- UT-12.3 apply_reuse_mask 50%/100% 与 golden 一致；跨精度 stats；维度异常。
- UT-12.4 evict 混合精度：字节下降、indices 减少、未触碰条目不变、无复活、group_size 不变。
- UT-12.5 单源存储：`store_dense_kv=False` 时 entry 无 dense，`_load_kv` 与 unpack 一致；
  开启时逐位兼容。
- UT-12.6 INT4 量化调用一次（compress→pack 复用）；group size 一致；entry_bytes 误差 <=1%。
- UT-12.7 resident_bytes 与理论值一致；manager 口径。
- UT-12.8 基准：512/1024 unpack >=5x、reuse >=3x、evict >=10x（脚本断言）。
