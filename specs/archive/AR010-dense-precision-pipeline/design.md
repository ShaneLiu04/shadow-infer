# [AR010] 设计说明：稠密精度码 KV 热路径

## 1. 总体方案

```
importance[seq,heads] ──vectorized──> codes int8
        │                              │
        │ drift top-k (scatter)       ├── admit_codes (RD, sink/reuse, evict)
        │                              ▼
        │                        codes (canonical)
        │                              ├── PrecisionPlan.from_codes → flat indices
        │                              ├── compress_kv(precision_codes) → k/v/payload
        │                              ├── pack(plan, payload)
        │                              └── codes_to_map (一次, API/统计)
```

v3.11：`allocate(dict) → admit(dict) → build(dict→codes→groups)`，每步多次 O(N) Python 解释。
AR010：codes 只转换一次到 dict，热路径全部走张量/扁平列表。

## 2. 数据结构与语义

- codes：`int8[seq, heads]`，值 0..4 与 `PRECISION_CODES` 对齐。
- `map_to_codes(map, seq, heads)`：缺失 token/head 取 fp16；未知精度按 strict 处理（默认抛错）。
- `codes_to_map(codes)`：`{t: {h: name}}`，保留全部条目。
- `from_codes(codes, reuse_mask)`：
  - `flat_codes = codes.reshape(-1)`（t-major）；
  - 每个精度 `pos = nonzero(flat==code)`，`t=pos//heads`、`h=pos%heads`，
    `flat_indices[p]=h*seq+t`，`groups[p]=zip(t.tolist(),h.tolist())`；
  - code=4 进入 `evicted_entries`，不进入 groups；
  - fingerprint 与 `build` 相同（对 codes+reuse 的 SHA-256）。
- `build(map)`：一次构造 codes 后委托 `from_codes`，保证两条路径完全一致。

## 3. 稠密准入 `admit_codes`

输入 codes、importance、sensitivity、预算、保护掩码；输出新 codes、predicted、evicted。

1. `score = importance*sensitivity`（或 importance）→ flatten；sink/protected 从候选集中排除。
2. `order = torch.argsort(score, stable=True)`；候选按 `(score, token, head)` 与 v3.11 同序。
3. 成本表 5 项（`entry_bytes`），初始 predicted 由 codes gather 求和。
4. 降档闭式求解：每个候选到 int4 的总节省为 `step1+step2+step3`；前缀和
   `cumsum` + `searchsorted` 定位预算边界；边界候选按剩余额度选择步数。
   与 v3.11 的“逐条降到底再下一条”完全同序，避免逐条 Python 解释。
5. 驱逐同理：对当前码的全额字节做前缀和，锁定需要驱逐的前缀（含边界）；
   `code != 4` 才计入 `evicted_count`；输入 codes 不被修改。

与 `admit_budget` 的差异仅在状态表示；排序、降档与驱逐顺序一致，保证逐项相同。

## 4. 生产接入

- `allocate_precision_codes`：复用 v3.10 的阈值/`torch.where` 分支，直接返回 codes。
- drift>0.5：沿用 v3.11 语义，将每 token top 20% head 的 code 置为 fp16（sink 前缀保护），
  与 dict 路径逐项一致。
- `on_step`：codes → drift → `_admit_precision_codes` → `compress_kv(precision_codes=codes)`；
  `precision_map = codes_to_map(codes)` 仅用于对外状态/统计/缓存副本。
- 决策缓存：同时保存 `precision_codes` 与准入后的精度映射；命中时直接复用；
  旧缓存无 codes 时用 `map_to_codes`。
- `compress_kv`：若传入 codes，`PrecisionPlan.from_codes`；否则维持 v3.11 `build(map)`。

## 5. workspace 修复

`k_out = torch.zeros(k_tensor.shape, dtype=k_tensor.dtype, device=k_tensor.device)`（v 同理），
保证 `.view(b, h*s, d)` 合法；输入 K/V 可为任意 stride。增加非连续输入 UT，验证与 legacy 逐位一致。

## 6. 接口

```python
# precision_plan.py
def map_to_codes(precision_map, seq_len, num_heads, strict=True) -> torch.Tensor
def codes_to_map(codes) -> Dict[int, Dict[int, str]]
PrecisionPlan.from_codes(codes, reuse_mask=None, device=None) -> PrecisionPlan

# rd_allocator.py
def admit_codes(codes, importance, sensitivity, budget_bytes, batch, head_dim,
                sink_tokens=0, rd_allocation=True, allow_eviction=False,
                protected_mask=None) -> (codes, predicted, evicted)

# ShadowKVAgent
def allocate_precision_codes(self, importance_map, mode) -> torch.Tensor
def _admit_precision_codes(self, codes, importance_map, kv_current, protected=None)
def compress_kv(..., precision_codes: Optional[torch.Tensor] = None)
```

## 7. 测试设计

- UT-10.1 codes↔map 往返、缺失/未知精度。
- UT-10.2 `from_codes` vs `build`：groups/flat/evicted/fingerprint 一致；512/2048 随机 map。
- UT-10.3 `admit_codes` vs `admit_budget`：多预算/rd 开关/保护/驱逐，逐项一致。
- UT-10.4 `admit_codes` 不修改输入。
- UT-10.5 非连续 K/V：plan 压缩与 legacy 逐位一致；无回退（捕获告警）。
- UT-10.6 生产 on_step 稠密路径 vs 关闭稠密开关的 dict 路径：map/k/v/packed 一致。
- UT-10.7 决策缓存命中复用 codes。
- UT-10.8 基准：512/2048/4096/8192 准入+计划耗时，8192×32 ≥10x。
