# [AR012] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR012 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-20 |

## 测试用例列表

### ST-001：向量化 unpack 与 FP32/FP16 逐位一致

**关联需求：** srs.md §3.1 FR12.1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 真实随机 K/V，`B=1/2,H=2,S=8,D=64`，全 FP32/FP16 map。

**测试步骤：**
1. `PackedKVCache.pack` 打包。
2. 调用 `unpack_reference()` 与 `unpack()`。

**期望结果：**
- 两者 `torch.equal` 逐位一致；EVICTED 位置为零。

**实际结果：** 通过；`test_unpack_matches_reference[fp32|fp16-...]` 全绿。

**状态：** PASS

---

### ST-002：混合精度 unpack 与半驱逐

**关联需求：** srs.md §3.1 FR12.1
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- 四种精度轮转 map；后半 token 置为 EVICTED。

**测试步骤：**
1. 打包并分别调用 reference / vectorized unpack。

**期望结果：**
- 两者一致；被驱逐位置 `count_nonzero == 0`。

**实际结果：** 通过；`test_unpack_matches_reference[*-True-*]` 全绿。

**状态：** PASS

---

### ST-003：INT4 B>1 scale 修正

**关联需求：** srs.md §3.1 FR12.1
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- 全 INT4，`B=2,S=8,H=2,D=64,group=16`。

**测试步骤：**
1. unpack 后按 batch 计算相对 MSE。

**期望结果：**
- 每个 batch 相对 MSE <= 0.02。

**实际结果：** 通过；`test_int4_batch_scale_layout` 与基准均给出 0.0075/0.0072。

**状态：** PASS

---

### ST-004：50% 复用

**关联需求：** srs.md §3.2 FR12.2
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 前后两步同 map、同形状 K/V；50% mask。

**测试步骤：**
1. 记录复用前 unpack 值。
2. `apply_reuse_mask(return_stats=True)`。

**期望结果：**
- 复用位置等于上一步反量化值；非复用位置逐位不变；`applied` 计数正确。

**实际结果：** 通过；`test_partial_reuse_50_percent`。

**状态：** PASS

---

### ST-005：100% 复用与默认返回

**关联需求：** srs.md §3.2 FR12.2
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- 全 True mask。

**测试步骤：**
1. 执行复用；2. 默认参数再执行一次。

**期望结果：**
- 整缓存与上一步一致；默认返回 `None`（向后兼容）。

**实际结果：** 通过；`test_full_reuse_100_percent`、`test_returns_none_by_default`。

**状态：** PASS

---

### ST-006：跨精度复用不静默

**关联需求：** srs.md §3.2 FR12.2
**测试类型：** 边界条件
**优先级：** Medium

**前置条件：**
- 上一步 FP16，当前同条目 INT4。

**测试步骤：**
1. 对该条目 mask=True 执行复用。

**期望结果：**
- 当前值保持不变；`skipped_precision >= 1`。

**实际结果：** 通过；`test_cross_precision_is_skipped`。

**状态：** PASS

---

### ST-007：维度/分组/批量不匹配异常

**关联需求：** srs.md §3.2 FR12.2；§4 非功能
**测试类型：** 异常处理
**优先级：** Medium

**前置条件：**
- head_dim 32 vs 64；INT4 group 16 vs 32；batch 1 vs 2。

**测试步骤：**
1. 分别调用 `apply_reuse_mask`。

**期望结果：**
- 均抛出 `ValueError`。

**实际结果：** 通过；三个异常用例全绿。

**状态：** PASS

---

### ST-008：桶内就地硬驱逐

**关联需求：** srs.md §3.4 FR12.4
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 混合精度、`B=2`，驱逐混合 map 每桶 1/4 条目。

**测试步骤：**
1. 记录字节与 unpack 值；2. 调用 `evict`；3. 再次驱逐同集合。

**期望结果：**
- 返回删除数；字节下降；被删位置为零；未触碰条目逐位不变；group_size 不变；
  二次驱逐返回 0 且不复活。

**实际结果：** 通过；`test_evict_drops_bytes_without_resurrection`；基准字节下降 25.0%。

**状态：** PASS

---

### ST-009：驻留口径与 dense shadow 释放

**关联需求：** srs.md §3.5 FR12.5
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- packed-only 与 dense+packed 两种 entry。

**测试步骤：**
1. 查询 `resident_bytes()` 与 manager 用量；2. 调用 `_release_dense_shadows`。

**期望结果：**
- 空缓存为 0；manager = packed resident（+ dense 若存在）；释放后 dense 字段为 None 且用量下降。

**实际结果：** 通过；`test_empty_cache_accounting`、`test_resident_bounds_and_manager_usage`、
`test_combined_dense_and_packed_accounting`、`test_release_dense_shadow_clears_fields`。

**状态：** PASS

---

### ST-010：单源存储与按需反量化

**关联需求：** srs.md §3.5 FR12.5
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 真实 ShadowKVAgent（4 层/4 头/dim64），`use_packed_cache=True`。

**测试步骤：**
1. on_step；2. 检查 entry；3. `_load_kv`；4. 开启 `store_dense_kv=True` 重跑。

**期望结果：**
- 默认 entry 无 dense、precision="mixed"；`_load_kv` 与 packed.unpack 一致；
  开启后保留 dense 且 `_load_kv` 返回 dense。

**实际结果：** 通过；`test_packed_only_store_and_load`、`test_store_dense_kv_enabled`、
`test_store_dense_default_without_packed`。

**状态：** PASS

---

### ST-011：INT4 单次量化、统一分组与成本模型

**关联需求：** srs.md §3.3 FR12.3
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- `int4_group_size=32`，混合 map；spy 统计 `Quantizer.quantize_int4` 调用。

**测试步骤：**
1. compress；2. 检查调用次数与 packed 分组；3. 对比 `_estimate_precision_bytes`/`predict_codes_bytes` 与实测字节。

**期望结果：**
- K/V 各一次量化（共 2 次）；group_size=32；成本模型与 packed 字节误差 <=1%。

**实际结果：** 通过；`test_int4_quantized_once_and_group_size_respected`、
`test_budget_model_uses_configured_group_size`。

**状态：** PASS

---

### ST-012：性能门槛与全量回归

**关联需求：** srs.md §4 非功能需求；FR12.6
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- 真实张量，S=512/1024/2048，`B=1/H=8/D=64`，3 次中位。

**测试步骤：**
1. 运行 `ar012_bench.py`（断言 5x/3x/10x 与 MSE）；
2. 运行全量 `pytest`。

**期望结果：**
- unpack >=5x、reuse >=3x、evict >=10x；INT4 MSE <=0.02；全量无失败。

**实际结果：** 通过；unpack 101-219x、reuse 7.3-8.2x、evict 49-109x；
853 passed / 1 skipped，覆盖率 83.23%。

**状态：** PASS

---

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-20 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 FR12.1 | 向量化 unpack + INT4 batch 修正 | ST-001, ST-002, ST-003 | PASS |
| §3.2 FR12.2 | 向量化复用 + 统计 | ST-004, ST-005, ST-006, ST-007 | PASS |
| §3.3 FR12.3 | 统一 INT4 group size + 单次量化 | ST-011 | PASS |
| §3.4 FR12.4 | 桶内就地硬驱逐 | ST-008 | PASS |
| §3.5 FR12.5 | 单源驻留与内存口径 | ST-009, ST-010 | PASS |
| §3.6 FR12.6 | 复现、基准与回退 | ST-012 | PASS |

**需求覆盖率：** 6 / 6（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 4 | 4 | 0 | 0 |
| 边界条件 | 4 | 4 | 0 | 0 |
| 异常处理 | 3 | 3 | 0 | 0 |
| 回归测试 | 1 | 1 | 0 | 0 |
| **合计** | **12** | **12** | **0** | **0** |

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor | GPU/7B 端到端驻留与延迟待服务器按 `server-experiment.md` 回填 | 已在协议中记录，不影响本机结论 |
| Minor | `apply_reuse_mask` 的 `return_stats` 允许位置参数（设计为 keyword-only） | 兼容性超集，记录不阻塞 |

### 结论

> Go：需求覆盖 100%，无 Critical/Major 缺陷，全量 853 passed / 1 skipped，覆盖率 83.23%。
