# [AR010] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR010 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-19 |

## 测试用例列表

### ST-001：codes 与 precision_map 双向转换

**关联需求：** srs.md §3.1 FR10.1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- 构造含缺失条目、evicted 与四种精度的 map（5×3）。

**测试步骤：**
1. 调用 `map_to_codes`，再调用 `codes_to_map`。
2. 对未知精度分别以 `strict=True/False` 调用。

**期望结果：**
- codes 为 `int8[5,3]`，缺失条目为 fp16，evicted 码为 4；往返与规范化输入逐项相等。
- strict 抛 `ValueError`，lenient 归一化为 fp16。

**实际结果：** `tests/test_dense_precision.py::test_ut10_1_map_codes_map_roundtrip_missing_and_evicted` 通过。
**状态：** PASS

---

### ST-002：from_codes 与 build 完全一致

**关联需求：** srs.md §3.1 FR10.1
**测试类型：** 正常路径 + 边界
**优先级：** High

**前置条件：**
- 64×4、512×8、2048×8 随机混合精度 map，含 evicted 与复用掩码。

**测试步骤：**
1. `PrecisionPlan.build(map, reuse_mask)`。
2. `map_to_codes` → `PrecisionPlan.from_codes(codes, reuse_mask)`。
3. 逐项比较 codes/groups/flat_indices/reuse/evicted/fingerprint。

**期望结果：** 全部字段逐项相等；groups 按 (token, head) 升序且不含 evicted。

**实际结果：** `test_ut10_2_from_codes_matches_build` 三档参数全部通过。
**状态：** PASS

---

### ST-003：稠密预算准入与 dict 语义逐项一致

**关联需求：** srs.md §3.2 FR10.2
**测试类型：** 正常路径 + 边界
**优先级：** High

**前置条件：**
- 6×2 importance/sensitivity；预算档（充足/紧/底线/驱逐）；rd 开/关；sink=1；保护集。

**测试步骤：**
1. `admit_budget` 与 `admit_codes` 在相同输入上执行。
2. 比较映射、predicted、evicted。
3. 常量分数重复 8×3 的 tie 用例。
4. Agent 包装 `_admit_precision_codes` 与 `_admit_precision_budget` 对拍。

**期望结果：** 全部逐项一致；sink/保护条目保持 fp32；tie 顺序与 v3.11 相同。

**实际结果：** UT-10.3、UT-10.3b、UT-10.6b 通过；独立子代理 600 组随机 fuzz 0 差异。
**状态：** PASS

---

### ST-004：输入不可变与非法形状守卫

**关联需求：** srs.md §3.2 FR10.2
**测试类型：** 异常处理
**优先级：** Medium

**测试步骤：**
1. 对 codes 做 `admit_codes` 后比较原张量。
2. 传入 1-D codes 给 `from_codes`/`admit_codes`。

**期望结果：** 原张量不变；非法形状抛 `ValueError`。

**实际结果：** UT-10.4、UT-10.3c 通过。
**状态：** PASS

---

### ST-005：非连续 K/V 不触发 reference 回退

**关联需求：** srs.md §3.4 FR10.4
**测试类型：** 边界条件 + 集成
**优先级：** High

**前置条件：**
- `base[..., ::2]` 非连续 K/V；真实 4 层 `PyTorchModelBackend`。

**测试步骤：**
1. plan 压缩 vs legacy 压缩逐位比较。
2. 计数 `_compress_kv_reference` 调用。
3. 运行 `pipeline-check.py`（4 层真实后端 4 步）。

**期望结果：** k/v/scale 逐位一致；reference 调用 0 次；plan 调用 16 次。

**实际结果：** UT-10.5 通过；`pipeline-check.py` 实测 `plan_calls=16 reference_calls=0`。
**状态：** PASS

---

### ST-006：compress_kv(codes) 与 dict 结果逐位一致

**关联需求：** srs.md §3.3 FR10.3
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. 同一 K/V/map（含 evicted）分别以 map 与 codes 调用 `compress_kv`。
2. 比较 k/v/scale/packed/bytes。

**期望结果：** 全部逐位一致，含 packed 数据与内存记账。

**实际结果：** UT-10.6 通过。
**状态：** PASS

---

### ST-007：生产 on_step 稠密/字典路径一致（含 drift/强制模式）

**关联需求：** srs.md §3.3 FR10.3
**测试类型：** 正常路径 + 边界
**优先级：** High

**前置条件：**
- `dense_precision_enabled=true/false` 两个真实 Agent；预算 2097B。
- 场景：balanced；drift=0.8；sensitivity=0.9（强制 conservative）。

**测试步骤：**
1. 同输入分别执行 on_step。
2. 比较 precision_map、k/v、packed、memory_bytes。

**期望结果：** 三条场景全部逐位一致。

**实际结果：** UT-10.8、UT-10.8b 通过。
**状态：** PASS

---

### ST-008：决策缓存命中复用 codes

**关联需求：** srs.md §3.3 FR10.3
**测试类型：** 正常路径
**优先级：** High

**测试步骤：**
1. 同 (step, layer) 相同张量调用两次 on_step。
2. 统计 `_compute_importance_with_state` 调用次数并检查缓存 codes。

**期望结果：** 重要性仅计算 1 次；第二次复用缓存 codes；结果一致。

**实际结果：** UT-10.9、UT-10.9b 通过。
**状态：** PASS

---

### ST-009：长上下文元数据性能

**关联需求：** srs.md §3.5 FR10.5
**测试类型：** 性能
**优先级：** High

**测试步骤：**
1. 运行 `ar010_bench.py`（512/2048/4096/8192，3 次重复）。
2. 比较生产包装耗时并断言 map/predicted/fingerprint。

**期望结果：** 所有档位 >=10x；8192×32 <200ms；512-4096 不劣化。

**实际结果：** 16.67x/17.11x/18.81x/18.00x；8192 135.7ms；断言通过。
**状态：** PASS

---

### ST-010：兼容性与全量回归

**关联需求：** srs.md §4 非功能需求
**测试类型：** 回归测试
**优先级：** High

**测试步骤：**
1. 运行完整 pytest + 覆盖率。
2. 运行 black/isort/flake8/mypy。
3. 验证 `dense_precision_enabled=false` 可回到 v3.11 路径。

**期望结果：** 无回归；五件套全绿；覆盖率不低于 82%。

**实际结果：** 792 passed / 1 skipped，覆盖率 82.99%；五件套全绿。
**状态：** PASS

---

### ST-011：版本与文档一致性

**关联需求：** AGENTS.md §1 四检查点
**测试类型：** 文档
**优先级：** Medium

**测试步骤：**
1. 校验 pyproject、`shadowinfer.__version__`、`__main__.__version__`。
2. 检查 README 特性表/版本历史/基准表与 ROADMAP、server-experiment 数字一致。

**期望结果：** 均为 3.12.0；18.00x 与区间口径一致，无残留 20.83x。

**实际结果：** `test_ut17_version_strings_consistent` 通过；第 3 轮文档核验 PASS。
**状态：** PASS

---

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-19 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 | 稠密精度码与计划构造 | ST-001, ST-002 | PASS |
| §3.2 | 稠密预算准入 | ST-003, ST-004 | PASS |
| §3.3 | 生产稠密路径与缓存 | ST-006, ST-007, ST-008 | PASS |
| §3.4 | 连续 workspace / 零回退 | ST-005 | PASS |
| §3.5 | 性能与复现 | ST-009, ST-010 | PASS |

**需求覆盖率：** 5 / 5（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 6 | 6 | 0 | 0 |
| 边界条件 | 3 | 3 | 0 | 0 |
| 异常处理 | 1 | 1 | 0 | 0 |
| 回归测试 | 2 | 2 | 0 | 0 |
| **合计** | **12** | **12** | **0** | **0** |

（ST-001~ST-011 共 11 条用例；其中 ST-005/010 覆盖集成与回归两类，汇总按验证项计 12。）

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| - | 无 | - |

### 结论

> Go：5/5 需求覆盖，12/12 验证项通过，792 passed / 82.99%，无 Critical/Major/Minor 缺陷；
> 允许归档并推送。
