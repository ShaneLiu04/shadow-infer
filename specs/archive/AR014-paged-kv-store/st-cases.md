# [AR014] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR014 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 生成日期 | 2026-10-21 |

## 测试用例列表

### ST-001：页表间接寻址与按精度分页池

**关联需求：** srs.md §3.1 FR14.1（页表/分页池/页为量化粒度边界）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 真实张量、`PagedKVStore`（B=2, H=3, D=8, page_size=8）

**测试步骤：**
1. 分段以 fp16/int8/int4 交替 append
2. 逐步 gather 并与逐段独立参考（每段单独全量 pack→unpack）对比

**期望结果：**
- Then 页按精度分段持有数据；gather 输出与逐段参考逐元素相等

**实际结果：** 自动化 `tests/test_paged_kv_execution.py::test_mixed_precision_pages_match_per_segment_references` 通过（torch.equal）

**状态：** PASS

---

### ST-002：page_size 可配置与非法值拒绝

**关联需求：** srs.md §3.1 FR14.1（page_size 默认 32、须 >=1）；§3.5 FR14.5
**测试类型：** 异常处理
**优先级：** Medium

**前置条件：**
- Given store 构造与后端 `configure_execution`

**测试步骤：**
1. 构造 `page_size=0` 的 store；2. 后端配置 `page_size=0` 与非法 `kv_execution`

**期望结果：**
- Then 均 `ValueError` 显式报错

**实际结果：** 自动化 `test_page_size_below_one_raises`、`TestPagedConfigValidation` 3 例通过

**状态：** PASS

---

### ST-003：空闲链槽位复用

**关联需求：** srs.md §3.1 FR14.1（free list）；§3.4 FR14.4
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 已有 2 页（1 封存 + 1 开放）

**测试步骤：**
1. 驱逐整页 token；2. 再 append 触发新页分配

**期望结果：**
- Then 归还槽位被复用（`len(pages)` 不增），页序与数据正确

**实际结果：** 自动化 `test_evict_full_page_returns_slot_to_free_list` 通过（slot 复用断言）

**状态：** PASS

---

### ST-004：双口径记账

**关联需求：** srs.md §3.1 FR14.1（data_plane_bytes/resident_bytes/统计）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given fp16/fp32/int8/int4 各精度 store

**测试步骤：**
1. 断言 fp16/fp32 data_plane_bytes 与 dense 基准精确相等；2. int8/int4 低于 dense fp16；3. resident 含 48B/token-head 元数据；4. 空 store 全零

**期望结果：**
- Then 双口径记账精确且空 store 为零基线

**实际结果：** 自动化 `test_data_plane_bytes_fp16_exact` / `test_data_plane_bytes_fp32_exact` / `test_data_plane_bytes_int8_below_dense_fp16` / `test_data_plane_bytes_int4_below_dense_fp16` / `test_resident_bytes_include_metadata_overhead` / `test_empty_store_accounting_is_zero` 通过

**状态：** PASS

---

### ST-005：四精度逐步 append 后 gather 的 bitwise golden 等价

**关联需求：** srs.md §3.2 FR14.2 + §3.3 FR14.3（与 PackedKVCache 逐元素一致）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 同一 token 序列、同一精度，两路径对照

**测试步骤：**
1. PagedKVStore 分步 append 后 gather；2. PackedKVCache 全量 pack 后 unpack；3. 逐元素对比

**期望结果：**
- Then fp32/fp16/int8/int4 四精度均 `torch.equal`

**实际结果：** 自动化 `test_gather_golden_fp32_stepwise_append` / `_fp16_` / `_int8_` / `_int4_` 4 例通过（bitwise 级）

**状态：** PASS

---

### ST-006：封存页不可变（对象身份保持）

**关联需求：** srs.md §3.2 FR14.2（不触及已封存页）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- Given 至少 1 个封存页

**测试步骤：**
1. 后续 append / 开放页填充；2. 断言封存页 `id()` 与 `data_ptr()` 不变

**期望结果：**
- Then 封存页对象身份跨 append 稳定

**实际结果：** 自动化 `test_sealed_pages_are_immutable_across_appends` 与 `::TestPagedIncrementalImmutability::test_sealed_page_identity_survives_next_append`（后端级）通过

**状态：** PASS

---

### ST-007：append 错误路径（显式报错、不变异、无静默丢弃）

**关联需求：** srs.md §3.2 FR14.2（append 失败须显式报错）
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given 正常 store

**测试步骤：**
1. 非法精度 append；2. 形状不匹配 append；3. 零 token append

**期望结果：**
- Then 1/2 抛 `ValueError` 且 store 状态不变异；3 为 noop

**实际结果：** 自动化 `test_invalid_precision_raises` / `test_append_shape_mismatch_raises_without_mutation`（记账与页序前后相等）/ `test_zero_token_append_is_noop` 通过

**状态：** PASS

---

### ST-008：gather 单遍输出与空 store

**关联需求：** srs.md §3.3 FR14.3（O(S_total) 单遍、无再量化）
**测试类型：** 边界条件
**优先级：** Medium

**前置条件：**
- Given 空 store 与含洞（驱逐后）store

**测试步骤：**
1. 空 store gather；2. 含洞 store gather

**期望结果：**
- Then 空 store 返回零长度 (k,v)；洞位零填充、布局与 unpack 一致

**实际结果：** 自动化 `test_empty_store_gather_returns_zero_length` 通过；含洞路径由驱逐 golden 组覆盖（ST-013）

**状态：** PASS

---

### ST-009：跨页 append 与 int4 group_size 自适应

**关联需求：** srs.md §3.2 FR14.2（跨页切分）；§3.1（量化语义约束）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- Given 单次 append 跨越页边界；int4 head_dim 不整除 group_size

**测试步骤：**
1. 单次 int8 append 跨 2 页后 gather golden 对比；2. int4 group_size 回退路径 golden 对比

**期望结果：**
- Then 跨页切分与 group 自适应均与 PackedKVCache 逐元素一致

**实际结果：** 自动化 `test_gather_golden_int8_single_append_spans_pages` / `test_gather_golden_int4_group_size_fallback_head_dim` 通过

**状态：** PASS

---

### ST-010：全 head token 驱逐字节回收

**关联需求：** srs.md §3.4 FR14.4（字节相应下降）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 已填 fp16 store

**测试步骤：**
1. 驱逐完整 token 区间；2. 断言 `data_plane_bytes` 精确下降

**期望结果：**
- Then 释放字节精确等于被驱逐 payload 字节；gather 排除该 token

**实际结果：** 自动化 `test_evict_full_tokens_release_exact_bytes` 通过

**状态：** PASS

---

### ST-011：部分 head 驱逐零值行（int8/int4 语义）

**关联需求：** srs.md §3.4 FR14.4
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- Given int8 与 int4 store

**测试步骤：**
1. 驱逐部分 head；2. gather 对比零值行；3. 断言字节不降（页保留）

**期望结果：**
- Then int8 q=0 解码 0.0、int4 打包字节 0x88 解码 0.0；未触碰元素逐位不变

**实际结果：** 自动化 `test_evict_int8_golden_and_zeroed_heads` / `test_evict_int4_golden_and_zeroed_heads` / `test_evict_partial_heads_save_no_page_bytes` 通过

**状态：** PASS

---

### ST-012：驱逐段拆分、二次驱逐与未触及页

**关联需求：** srs.md §3.4 FR14.4（摊销驱逐、不复活）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- Given 多页 store 含开放页

**测试步骤：**
1. 驱逐页中部 token；2. 二次驱逐同集合；3. 驱逐空集；4. 断言未触及页身份

**期望结果：**
- Then 段拆分保 start_token 正确（segments 断言）；二次/空集驱逐返回 0 不复活；未触及页 `id()`/数据不变；驱逐后可继续 append

**实际结果：** 自动化 `test_evict_middle_token_splits_page_into_segments`（segments==[(0,3),(4,8),(8,16),(16,19)]）/ `test_evict_absent_and_already_evicted_positions_returns_zero` / `test_evict_empty_set_is_noop` / `test_evict_leaves_unaffected_pages_identity_intact` / `test_evict_preserves_max_seq_and_supports_further_append` 通过

**状态：** PASS

---

### ST-013：驱逐 golden 等价（三精度 vs PackedKVCache.evict）

**关联需求：** srs.md §3.4 FR14.4（语义对齐）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given fp16/int8/int4 同数据两路径

**测试步骤：**
1. 两路径执行同集合驱逐后各自解包；2. 逐元素对比

**期望结果：**
- Then 驱逐后输出与 PackedKVCache.pack→evict→unpack 逐元素一致

**实际结果：** 自动化 `test_evict_golden_fp16_mixed_full_partial_untouched`（含开放页驱逐边界）等驱逐 golden 组通过

**状态：** PASS

---

### ST-014：append 生命周期端到端 logits 等价

**关联需求：** srs.md §3.5 FR14.5（append 每步 gather 构建 raw_cache、仅新 token 入页）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 真实 `SimpleDiffusionLLM`，paged vs packed 同种子同输入

**测试步骤：**
1. 两路径各跑 3 步（步长 5/4/4 跨页）；2. 每步 logits 对比

**期望结果：**
- Then fp16 `torch.equal`；int8 `torch.equal`（同行级量化口径）

**实际结果：** 自动化 `test_paged_matches_packed_logits_fp16` / `test_paged_matches_packed_logits_int8` 通过

**状态：** PASS

---

### ST-015：replace 兼容、reset 覆盖、shape mismatch 清页

**关联需求：** srs.md §3.5 FR14.5（生命周期/状态管理）
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- Given paged 后端

**测试步骤：**
1. replace + paged 连续步；2. `reset_execution_state`；3. 注入形状突变输入

**期望结果：**
- Then replace 每步整页集重写且语义正确；reset 后页集清空统计归零；mismatch 清页并置 `reset_reason="shape_mismatch"`

**实际结果：** 自动化 `test_replace_paged_repages_each_step` / `test_reset_clears_paged_state` / `test_paged_shape_mismatch_clears_cache_with_reason` 通过

**状态：** PASS

---

### ST-016：fallback 回退 dense 并恢复（写路径口径）

**关联需求：** srs.md §3.5 FR14.5（fallback 置位与恢复；口径见 tasks.md Review 契约裁定：保护写路径）
**测试类型：** 异常处理
**优先级：** High

**前置条件：**
- Given paged 后端，monkeypatch 注入 append 失败（AR013 先例允许的故障注入）

**测试步骤：**
1. 注入失败步执行；2. 撤销注入继续步

**期望结果：**
- Then 失败步 `fallback=True` 且输出可用；后续步页历史恢复（max_seq 保持已见总 token，历史不丢）

**实际结果：** 自动化 `test_append_failure_falls_back_to_dense_and_recovers` 通过

**状态：** PASS

---

### ST-017：执行统计键正确性

**关联需求：** srs.md §3.5 FR14.5（paged/page_size/page_count/open_pages/appended_tokens/appended_tokens_total/repack_tokens_saved/page_evictions 八键）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given paged 后端跑 3 步（步长 5/4/4，PAGE_SIZE=4，L=2）

**测试步骤：**
1. 逐步断言八键数值与解析预期（pages=ceil(total/4)、open=total%4?1:0、repack_saved=past）
2. 断言双口径记账 == 逐层 store 求和

**期望结果：**
- Then 全键逐步正确且单调；双口径一致

**实际结果：** 自动化 `test_stats_track_paged_append_accounting` 通过（含 open_pages==2 断言）

**状态：** PASS

---

### ST-018：orchestrator 透传与统计透传

**关联需求：** srs.md §3.6 FR14.6（execution 配置透传、execution_stats 双通道）
**测试类型：** 集成
**优先级：** High

**前置条件：**
- Given Orchestrator + paged execution 配置

**测试步骤：**
1. orchestrator 配置 `kv_execution="paged"` + `page_size` 下发；2. 运行后检查 `state.kv_result["execution"]` 与 profiler 结果

**期望结果：**
- Then 配置到达后端（统计含 paged=True）；统计沿双通道透传

**实际结果：** 自动化 `TestOrchestratorPagedWiring::test_paged_execution_config_reaches_backend_and_results` 通过

**状态：** PASS

---

### ST-019：基准预注册断言实跑

**关联需求：** srs.md §3.6 FR14.6（2 规模 × 3 seeds、预注册断言、落档）
**测试类型：** 集成（真实基准脚本）
**优先级：** High

**前置条件：**
- Given `ar014_bench.py`（append 簿记平面，baseline=packed 全量重打包）

**测试步骤：**
1. 运行脚本（2 规模 × 3 seeds）
2. 检查内置断言：加速 >=1.5x、斜率比 <0.60、堆峰值 paged<baseline、fp16 `torch.equal`

**期望结果：**
- Then 全部断言通过并输出 median/区间

**实际结果：** 2026-10-21 实跑通过：(8,64,512) 加速 9.303x [9.209-9.751]、斜率比 0.088、堆 2408KB→47KB；(16,128,2048) 加速 10.162x [7.684-10.278]、斜率比 0.114、堆 9975KB→175KB；输出 "all pre-registered assertions passed"；逐 seed 原始数据落档 bench-results.md

**状态：** PASS

---

### ST-020：服务器复现协议资产完整性

**关联需求：** srs.md §3.6 FR14.6（s9 yaml + server-experiment.md）
**测试类型：** 文档/配置检查（MANUAL）
**优先级：** Medium

**前置条件：**
- Given `configs/experiments/s9_paged_longctx_gpu.yaml` 与 `server-experiment.md`

**测试步骤：**
1. 检查 yaml：7B 维度注释、三对照臂说明、--seed 完整命令、统计键清单
2. 检查 server-experiment.md：判定表、复现命令（changes/archive 双路径）、质量/显存模板

**期望结果：**
- Then 可按文档在服务器复现

**实际结果：** 人工核验通过（yaml 含 7B 维度与完整命令、append+paged+page_size=32 主臂；server-experiment.md §1 判定含斜率比与 fp16 逐位要求、§3 双路径、§4 三对照臂与模板、§6 落档规则）

**状态：** PASS

---

### ST-021：版本同步、默认行为回归与全量门禁

**关联需求：** srs.md §4 非功能（默认不变/覆盖率 >=83.37%/版本三处同步）
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- Given 仓库全量测试套件与五件套门禁

**测试步骤：**
1. `test_ffn_hotpath.py::test_ut17` 版本一致性；2. 默认 `kv_execution="dense"` 回归；3. 全量 pytest（含覆盖率门禁）+ black/isort/flake8/mypy

**期望结果：**
- Then 版本四处 == 3.16.0；默认路径不变；全量通过且覆盖率 >= 83.37%（AR013 基线）

**实际结果：** 2026-10-21 实跑：test_ffn_hotpath.py 34 passed（含 test_ut17=3.16.0）；`test_default_execution_stays_dense` 通过；全量 **921 passed / 1 skipped，覆盖率 83.78%**（基线 876/83.37%，+45 用例、+0.41pp）；black 135 文件/isort/flake8/mypy 73 文件全绿；AR013 回归 `tests/test_step_reuse_execution.py` 23 passed

**状态：** PASS

---

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-21 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 FR14.1 | PagedKVStore 核心数据结构 | ST-001, ST-002, ST-003, ST-004 | PASS |
| §3.2 FR14.2 | 增量 append | ST-005, ST-006, ST-007, ST-009 | PASS |
| §3.3 FR14.3 | 单遍 gather | ST-005, ST-008, ST-009 | PASS |
| §3.4 FR14.4 | 页粒度驱逐与槽位复用 | ST-003, ST-010, ST-011, ST-012, ST-013 | PASS |
| §3.5 FR14.5 | 后端 paged 执行模式 | ST-002, ST-014, ST-015, ST-016, ST-017 | PASS |
| §3.6 FR14.6 | 接线/UT/基准/服务器协议 | ST-018, ST-019, ST-020 | PASS |
| §4 非功能 | 默认不变/覆盖率/版本同步 | ST-021 | PASS |

**需求覆盖率：** 7 / 7（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 10 | 10 | 0 | 0 |
| 边界条件 | 6 | 6 | 0 | 0 |
| 异常处理 | 3 | 3 | 0 | 0 |
| 集成/回归 | 2 | 2 | 0 | 0 |
| **合计** | **21** | **21** | **0** | **0** |

### 缺陷记录

| 编号 | 严重性 | 描述 | 处理方式 |
|------|-------|------|---------|
| D-01 | Minor（已修） | Review D4：design 声明的 `open_pages` 统计键未实现、srs 声明的 `shadowkv` 包导出未做、FR14.5 fallback 条款「gather 失败回退」与 design §4.2.1 不一致 | 已修复：补 open_pages 第 8 键+断言、补包导出、fallback 口径按 design §4.2.1 记契约裁定（tasks.md Review 记录）；复测全绿 |
| D-02 | Minor | AR002 遗留 `test_m1_vectorization.py::test_compress_kv_perf_5x` 负载敏感断言偶发抖动（AR013 D-02 延续，非 AR014 缺陷） | 延后处理 |

### 结论

> Go。21/21 用例通过，需求覆盖 7/7（100%），无未修复 Critical/Major 缺陷；全量回归 921 passed / 1 skipped、覆盖率 83.78% 无回退，五件套全绿；bench 预注册断言实跑通过且逐 seed 数据落档（含 seed=1 抖动如实保留）。D-02 为 AR002 遗留 Minor，延续延后处理。
