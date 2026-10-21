# [AR014] 软件需求规格说明书：分页增量 KV 驻留

## 1. 背景与目标

### 1.1 痛点实证

当前 append 语义下（服务/LM 生成与 orchestrator 默认路径），无论 `kv_execution` 取 `dense` 还是 `packed`，每个执行步都对**全量** KV 执行去量化与再量化：

- `dense` 路径：`pytorch_backend.py:239` 每步 `_dequantize_kv(entry)` 全量去量化（O(S_total)），`pytorch_backend.py:352/368` 每步 `_quantize_kv(k, v, precision)` 全量再量化（O(S_total)）；
- `packed` 路径：`pytorch_backend.py:229-239` 每步 `entry.packed_kv.unpack()` 全量解包（O(S_total)），随后 `PackedKVCache.pack(...)` 对 past+new 全量重打包（O(S_total)），并伴随全量稠密瞬态分配。

即每步工作量为 2×O(S_total)，N 步累计 O(N^2)；上下文越长，每步延迟越高。这是服务路径的核心可扩展性缺陷。

### 1.2 SOTA 参照与选题依据

FlashInfer 的核心工程思想之一是**页表间接寻址（page table indirection）**：KV 按固定大小页存储，append 仅写入新页（摊销 O(1)），读取经页表批量 gather，驱逐按页归还槽位。本项目已在 AR008 落地 plan/run 分离、AR012 落地向量化紧凑驻留，但缺 FlashInfer 的另一半：**页表间接层**。ROADMAP §3.4 已将「Paged KV layout (future): FlashInfer-style page table indirection for O(1) append/eviction」列为规划项。

### 1.3 目标

新增 `kv_execution="paged"` 执行模式：KV 以按精度分页的方式驻留，append 步仅对**新 token** 执行量化入页，已封存页不可变且永不重打包；读取经页表批量 gather 解包；驱逐按页粒度执行且不触发全量重打包。在同等精度语义下，实现与现有全量重打包路径的**逐元素等价**，并以实测数据验证：每步打包工作量从 O(S_total) 降至 O(S_new)、瞬态分配显著下降、总时间收益随上下文长度增长（预期 ~1.8-2x，以实测为准）。

## 2. 需求范围

| 项 | 内容 |
|---|---|
| AR 编号 | AR014 |
| 版本 | v3.16.0 |
| 影响面 | `shadowinfer/shadowkv/paged_kv_store.py`（新增）、`pytorch_backend.py`、`orchestrator.py`、`shadowkv` 包导出、`tests/test_paged_kv_execution.py`（新增）、`specs/changes/AR014-paged-kv-store/ar014_bench.py`、`configs/experiments/s9_paged_longctx_gpu.yaml`、README.md、版本号三处同步 |
| 不破坏对外 API | 新增可选配置项与统计键允许；现有 `dense`/`packed` 行为不变 |

## 3. 功能需求

### 3.1 FR14.1 PagedKVStore 核心数据结构

新增 `shadowinfer/shadowkv/paged_kv_store.py`，提供 `PagedKVStore`：

- **页表间接寻址**：per-layer 维护 token 位置到页的映射；页大小 `page_size`（默认 32，可配置，须 >=1）；
- **按精度分页池**：页按其 token 的精度语义（fp16/int8/int4）持有量化数据；同一页内 token 精度一致（页为量化粒度边界）；
- **空闲链（free list）**：驱逐归还的页槽位可被后续 append 复用；
- **量化语义约束**：行级量化/去量化必须与 `PackedKVCache` 现有 per-token-head 语义逐元素一致（复用同一量化实现），保证 golden 等价可达 bitwise 级；
- **记账**：`data_plane_bytes()`（张量字节，含 scale）与 `resident_bytes()`（另含元数据估计，口径与 AR013 一致：per token-head 48B + 页表槽位开销），统计含 `page_count`、`open_pages`、`appended_tokens` 等。

### 3.2 FR14.2 增量 append（O(S_new)）

- `append(layer_id, k_new, v_new, precision)` 仅对新 token 执行量化入页：摊销 O(S_new)，**不触及已封存页**（sealed page 对象不可变）；
- 允许尾部开放页（open page）填充以摊销成本；封存页永不重写；
- 页槽位优先复用空闲链；
- append 失败（shape/精度不兼容等）须显式报错或按后端约定走 fallback，禁止静默丢弃数据。

### 3.3 FR14.3 批量 gather 解包（单遍）

- `gather(layer_id)`（或等价 API）经页表在单遍内向量化拼出与 `PackedKVCache.unpack()` 同布局的稠密 (k, v)，供 attention 消费；
- 复杂度 O(S_total) 读取（不可消除的下界），但**无再量化**、无第二次全量分配；
- 等价性要求：对同一 token 序列与同一精度映射，`append 逐步执行后 gather` 的输出与 `PackedKVCache.pack(全量)` 后 `unpack()` 的输出**逐元素相等**（fp16/int8/int4 全路径）。

### 3.4 FR14.4 页粒度驱逐与槽位复用

- 提供 token 粒度驱逐入口（与现有 `PackedKVCache.evict(token_head_set)` 语义对齐，供 RD 驱逐路径复用）：定位受影响页，页内存活 token 重打包进新页，空页槽位归还空闲链；
- 驱逐成本摊销 O(受影响页数 × page_size)，**不触发全量重打包**；
- 驱逐后 `gather` 输出必须排除已驱逐 token，`resident_bytes`/`data_plane_bytes` 相应下降。

### 3.5 FR14.5 后端 paged 执行模式

`pytorch_backend.py` 的 `configure_execution` 新增 `kv_execution="paged"`：

- **append 生命周期**：每步 `raw_cache` 经页表 gather 构建（替代全量 unpack/dequant）；模型产出后仅新 token 入页（替代全量 pack/requant）；旧页零重打包；
- **replace 生命周期**：允许 paged（整页集重写，语义正确），但不对其宣称性能收益；
- **reset/形状不匹配**：`reset_execution_state()` 覆盖 paged 状态；shape mismatch 清空页集并置 `reset_reason="shape_mismatch"`（口径沿用 AR013）；
- **fallback**：页 append/gather 失败时回退 dense 路径并置 `fallback=True` + 原因（口径沿用 AR013）；
- **执行统计**：`execution_stats` 新增 `paged` 相关键（`page_count`、`appended_tokens`、`repack_tokens_saved`（每步避免重打包的 past token 数）、`page_evictions`），并保持既有双口径记账键的正确性。

### 3.6 FR14.6 Orchestrator 接线、等价性验证、基准与服务器协议

- **Orchestrator**：`_maybe_configure_execution` 透传 `execution.page_size` 与 `kv_execution="paged"`；run_* 起始 reset 覆盖 paged 状态；`execution_stats` 沿既有通道透传至 `state.kv_result["execution"]` 与 profiler 结果；
- **UT（golden）**：新增 `tests/test_paged_kv_execution.py`，至少覆盖：
  - 逐步 append 后 gather 与全量 pack/unpack 的逐元素等价（fp16/int8/int4）；
  - 已封存页对象身份在后续步保持不变（不可变性实证）；
  - token 粒度驱逐：字节回收、gather 排除、空闲链复用；
  - append/gather/驱逐的参数校验与错误路径；
  - 后端 paged 模式端到端 logits 等价（vs packed 路径同种子同输入）、replace 兼容、reset、fallback、统计键存在性与正确性；
- **基准（`specs/changes/AR014-paged-kv-store/ar014_bench.py`）**：append 模式 N 步生成，baseline = 现行 packed 全量重打包路径，对照 = paged；至少 2 个规模点（如 (N=8, S_new=64) 与 (N=16, S_new=128)），速度 3 seeds 报告中位与区间；产出：总时间加速比、每步延迟曲线（paged 斜率显著低于 baseline）、瞬态峰值分配对比、logits relL2（fp16 期望 0）；所有断言基于实测；
- **服务器协议**：`configs/experiments/s9_paged_longctx_gpu.yaml`（7B 长上下文 append 服务协议，含维度注释与 --seed 复现命令）与 `specs/changes/AR014-paged-kv-store/server-experiment.md`；
- **实验落档**：完整配置、原始数值、复现命令存入 `specs/changes/AR014-paged-kv-store/`，README 仅引用结论性表格（规则 1.1）。

## 4. 非功能需求

- **默认行为不变**：`kv_execution` 默认仍为 `dense`；paged 为显式 opt-in；
- **性能主张纪律**：README 中任何优越性数字必须来自本 AR 实跑（规则 1.1：对照、多规模、3 seeds、实档），未验证主张只能以「机制说明（未验证）」口径表述；
- **质量门禁**：black / isort / flake8 / mypy / pytest 五件套全绿；总用例数与覆盖率不低于 AR013 基线（876 passed / 83.37%）；
- **版本同步**：`pyproject.toml`、`shadowinfer/__init__.py`、`shadowinfer/__main__.py`（及 `test_ffn_hotpath.test_ut17` 中的版本断言）同步至 3.16.0；
- **文档**：README 四检查点更新（特性表、版本历史 3.16.0、基准表新增 paged 实测数据、TOC/口径零残留）；全文无 emoji。

## 5. 约束与边界

- **Out of scope**：
  - 不在页上直接执行 fused attention（dense 物化仍存在于 gather 之后；这是候选后续创新）；
  - 宏块 SKIP/DELTA/REFRESH 模式控制（AR015 候选）；
  - HF 后端、CUDA kernel、GPU 专属路径（GPU 侧仅落配置与复现协议，待服务器）；
  - 不破坏对外 API（新增可选参数/键允许）。
- **诚实收益边界**：paged 每步仍须 O(S_total) 的 gather 读取（attention 消费下界）；收益来自消除全量再打包与瞬态分配、以及 O(1) 级驱逐。所有加速比以实测报告区间，禁止只报最好点；CPU 实验须标明实际执行设备。

## 6. 术语

| 术语 | 定义 |
|---|---|
| 页（page） | 固定 `page_size` 个 token 位置的 KV 量化数据存储单元 |
| 封存页 / 开放页 | 已写满且不可变的页 / 尾部可继续填充的页 |
| 页表（page table） | token 位置到页槽位的间接映射 |
| 空闲链（free list) | 已驱逐页槽位的可复用列表 |
| golden 等价 | 同输入、同精度映射下与全量 pack/unpack 路径的逐元素相等 |
