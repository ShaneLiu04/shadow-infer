# [AR008] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR008 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（待生成） |
| 创建日期 | 2026-10-19 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 基线取证：compress/pack 重复量化与分组（多规模微基准） | - | passing | 2026-10-19：512×8×64 27.71 ms；1024×8×64 50.41 ms |
| T002 | TDD Red：PrecisionPlan + 融合路径 UT（golden/边界） | T001 | passing | 2026-10-19：UT-8.1~8.8（子代理 Red） |
| T003 | 实现 `precision_plan.py` + `_compress_kv_vectorized` 消费 plan | T002 | pending | |
| T004 | `PackedKVCache.pack` 预计算分组路径 + `compress_kv` 融合集成 | T003 | pending | |
| T005 | 性能微基准（≥3 规模点、3 次中位）+ 全量回归 + 五件套 | T004 | passing | 2026-10-19：759 passed / 82.87%；四形状 2.81-3.74x |
| T006 | README v3.10.0 / ROADMAP / 服务器复现配置与协议 | T005 | passing | 2026-10-19：README 四检查点、server-experiment.md、s2/s3 use_packed_cache |
| T007 | Review + ST + 归档 + Gitee 推送 | T006 | passing | 2026-10-19：归档至 specs/archive/AR008-fused-kv-plan；推送见仓库提交 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

### 2026-10-19 T001 基线取证

- 环境：Windows / CPU，torch 2.5.1+cu121，seed=42，5 次中位。
- `_compress_kv_vectorized`：seq=256/512/1024（heads=8,dim=64）分别 11.08 / 23.15 / 41.17 ms。
- `PackedKVCache.pack`：3.56 / 4.84 / 9.24 ms。
- seq=512/heads=16/dim=128：compress 47.22 ms + pack 9.61 ms。
- cProfile：compress 自耗时 0.069 s / 3 次，主要是逐条 dict/切片；单组量化约 0.14 ms。
- 复现脚本：`specs/archive/AR008-fused-kv-plan/ar008_bench.py`（待随实现补齐）。

### 2026-10-19 T002-T004 实现与 golden

- 新增 `PrecisionPlan`（稠密精度码 + 分组 + flat 索引 + 复用索引 + SHA-256 指纹）。
- `_compress_kv_with_plan` 用 flat gather/scatter 替代二维 advanced indexing；复用条目在分组写回后由上一步张量覆盖（与 legacy 语义一致）。
- `PackedKVCache._do_pack_with_plan` 复用分组与压缩阶段 fp/int8 结果；int4 按 pack 自己的 group size 重新量化，保证字节布局不变。
- `on_step` 生产路径传 `build_scale_map=False`；scale map 仅测试/golden 构建。
- UT-8.1~8.8 全通过；`test_m1_vectorization`/`test_shadowkv`/`test_budget_eviction` 回归通过。

### 2026-10-19 T005 性能微基准

`ar008_bench.py`，CPU，同进程 5 次中位，fused 与 legacy 逐位一致：

| 形状（seq×heads×dim） | compress legacy→plan | pack legacy→plan | 合计 legacy→fused | 比值 |
|---|---|---|---|---|
| 256×8×64 | 8.43→2.51 ms | 3.72→0.75 ms | 14.66→4.22 ms | 3.48x |
| 512×8×64 | 20.60→2.68 ms | 3.91→1.11 ms | 26.23→9.33 ms | 2.81x |
| 1024×8×64 | 43.15→3.77 ms | 7.73→1.41 ms | 47.26→12.65 ms | 3.74x |
| 512×16×128 | 39.47→5.63 ms | 9.86→1.71 ms | 53.98→15.54 ms | 3.47x |

## Review 记录（2026-10-19）

- 设计符合性：`PrecisionPlan` 与 design §4.3 接口一致；`_compress_kv_with_plan` / `_do_pack_with_plan` 均按 §4.2 流程实现。
- golden：legacy 压缩、reference 压缩、原 pack 三条路径保留；UT 覆盖逐位一致、复用、batch=2、奇数 head_dim、int4。
- 回退：plan 构建/执行异常回退 reference；pack 缺 precomputed 自动自行 gather/quantize。
- 兼容性：新增参数均有默认值；既有 `pack`/`compress_kv` 调用不变；仅测试中的私有 pack 计数器补 `**kwargs`。
- 风险：int4 因压缩工作副本与 pack 的 group size 语义不同，未跨阶段复用码值（仅复用 gather），已在 design/README 说明；paged KV 留作后续 AR。
- 结论：PASS。

## ST 记录（2026-10-19）

- `st-cases.md` ST-8.1~ST-8.7：本机项全 PASS；ST-8.7 服务器执行待回填。
- 全量：759 passed / 1 skipped / 82.87%；五件套全绿。

## 阶段门控记录

- 2026-10-19：用户要求自动化执行 harnessX，跳过交互式 req/design 门控（无 FAIL 记录）；Review/ST 按本文件记录执行。


## 阶段门控记录
