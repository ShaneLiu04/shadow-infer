# [AR005] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR005 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-19 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 基线取证：loader 逐次重试与 int4 pack 逐条循环 → baseline-profile.md | - | passing（已核验） | 2026-10-19（ar005_bench.py） |
| T002 | FR5.1 加载记忆化 + 双检锁 + 告警一次 | T001 | passing | 2026-10-19 |
| T003 | FR5.2 `_pack_int4_rows` 批量打包 + 参考实现保留 | T001 | passing | 2026-10-19 |
| T004 | FR5.3 测试（UT-1~UT-6）+ 全量回归 + 五件套 | T002, T003 | passing | 2026-10-19：tests/test_pipeline_overhead.py 6 passed；全量 738 passed / 82.41%；五件套全绿 |
| T005 | FR5.4 README v3.7.0 四检查点 + 版本同步 + ROADMAP | T004 | passing | 2026-10-19：特性表/版本历史/基准/复现路径；版本三处 3.7.0 |
| T006 | Review + ST + 归档 + Gitee 推送 | T005 | pending | 规则二 |

## 进度记录

### 2026-10-19 T001 基线取证

- 7B 配置 6 步管线 cProfile：`_load_cuda_extensions` 3816 次 / 12.4s（占总 27.6s 的 45%），单次约 2.8ms（本机 CUDA 扩展加载失败导致逐次重试）。
- `PackedKVCache` int4 pack（128 seq × 8 heads × head_dim 128）：逐条实现 72.7ms vs 参考 189.17ms（0.384x）。
- 脚本：`specs/changes/AR005-overhead-memoization/ar005_bench.py`（seed=42，CPU）。
