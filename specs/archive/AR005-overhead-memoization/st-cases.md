# AR005 ST 验收用例（2026-10-19）

环境：Windows / Python 3.11.9 / torch 2.5.1 CPU / pytest 9.1.1。

| 编号 | 场景 | 步骤 | 期望 | 结果 |
|------|------|------|------|------|
| ST-01 | 全量回归 | `python -m pytest` | 0 failed；覆盖率 >= 75% | PASS（738 passed / 1 skipped / 82.42%） |
| ST-02 | 质量门禁 | black / isort / flake8 / mypy | 零告警 | PASS |
| ST-03 | loader 记忆化 | `test_ut1_loader_attempted_once` | 200 次 kernel 调用仅 1 次加载尝试 | PASS |
| ST-04 | 并发首调用 | `test_ut2_concurrent_first_call` | 8 线程仅 1 次尝试 | PASS |
| ST-05 | 告警一次 | `test_ut3_unavailable_warns_once` | 仅 1 条告警 | PASS |
| ST-06 | 打包 golden | `test_ut4_pack_golden_bytes_identical` | k/v 字节、byte_offsets、scales、memory 一致（含奇数行、batch=2） | PASS |
| ST-07 | 打包性能 | `test_ut5_pack_vectorized_speedup` | 128×8×128，3 次中位 vec <= 参考 30% | PASS（实测约 1.6%） |
| ST-08 | 管线 smoke | `test_ut6_pipeline_smoke_after_optimization` | 2 步管线通过，step_results 完整 | PASS |
| ST-09 | 兼容回归 | 既有 shadowkv/kernels/vectorization 测试 | 全部通过 | PASS |
| ST-10 | 微基准复现 | `python specs/archive/AR005-overhead-memoization/ar005_bench.py` | loader <=0.05、pack <=0.30、golden identical | PASS |
| ST-11 | 文档/版本/归档 | README/ROADMAP/版本三处；归档 | 3.7.0 一致、链接有效、无 emoji | PASS |

结论：AR005 验收通过。
