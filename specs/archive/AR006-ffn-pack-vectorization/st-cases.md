# AR006 ST 验收用例（2026-10-19）

环境：Windows / Python 3.11.9 / torch 2.5.1 CPU / pytest 9.1.1。

| 编号 | 场景 | 步骤 | 期望 | 结果 |
|------|------|------|------|------|
| ST-01 | 全量回归 | `python -m pytest` | 0 failed；覆盖率 >= 75% | PASS（743 passed / 1 skipped / 82.53%） |
| ST-02 | 质量门禁 | black / isort / flake8 / mypy | 零告警 | PASS |
| ST-03 | 打包 golden | `test_ut61_pack_golden_mixed_precision`、`test_ut61b_..._fp32_and_unknown_precision` | `_data`/dtype/shape/scales/memory/fingerprint 一致 | PASS |
| ST-04 | 奇数长度/空/维度 | `test_ut62`、`test_ut63`、3D ValueError | 行内 padding、空权重、2D 约束 | PASS |
| ST-05 | unpack 等价 | `test_ut64_pack_then_unpack_matches_reference` | <= 1e-6 | PASS |
| ST-06 | 性能 | `test_ut65_pack_vectorized_speedup` | 1024×256，3 次中位 <= 参考 50% | PASS（实测约 7.2%） |
| ST-07 | 微基准复现 | `python specs/archive/AR006-ffn-pack-vectorization/ar006_bench.py` | ratio <= 0.50；golden identical | PASS |
| ST-08 | 文档/版本 | README/ROADMAP/pyproject/`__version__` | 3.8.0 三处一致、四检查点、无失效引用 | PASS |
| ST-09 | 兼容回归 | 既有 FFN/unpack/缓存测试 | 全部通过 | PASS |

结论：AR006 验收通过。
