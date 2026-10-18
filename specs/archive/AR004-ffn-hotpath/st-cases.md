# AR004 ST 验收用例（2026-10-18）

环境：Windows / Python 3.11.9 / torch 2.5.1 CPU / pytest 9.1.1。

| 编号 | 场景 | 步骤 | 期望 | 结果 |
|------|------|------|------|------|
| ST-01 | 全量回归 | `python -m pytest` | >= 732 passed / 0 failed；覆盖率 >= 75% | PASS（732 passed / 1 skipped / 82.37%） |
| ST-02 | 质量门禁 | black / isort / flake8 / mypy | 全部零告警 | PASS |
| ST-03 | 指纹 O(1) | `pytest tests/test_ffn_hotpath.py::test_ut1_fingerprint_is_o1 test_ut2_repack_invalidates_cache` | 1024 ch <= 10µs；重打包缓存失效 | PASS |
| ST-04 | unpack golden | `test_ut4_unpack_golden_vec_vs_reference`、`test_ut4_int4_non_divisible_group` | vec vs 参考 <= 1e-6（pack_dim 0/1、int4 非整除） | PASS |
| ST-05 | 量化 golden | `test_ut10_quantize_golden_vec_vs_reference`、`test_ut10_zero_channel_preserved`、`test_ut14_threshold_boundary_matches_reference` | <= 1e-6；scales 一致；0.7/0.3 边界同侧 | PASS |
| ST-06 | 层隔离 | `test_ut6_layer_isolation_full_path`、`test_ut6b_lifecycle_and_overflow_warning`、`test_ut6c_sparse_and_incremental_layer_isolation`、`test_ut16_static_pruning_uses_per_layer_importance` | 各层输出等于直接计算；on_init 清空；超层告警一次 | PASS |
| ST-07 | 掩码修复 | `test_ut8_mixed_mask_golden`、`test_ut9_all_false_mask_zero_with_single_warning`、`test_ut9_mask_exception_resets_state`、`test_ut21_full_all_false_mask` | packed + mask 不抛错且与 full golden 一致；全 False 零输出/告警一次 | PASS |
| ST-08 | 兼容回归 | `tests/test_ffn_channel_pruning.py`、`test_ffn_optimizer.py`、`test_m5_memoization.py`、`test_m1_vectorization.py` | 全部通过；property 赋值兼容 | PASS |
| ST-09 | 微基准 | `python specs/archive/AR004-ffn-hotpath/ar004_bench.py` | 同进程 A/B：指纹 0.001-0.004x；unpack 0.044-0.070x；量化 0.053-0.077x；on_step 0.59-0.77ms；缺陷复现转通过 | PASS |
| ST-10 | 文档/版本 | 检查 README/pyproject/`__version__`/ROADMAP | 版本三处 3.6.0；README 四检查点；无 emoji；能力与实测一致 | PASS |
| ST-11 | 归档 | `specs/archive/AR004-ffn-hotpath/` -> `specs/archive/AR004-ffn-hotpath/` | 交付件齐全、引用更新 | PASS |

结论：AR004 验收通过。
