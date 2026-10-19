# [AR008] ST 用例与结果

| 用例 | 步骤 | 期望 | 结果 |
|------|------|------|------|
| ST-8.1 | `pytest tests/test_kv_plan.py -q --no-cov` | UT-8.1~8.8 全通过（真实张量） | PASS（8 passed） |
| ST-8.2 | `python specs/archive/AR008-fused-kv-plan/ar008_bench.py` | 4 个形状 fused 与 legacy 逐位一致，输出中位/比值 | PASS |
| ST-8.3 | `pytest tests/test_m1_vectorization.py tests/test_shadowkv.py tests/test_m5_memoization.py -q --no-cov` | 既有压缩/打包/记忆化回归全通过 | PASS |
| ST-8.4 | `pytest tests/test_budget_eviction.py -q --no-cov` | 预算准入与 pack 次数回归 | PASS（10 passed） |
| ST-8.5 | 全量 `pytest` | 无回归、覆盖率 >=75% | PASS（759 passed / 1 skipped / 82.87%） |
| ST-8.6 | 五件套 | black/isort/flake8/mypy/pytest 全绿 | PASS |
| ST-8.7 | 服务器协议审阅 | `server-experiment.md` 命令可复制、产物/判定/回填清晰 | PASS（待服务器执行） |

## 实测数据（CPU，2026-10-19）

`ar008_bench.py`，seed=42，同进程 5 次中位，逐位断言通过：

| 形状（seq×heads×dim） | 合计 legacy→fused | 比值 |
|---|---|---|
| 256×8×64 | 14.66→4.22 ms | 3.48x |
| 512×8×64 | 26.23→9.33 ms | 2.81x |
| 1024×8×64 | 47.26→12.65 ms | 3.74x |
| 512×16×128 | 53.98→15.54 ms | 3.47x |

## 正确性覆盖

- 混合精度 fp32/fp16/int8/int4；int4 奇数 head_dim=48；
- batch=2；
- reuse_mask 命中/未命中，复用条目最终值来自上一步；
- `build_scale_map=True/False`；
- plan/precomputed 缺失回退、形状不匹配、设备不匹配；
- 冻结 golden：legacy `_compress_kv_vectorized`、`_compress_kv_reference`、`PackedKVCache` 原路径。
