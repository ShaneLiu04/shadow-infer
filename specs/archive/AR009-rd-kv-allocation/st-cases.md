# [AR009] ST 用例与结果

| 用例 | 步骤 | 期望 | 结果 |
|------|------|------|------|
| ST-9.1 | `pytest tests/test_rd_allocator.py -q --no-cov` | UT-9.1~9.13 全通过（真实张量） | PASS（13 passed） |
| ST-9.2 | `python specs/archive/AR009-rd-kv-allocation/ar009_bench.py` | 同字节 1.2/1.5MB RD 误差下降；低预算合规 | PASS |
| ST-9.3 | `pytest tests/test_budget_eviction.py -q --no-cov` | v3.10.0 语义（RD 关闭）回归 | PASS（10 passed） |
| ST-9.4 | `pytest tests/test_kv_plan.py tests/test_shadowkv.py tests/test_m1_vectorization.py -q --no-cov` | AR008/压缩/打包回归 | PASS |
| ST-9.5 | 全量 `pytest` | 无回归、覆盖率 >=75% | PASS（772 passed / 1 skipped / 82.93%） |
| ST-9.6 | 五件套 | black/isort/flake8/mypy/pytest 全绿 | PASS |
| ST-9.7 | 服务器协议审阅 | s2/s3/s4 命令、证据清单、回填规则清晰 | PASS（待服务器执行） |

## 实测（CPU，2026-10-19，5 seed 中位）

| 总预算 | baseline 误差 | RD 误差 | 变化 | 字节 |
|---|---|---|---|---|
| 0.9MB | 0.011157 | 0.011128 | -0.26% | 899904 / 899904 |
| 1.2MB | 0.009997 | 0.009603 | -3.94% | 1199424 / 1199424 |
| 1.5MB | 0.008910 | 0.008324 | -6.58% | 1499904 / 1499904 |

低于 int4 底线（RD+硬驱逐）：baseline 实际 892928B（超限 1.28-2.98x）；
RD+驱逐 299264/499712/699392B，全部合规；误差 0.7345/0.4736/0.2284。
分配耗时 RD 与 baseline 同量级（75-101ms / 8 层）。

## 正确性覆盖

- RD 与 legacy 排序；sink/复用保护；只降不升；预算合规；
- 硬驱逐：predicted/packed 合规、零字节、解包为零、不复活；
- `PrecisionPlan` codes=4、groups 排除、指纹敏感；
- 默认配置（RD on / 驱逐 off）与 v3.10.0 调用兼容。
