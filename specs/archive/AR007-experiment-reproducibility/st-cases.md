# [AR007] ST 用例与结果

| 用例 | 步骤 | 期望 | 结果 |
|------|------|------|------|
| ST-7.1 | `python scripts/run_experiment.py --profile s1 --steps 6 --repeats 3 --device cpu --out results/ar007_s1` | 生成 result.json/summary.md，6 步 × 3 次 | PASS |
| ST-7.2 | 查看 `results/ar007_s1/run_*/profile_*.json` | 三份 profiler 产物均在各 run 目录，无全局 outputs 泄漏 | PASS |
| ST-7.3 | `--dry-run` | 仅输出 plan（含 config_digest），不创建输出目录 | PASS |
| ST-7.4 | `--backend mock` | 非零退出/抛错，提示必须真实后端 | PASS |
| ST-7.5 | 相同 seed 两次重复 | baseline KV 层内存结构一致；optimized 执行步数=请求步数 | PASS |
| ST-7.6 | `--model-config configs/experiments/model_7b_longctx.yaml`（dry-run） | 嵌套模板展平，无结构错误 | PASS |
| ST-7.7 | 五件套 | black/isort/flake8/mypy/pytest 全绿 | PASS |

## 实测数据（s1，2026-10-19）

- 设备：Windows / CPU（torch 2.5.1+cu121，CUDA 不可用回退）
- 配置摘要：`00a45483…`（s1 内置 + 6 步）
- baseline e2e：41.89 ms（39.70 / 46.91）
- optimized e2e：1029.81 ms（1016.80 / 1060.44）
- speedup：0.041x（0.039 / 0.044）
- KV：1.75 MB → 1.65 MB（压缩比 1.061x）
- 原始数据：`results/ar007_s1/result.json`（本地运行产物，不入库）
