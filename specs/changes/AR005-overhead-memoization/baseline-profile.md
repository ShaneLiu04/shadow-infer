# AR005 基线取证与优化对照（baseline-profile.md）

| 字段 | 内容 |
|------|------|
| 取证日期 | 2026-10-19 |
| 执行设备 | CPU（Windows，torch 2.5.1 CPU-only；CUDA 扩展加载失败触发 CPU fallback） |
| 性质 | 常数开销取证（微基准 + 6 步管线 cProfile；非大型模型实验） |
| 复现脚本 | `specs/changes/AR005-overhead-memoization/ar005_bench.py`（seed=42） |

## 1. 管线 cProfile（优化前）

配置：真实 7B 形状（32 层 / 32 heads / head_dim 128 / hidden 4096 / intermediate 11008），6 步，CPU。

| 指标 | 实测 |
|------|------|
| 总耗时（cProfile 内） | 27.63 s |
| `_load_cuda_extensions` 调用次数 | 3816 |
| 该函数累计耗时 | 12.38 s（44.8%） |
| 失败重试单次 | 约 2.8 ms（`cpp_extension.load` 每次重试） |
| ShadowKV 每步延迟 | 约 289.9 ms（优化前） |

根因：`_cuda_ops` 仅在成功时缓存；本机加载失败后每调用重试。

## 2. INT4 打包微基准（优化前）

`PackedKVCache`，batch=1 / heads=8 / seq=128 / head_dim=128，group_size=16，3 次中位：

| 实现 | 耗时 |
|------|------|
| AR002 向量化量化 + 逐条 `_pack_int4` | 72.71 ms |
| `_do_pack_reference`（逐条全流程） | 189.17 ms |
| 比值 | 0.384x |

## 3. 优化后同进程对照

| 指标 | 优化路径 | 参考/旧路径 | 比值 |
|------|----------|-------------|------|
| loader（200 次调用） | 64.2 µs/次 | 2866.1 µs/次（逐次重试） | 0.022x |
| int4 pack（128×8×128） | 3.65 ms | 232.83 ms | 0.016x |
| 字节布局 | `torch.equal` + `byte_offsets` 一致 | — | golden |
| 管线 smoke（2 步，小模型） | 通过 | — | 键集合不变 |


## 4. 目标（验收硬门禁）

| 指标 | 目标 |
|------|------|
| loader 记忆化 | <= 5% 旧逐次重试耗时（同进程 200 次对照） |
| int4 pack | <= 参考 30%（同进程 3 次中位） |
| 正确性 | 字节张量 `torch.equal`、`byte_offsets`/scales/memory_bytes 一致 |
| 兼容性 | 公开 API 与存储布局不变 |

## 5. 服务器复现（可选 GPU 确认）

```bash
python specs/archive/AR005-overhead-memoization/ar005_bench.py
```

预期输出：`loader: ... ratio <= 0.05`、`pack: ... ratio <= 0.30`、`golden: ... identical`。
GPU 环境：安装 CUDA 扩展后 k/v 字节布局与 CPU 必一致（由 golden 测试约束），打包耗时比值应仍显著小于 1。
