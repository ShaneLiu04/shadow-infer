# AR006 基线取证与优化对照（baseline-profile.md）

| 字段 | 内容 |
|------|------|
| 取证日期 | 2026-10-19 |
| 执行设备 | CPU（Windows，torch 2.5.1 CPU-only） |
| 性质 | 冷启动打包开销取证（微基准 + 4 步小模型 cProfile；非大型实验） |
| 复现脚本 | `specs/archive/AR006-ffn-pack-vectorization/ar006_bench.py`（seed=42） |

## 1. 打包微基准（优化前，逐通道）

3 次中位，混合精度 fp16/int8/int4（1:1:2），group_size=64：

| 规模 | 耗时 |
|------|------|
| 256×256 | 31.41 ms |
| 1024×256 | 116.56 ms |

## 2. 管线内占比（AR005 后，4 步小模型）

`PackedFFNWeight._do_pack` 自耗时 0.071 s（8192 次 `select`），为 AR005 修复后最大剩余项。

## 3. 优化后同进程对照

| 规模 | 向量化 | 参考 | 比值 |
|------|--------|------|------|
| 256×256 | 2.59 ms | 31.41 ms | 0.082x |
| 1024×256 | 8.41 ms | 116.56 ms | 0.072x |

逐位一致性（脚本 `golden_check`）：`_data` 每通道键集合、张量 `torch.equal`、dtype/shape、`memory_bytes`、`_plan_fingerprint` 全部一致（含 pack_dim 0/1、奇数长度 30/35、int4 单组自适应）。

## 4. 目标（验收硬门禁）

| 指标 | 目标 |
|------|------|
| pack 性能（1024×256） | <= 参考 50%（3 次中位） |
| `_data`/字节/内存/指纹 | 与参考完全一致 |
| unpack 等价 | vec pack 后 `unpack` vs `_unpack_reference` <= 1e-6 |
| 兼容性 | API/存储布局不变 |

## 5. 服务器复现

```bash
python specs/archive/AR006-ffn-pack-vectorization/ar006_bench.py
```

预期：`pack 1024x256 ... ratio <= 0.50`；`golden: ... identical`。
7B 形状（11008×4096）可按脚本参数扩展复跑；该成本为冷启动一次性开销，端到端影响取决于层数与重复打包频率。
