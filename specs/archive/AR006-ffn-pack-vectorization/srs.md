# AR006 SRS — FFN 打包冷启动向量化

## 1. 背景与目标

AR004 已向量化 `PackedFFNWeight.unpack`，但**打包路径 `_do_pack` 仍逐通道 Python 循环**：每个通道执行 `select`/`reshape`/`contiguous`/量化，1024 通道约 116.6 ms（CPU，同进程 3 次中位）。AR005 后的 4 步小模型管线 cProfile 中，`_do_pack` 是剩余最大自耗时（0.071 s，8192 次 `select`）。

在真实 7B 形状（intermediate_dim=11008）下，该冷启动成本按层线性放大（4 层约 5 s），直接拖慢首次推理与实验迭代。

目标：按精度分组批量打包，保持字节/scale/统计/内存口径**逐位一致**，保留逐通道参考实现作为 golden。

## 2. 范围

**In Scope**
- `PackedFFNWeight._do_pack` 分组向量化（fp32/fp16/int8/int4，pack_dim 0/1）。
- 新增 `_do_pack_reference`（旧逐通道实现，golden）。
- 单元/性能/接口测试与复现脚本；README/版本/ROADMAP。

**Out of Scope**
- 不改变存储字典结构、字节布局、精度计划语义、公开 API。
- 不做大模型实验（服务器协议另行落档）。

## 3. 功能需求

### 3.1 FR6.1 分组向量化打包

**描述：** 按 precision 分组，批量 gather → 批量量化/半精度转换 → 逐通道写回存储；int4 按行批量打包字节（行内补零）。

**触发条件：** `PackedFFNWeight.pack(...)` / `_do_pack(...)`。

**期望行为：**
- `_data` 中每个通道的键集合、张量形状、dtype、数值与参考实现完全一致；
- `memory_bytes()` / `_plan_fingerprint` / `_memory_bytes` 与参考一致；
- `unpack()` 与 `_unpack_reference()` 结果一致（<= 1e-6）；
- pack_dim=0（行）与 pack_dim=1（列）均支持；
- int4 通道长度非 group_size 整除时沿用单组自适应；
- 未知精度回退 fp16（现状）。

**异常处理：** 空权重（0 通道）返回空 `_data` 与 0 字节；量化 scale=0 的通道保持原值（现状）。

**验收标准：**
- Given 混合精度权重（含奇数通道长度、pack_dim 0/1、batch 无关），When `_do_pack` vs `_do_pack_reference`，Then `_data` 逐键 `torch.equal`、dtype/shape 一致、`memory_bytes` 一致；
- Given 1024 通道，When 3 次中位对比，Then 向量化 pack <= 参考 50%（同进程）。

### 3.2 FR6.2 回归与可观测性

- 全量测试通过、覆盖率不低于当次基线；
- 复现脚本输出 pack 同批比值与 golden 结果。

### 3.3 FR6.3 文档与版本

- README v3.8.0 四检查点；版本三处同步；ROADMAP 勾选。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | FFN pack（1024×256） | <= 参考 50%（同进程 3 次中位） |
| 正确性 | `_data` / 字节 / 内存 | 与参考完全一致 |
| 兼容性 | API/布局 | 不变 |
| 质量 | 五件套 | 全绿；覆盖率 >= 75% |
| 环境 | 验证 | CPU-only |

## 5. 约束与假设

- 参考实现保留，禁止删除。
- 通道长度可能为奇数（int4 打包需行内 padding）。
- 仅单线程打包语义；并发打包仍不保证。

## 6. 术语

- 分组向量化：按 precision 把通道 gather 成 2D 批，批量运算后 scatter 回逐通道存储。
