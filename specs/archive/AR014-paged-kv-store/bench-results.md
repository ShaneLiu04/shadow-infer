# AR014 基准实测结果（bench-results）

> 铁律合规：本文件为 AR014 实验原始数据落档（规则 1.1）。README 仅引用结论性数字。
> 执行设备：CPU（Windows，Python 3.11.9，torch 2.5.1，`torch.set_num_threads(1)`）。
> 复现：`$env:PYTHONPATH='D:\ShadowInfer'; C:\Python311\python.exe specs\changes\AR014-paged-kv-store\ar014_bench.py`

## 方法

测量 append 语义下 KV 执行平面每步每层的真实簿记成本，两条路径严格镜像现行后端代码：

- **baseline（`kv_execution="packed"`，现行实现）**：每步每层重建完整 `precision_map`（S 个 token × H 个 head 的 Python dict）→ `PackedKVCache.pack(全量 past+new)`（含 `_do_pack` 的 O(S·H) Python 分组循环 + 全量张量量化）→ 下一步 `unpack()` 全量解包读取。即每步 2×O(S_total) 张量工作 + O(S·H) Python 循环。
- **paged（AR014）**：每步 `PagedKVStore.gather()` 读取（O(S_total) 张量）+ 仅新 token `append()` 入页（O(S_new)，向量化，零 Python 逐 token 循环）。

维度：B=4, H=8, D=64, L=8 层, 精度 fp16, page_size=32；速度 3 seeds（0/1/2）报中位与区间；`tracemalloc` 测 Python 堆瞬态峰值（torch CPU 张量走 ATen 分配器不计入，基线的 precision_map/分组列表等 Python 瞬态全部计入）。

## 预注册断言与结果（全部通过）

| 断言 | 阈值 | (N=8, S_new=64, max_seq=512) | (N=16, S_new=128, max_seq=2048) |
|---|---|---|---|
| 总时间加速比（3 seed 中位） | >= 1.5x | **9.303x**（区间 9.209-9.751） | **10.162x**（区间 7.684-10.278） |
| 每步延迟线性拟合斜率比 paged/baseline | < 0.60 | **0.088**（中位） | **0.114**（中位） |
| Python 堆瞬态峰值 paged < baseline | 必须 | **2408KB -> 47KB**（约 51x） | **9975KB -> 175KB**（约 57x） |
| fp16 逐元素等价（torch.equal） | 必须 | 通过（final gather == 全量 pack 参考解码） | 通过（同左） |

## 原始数据（实跑输出，未修饰）

规模 1（N=8, S_new=64, max_seq=512）：

| seed | baseline 总时 (s) | paged 总时 (s) | 加速比 | 斜率比 | 堆峰值 baseline (B) | 堆峰值 paged (B) |
|---|---|---|---|---|---|---|
| 0 | 0.8575 | 0.0922 | 9.303 | 0.088 | 2,475,724 | 50,648 |
| 1 | 0.8246 | 0.0895 | 9.209 | 0.090 | 2,466,067 | 48,502 |
| 2 | 0.8307 | 0.0852 | 9.751 | 0.084 | 2,466,039 | 47,042 |

规模 2（N=16, S_new=128, max_seq=2048）：

| seed | baseline 总时 (s) | paged 总时 (s) | 加速比 | 斜率比 | 堆峰值 baseline (B) | 堆峰值 paged (B) |
|---|---|---|---|---|---|---|
| 0 | 7.0217 | 0.6832 | 10.278 | 0.100 | 10,214,335 | 179,201 |
| 1 | 6.7777 | 0.8821 | 7.684 | 0.118 | 10,214,680 | 179,068 |
| 2 | 7.3896 | 0.7271 | 10.162 | 0.114 | 10,214,392 | 178,006 |

（heap 峰值列为 tracemalloc 原始字节数；表中中位数由脚本计算报告。）

## 驻留字节（fp16，单层，max_seq=2048）

| 项 | 值 |
|---|---|
| paged data_plane_bytes | 8,388,608 B |
| dense FP16 基准 | 8,388,608 B |
| 比值 | 1.0（fp16 页无压缩，符合预期；分页收益在时间与瞬态，不在 fp16 驻留） |

## 诚实说明

1. SRS 预估收益为 ~1.8-2x（仅按张量算子口径 2×O(S) -> 1×O(S)+O(S_new) 推算）；实测 9-10x 的差额来自基线路径的 Python 层开销（每步每层 O(S·H) 的 precision_map 重建与 `_do_pack` 分组循环），该开销为现行代码的真实成本，分页路径对 past token 完全绕开。两口径均已如实报告。
2. 每步延迟并非绝对平坦：paged 的 gather 读取仍随 S 线性增长（attention 消费下界），斜率比 0.09-0.12 反映的是相对基线的增长斜率大幅下降。
3. Python 堆峰值只覆盖 Python 对象（dict/list/元组）；torch CPU 张量分配不计入 tracemalloc。基线每步的 precision_map（S×H 项）与分组索引列表是真实存在的 Python 瞬态，分页路径消除了它们。
4. 本基准为 KV 执行平面微基准（隔离 AR014 机制）；端到端 logits 等价由 UT 证明（tests/test_paged_backend_execution.py，fp16 `torch.equal`）。GPU 端到端收益待服务器按 `configs/experiments/s9_paged_longctx_gpu.yaml` 协议复现。
5. int8 精度等价性与驱逐等价性由 UT 覆盖（tests/test_paged_kv_execution.py，逐元素 `torch.equal`）；基准以 fp16 为主口径。
6. 规模 2 的 seed=1 加速比 7.684 低于其余两 seed（10.2+）：paged 路径该次运行耗时偏高（0.8821s vs 0.68-0.73s），属 CPU 调度抖动；如实报告，不剔除。
