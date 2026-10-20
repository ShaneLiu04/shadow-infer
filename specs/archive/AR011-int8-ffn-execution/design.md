# [AR011] 设计说明：可执行 INT8 FFN 低比特运行路径

## 1. 总体方案

```
FP32 Linear ──quantize per output channel──> INT8Linear
                                              ├── W_q int8[K,N]
                                              ├── scale_w float32[N]
                                              └── bias float32[N] (可选)
forward(x):
  s_x = |x|.amax(-1)/127
  x_q = round(x/s_x)
  y_i32 = _int_mm(x_q.view(-1,K), W_q)
  y = y_i32 * (s_x*s_w) + bias
```

FlashInfer 的 Plan/Run 思想落在 FFN：量化计划（权重/scale/layout）一次构建、不可变；
运行时不再反量化整块权重，而是直接执行 INT8 GEMM。PyramidInfer 的启示是“必须真正少算/少存”，
本 AR 以真实低比特执行替代原有 packed 记账路径。

## 2. 数据结构与 API

### 2.1 `shadowinfer/ffn_optimizer/int8_linear.py`

```python
class INT8Linear(nn.Module):
    def __init__(self, qweight, weight_scale, bias=None, in_features=None, out_features=None)
    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "INT8Linear"
    def forward(self, x: torch.Tensor) -> torch.Tensor
    def memory_bytes(self) -> int
    def dense_equivalent_bytes(self) -> int
    @property
    def using_int_mm(self) -> bool
```

- `qweight`：`int8[K, N]`（`linear.weight.T.contiguous()` 的量化结果）。
- `weight_scale`：`float32[N]`，`max|w_row|/127`，`clamp_min(eps)`。
- bias：保留 FP32（若原层有 bias）。
- 输入展平：`flat = x.reshape(-1, K)`；输出 `y.reshape(*x.shape[:-1], N)`。
- 激活 scale：`s_x = flat.abs().amax(dim=1)/127`，`clamp_min(eps)`；`x_q = round(flat/s_x[:,None]).clamp(-127,127).to(int8)`。
- INT8 GEMM：`torch._int_mm(x_q, qweight)`；结果 `int32`。
- 反量化：`y = y_i32.to(float32) * (s_x[:,None] * weight_scale[None,:]) + bias`；最后 `.to(x.dtype)`。
- 回退：捕获 `RuntimeError/AttributeError`（`_int_mm` 不存在/形状不支持），设置 `_int_mm_available=False`，
  使用 `F.linear(flat, (qweight.float() * weight_scale[None,:]).T, bias)`；同一模块后续直接走回退，避免反复异常。
- `memory_bytes = qweight.numel() + weight_scale.numel()*4 + (bias.numel()*4 if bias)`
- `dense_equivalent_bytes = K*N*4 + (N*4 if bias)`

### 2.2 `SimpleDiffusionLLM`

```python
def enable_int8_ffn(self, keep_fp32_reference: bool = False) -> Dict[str, Any]
def restore_fp32_ffn(self) -> None
```

- 遍历 `self.blocks`，对每个 `block.up_proj` / `block.down_proj` 调用 `INT8Linear.from_linear`。
- 默认 `keep_fp32_reference=False`：替换后删除原模块引用，FP32 权重可被回收。
- `keep_fp32_reference=True`：保存 `(block_index, name, module)` 列表供 golden / 恢复。
- 幂等：已是 `INT8Linear` 时跳过并计入统计。
- 返回：`{"layers": L, "modules": 2L, "weight_bytes": ..., "dense_bytes": ..., "fallback": bool}`。

### 2.3 后端

`ModelBackend.enable_int8_ffn(keep_fp32_reference=False) -> Dict[str, Any]` 默认返回
`{"supported": False}`。`PyTorchModelBackend` 覆写：

- 模型未加载时抛 `RuntimeError`；
- 调用 `self._model.enable_int8_ffn(...)`；
- 缓存 `self._int8_ffn_stats`，`get_int8_ffn_stats()` 返回副本。

HF 后端使用同一 `quantize_ffn_modules` 按常见 FFN 命名后缀
（`mlp.up_proj/down_proj/gate_proj`、`mlp.c_fc/c_proj`）替换线性层；无匹配时返回
`supported=False`。`keep_fp32_reference=True` 时保存引用并可通过 `restore_fp32_ffn()`
恢复；默认不保留，FP32 权重可回收。

### 2.4 Orchestrator

- 读取配置：`ffn_cfg = config.get("ffn", config.get("ffn_optimizer", {}))`；
  兼容顶层 `int8_ffn_enabled`。
- `initialize()` 末尾：若启用，调用 `backend.enable_int8_ffn()` 并保存 `self._int8_ffn_stats`；
  记录事件 `int8_ffn_enabled`。
- `_run_ffn`：若 `self._int8_ffn_stats` 存在，返回真实执行统计（`compute_path="int8_exec"`，
  `weight_bytes`/`dense_bytes`），不调用 `FFNOptimizerAgent.on_step`，避免重复打包。
- 未启用时保持既有行为（逐位不变）。

## 3. 量化与数值语义

- 权重量化：`scale_w = |W|.amax(dim=0)/127`（`dim=0` 对应输入维，得到每输出通道 scale）；
  `W_q = round(W / scale_w[None,:]).clamp(-127,127).to(int8)`。
- 激活量化：每 token 一行独立 scale，动态范围自适应。
- 反量化严格乘回 `s_x * s_w`，bias 最后加。
- 误差来源：权重/激活独立量化；对称 INT8 理论相对误差约 0.4%-2%，实测阈值 3%。

## 4. 回退与错误处理

| 场景 | 行为 |
|------|------|
| `_int_mm` 不存在 | 一次 warning，float32 整数乘积回退（与 INT8 整数语义一致） |
| `_int_mm` 抛 RuntimeError | 同上，并缓存不可用状态 |
| 输入最后一维 != in_features | `ValueError` |
| 权重非 2D / scale 长度不匹配 | `ValueError` |
| `restore_fp32_ffn` 无参考 | `RuntimeError`，提示启用 `keep_fp32_reference=True` |
| 重复 enable | 幂等，返回同一统计 |

## 5. 影响范围

- 新增：`shadowinfer/ffn_optimizer/int8_linear.py`
- 修改：`shadowinfer/models/simple_diffusion_llm.py`、`shadowinfer/core/model_backend.py`、
  `shadowinfer/core/backends/pytorch_backend.py`、`shadowinfer/core/backends/huggingface_backend.py`、
  `shadowinfer/orchestrator.py`
- 测试：`tests/test_int8_ffn.py`（新增）
- 资产：`specs/archive/AR011-int8-ffn-execution/`、`configs/experiments/s6_int8_ffn_gpu.yaml`

## 6. 测试设计

- UT-11.1 `INT8Linear` 与 FP32 相对误差 <=3%，输出 dtype/shape 正确。
- UT-11.2 批维 `[B,S,K]` 与 2D 展平逐位一致。
- UT-11.3 权重/激活 scale 极端值（全零、常量、超大值）无 NaN。
- UT-11.4 `memory_bytes` 与理论值一致，约 FP32 的 1/4。
- UT-11.5 回退路径（注入 `_int_mm` 异常）结果与真实 INT8 一致，且只告警一次。
- UT-11.6 模型替换：每层 up/down 为 INT8Linear；幂等；`keep_fp32_reference=False` 后无 FP32 引用。
- UT-11.7 保留参考时同输入 logits 相对误差 <=3%，top-1 一致率 >=95%（真实模型，3 seed）。
- UT-11.8 后端 `enable_int8_ffn` 与 orchestrator 接线；未启用时逐位不变。
- UT-11.9 配置 `ffn:` 段能驱动 FFN Agent 键（修复键名不一致）。
- UT-11.10 小尺寸/7B 形状基准：脚本断言 INT8 内存与加速比，原始数据落档。

## 7. 基准设计

### 7.1 微基准（CPU）

- 形状：`(M,K,N) ∈ {(1,4096,11008),(16,4096,11008),(64,4096,11008),(128,4096,11008),(64,11008,4096)}`。
- 对比：FP32 `F.linear` vs `INT8Linear.forward`；3 次重复取中位[min-max]。
- 断言：INT8 权重字节 == 理论值；7B 上投影（M>=16）加速比 >=3x。
- 额外记录小尺寸 `(128,128,512),(512,256,1024)` 的交叉点。
- 中等模型（hidden=1024/intermediate=4096/seq=256）端到端对比，断言不慢于 FP32 且 logits 误差 <=3%。

### 7.2 小模型精度

- `SimpleDiffusionLLM` 4 层/8 头/dim64/seq128，3 seed；同一输入 FP32 vs INT8 logits。
- 指标：相对 L2 误差、max 绝对误差、top-1 一致率。
- 端到端延迟：如实报告，不宣称小模型加速。

### 7.3 服务器（7B / GPU）

见 `server-experiment.md`：预训练模型、s6 档位、3 次重复、logits/困惑度、峰值显存、`_int_mm`/oneDNN 后端信息。
