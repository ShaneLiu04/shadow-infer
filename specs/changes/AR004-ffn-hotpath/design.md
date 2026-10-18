# 1 AR概述

| 组件名称 | ShadowInfer / shadowinfer.ffn_optimizer（PackedFFNWeight + FFNOptimizerAgent 热路径与层状态） |
| --- | --- |
| AR系统流水号 | AR004 |
| AR描述 | 消除 FFN 支线热路径（O(channels) 指纹与每步统计）、向量化 unpack/量化、修复逐层状态共享（FFN-P1）与混合路径通道掩码缺陷。全部 CPU 可验证，golden 等价。 |

# 2 动态行为

## 交互时序图

```plantuml
@startuml
title AR004 FFN 热路径（单层 on_step）
participant ORC as Orchestrator
participant FA as FFNOptimizerAgent
participant PC as _LayerState(layer_id)
participant PW as PackedFFNWeight

ORC -> FA: on_step(step_config, inputs{layer_id})
FA -> FA: 权重缺失校验（ValueError）
FA -> PC: 取/建 层状态（权重、量化态、统计缓存）
FA -> FA: sensitivity>=0.7 → ffn_mode=full
alt channel_pruning 且 sensitivity<0.7
  FA -> FA: analyze_channel_importance（本层，惰性一次）
  FA -> FA: channel_mask 计算（:174）+ 写回层状态
end
FA -> FA: select_compute_path（:191）
alt reuse / incremental / sparse 但缺 previous
  FA -> FA: compute_path = full（安全回退）
end
alt 路径 ∈ {mixed_full, sparse, incremental}
  FA -> FA: quantize_mixed_precision（向量化，一次）+ 写回层状态
  FA -> PC: 写入 quantization_stats（量化时一次）
  FA -> PW: pack（quantize 内部；仅缓存 fingerprint）
end
alt 各计算分支内（mixed_full/sparse/incremental）
  FA -> PW: dequantize_for_matmul（首次经向量化 unpack）
  PW -> PW: 指纹缓存命中（先 is 后 ==）
  PW --> FA: dense
end
alt reuse
  FA -> FA: output = ffn_output_previous
  else incremental
    FA -> FA: output = prev + reconstruct
  else sparse
    FA -> FA: sparse_update
  else mixed_full
    FA -> FA: _mixed_compute（up 行掩码 / down 列掩码）
  else full
    FA -> FA: _full_compute
end
FA --> ORC: output + quantization_info（缓存统计副本）
@enduml
```

# 3 功能点分解

| 序号 | 功能点名称 | 功能点描述 | 对应 srs |
| --- | --- | --- | --- |
| 1 | 指纹 O(1) | `_do_pack` 时缓存精度计划指纹与清空反量化缓存；dequant 校验 O(1) | FR4.1 |
| 2 | 统计缓存 | 层级 quantization_info 聚合在量化时计算一次，on_step 只读副本 | FR4.2 |
| 3 | 向量化 unpack | 按精度分组批量反量化；保留 `_unpack_reference` golden | FR4.3 |
| 4 | 层状态隔离 | `_layer_states[layer_id]`：权重/量化态/重要性/mask/统计；on_init 清空 | FR4.4 |
| 5 | 掩码修复 | `_mixed_compute` down 投影列掩码；全 False 告警 | FR4.5 |
| 6 | 向量化量化 | `quantize_mixed_precision` 分组批量；保留参考实现 | FR4.6 |
| 7 | 文档与版本 | README v3.6.0 四检查点 + 版本号三处同步 + ROADMAP | FR4.7 |

# 4 实现设计

## 4.1 功能实现思路

**FR4.1 指纹 O(1)：**

```
__init__：self._plan_fingerprint = ()
_do_pack 末尾：
  self._plan_fingerprint = tuple((ch, precision) ...)   # 一次构建
  self._dequant_cache = None                            # AR002 语义保持
  self._dequant_cache_fingerprint = None
_precision_plan_fingerprint() -> self._plan_fingerprint  # O(1)，保留方法名兼容
缓存指纹与 _plan_fingerprint 为同一对象（先 is 后 ==），避免 O(N) 元组比较
dequantize_for_matmul：cache 存在且 (cache_fp is plan_fp or cache_fp == plan_fp) → 直接返回
```

正确性用值语义（is 快路径失败后 `==`）；重打包同时清空反量化缓存，与
`tests/test_m5_memoization.py:263` 的 `_dequant_cache is None` 断言兼容。

**FR4.2 统计缓存（层级唯一归属）：**

```
_LayerState.quantization_stats：量化/pack 时填充一次
on_step：quantization_info = dict(st.quantization_stats)（浅拷贝，零迭代）
内存累计（_total_memory_bytes_*）：仅层状态首次量化时累加一次（不随步重复）；
  每层各累加一次；on_init 清空层状态时同步重置累计为 0
quantization_info 键集合与数值口径不变（含 memory_savings_bytes）
```

**FR4.3 向量化 unpack：**

```
按 precision 分组 channel 索引 → 批量取切片（stack 成 [N, channel_len]）
fp32/fp16：整组 .float()
int8：整组 dequantize_int8（scale 广播）
int4：整组 unpack_int4 → reshape(N, num_groups, group) → dequantize_int4
最后 scatter 回 out（pack_dim=0 行赋值 / pack_dim=1 列赋值）
参考实现 `_unpack_reference` 保留（逐 channel，golden）
```

**FR4.4 层状态隔离：**

```
@dataclass _LayerState：weight_up, weight_down, channel_importance, channel_mask,
    active_channels, weight_up_quantized, weight_down_quantized, precision_scales,
    quantization_stats
self._layer_states: Dict[int, _LayerState]
on_step：st = self._layer_state(layer_id)（首建时缓存该层权重）
compute_channel_mask(..., importance=st.channel_importance)  # 按当前层重要性
五处调用点必须同步改造（逐点闭合，行为断言见 UT-6/UT-16）：
  :159-163 重要性预计算 → 读写 st.channel_importance
  :174 compute_channel_mask 调用 → 传 importance=st.channel_importance
  :209-212 量化入口 → 使用 st.channel_importance
  :219 quantize_mixed_precision 调用 → 使用 st.channel_importance
  :858-864 静态掩码分支 → 优先使用传入 importance
量化结果写回与读取同样归层（新增闭合点）：
  :214-219 量化结果写回 st.weight_up_quantized / st.weight_down_quantized / st.precision_scales
  :221-239 原实现（仅参考）；目标：量化时建 st.quantization_stats，on_step 仅 dict() 浅拷贝，零迭代
  :269-272 计算取 st.weight_up_quantized / st.weight_down_quantized
  :180-181 mask/active_channels 写入 st.channel_mask / st.active_channels
  :373 返回 active_channels 读取 st.active_channels
compute_channel_mask 语义：importance 不为 None 时仅用它；为 None 时无条件回退
  _layer_state(0).channel_importance（仅兼容直接调用；on_step 内恒传入本层 importance）
兼容 property：channel_importance / weight_up_quantized / weight_down_quantized /
    precision_scales 代理 _layer_state(0)（现有测试直接赋值仍可用）
__init__ 改造顺序：先建 self._layer_states = {}、self._warned = set()，再定义 property（_plan_fingerprint 属 PackedFFNWeight，在其 __init__ 初始化）；移除对应直接赋值字段
_weight_up / _weight_down 私有属性删除
on_init 顺序：先 clear() 层状态与重置内存累计，再 _precompute_weights(..., layer_id=0)
layer_id < 0 → ValueError
层状态数 > max(1, num_layers) → log_alert("warning") 一次，继续运行
num_layers 解析：依次取 is not None 的 model_config/config 值，均缺失或 <=0 时按 0 处理（阈值 max(1,0)）；layer_id 非 int → ValueError
```

**FR4.5 掩码修复：**

```
_linear_forward(mask_dim)：up 投影输出维度 = intermediate_dim；down = hidden_dim
  mask_dim=0 → dense_weight = dense_weight[mask]（up 行）
  mask_dim=1 → dense_weight = dense_weight[:, mask]（down 列）
  mask_dim ∉ {0,1} → ValueError
_mixed_compute / _full_compute：接收 warning_key；将提供的掩码传给各自的 up/down 计算
全 False：FFN 输出 zeros（形状 [..., hidden_dim]）+ 告警一次（键 (layer_id, "all_false_mask")）
掩码计算异常：channel_mask=None + 告警，并重置 st.channel_mask / st.active_channels
静态剪枝：ratio clamp 到 [0,1]；k = max(1, int(...))
```

**FR4.6 向量化量化（阈值参数化）：**

```
分组条件：imp >= high_precision_threshold → fp16；0.3 <= imp < high_precision_threshold → int8；否则 int4
fp16：整组 half().float()
int8/int4：per-channel scale = abs().max(dim=-1)/qmax（up 行 / down 列，向量化）
  q = round(w / scale).clamp(...)；dequant = q * scale
scales 字典逐 channel 填 (precision, (scale_up, scale_down))
全零通道：scale=0 → 保留原值（参考语义）
```

## 4.2 功能实现设计

### 4.2.1 流程图

```plantuml
@startuml
title FFNOptimizerAgent.on_step（AR004）
start
:layer_id 解析（<0 → ValueError）;
:缺 up/down 权重 → ValueError;
:st = _layer_state(layer_id)（首建：缓存该层权重）;
if (sensitivity >= 0.7?) then (yes)
  :ffn_mode = full（禁用剪枝）;
endif
if (channel_pruning_enabled 且 ratio>0 且 sensitivity < 0.7?) then (yes)
  if (st.channel_importance is None?) then (yes)
    :analyze_channel_importance（本层权重，:159-163）;
  endif
  :channel_mask = compute_channel_mask(importance=st.channel_importance)（:174）;
  :st.channel_mask / st.active_channels 写入（:180-181）;
endif
:select_compute_path（:191-198）;
if (reuse/incremental/sparse 但缺 previous?) then (yes)
  :compute_path = full（安全回退）;
endif
if (compute_path ∈ {mixed_full, sparse, incremental}?) then (yes)
  if (st.channel_importance is None?) then (yes)
    :analyze_channel_importance（本层，:209-212）;
  endif
  if (st.weight_up_quantized is None?) then (yes)
    :quantize_mixed_precision（向量化，:219）;
    :写回 st.*_quantized / st.precision_scales（:214-219）;
    :pack + 缓存 fingerprint/stats（内存累计一次）;
  endif
  :quantization_info = dict(st.quantization_stats)（零迭代，:221-239）;
endif
alt reuse
  :output = ffn_output_previous（直接返回，不重算）;
else incremental
  :output = ffn_output_previous + incremental_reconstruct（UT-6c）;
else sparse
  :output = sparse_update(...)（UT-6c）;
else mixed_full
  :_mixed_compute(..., warning_key=(layer_id, "all_false_mask"))（up 行掩码 / gelu / down 列掩码）;
else full
  :_full_compute(..., warning_key=(layer_id, "all_false_mask"))（up 行掩码 / down 列掩码）;
end
stop
@enduml
```

### 4.2.2 流程说明

1. 层状态首建时仅缓存该层权重；量化/重要性按需惰性初始化（时序与现状一致，但状态归属层）；
2. 统计缓存与指纹在量化/打包时一次性生成，热路径只做字典读取；
3. 掩码在 full/mixed 两条路径语义一致（up 行、down 列）；
4. 参考实现（`_unpack_reference`、`_quantize_mixed_precision_reference`）保留作 golden；
5. `on_init` 清空层状态（保留配置，AR002 M4 口径）。

### 4.2.3 边界条件矩阵

| 类别 | 场景 | 期望 |
|------|------|------|
| 正常 | 混合精度（4 档）、pack_dim 0/1 | golden 等价（≤1e-6） |
| 边界 | 重要性 = 0.7 / 0.3（阈值临界） | 与参考实现同侧（>= 语义） |
| 边界 | int4 channel_len 不能被 group_size 整除 | 单组（channel_len） |
| 边界 | num_layers = 0 | 告警阈值 max(1, num_layers)，不除零 |
| 边界 | layer_id 缺省 | 按 0 处理（既有行为） |
| 边界 | ratio 越界（-0.1/1.5） | clamp 到 0/1 |
| 异常 | mask_dim ∉ {0,1} | ValueError |
| 异常 | layer_id < 0 | ValueError |
| 异常 | 层状态数 > max(1, num_layers) | 告警一次，继续运行 |
| 异常 | on_step 缺少 up/down 权重 | ValueError（现状保留） |
| 异常 | 缺 channel_importance 且无法计算（RuntimeError 现状） | 沿用现有分支，不新增崩溃 |
| 边界 | sensitivity = 0.7 / 0.69 | 0.7 → 强制 full 且无掩码；0.69 → 保留原路径 |
| 异常 | compute_channel_mask 非法 method | importance 存在 → 静态回退；为空 → RuntimeError |
| 异常 | 全零权重通道（int8/int4） | scale=0 → 保留原值 |
| 异常 | 全 False 通道掩码 | 输出全 0 + 告警一次 |
| 异常 | 掩码计算抛异常（:182-189） | channel_mask=None + 告警，保持 select_compute_path 结果继续（mixed 仍走 mixed，无掩码） |
| 异常 | 空 `_data`（未打包即 unpack） | 返回全零 dense（现状） |
| 异常 | packed 与输入 device/dtype 不一致 | dequant 后 `.to(device)`（现状保留） |

## 4.3 接口描述

### 4.3.0 Agent 公共接口契约

| 接口 | 签名 | 返回 | 异常/边界 |
|------|------|------|-----------|
| `on_init` | `(model_config: Dict) -> None` | None | 清空 `_layer_states` 与内存累计；预载权重归 layer 0 |
| `_precompute_weights` | `(weight_up, weight_down, layer_id: int = 0) -> None` | None | 非 2D 权重按现状处理（不新增崩溃） |
| `on_step` | `(step_config: StepConfig, inputs: Dict) -> Dict` | FFN 输出 + 指标（键集合不变） | `layer_id < 0`/非 int → ValueError；缺 up/down 权重 → ValueError；调用 _mixed_compute/_full_compute 时传 warning_key=(layer_id, "all_false_mask") |
| `compute_channel_mask` | `(ffn_input, weight_up, method=None, importance=None) -> Tensor` | `[channels]` bool | importance=None 回退 layer 0；method 非法值按现状处理 |
| 告警一次 | `self._warned: Set[Tuple[int, str]]` | 键为 (layer_id, warning_kind) | 首次触发 log_alert("warning")，同键后续静默；on_init 清空 |
| layer_id 解析 | `inputs.get("layer_id", 0)`；非 int → ValueError；<0 → ValueError | int | 缺省按 0（既有行为） |
| num_layers 解析 | 依次取 is not None 的 model_config/config 值，均缺失/<=0 按 0（阈值 max(1,0)） | int | 不除零 |

### 4.3.0.1 `_LayerState` 字段（含类型）

```
weight_up: Optional[Tensor]
weight_down: Optional[Tensor]
channel_importance: Optional[Tensor]      # [channels]
channel_mask: Optional[Tensor]            # [channels] bool
active_channels: int = 0                  # 活跃通道数（未剪枝 = 0，保持现有返回语义）
weight_up_quantized: Optional[Union[Tensor, PackedFFNWeight]]
    # packed 模式：PackedFFNWeight；非 packed 回退：反量化后的 float dense
weight_down_quantized: Optional[Union[Tensor, PackedFFNWeight]]
precision_scales: Dict[int, Tuple[str, Optional[Tuple[float, float]]]]
    # 逐通道 (precision 名, (scale_up, scale_down) 或 None)
quantization_stats: Dict[str, Any]        # 键集合同现状（值可为 int/dict）
```

### 4.3.1 PackedFFNWeight（内部接口）

| 接口 | 签名 | 返回 | 异常/边界 |
|------|------|------|-----------|
| `_precision_plan_fingerprint` | `() -> tuple` | pack 时缓存的 (ch, precision) 元组（O(1)） | 未打包时为空元组 |
| `unpack` | `() -> Tensor` | 原始形状 float32 dense | 未知精度在 pack 时归一为 fp16；空 `_data` 返回全零 |
| `_unpack_reference` | `() -> Tensor` | 逐 channel golden 参考 | 同上 |
| `dequantize_for_matmul` | `() -> Tensor` | dense（缓存命中直接返回） | 指纹变化时重算，无异常外抛 |

### 4.3.2 FFNOptimizerAgent（内部接口）

| 接口 | 签名 | 返回 | 异常/边界 |
|------|------|------|-----------|
| `_precompute_weights` | `(weight_up, weight_down, layer_id: int = 0) -> None` | 写入该层状态（on_init 预载归 layer 0） | — |
| `_layer_states` | `Dict[int, _LayerState]` | 每层状态 | — |
| `_layer_state` | `(layer_id: int) -> _LayerState` | 惰性创建/取回；on_step 取得 st 后，仅当 st.weight_up is None 时写入本层权重（不覆盖 on_init 预载的 layer 0） | `layer_id < 0` → ValueError；layer_id >= max(1,num_layers) 或状态数超限 → 告警一次 |
| `_mixed_compute` | `(x, up, down, channel_mask=None, warning_key=None) -> Tensor` | FFN 输出 [..., hidden_dim] | 全 False 掩码 → 全 0 + warning_key 告警一次 |
| `_full_compute` | `(x, up, down, channel_mask=None, warning_key=None) -> Tensor` | FFN 输出 [..., hidden_dim] | 全 False 掩码 → 全 0 + warning_key 告警一次 |
| `_linear_forward` | `(x, weight, use_sparse_gemm=False, channel_mask=None, mask_dim=0) -> Tensor` | 线性输出（up: intermediate_dim；down: hidden_dim） | `mask_dim ∉ {0,1}` → ValueError |
| `compute_channel_mask` | 签名新增可选 `importance=None` | `[channels]` bool mask | 缺省回退 `_layer_state(0).channel_importance` |
| `quantize_mixed_precision` | 签名不变（`high_precision_threshold=0.7`） | `(up, down, scales)` | 全零通道保留原值 |
| `_quantize_mixed_precision_reference` | 同签名 | golden 参考 | 同上 |
| 兼容 property | `channel_importance` / `weight_up_quantized` / `weight_down_quantized` / `precision_scales` | 代理 `_layer_state(0)` 对应字段 | 赋值写入 layer 0（现有测试兼容） |

向后兼容：`_weight_up/_weight_down` 私有属性删除（无测试引用，已核实）；`channel_importance` 等改为 property（tests/test_ffn_channel_pruning.py 直接赋值仍可用）。

## 4.4 代码设计

| 文件 | 变更 |
|------|------|
| `shadowinfer/ffn_optimizer/packed_weight.py` | 指纹缓存、向量化 unpack、参考实现保留 |
| `shadowinfer/ffn_optimizer/ffn_optimizer_agent.py` | `_LayerState`、统计缓存、向量化量化、掩码修复 |
| `tests/test_ffn_hotpath.py`（新） | FR4.1-4.6 用例 |
| `tests/test_ffn_channel_pruning.py` | `channel_importance` property 兼容验证（回归） |
| `tests/test_m5_memoization.py` | 反量化缓存用例随指纹/重打包清空语义复核（回归验证） |
| `tests/test_ffn_optimizer.py` | `memory_savings_bytes` 口径回归（量化一次累计语义） |
| `README.md` / `ROADMAP.md` / `pyproject.toml` / `__init__.py` | v3.6.0 四检查点 |

# 5 重构设计

1. 全局单份权重/量化状态 → 逐层 `_LayerState`（FFN-P1）；
2. 每次 matmul 重建指纹 → pack 时一次性指纹；
3. 每步 O(channels) 统计 → 量化时缓存；
4. 逐 channel unpack/量化循环 → 分组批量向量化（参考实现保留 golden）。

# 6 测试设计

覆盖率目标：当次 pytest --cov 实测不低于项目门禁 75%（fail-under 75 不变），并同步 README 两处数字为实跑值；82.04% 仅作 v3.5.0 基线参考。

## 6.1 单元测试（UT）

| 用例 | 断言 | 对应 |
|------|------|------|
| UT-1 指纹 O(1) | 100 次 dequant 仅 1 次 unpack（计数）；1024 ch 下 fingerprint 耗时 ≤10µs（3 次中位） | FR4.1 |
| UT-2 指纹失效 | 重 pack 后 `_dequant_cache is None` 且 dequant 结果更新（golden） | FR4.1 |
| UT-3 统计缓存 | 连续 3 步 quantization_info 完全一致；计数和 = 通道数；压缩比正确 | FR4.2 |
| UT-3b 统计 O(1) | 首步完成量化后，连续 3 步 quantization_info 数值/键集合完全一致；强断言：热路径 spy 计数 `precision_scales` 迭代 = 0（首步允许构建）；计时 3 次中位 ≤20µs（诊断） | FR4.2 |
| UT-4 unpack golden | 混合精度（含 int4 非整除）vec vs `_unpack_reference` ≤1e-6（pack_dim 0/1） | FR4.3 |
| UT-5 unpack 性能 | 256/1024 两个规模点：3 次中位 vec ≤ 参考 50%（异常抖动允许 1 次重试，报告区间） | FR4.3 |
| UT-6 层隔离 | full 路径（ffn_mode=full）两层不同权重 → 第 2 层输出与直接计算 max|Delta| ≤1e-6；回到第 1 层与首次输出 ==；断言 st.channel_importance/weight_up 指向本层对象 | FR4.4 |
| UT-6b 层状态生命周期 | on_init 清空；`_layer_state` 超 num_layers 告警一次；`layer_id<0` ValueError | FR4.4 |
| UT-6c 分支隔离 | sparse / incremental 路径下两层状态互不污染 | FR4.4/G22 |
| UT-7 空态边界 | 空 `_data` unpack/参考实现均返回全零 dense；未知精度字符串回退 fp16；packed 与输入 device 不一致时 dequant 后 `.to(device)` 正常（CPU-only 环境该断言 skip；dtype 不做转换承诺） | FR4.3 |
| UT-8 掩码 golden | channel_mask + packed：`_mixed_compute(packed)` 与「packed.unpack() dense + 同掩码 full」≤1e-6；不抛错 | FR4.5 |
| UT-9 全 False 掩码与异常回退 | 全 False → 输出全 0 + 告警一次；compute_channel_mask 抛异常 → channel_mask=None/st.active_channels=0 + 告警，保持原路径（mixed 无掩码）不崩溃 | FR4.5 |
| UT-10 量化 golden | vec vs `_quantize_mixed_precision_reference`：`use_packed_weights=False` 下比较反量化后 dense 输出 ≤1e-6；packed 模式先 `.unpack()` 再比较；scales 逐位一致（含全零通道） | FR4.6 |
| UT-11 量化性能 | 256/1024 两个规模点：3 次中位 vec ≤ 参考 50%（允许 1 次重试） | FR4.6 |
| UT-12 端到端微基准（报告型，@pytest.mark.skip 或脚本化） | 同会话同批：优化路径 vs 参考路径（`_unpack_reference` + 参考量化）各 3 次，报告中位与区间；结构指标硬门禁（指纹 ≤10µs、vec ≤50% 参考）随 T007 落档，不设跨会话固定基线；报告型用例（UT-12）不设 pass/fail，不计门禁（结构硬门禁单独断言） | NFR |
| UT-13a 接口边界 | `mask_dim=2` → ValueError；缺权重 → ValueError；layer_id 非 int/-1 → ValueError；ratio=-0.1/1.5 → clamp 0/1 | G24 |
| UT-13b 兼容回退 | importance=None 回退 layer 0；`_precompute_weights` 非 2D 原样缓存；非法 method：importance 存在 → 静态回退，为空 → RuntimeError | G24 |
| UT-14 阈值临界 | 重要性 0.7/0.3 与参考实现同侧（>= 语义） | G22/G24 |
| UT-15 reuse 分支 | reuse 输出与上一步输出值相等（max|Delta|==0），且与直接重算结果不同（证明未重算）；尺寸正确 | G22 |
| UT-16 层隔离×剪枝 | 静态剪枝：两层不同 channel_importance，各层 mask/量化结果/active_channels 均归本层（行为断言；行号仅设计注记）；动态剪枝：按各层输入激活能量独立计算 | G21/G22 |
| UT-17 FR4.7 文档 | 版本号三处一致（3.6.0）；README 无 emoji；TOC 锚点有效；ROADMAP AR004 完成项已更新 | FR4.7 |
| UT-19 安全回退 | monkeypatch `select_compute_path` 返回 reuse/incremental/sparse 且 previous=None → 断言 compute_path=='full' | G22 |
| UT-18a 配置边界 | num_layers=0 → 阈值 max(1,0)，不除零；layer_id >= max(1,num_layers) 或状态数超限 → 告警一次（on_init 清空） | G22 |
| UT-18b 内存计账 | 内存累计每层一次，重复 on_step 不重复累加；on_init 归零 | FR4.2 |
| UT-21 full 全 False | full 路径 + 全 False 掩码 → FFN 输出全 0 + warning_key 告警一次（重复调用不重复告警） | FR4.5 |
| UT-20 sensitivity 阈值 | sensitivity=0.7 → compute_path=='full'、无掩码、active_ratio=1.0、result["quantization"] == {}；0.69 + ffn_mode=mixed → compute_path=='mixed_full' 且 quantization != {} | G22 |

## 6.2 接口测试

- on_step 输入/返回结构不变（键集合与类型断言）；
- 未提供 layer_id 时按 0 处理；`layer_id=-1` → ValueError；
- `use_packed_weights=False` 路径返回 dense 张量（不受指纹/层状态影响）；
- `mask_dim` 非法值 → ValueError；缺失 up/down 权重 → ValueError；
- 超 num_layers 的 layer_id：告警一次且不崩溃；
- 空 `_data` unpack 返回全零；多层 quantization_info 按层独立；
- `_precompute_weights` 非 2D 按现状；`compute_channel_mask` importance=None 回退 layer 0；非法 method（importance 存在→静态回退，为空→RuntimeError）。

## 6.3 业务场景测试

- 多层真实权重差异场景（4 层不同随机权重）：各层输出与直接计算一致；
- 混合路径 + 通道剪枝组合（修复 4.5 后新增覆盖）；
- mock 全链路（orchestrator 6 步）无回归。

## 6.4 异常场景测试

| 场景 | 期望 |
|------|------|
| 未知精度字符串 | 回退 fp16（现状） |
| 全零权重通道（int8/int4） | scale=0 → 保留原值，不产生 NaN |
| 掩码全 False | 输出 0 + 告警一次 |
| 层数超过 num_layers | 告警一次，继续运行 |
| 重 pack 后 dequant | 缓存清空并重算（单线程语义） |

# 7 决策记录（2026-10-18）

| 方案 | 思路 | 优点 | 缺点 | 结论 |
| --- | --- | --- | --- | --- |
| A 局部修补 | 仅缓存指纹 | 改动最小 | 统计与 unpack 热点仍在；层状态缺陷未修 | 弃 |
| B 热路径 + 层状态 | 指纹 O(1) + 统计缓存 + 向量化 unpack/量化 + `_LayerState` + 掩码修复 | 覆盖全部取证根因；golden 可证；为真实多层模型扫清正确性障碍 | 中等工作量；需保留参考实现 | **采用** |
| C 全面重写 FFN 引擎 | 重写为 kernel/编译路径 | 上限最高 | GPU 内核不在本轮范围；风险高 | 弃（AR005+ 视 GPU 结果） |

关键子决策：
1. 参考实现保留为 golden（`_unpack_reference` / `_quantize_mixed_precision_reference`）——延续 AR002 双路径纪律；
2. 指纹在 pack 时生成而非惰性缓存——不可能出现"每步重建"回归；
3. 层状态用 dataclass（显式字段）而非散落字典——便于类型检查与测试；
4. 掩码以 `mask_dim` 表达（0/1）而非特判 down——统一 full/mixed 语义。
