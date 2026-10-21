# [AR013] 设计说明：跨步复用与紧凑 KV 闭环执行

## 1. 总体方案

```
Orchestrator config["execution"]
        │ configure_execution / reset_execution_state
        ▼
PyTorchModelBackend
  ├─ kv_lifecycle: append | replace
  ├─ kv_execution:  dense  | packed
  ├─ StepReuseState (per-layer input/output/K/V + last attention)
  └─ forward_step:
        raw_cache = packed? unpack : dequant dense   (replace 时为空)
        model.forward_step(..., reuse_state)
        for layer:
          reused? keep previous entry (no repack)
          else    pack/quantize new K/V
        execution_stats
```

FlashInfer 的 plan/run 思路用于“存储表示与执行表示分离”：packed 是唯一驻留表示，执行时一次性
读取。DeepCache / Δ-DiT 的“跨步特征复用”由一个显式漂移门控实现，并且必须给出误差-跳过率曲线。

## 2. 数据结构

### 2.1 `shadowinfer/execution/step_reuse.py`

```python
@dataclass(frozen=True)
class StepReuseConfig:
    enabled: bool = False
    threshold: float = 0.1
    warmup_steps: int = 1
    def validate(self) -> None: ...

@dataclass
class LayerReuseEntry:
    hidden_in: Tensor      # 上次完整计算的 block 输入（推理时直接引用）
    hidden_out: Tensor     # 上次完整计算的 block 输出
    k: Tensor
    v: Tensor

class StepReuseState:
    entries: Dict[int, LayerReuseEntry]
    last_attention: Optional[Tensor]
    last_reused_layers: Set[int]
    last_computed_layers: Set[int]
    last_drifts: Dict[int, float]
    def reset(self) -> None
    def begin_step(self) -> None
    def drift(self, layer_id, hidden) -> Optional[float]
    def should_reuse(self, layer_id, hidden, step_id, cfg) -> Tuple[bool, Optional[float]]
    def record(self, layer_id, hidden_in, hidden_out, k, v) -> None
    def stats(self) -> Dict[str, Any]
```

- `should_reuse`：无缓存、形状不匹配或 warmup 内 → False；否则相对 L2 漂移 <= threshold。
- `record` 只在完整计算后调用；复用时不刷新缓存，因此漂移始终相对上次完整计算有界。
- `last_attention` 只保存最近一次计算的注意力矩阵（一个 `[B,H,S,S]`），避免 O(L·S²) 缓存。
- `begin_step` 清空本步的 reused/computed/drift 集合。

### 2.2 模型接口

`DiffusionTransformerBlock.forward(hidden, kv_cache=None, layer_id=0, reuse_state=None,
reuse_config=None, step_id=0)`：

```
if reuse_state and reuse_config:
    should, drift = state.should_reuse(layer_id, hidden, step_id, config)
    if should:
        hidden_out = cached.hidden_out + (hidden - cached.hidden_in)
        k, v = cached.k, cached.v
        attn = state.last_attention or empty
        state.last_reused_layers.add(layer_id)
        return hidden_out, (k, v), attn

# 原实现：LayerNorm + Q/K/V + attention + FFN
state.record(layer_id, block_input, hidden, k, v)
state.last_computed_layers.add(layer_id)
state.last_attention = attn_weights.detach()
```

`SimpleDiffusionLLM.forward_step(..., reuse_state=None, reuse_config=None)`：
- 默认 `None` 时与 v3.14.0 逐位一致。
- 每步入口调用 `reuse_state.begin_step()`，逐层传递 `layer_id/step_id`。
- 最后若干层被复用时，`attention_scores` 返回 `state.last_attention`（上一步最近计算的矩阵），
  stats 标记 `attention_reused=True`。
- 计算/复用集合写入 `state`，不改变返回三元组签名。

## 3. 后端执行

### 3.1 配置

```python
configure_execution(
    kv_lifecycle: str = "append",     # append | replace
    kv_execution: str = "dense",      # dense | packed
    packed_precision: Optional[str] = None,
    step_reuse_enabled: bool = False,
    step_reuse_threshold: float = 0.1,
    step_reuse_warmup_steps: int = 1,
    int4_group_size: int = 16,
) -> None
reset_execution_state() -> None
```

- 非法枚举/负阈值/非法精度立即 `ValueError`。
- 复用仅在 `kv_lifecycle="replace"` 时启用；append 时请求复用会 warning 并禁用。
- `reset_execution_state()` 清空 `_kv_cache`、`_packed_cache`、`_reuse_state` 与上一步统计。

### 3.2 `forward_step`

```
raw_cache = {}
if lifecycle == "append":
    raw_cache = {lid: unpack(entry) if packed else dequant(entry)}
reuse_state = self._reuse_state if step_reuse else None
logits, new_raw_kv, attn = model.forward_step(..., reuse_state=reuse_state,
                                              reuse_config=cfg)
reused = reuse_state.last_reused_layers
for lid:
    if lid in reused and lid in self._kv_cache:
        updated[lid] = self._kv_cache[lid]       # 同一对象，不 repack
        continue
    k, v = new_raw_kv[lid]
    if kv_execution == "packed":
        packed = PackedKVCache.pack(k, v, uniform_map(precision), group_size=...)
        entry = KVCacheEntry(None, None, "mixed", packed_kv=packed)
    else:
        entry = _quantize_kv(k, v, precision)
```

- packed 精度：`conservative=fp32`、`balanced=fp16`、`aggressive=int8`；显式
  `packed_precision` 覆盖（支持 int4）。
- `replace` 时模型不接收 past，K/V 长度恒为 S；packed 作为本步驻留表示，并在复用层跨步保留。
- `append` + packed 会打包累计 K/V，此时下一步 attention 从 packed 反量化读取 past（闭合回路）。
- 任何 pack 异常：该层回退 dense，`fallback=True` 并告警，不静默。

### 3.3 `execution_stats`

```python
{
  "kv_lifecycle": "replace",
  "kv_execution": "packed",
  "reused_layers": [..],
  "computed_layers": [..],
  "reused_ratio": 0.5,
  "packed_layers": 4,
  "skipped_repack": 4,
  "resident_bytes": 12345,
  "dense_bytes": 32768,
  "compression_ratio": 0.36,   # resident / dense
  "fallback": False,
  "attention_reused": True,
}
```

`resident_bytes` 来自 `packed_kv.resident_bytes()`；`dense_bytes` 为 K/V 的 FP32 等价字节。
返回 dict 携带该统计，Orchestrator 透传到 profiler。

## 4. Orchestrator 接线

- `initialize()` 在 `_maybe_enable_int8_ffn()` 后调用 `_maybe_configure_execution()`：
  - 读取 `config["execution"]`，兼容 `config["shadowkv"]["execution"]`；
  - 仅对实现 `configure_execution` 的后端调用；不支持的 backend 记录一次事件。
- `run_baseline` / `run_optimized` 开始时调用 `reset_execution_state()`，防止跨 run 共享。
- `_prepare_backend_inputs_from_result` 透传 `execution_stats`；profiler 的 custom metrics
  可记录 `reused_ratio` / `resident_bytes`。
- 现有 forward 异常告警路径保持不变；execution fallback 标记在统计中。

## 5. 测试设计

- UT-13.1 生命周期：replace 连续 3 步长度恒为 S；append 第 2 步 2S。
- UT-13.2 复用关闭：两次 forward logits 逐位一致，无状态、无复用层。
- UT-13.3 复用开启：受控小漂移、warmup、全部层复用、relL2 <=5%。
- UT-13.4 复用公式：手工设置 cached in/out，验证 `cached_out + (h - cached_in)` 经
  output_proj 后与 logits 逐位一致。
- UT-13.5 packed replace：entry 无 dense、resident<dense、logits 与 dense allclose。
- UT-13.6 packed append：第二步从 packed past 反量化，logits 与 dense allclose（闭合回路）。
- UT-13.7 int8 packed：resident/dense <=0.6、relL2 <=3%。
- UT-13.8 组合：跳过层保持同一 packed 对象、`packed_layers == len(computed_layers)`、
  `skipped_repack == len(reused_layers)`、全复用驻留不变。
- UT-13.9 配置校验、reset、Orchestrator 下发与每 run reset。

## 6. 基准与服务器

`ar013_bench.py`：
- 规模 `(L,H,D,HN,S)`：`(8,8,64,512,256)`、`(8,8,64,512,512)`、`(8,16,64,1024,256)`；
  3 seed × 3 repeats。
- 轨迹 `x_t = x0 + sigma_t * noise`，sigma 线性 0.9→0.02，10 步，固定噪声。
- 四组：full baseline（replace+dense）、reuse（threshold 0.1）、packed（int8）、
  combined（packed+reuse）。
- 实测：reuse 1.77-2.19x、relL2 0.94-0.96%、跳过 50%；packed 驻留 36%、relL2 0；
  combined 1.32-2.40x。

`s8_step_reuse_gpu.yaml`：7B 维度（32 层/32 头/128/4096/11008），replace + reuse(0.1) +
packed(fp16)，30 步。`server-experiment.md` 提供 5 组对照、质量/延迟/显存模板与结论口径。
