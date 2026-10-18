# AR003 T001 基线取证（baseline-profile.md）

| 字段 | 内容 |
|------|------|
| 取证日期 | 2026-10-18 |
| 执行设备 | CPU（Windows，torch 2.5.1+cu121 CPU-only 执行） |
| 性质 | 缺陷现状取证（非创新收益实验；收益验证按规则 1.1 于 T007/T008 执行） |
| 基线代码 | HEAD（AR002 归档后；69f666b 仅 README，代码与 AR002 T008 实测时一致） |

## 0. 基线锚点（冒烟）

- 全量测试：**630 passed / 1 skipped / 81.68%**（2026-10-18 实跑，238.37s）——与 AR002 T009 收录的 README v3.4.0 锚点一致，无回归。
- 端到端：**1.61x 中位**（1.585/1.594/1.616/1.639）引用 AR002 T008 实测（specs/archive/AR002-hotpath-metrics-drift/baseline-profile.md §8）；本取证未重跑（代码未变）。

## 1. E1/E1b attention 契约双重性与熵饱和（KV-P2）

代码事实：
- 真实后端 `SimpleDiffusionLLM` 返回 post-softmax 权重（`simple_diffusion_llm.py:83`）；
- mock 路径 `_simulate_step_tensors` 生成 randn logits（`orchestrator.py:1922`）；
- agent 无条件再 softmax（`shadowkv_agent.py:439` 参考路径 / `:482` 向量化路径）——同一入口两种语义。

实测（seed=20261017，B=1/H=8/S=128，异质注意力构造：偶数 query 行 ×6 锐化）：

| 路径 | imp mean | imp min | p05 | 精度分布 |
|------|---------|---------|------|---------|
| 真实后端契约（probs→再 softmax，现状） | **1.0000** | **1.0000** | 1.0000 | **fp32: 100%** |
| mock（logits→单次 softmax，现状） | 0.5642 | 0.0001 | 0.0223 | fp32 50.0% / int4 25.9% / int8 23.4% / fp16 0.7% |
| FR3.1 目标（probs 直接使用） | 0.5642 | 0.0001 | 0.0223 | 同上（与 mock 路径逐位一致） |

结论：真实后端路径下重要性**零动态范围**（min=max=1.0），精度分配完全失效（全 fp32，压缩收益归零）；FR3.1 单次归一化后恢复与 mock 路径完全一致的语义。随机均匀注意力下（非异质构造）两路径均高熵全 fp32——档位散开断言必须用异质构造（srs FR3.1 验收已按此口径）。

## 2. E2/E2b 重要性轴向反转（KV-P3）

实测（sink 构造：全部 query 行集中于 token 0，seed 同上）：

| 量 | 值 |
|----|----|
| token 0 接收注意力（真实重要性） | **0.9591**（次高 token 仅 0.000322，差 2978 倍） |
| token 0 现行重要性（行熵语义） | **0.0934** → int4 |
| 全序列精度分布 | **int4: 100%** |

结论：接收 95.9% 总注意力的 sink token 被判最低档（0.093，仅因位置加成略高于其余同塌缩 token）；现行信号与 KV token 的真实被关注度**零相关**（轴向错误），预算紧张时最先被摧毁的恰是最关键 token。修复方向（列向接收注意力）与 H2O/StreamingLLM 一致。

## 3. E3 无界 importance_history（KV-P9）

实测（seq=32/heads=8，连续 20 步 on_step）：
- 20 步后列表长度 20、累计 20480 B（每步 append 256 元素 ×4B）——**线性增长**；
- 投影（seq=128/heads=32/1000 步）：**15.62 MB/层**，多层数十倍放大；
- 对照 IRAP 状态：常数 128×32×4B = **16 KB/层**（O(1)，与步数无关）。

## 4. E4 预算/驱逐死代码（KV-P4/P8/P10）

实测（memory_budget_mb=0.05，memory_pressure=0.95 显式传入，packed cache 开启）：

| 检查项 | 结果 |
|--------|------|
| 存储后实际占用 | **0.4998 MB（超预算 10 倍）** |
| `token_head_meta` 条目数 | **0**（`set_token_head_meta` 生产路径零调用） |
| 驱逐标记（importance==-1） | 无 |
| `memory_pressure` 输入 | 已由 orchestrator 传入（`orchestrator.py:1361`）但 agent 侧无任何读取——纯死输入 |
| `_enforce_budget` | 因 meta 恒空在 `:119` 提前返回，从未执行过驱逐 |
| 字节估算缺陷 | `kv_cache_manager.py:126` 以 `batch*head_dim`≈1B/值 估算（fp32 实际 8B/值，**8x 低估**） |

## 5. 参数默认值校准依据（design §4.1 落地值）

| 参数 | 默认 | 依据 |
|------|------|------|
| `irap_decay` | 0.9 | 标准 EMA；有效记忆视野 1/(1-γ)=10 步（近期 10 步覆盖 65% 权重），兼顾跨步稳定与响应速度 |
| `sink_tokens` | 4 | StreamingLLM 惯例值；现行 pos_factor 首 5 位加成口径的收敛替代 |
| `entropy_weight` | 0.3 | rank 主信号占 0.7 为决策主力，熵辅助提供「关注广度」信息；0.3 保证辅助不主导 |
| `eviction_pressure_threshold` | 0.9 | 10% 迟滞带（触发后压至 ≤0.9·预算），避免抖动 |
| 层预算 | budget//num_layers | 全局预算均摊；配合触底告警与 sink 保护 |

## 6. 复现命令

```powershell
python "C:\Users\L30086~1\AppData\Local\Temp\opencode\t001_forensics.py"    # E1/E3/E4
python "C:\Users\L30086~1\AppData\Local\Temp\opencode\t001_forensics2.py"   # E1b/E2b（异质/sink 构造）
python -m pytest -q                                                          # 基线冒烟（§0）
```

脚本内置 seed=20261017，结果确定性可复现。
