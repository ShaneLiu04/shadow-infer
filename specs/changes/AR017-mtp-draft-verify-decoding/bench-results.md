# AR017 基准结果（FR17.4）

- 设备：CPU（本机 Windows，单线程 `torch.set_num_threads(1)`），Python 3.11.9，torch 2.5.1
- 口径：真实 SimpleDiffusionLLM 前向，真实随机 prompt（generator seed 2026+seed），无 mock
- 脚本：`ar017_bench.py`（本目录）；复现：`$env:PYTHONPATH='D:\ShadowInfer'; python specs/changes/AR017-mtp-draft-verify-decoding/ar017_bench.py`
- 纪律：沿 AR015/016 惯例——校准探针先行（不入库断言）、实测固化注册断言、跑前登记、全量实跑、逐 seed 落档

## 1. 校准探针（跑前，单 seed，未注册）

探针脚本为临时文件（`ar017_probe1.py` / `ar017_probe2.py`，非注册资产，不随 AR 交付）。

### 1.1 probe1：两臂轮数与接受率（threshold=0.9, W=32, seed=0）

| 臂 | G | 每块轮数 | forced 块 | acceptance | 结论 |
|---|---|---|---|---|---|
| mask-init paged | 192 | [8]x6 | 6/6 | 不适用 | 逐轮 top-1 兜底，8 轮解 8/32 位后强制收尾 |
| draft-verify paged (max_rounds=8) | 192 | 8（全 forced） | 6/6 | 0.039 | 纯 Jacobi 不定点 |
| draft-verify paged (max_rounds=16) | 192 | 16（全 forced） | 6/6 | 0.086 | cap 提高仅略升接受率，仍不定点 |
| mask-init paged | 448 | 8/块 | 14/14 | 不适用 | 同型 |
| draft-verify paged (max_rounds=8/16) | 448 | 8/16（全 forced） | 14/14 | 0.038 / 0.077 | 同型 |

两臂每块调度前向数均为 8 + 1 封存 = 9 前向/块：**持平**。预注册 v1 门槛
（端到端 >= 1.5x、轮数削减 >= 2x）在随机权重玩具模型上机理不可达。

### 1.2 probe2：震荡结构与块宽敏感性（threshold=0.9, cap=16, G=96, seed=0）

changed 位比例逐轮轨迹（各块尾 8 轮平台区间）：

| W | forced 块 | changed 平台区间 | 形态 |
|---|---|---|---|
| 8 | 12/12 | 0.62 - 0.81 | 震荡平台，不衰减 |
| 16 | 6/6 | 0.62 - 0.91 | 震荡平台，不衰减 |
| 32 | 3/3 | 0.78 - 0.97 | 震荡平台，不衰减 |

**机理结论**：随机权重模型无稳定 argmax 不动点——全位同步替换使 62-97% 的位
持续翻转，changed 轨迹不衰减、块宽不敏感，任何 cap 下均不收敛。MTP draft-verify
的轮数削减收益 contingent 于草稿质量 = 真实预训练模型属性（Fast-dLLM/Jacobi 文献
证据全部来自预训练模型；LLaDA 块内 Jacobi 收敛为 O(1)-O(3) 轮）。自一致校验本身
工作正常：低质草稿被持续替换而非盲目锁定（acceptance 0.039-0.086 即拒收证据）。

### 1.3 预注册修订记录（v1 → v2）

**v1（srs/design 原定，用户 2026-10-23 确认口径）**：draft-verify vs mask-init
端到端 >= 1.5x（threshold=0.9）、每块轮数削减 >= 2x、token 一致率 >= 90%。

**v2（本次修订，用户 2026-10-23 确认转向诚实口径）**：probe1/probe2 证实随机权重
玩具模型上纯 Jacobi draft-verify 不收敛（§1.1/§1.2），v1 三门槛机理不可达；且
draft-verify 与 mask-init 收敛到不同终态（草稿起步 vs 全 MASK 起步），玩具无语义
下一致率为随机水平，>= 90% 同样不成立。按预注册修订纪律（沿 AR015/016 校准史）
修订为**诚实口径注册断言**（§2），性能主张（1.5x/2x/90%）**转入 s12 GPU 协议
作为服务器场地断言**（需 AR018 真实 dLLM 权重；Fast-dLLM 发表证据为该主张的
文献支撑，本仓库不提前引用为实测）。

修订依据：与 AR015 负结果先例同型——机制在玩具负载上的收益 contingent 于真实
模型属性时，本机注册结构性/精确性断言 + 零回退界，性能主张移交真实权重场地。

## 2. 注册断言（v2，跑前登记）

四臂：**baseline**（mask-init 全量重算，AR016 参考臂）/ **baseline-dv**
（draft-verify 全量重算，决策等价参考臂）/ **mask-init**（paged fp32）/
**draft-verify**（paged fp32）。baseline-dv 臂为 A1/A2 精确性锚点服务
（AR016 两臂结构沿承；srs FR17.2 已定义 baseline draft_mode 分支，T002 交付）。

注册配置：threshold=0.9（多轮主口径，沿 srs），G∈{192,448,960}，seeds={0,1,2}，
B=2，W=32，P=64，max_rounds=8，page_size=32，(L,H,D)=(8,8,64)，vocab=1000，
单线程 CPU。敏感性扫描（仅报告）：threshold=0.005，G=192 seed 0（阈值塌缩态
两臂轮数对照）。

| # | 断言 | 注册值 | 探针/UT 实测 | 裕度 |
|---|---|---|---|---|
| A1 | draft-verify paged vs baseline-dv tokens 逐位一致（9 点全） | 全部 True | UT T004 2/2 规模 True | 二值 |
| A2 | draft-verify paged vs baseline-dv 调度统计整表一致（rounds_per_block / total_rounds / rounds_to_fixpoint / changed_per_round / forced_completions / draft_acceptance_rate） | 全部相等 | UT T004 2/2 规模相等 | 二值 |
| A3 | 零回退界：draft-verify paged 端到端 / mask-init paged 端到端 | <= 1.2（探针结构持平 9=9 前向/块，预期 ~1.0） | probe1 持平 | ~20% 裕量 |
| A4 | 坏草稿拒收：draft_acceptance_rate（draft-verify paged，9 点） | <= 0.25 | probe1 0.038-0.086 | ~3x |
| A5 | 结构记账：draft-verify paged `computed_rows_total == Σ W×(调度前向+封存前向)` 闭式回代；`forced_completions == #{rounds_to_fixpoint == -1}`；`kv_write_rows_total == P+G` | 逐点成立（确定性） | UT T003 闭式回代通过 | 确定性 |
| A6 | 震荡特征（仅报告，不设门槛）：changed_per_round 末轮 > 0 的 forced 块占比与平台区间逐 seed 落档 | report-only | probe2 平台 0.62-0.97 | 不适用 |

未注册（v1 撤回，report-only 落档实际值）：draft-verify vs mask-init 端到端加速
（玩具持平，无收益可注册）；轮数削减（同）；token 一致率 vs mask-init（玩具无
语义，随机水平）。

## 3. 全量实跑结果

（T006 实跑后落档）
