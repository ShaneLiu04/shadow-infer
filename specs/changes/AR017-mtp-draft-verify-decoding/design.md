# 1 AR概述

| 组件名称 | shadowinfer（execution 调度层 / backend 生成管线） |
| --- | --- |
| AR系统流水号 | AR017 |
| AR描述 | 块级多 token 草拟-验证并行解码（MTP draft-verify）：round-0 全位并行草拟替代全 MASK 初始化，验证轮自一致迭代至不动点，块内轮数从 O(W) 塌缩至 O(1)-O(3)；定点确认前向的 K/V 复用为封存 append（结构性省一次前向）。Fast-dLLM MTP + Jacobi 式收敛的对标实现，与 AR016 精确前缀 KV 协同（零额外页写）。 |

# 2 动态行为

## 交互时序图

```plantuml
@startuml
title draft-verify 模式单块生成时序（generate 臂）
participant Sched as block_decoding\n(纯函数)
participant Gen as PyTorchModelBackend.generate
participant FB as SimpleDiffusionLLM.forward_block
participant Store as PagedKVStore (per-layer)

Gen -> FB : forward_block(MASK块, prefix_kv)
FB --> Gen : logits (round-0)
Gen -> Sched : draft_round(logits)
Sched --> Gen : 草稿 tokens [B,W]（全位 argmax 填充）
Gen -> Gen : block_tokens = 草稿（无 MASK 残留）

loop 验证轮 (rounds < max_rounds)
  Gen -> FB : forward_block(草稿块, prefix_kv)
  FB --> Gen : logits, block_kv
  Gen -> Sched : verify_round(logits, block_tokens)
  Sched --> Gen : changed / new_tokens
  alt changed.sum() == 0（不动点）
    Gen -> Store : append(最后一次前向的 block_kv)  «定点 KV 复用，零额外前向»
    note right : rounds_to_fixpoint 记账
  else 有替换位
    Gen -> Gen : block_tokens[changed] = new_tokens
  end
end

alt 达 max_rounds 未不动点（forced）
  Gen -> FB : forward_block(最终块状态, prefix_kv)  «封存前向»
  FB --> Gen : block_kv
  Gen -> Store : append(block_kv)
  note right : forced_completions += 1（单位=块）
end
@enduml
```

# 3 功能点分解

| 序号 | 功能点名称 | 功能点描述 | 追溯 |
| --- | --- | --- | --- |
| 1 | 草拟轮原语 | round-0 logits 全位 argmax 填充；返回草稿 tokens 与置信度 | FR17.1 |
| 2 | 验证轮原语 | 逐位 `argmax(logits) == current_token` 自一致校验；不匹配位替换；返回 changed/new_tokens/置信度 | FR17.1 |
| 3 | 不动点检测与收尾 | changed==0 判定 rounds_to_fixpoint；max_rounds 耗尽强制封存 | FR17.1 |
| 4 | 配置扩展 | `BlockDecodingConfig.draft_mode: bool = False` + bool 校验 | FR17.1/17.2 |
| 5 | generate draft_mode 分支 | 管线骨架沿 AR016；草拟-验证调度替换 mask-init 调度；定点 KV 复用封存 | FR17.2 |
| 6 | baseline draft_mode 分支 | `blockwise_generate_baseline` 同调度全量重算（决策等价参考臂） | FR17.2 |
| 7 | 统计扩展 | draft_acceptance_rate / changed_per_round / rounds_to_fixpoint；两模式 17 键集一致 | FR17.3 |
| 8 | 三臂基准 | ar017_bench.py：baseline / mask-init / draft-verify；校准探针 + 预注册断言 | FR17.4 |
| 9 | s12 GPU 协议 | s12 yaml + server-experiment + serve_arm --mode；AR018 前向兼容 | FR17.5 |
| 10 | README + 版本同步 | 四检查点 + 3.19.0 四处 | FR17.6 |

# 4 实现设计

## 4.1 功能实现思路

**方案 A（选定）：独立纯函数 + 管线内联分支 + 定点 KV 复用。**

- 调度层新增两个**无状态纯函数**（`draft_round` / `verify_round`），与 `schedule_round` 并列；AR016 mask-init 语义零改动（108 用例零回归）；
- `generate` 与 `blockwise_generate_baseline` 在块内循环按 `config.draft_mode` 分支：mask-init 走既有 `schedule_round` 路径（逐行保持），draft-verify 走草拟-验证路径；
- **定点 KV 复用**：验证轮的 `forward_block` 返回 `(logits, block_kv)`；当该轮 changed==0（不动点确认），此最后一次前向的 block_kv 即最终块状态的 K/V，直接 append 入页——省一次封存前向。forced 收尾（max_rounds 耗尽仍有替换）时最后前向后状态已变，需一次封存前向；
- **精确性锚点**：段因果掩码（AR016 §4.5 erratum）保证前缀行感受野时间不变 → draft-verify paged 臂 vs baseline draft-verify 臂（同调度全量重算）**fp32 逐位一致**（UT 锚点）；draft-verify vs mask-init 为一致率口径（srs 约束，不伪称 bitwise）；
- baseline draft_mode 分支存在的意义：为 draft-verify 提供 bitwise 参考臂（同 AR016 两臂结构），基准三臂中 baseline 默认 mask-init（token 参考口径），UT 中 baseline+draft_mode 用于逐位验证。

**与 AR016 的轮数口径对齐**：`max_rounds` 约束块内调度前向总数（draft + verify）。mask-init：最多 max_rounds 轮 + 恒定 1 次封存前向；draft-verify：1 次草拟 + 最多 (max_rounds-1) 次验证，定点则零封存前向、forced 则 +1。`computed_rows_total = Σ W×(调度前向数 + 封存前向数)` 两模式同式。

## 4.2 功能实现设计

### 4.2.1 流程图

```plantuml
@startuml
title generate 块内循环（draft_mode 分支）
start
:块初始化为 MASK×W;
:rounds = 0;
repeat
  :forward_block(block_tokens, prefix_kv)\n获得 logits;
  if (rounds == 0) then (草拟轮)
    :tokens, conf ← draft_round(logits);
    :block_tokens = tokens（全位填充）;
    :changed_count = W;
  else (验证轮)
    :changed, tokens, conf ← verify_round(logits, block_tokens);
    :block_tokens[changed] = tokens[changed];
    :changed_count = changed.sum();
    if (changed_count == 0) then (不动点)
      :rounds_to_fixpoint = rounds + 1;
      :seal_kv = 本次前向的 block_kv «复用»;
      break
    endif
  endif
  :rounds += 1;
repeat while (rounds < max_rounds?) is (是)
->否（forced）;
:forced_completions += 1;
:封存前向 forward_block(最终状态)\n获得 block_kv;
endif
:append block_kv 入各层页（一次）;
:记账 changed_per_round / rounds_per_block 等;
stop
@enduml
```

### 4.2.2 流程说明

1. **预填充**（与 AR016 完全一致，两模式共享）：prompt 全量前向一次，K/V 按层 append 入页（P 行）；此后进入逐块循环。
2. **块内循环 draft-verify**：
   - rounds==0（草拟轮）：前向输入为全 MASK 块（MASK 条件化），`draft_round` 取 [B,W,V] logits 的全位 argmax，整块填充。changed_per_round[0] = W（全部位获得草稿 token）。草拟后块内恒无 MASK——进度保证的结构性来源；
   - rounds>=1（验证轮）：前向输入为当前草稿块，`verify_round` 逐位比较 argmax 与当前 token：自一致位保留、不匹配位以新 argmax 原地替换（单调替换，无重掩码）；
   - **不动点**：changed==0 → 该轮前向即最终状态的 forward，其 block_kv 直接用于封存 append（零额外前向）；`rounds_to_fixpoint = rounds+1`（含草拟轮与本次确认轮的总调度前向数）；
   - **收尾**：rounds 达 max_rounds 仍有替换 → 以当前状态强制封存（需一次封存前向），`forced_completions += 1`（单位=块，沿 AR016 design §4.6 erratum 口径）；
   - batch 元素独立：changed/替换决策逐元素计算，互不影响；并列 argmax 取最小 token id（torch.argmax 语义，沿 AR016 确定性惯例）。
3. **封存**：block_kv append 入各层页（W 行一次）；prefix_len += W；下一块。
4. **mask-init 分支**（draft_mode=False）：逐行保持 AR016 §4.2.2 原文语义（schedule_round 过阈解掩 + 保底 top-1 + max_rounds 强制收尾 + 恒定封存前向），本文档不重复。
5. **fail-fast**（两模式共享，沿 AR016 §4.2.2(7) 清单零改动）：prompt 2-D/整型、P+G<=max_seq_len、mask_token_id ∈ [0, vocab)、prompt 不含 mask、G%W==0、page_precision 合法 → `ValueError` 且零副作用；`draft_mode` 非 bool → `ValueError`（配置构造期）。
6. **统计记账**（见 4.3 stats 契约）。

### 4.3 接口描述

**新增公开 API（`shadowinfer/execution/block_decoding.py`，经 `execution/__init__.py` 导出）：**

```python
@dataclass(frozen=True)
class BlockDecodingConfig:
    # ... AR016 六字段零改动 ...
    draft_mode: bool = False          # 新增：草拟-验证解码模式开关

def draft_round(
    logits_block: torch.Tensor,       # [B, W, V] round-0（全 MASK 条件）logits
) -> Tuple[torch.Tensor, torch.Tensor]:
    """草拟轮：全位 argmax 填充。

    Returns:
        (tokens [B, W] long, confidences [B, W] float)  # conf = softmax 最大值
    Raises:
        ValueError: logits_block 非 3-D。
    """

def verify_round(
    logits_block: torch.Tensor,       # [B, W, V] 验证轮 logits
    block_tokens: torch.Tensor,       # [B, W] long 当前块 token
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """验证轮：自一致校验与原地替换。

    Returns:
        (changed [B, W] bool, tokens [B, W] long（全位 argmax，调用方仅取
         changed 位）, confidences [B, W] float)
    Raises:
        ValueError: 形状不符 / block_tokens 非 2-D 整型。
    """
```

**既有 API 语义扩展（签名不变）：**

- `PyTorchModelBackend.generate(...)`：`config.draft_mode=True` 时走草拟-验证分支；`GenerateResult` 结构不变；
- `blockwise_generate_baseline(...)`：同上（决策等价参考臂）。

**stats 契约（17 键，两模式键集一致；既有 14 键语义零改动）：**

| 键 | mask-init | draft-verify |
|---|---|---|
| draft_acceptance_rate | None（不适用） | float ∈ [0,1]：各块 round-0 草稿 token 被最终封存保留的位比例（batch×blocks 总均值） |
| rounds_to_fixpoint | None | List[int] 逐块：定点确认时的调度前向总数（草拟+验证）；forced 块记 -1 |
| changed_per_round | None | List[List[int]] 逐块逐轮替换位数；round 0 恒 W（草拟填充） |
| forced_completions | 沿 AR016（单位=块） | 同口径：达 max_rounds 未定点的块数 |
| computed_rows_total | Σ W×(rounds+1) | Σ W×(调度前向数 + 封存前向数)：定点块封存前向=0，forced 块=1 |

**serve_arm.py**：新增 `--mode {mask-init, draft-verify}`（默认 mask-init），映射 `config.draft_mode`；artifacts/stats.json 落盘含新键。

### 4.4 代码设计

```
shadowinfer/
├── execution/
│   ├── block_decoding.py      # +draft_mode 字段；+draft_round/verify_round 纯函数
│   │                          #  baseline 块内循环 draft_mode 分支（~25 行）
│   └── __init__.py            # 导出 draft_round / verify_round（BlockDecodingConfig 等已有）
├── core/backends/
│   └── pytorch_backend.py     # generate 块内循环 draft_mode 分支 + 定点 KV 复用（~40 行）
tests/
└── test_draft_verify_decoding.py   # 新增 UT 文件（五分区，见 6.1）
specs/changes/AR017-mtp-draft-verify-decoding/
├── ar017_bench.py             # 三臂基准（注册化断言内置）
└── server-experiment.md       # s12 GPU 复现协议
configs/experiments/
└── s12_draft_verify_gpu.yaml  # 7B 三臂协议（model_weights 位预留 AR018）
```

- **分层不变**：调度纯函数无状态（仅 torch 依赖，TYPE_CHECKING 守卫沿 AR016）；backend 编排与 store 装配；模型侧零改动（`forward_block` 已返回 (logits, block_kv)，无需新接口）；
- **无循环导入**：新函数与 schedule_round 同层同依赖面；
- **mask-init 路径逐行保持**：draft_mode 分支为纯增量，默认 False 回归锚点 = v3.18.0 全部 1128 用例零改动通过。

# 5 重构设计

无。本 AR 为纯增量扩展：AR016 全部代码路径、签名、统计语义逐行保持。

# 6 测试设计

## 6.1 单元测试（UT）

`tests/test_draft_verify_decoding.py`，五分区（沿 AR016 文件组织惯例）：

- **T001 调度原语**（FR17.1）：草拟轮全位 argmax 填充 + 置信度正确；验证轮构造自一致/替换/并列最小 token id 三态；changed==0 不动点判定；batch 独立（元素 A 替换不影响 B）；退化输入（空 logits/形状不符）ValueError；draft_mode 非 bool 配置 ValueError；配置合法边界（draft_mode=True 构造通过）。
- **T002 管线分支**（FR17.2）：draft-verify 全流程完成（多块/两规模/B=2）；序列形状 [B,P+G] 与 prompt 逐位保留；页驻留不变量（轮间零页写、append 恰 P+G 各一次、`last_generate_stores` 账本核验）；**mask-init 回归锚点**（draft_mode=False 与 v3.18.0 逐位一致，随机 prompt 多块）；fail-fast 清单全项 + 零副作用（`last_generate_stores is None`）。
- **T003 统计契约**（FR17.3）：17 键恒存在；draft-verify 模式 `draft_acceptance_rate ∈ [0,1]` 且与逐位核对一致（终态 == 草稿的位比例）；`rounds_to_fixpoint` 与 changed_per_round 解析预期；`computed_rows_total` 闭式回代（定点块封存 0 前向、forced 块 1 前向）；mask-init 模式三新键 None。
- **T004 e2e 行为**（FR17.1/17.2 验收）：**draft-verify paged vs baseline draft-verify 逐位一致**（fp32，段因果时间不变性锚点，多块两规模）；构造两轮收敛轨迹核验 rounds_to_fixpoint==2；永不收敛轨迹 + max_rounds 强制收尾（forced_completions 计数 + 封存前向回代）；batch 元素不同收敛轮数独立；max_rounds=1 退化（仅草拟，forced 记账）。
- **T005 基准接口冒烟**：ar017_bench.py 探针模式可跑（小规模单 seed），断言机制存在。

## 6.2 接口测试

`draft_round`/`verify_round` 签名与返回契约（形状/dtype）；`BlockDecodingConfig.draft_mode` 默认 False 与校验消息；`generate`/`blockwise_generate_baseline` 对 `draft_mode` 的分支等价性（同输入两实现调度决策一致）；`execution/__init__` 导出名可达。

## 6.3 业务场景测试

- 受控生成任务：随机 prompt（seed 固定）+ 多块生成，全程真实前向，draft-verify 正常完成；
- 精确性场景：draft-verify paged vs baseline 逐位一致（6.1 T004 锚点）；
- 一致率场景：draft-verify vs mask-init token 一致率实测（基准量化，>= 90% 预注册口径）；
- 性能场景：轮数削减（rounds_per_block 对比）与端到端加速（基准量化）。

## 6.4 异常场景测试

- 配置非法（draft_mode 非 bool 等）→ 配置构造期 ValueError（fail-fast，无半执行状态）；
- generate 非法输入（超长/prompt 含 mask/词表外 mask_token_id/gen_len 不整除/1-D/float dtype）→ ValueError 且 `last_generate_stores is None`（沿 AR016 清单，两模式各覆盖一遍）；
- 退化输入（空 logits）→ 调度原语显式 ValueError 不崩溃；
- forced 路径：max_rounds=1 强制收尾记账；永不收敛轨迹有界终止（max_rounds 上界保证）。

## 6.5 基准测试（T005/T006，`ar017_bench.py`）

- 三臂：**baseline**（mask-init 全量重算）/ **mask-init**（AR016 优化管线）/ **draft-verify**（本 AR）；真实 `SimpleDiffusionLLM`，真实随机 prompt，`torch.set_num_threads(1)`，3 seeds（0/1/2）；
- 规模：沿 AR016 口径 (L,H,D)=(8,8,64)、P=64、G∈{192,448,960}、B=2、W=32、max_rounds=8；**threshold=0.9 主口径**（多轮场景，MTP 机制所在）；探针阈值扫描（如 {0.9, 0.5, 0.005}）落档校准史；
- 指标：三臂端到端时间、mask-init vs draft-verify 每块轮数（rounds_per_block / rounds_to_fixpoint）、draft_acceptance_rate、draft-verify vs mask-init token 一致率、draft-verify vs baseline 一致率、computed_rows、kv_write_rows；
- **校准先行**（AR015/016 纪律）：正式断言登记前跑校准探针（不入库），按实测固化阈值（裕量 >= 20%），断言集跑前登记于 bench-results.md；
- 预注册门槛（srs 用户确认口径，探针后修订需落档）：draft-verify vs mask-init 端到端 >= 1.5x（threshold=0.9）、每块轮数削减 >= 2x、token 一致率 >= 90%；未达标如实报告（三轮预注册历史模式沿用）。
