# [AR004] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR004 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-18 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 微基准取证落档：on_step 中位 1.88ms、fingerprint 145.4-194.6µs、统计 cProfile 20%、掩码与 FFN-P1 缺陷复现 → baseline-profile.md | - | passing（已核验） | 2026-10-18（baseline-profile.md + ar004_bench.py） |
| T002a | FR4.1 指纹 O(1)（packed_weight + 集成验证），golden 测试 | T001 | passing | 2026-10-18（commit c027ee0；UT-1/UT-2） |
| T002b | FR4.2 统计缓存（层级唯一归属 + 内存累计），golden 测试 | T004 | passing | 2026-10-18（quantization_stats 按层一次；UT-3/3b/18b） |
| T003 | FR4.3 向量化 unpack + `_unpack_reference` golden；性能微基准 | T002b, T004 | passing | 2026-10-18（20x，UT-4/5/7） |
| T004 | FR4.4 逐层状态隔离（`_LayerState` 骨架 + 层级读写点）+ on_init 清空 + 超层告警 | T002a | passing | 2026-10-18（UT-6/6b/6c/16） |
| T005 | FR4.5 `_mixed_compute` 列掩码修复 + 全 False 告警 | T004 | passing | 2026-10-18（缺陷复现转通过；UT-8/9/21） |
| T006 | FR4.6 向量化 quantize_mixed_precision + 参考 golden；性能微基准 | T003, T005 | passing | 2026-10-18（20x，UT-10/11） |
| T007 | 微基准对照 + 全量回归 + 五件套 | T006 | passing | 2026-10-18 同进程同批：指纹 256/1024 ch = 0.004x/0.001x；unpack 0.070x/0.044x；quantize 0.077x/0.053x；on_step 0.66-0.72ms（绝对中位）；719 passed / 82.37% |
| T008 | README v3.6.0 四检查点 + pyproject/`__version__`/ROADMAP | T007 | passing | 2026-10-18：特性表/版本历史/基准/复现协议；版本号三处 3.6.0；当前口径 719/82.37%，历史 v3.4.0 行保留 |
| T009 | Review + ST（st-cases.md）+ 归档 + Gitee 推送（a Review；b ST；c 归档；d 推送三核验） | T008 | pending | 规则二 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

### 2026-10-18 T001 基线取证

- 微基准（1024 ch / 4 层 / 12 calls，CPU，3 次独立 agent + warmup）：per_on_step 中位 1.88ms（复跑 1.97ms）
- cProfile：dequantize_for_matmul 39%（其中 `_precision_plan_fingerprint` 37%）、quantization_info 统计 20%
- 缺陷复现：`_mixed_compute` + channel_mask + packed → shape 错误（down 投影未应用列掩码）
- 脚本归档：`specs/changes/AR004-ffn-hotpath/ar004_bench.py`（seed=42）

## 阶段门控记录

### 2026-10-18 req+design 门控记录（第 1 轮）

- 门控结果：FAIL
- 失败项：I-1 tasks.md 预填 PASS 记录；I-2 基线未落档
- 修复：删除预填记录；新增 baseline-profile.md 并归档脚本；修正 design 缺失
- 审查代理：sdd-gate-reviewer

<!-- 第 2 轮门控结论由实际审查后追加，禁止预填 -->

### 2026-10-18 req+design 门控记录（第 2 轮）

- 门控结果：FAIL
- 阻断项：① UT-12/NFR 固定基线口径不成立（违反规则 1.1）；② 层隔离三处调用点未逐点闭合（I-3）；③ G20 用例有效性、G22 边界/流程分支覆盖不足
- 残留 WARN：接口表不全（on_step/on_init/config）、内存累计口径、FR4.2 计时、UT-17 ROADMAP、srs 影响模块漏 pyproject/__init__
- 修复：NFR/UT-12 改为同会话同批 A/B + 结构硬门禁；design 增 4.3.0 契约与 `_LayerState` 类型、三处入口逐点、流程图 alt/else 分支、UT-7/UT-18、UT-3b 计时、UT-15/16/17 增强；srs 影响模块与 FR4.5 比较对象修订；tasks T001 已核验、T007/T009 拆分
- 审查代理：sdd-gate-reviewer

### 2026-10-18 req+design 门控记录（第 3 轮）

- 门控结果：FAIL
- 阻断项：G12 `_LayerState` 三字段类型不符；G15 流程图重复分支；G22 UT-15 与 reuse 实现冲突；T001 证据不足（缺陷未入脚本/无 warmup）
- 修复：srs FR4.4/FR4.5 口径统一；流程图单 alt + sensitivity/缺权重分支；字段类型对齐代码；UT-15 改为“直接返回旧输出”；bench 增两条缺陷复现与 warmup（实跑输出：RuntimeError 32x716 vs 1024x256；|out-direct|=0.5771、|out-shared|=0.0）；T002 拆 a/b
- 审查代理：sdd-gate-reviewer

### 2026-10-18 req+design 门控记录（第 4 轮）

- 门控结果：FAIL
- 阻断项：P1 层隔离调用点漏 :174/:219；P2 时序图 `_mixed_compute` 无条件调用、流程图顺序与代码不符；P3 `weight_up_quantized` 类型应为 Union[Tensor, PackedFFNWeight]
- 修复：补 :174/:219 与 UT-16 五处断言；两张图按代码顺序重画（sensitivity→select→回退→惰性量化→掩码→分支）；类型改 Union；UT-19/UT-20 覆盖安全回退与 sensitivity 阈值；UT-3b 改行为级；UT-13 明确非法 method 期望；srs 统计基线口径修正；覆盖率来源与 T007 落档说明
- 审查代理：sdd-gate-reviewer

### 2026-10-18 req+design 门控记录（第 5 轮）

- 门控结果：FAIL
- 阻断项：G15/P2 两图顺序仍与代码不符（importance/mask 在 select 之前；:154-163 早期预计算遗漏）；P1 层隔离仅覆盖 importance，量化写回/读取与 `_channel_mask/_active_channels` 未归层；P4 统计基线与 on_step 实测口径矛盾（后续按实跑更正）；G22 UT-19/20 重复、UT-6 未限定 full、UT-16 动态剪枝语义不清
- 修复：两图按代码顺序重画（sensitivity→早期 importance/mask→select→回退→惰性量化/统计→五分支）；design 补 10 个层状态读写点、None 回退仅限 layer 0、on_init 先清空后预载；srs 统计验收改为「读取 ≤20µs + cProfile 20% 取证口径，绝对值以 T007 同批为准」；UT-6 限定 full、UT-16 拆静态/动态、UT-19 注明 monkeypatch、删除重复行；边界矩阵删除不存在的「通道<4」行并补 sensitivity/method 边界；UT-7 明确仅 device 转换
- 审查代理：sdd-gate-reviewer

### 2026-10-18 req+design 门控记录（第 6 轮）

- 门控结果：FAIL
- 阻断项：I-1 基线旧数残留（tasks/srs）；I-2 UT-19 未注明 monkeypatch；I-3 UT-18 保留不存在的「通道<4」；I-4 compute_channel_mask 惰性语义与签名矛盾；I-5 sensitivity 分支无用例
- 修复：按实跑更正所有基线数字（1.88/1.97ms，39/37/20%）；UT-19 注明 monkeypatch；UT-18 改为 topk/ratio/内存边界；UT-20 恢复 sensitivity 0.7/0.69；compute_channel_mask 明确恒回退 layer 0（on_step 恒传本层）；时序图统计归属 _LayerState；掩码异常入矩阵与 UT-9
- 审查代理：sdd-gate-reviewer

### 2026-10-18 req+design 门控记录（第 7 轮）

- 门控结果：FAIL
- 阻断项：P1 基线旧数残留；P2 统计缓存零迭代口径与实现冲突；P3 掩码异常回退写成 full；P4 时序图 alt 结构；P5 T002b/T004 依赖倒置；P6 compute_channel_mask 回退条件不可实现；P7 缺第 6 轮记录；P8 覆盖率冲突未登记；P9 初始化迁移；P10 单规模点；P11 告警键歧义；P12 UT-3b 断言；P13 指纹对象同一性；P14 全 False 告警落点
- 修复：交付件统一实跑基线（1.88/1.97ms、39/37/20%）；统计改为量化时构建+on_step 浅拷贝；掩码异常保持原路径；时序图五分支单 alt；任务链 T002a→T004→T002b；compute_channel_mask 恒回退 layer 0（on_step 恒传本层）；补第 6 轮记录；T008 显式登记覆盖率冲突；__init__ 顺序与 _plan_fingerprint 初值；UT-5/11 双规模；_warned 键 (layer_id, kind)；UT-3b 预热后强断言；指纹先 is 后 ==
- 审查代理：sdd-gate-reviewer

### 2026-10-18 req+design 门控记录（第 8 轮）

- 门控结果：FAIL
- 阻断项：I-1 时序图重复 else incremental；I-2 _warned 命名/解析规则；I-3 full 路径全 False 告警链断裂；I-4 UT-3b 未限定预热后
- 修复：时序图单 alt 五分支；统一 _warned + 解析规则入接口表；warning_key 由 on_step 传 _mixed_compute/_full_compute（_linear_forward 仅应用掩码），补 _full_compute 接口与 UT-21；UT-3b 限定首步量化完成后
- 审查代理：sdd-gate-reviewer

### 2026-10-18 req+design 门控记录（第 9 轮）

- 门控结果：FAIL
- 阻断项：I-3 full 全 False 告警链（_full_compute 未入接口/无 warning_key 传参）；G23 缺第 8 轮记录；G17 指纹区间不一致；G12/G24 active_channels 默认值、_layer_state 权重写入、超层规则、_plan_fingerprint 归属
- 修复：补 _full_compute 接口与 warning_key（on_step 传入）；补第 8 轮记录；baseline 指纹统一 145.4~194.6；active_channels int=0；_layer_state 后写权重；超层规则统一；移除 agent 侧 _plan_fingerprint；UT-13/18 拆分；UT-20/UT-21；图与 dequant 链修正；bench 路径改 parents[3]
- 审查代理：sdd-gate-reviewer

### 2026-10-18 req+design 门控记录（第 10 轮）

- 门控结果：FAIL
- 阻断项：J1 on_step 未传 warning_key；J4 srs 覆盖率 82.04 与 75 门禁冲突；J13 指纹 is/== 表述矛盾；J15 bench parents[2] 层级错误
- 修复：on_step 契约与流程图传入 warning_key；srs NFR 改 75% 门禁 + README 实跑同步（82.04 仅参考）；指纹统一「is 快路径，否则 ==」；bench 改 parents[3] 并复跑成功；补 J3 样本（145.4/151.4/170.1/185.3/194.6）、J6 写入条件、J12 重置值/ratio 用例、UT-12 报告型、UT-15 行为断言、矩阵 ratio/mask_dim 行
- 审查代理：sdd-gate-reviewer

### 2026-10-18 req+design 门控记录（第 11/12 轮）

- 第 11 轮：FAIL（唯一 Important：G23 缺第 10 轮门控记录；K5 指纹样本 Minor）
- 修复：补第 10 轮记录、baseline 五样本、T002a 措辞、UT-3b 口径统一
- 第 12 轮：PASS（0 Critical / 0 Important；3 WARN + 9 Minor 由 T007/T008 与实现期处理）
- 门控结论：**req+design 阶段通过，放行 T002a 开发**
- 审查代理：sdd-gate-reviewer

## 实现与验证记录（2026-10-18）

- T002a：`PackedFFNWeight._plan_fingerprint` pack 时构建；dequant `is` 快路径 + `==` 值语义。
- T004/T002b：`_LayerState` + 10 处读写点归层；`on_init` 先清空/重置再预载 layer 0；`_warned` 键去重。
- T003/T006：unpack 分组批量（int4 批量解包）+ 参考实现保留；量化按重要性掩码向量化 + `_quantize_mixed_precision_reference`。
- T005：`_linear_forward(mask_dim)`、down 列掩码、全 False 零输出 + 告警一次（mixed/full）。
- T007 实测（同会话同批，CPU）：on_step 1.95→0.71ms（0.36x）；指纹 151.4→0.3µs；unpack 82.29→3.91ms（0.047x）；quantize 189.09→9.43ms（0.050x）；`732 passed, 1 skipped`，覆盖率 82.37%。
- T008：README v3.6.0 四检查点 + 服务器复现协议；pyproject/`__version__`/README 徽章 3.6.0 同步；ROADMAP AR004 勾选。
