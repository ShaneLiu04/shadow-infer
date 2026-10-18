# AR003 大规模复现实验计划（experiment-plan.md）

| 字段 | 内容 |
|------|------|
| 目的 | 在高速算力服务器（GPU）上验证 IRAP 决策平面与预算闭环的收益/精度影响（规则 1.1 实验铁律） |
| 现状 | 本机（Windows / CPU fallback）仅完成单元级与 golden 验证；收益类结论**未验证** |
| 关联 | srs.md §4/§5、design.md §6、README《大规模复现实验指南》 |

## 1. 实验问题（Hypotheses）

| ID | 假设 | 判定指标 | 判定线（需实测） |
|----|------|----------|------------------|
| H1 | IRAP 使 KV 压缩在同等预算下保留更关键的 KV（sink / 高接收注意力 token），质量损失下降 | perplexity 相对基线变化、任务准确率 | 3 种子中位，差异置信区间不跨 0 才算收益 |
| H2 | 预算准入使实际峰值显存稳定 <= 配置预算 | packed bytes / 进程峰值显存 | <= 预算（S2/S3），偏差 <= 1% |
| H3 | `delta_a` 信号接入 Q-drift（`use_importance_drift=true`）优于仅 query/activation 漂移 | 调度命中率、端到端质量-延迟帕累托 | 对照 3 次中位 |
| H4 | IRAP 常数内存 + 单次打包在长会话下不劣化延迟 | per_step_ms 斜率（步 1-5 预热后） | 无正斜率趋势（回归 p > 0.05） |

## 2. 硬件与软件环境

- GPU：单卡 >= 24 GB（A100 / L20 / 910B 等），CUDA >= 12.1；
- Python 3.10+，torch >= 2.2（按 pyproject 安装），启用 CUDA kernel（`shadowinfer.kernels` 编译成功日志无 fallback 警告）；
- 固定环境变量：`OMP_NUM_THREADS=8`、`CUDA_VISIBLE_DEVICES=0`、`PYTHONHASHSEED=0`。

## 3. 受测规模点

| 规模点 | 模型 | 后端 | seq_len | steps | batch | 预算 | 目的 |
|--------|------|------|---------|-------|-------|------|------|
| S1 | SimpleDiffusionLLM（内置） | pytorch | 128 | 6 | 1 | 不设 | 冒烟 / 回归对照（与 CPU 结果可比性） |
| S2 | 7B Diffusion LLM（LLaDA-8B-Instruct 或 Dream-7B） | huggingface | 512 | 32 | 1 | 6144 MB | 主实验（H1/H2/H3） |
| S3 | 同上 | huggingface | 2048 | 32 | 1 | 6144 MB | 长上下文 / 压力驱逐（H2/H4） |

模型配置模板：`configs/model_fast_dllm_7b.yaml`（按实际模型覆盖 num_layers/heads/head_dim）。

## 4. 配置矩阵（对照维度）

| 维度 | 取值 | 说明 |
|------|------|------|
| 决策信号 | `attn_contract_fixed`（True + IRAP） vs `legacy_double_softmax`（False 传入 probs，复现旧语义） | 核心对照（H1） |
| 预算 | 不设 / 6144 MB | 准入与驱逐（H2） |
| ΔA 融合 | `use_importance_drift` false / true（权重 0.3） | H3 |
| 种子 | 42 / 43 / 44 | 每组合 3 次，取中位 + 区间 |

配置落点：`configs/optimize_full.yaml`（shadowkv / qdrift 段，v3.5.0 已含全部键）。

## 5. 运行命令（每组 3 次）

```bash
for seed in 42 43 44; do
  python -m shadowinfer optimize \
    --model LLaDA-8B-Instruct --backend huggingface --device cuda \
    --num-steps 32 --seq-len 512 --config configs/optimize_full.yaml \
    --prompt prompts/eval_128.txt --seed $seed \
    --output outputs/S2/irap_s${seed}
done
```

基线对照：同命令附 `--baseline`（或独立运行 profiler 全量路径），确保同机同进程参数。

## 6. 采集与落档

每次运行输出 `outputs/<scale>/<variant>_s<seed>/`，汇总为 `outputs/summary.csv`：

```
scale,variant,seed,e2e_ms,per_step_ms_median,peak_mem_mb,kv_bytes,
sink_tier_fp32_ratio,precision_dist,spearman_received,delta_a_mean,evictions,alerts,perplexity
```

必须保存：原始 JSON 指标、`profile` 采样（cProfile / torch profiler）、环境指纹
（`nvidia-smi`、torch 版本、git SHA）。落档目录随 AR 归档至 `specs/archive/`。

## 7. 统计与判定

1. 延迟/显存：3 种子中位，报告中位 [min, max]；
2. 质量：perplexity/准确率差异用配对检验（>=3 配对，报告 p 值；样本不足时只报区间）；
3. 规模趋势：S1 -> S2 -> S3 同向变化才可称「多规模成立」；
4. 失败/回退：若某规模触发预算触底告警或驱逐率异常（> 5% token-head），如实记录，
   不得只报最优点；
5. 结论落档后才允许更新 README「性能基准」为已验证口径（规则 1.1）。

## 8. 预算与时长估算

| 规模点 | 单次运行（估） | 3 种子 x 2 对照 x 3 预算组合 |
|--------|----------------|------------------------------|
| S1 | < 2 min（GPU） | ~20 min |
| S2 | 15-25 min | 4.5-7.5 h |
| S3 | 45-90 min | 13-27 h |

建议 S1/S2 先行，S3 按需（长上下文收益确认后）。
