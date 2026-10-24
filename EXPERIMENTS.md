# EXPERIMENTS.md — s13 服务器实验手册（真实 dLLM 权重四臂协议）

> 本手册属 AR018（v3.20.0，FR18.6）：在服务器 GPU 上以**真实预训练 dLLM 权重**
> （Dream-7B 主验收 / LLaDA-8B 次之）运行四臂对照协议 s13，判定 AR016/AR017
> 悬置的场地断言 S1-S3。本机开发环境（Windows / CPU / 网关拦截 HF）**零性能
> 主张**：下文所有「结果」小节均为**服务器实测后回填**的占位；本地仅以 fixture
> 权重冒烟证明命令形态可执行（fixture 与真实权重走同一代码路径，设计 D12）。

---

## 目录

| 章节 | 内容 |
|------|------|
| [1. 目的与范围](#1-目的与范围) | 四臂定义、主/次验收模型 |
| [2. 权重获取](#2-权重获取) | HF 下载 / 离线传输、PRESETS 核实 |
| [3. 环境要求](#3-环境要求) | CUDA / torch / 依赖版本登记 |
| [4. 运行前自检](#4-运行前自检) | fixture 冒烟（CPU 可跑） |
| [5. 四臂运行命令](#5-四臂运行命令) | 服务器正式命令（seed 0/1/2） |
| [6. 取证要求](#6-取证要求) | stats.json / artifacts.pt / 显存与页统计 |
| [7. S1-S3 判定标准](#7-s1-s3-判定标准) | 判定口径与计算方法 |
| [8. 结果落档](#8-结果落档) | 回填模板（待服务器实测） |
| [9. 诚实纪律](#9-诚实纪律) | 未达标如实标注、口径声明 |

---

## 1. 目的与范围

| 臂 | 调度实现 | 语义 |
|----|----------|------|
| `baseline` | `blockwise_generate_baseline` | 块式全量重算参考（AR016 精确性锚点；每轮全 [B, L+W] 重算，无页存储） |
| `mask-init` | `blockwise_generate`（`draft_mode=False`） | AR016 块式管线：页式 KV + 逐轮 mask-init 解码 |
| `draft-verify` | `blockwise_generate`（`draft_mode=True`） | AR017 草拟-验证：轮 0 全位并行草拟 + 自洽性验证至不动点（**本协议测量臂**） |
| `remasking` | `remasking_generate` | AR018 dLLM 原生基线：全序列双向重算 + 低置信 remasking 解掩（LLaDA/Dream 官方生成语义） |

- **主验收模型**：Dream-7B；**次验收模型**：LLaDA-8B-Base。两者均为 Llama 架构
  双向 dLLM，由 `load_diffusion_llama` 以 config.json 驱动加载（代码零硬编码，
  设计 D13）。
- 同一 prompt（seed 决定）跨四臂对照；`page_precision` 固定 fp32，使 S1/S2 是
  **调度对比**而非精度对比。
- 口径声明：块式三臂是全双向 dLLM 的**半 AR 近似**（早期块不可见后期块），
  与 remasking 臂的一致率（S3）是一等指标，任何文档不得伪称等价。

## 2. 权重获取

### 2.1 HF 下载（服务器可直连时）

文献口径的仓库标识（**本机网关无法核实，服务器侧以实际页面为准**）：

| 模型 | HF 仓库（文献口径，待核实） | mask_token_id（PRESETS，待核实） |
|------|------------------------------|----------------------------------|
| Dream-7B | `Dream-Projects/Dream-7B` | 151669 |
| LLaDA-8B-Base | `GSAI-ML/LLaDA-8B-Base` | 126336 |

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Dream-Projects/Dream-7B \
  --local-dir /data/weights/Dream-7B
huggingface-cli download GSAI-ML/LLaDA-8B-Base \
  --local-dir /data/weights/LLaDA-8B-Base
```

### 2.2 离线传输（服务器不通外网时）

在任意可联网机器执行上述 `huggingface-cli download`，打包传输：

```bash
tar -czf dream-7b.tar.gz -C /data/weights Dream-7B
scp dream-7b.tar.gz <user>@<server>:/data/weights/
ssh <user>@<server> "tar -xzf /data/weights/dream-7b.tar.gz -C /data/weights"
```

校验：目录内须含 `config.json` 与（`model.safetensors`，或
`model.safetensors.index.json` + 分片）；加载器对缺文件/坏索引 fail-fast。

### 2.3 PRESETS 核实（必做，一次）

加载前将真实 `config.json` 与源码 `PRESETS`（`shadowinfer/models/diffusion_llama.py`）
逐字段对照（文献口径登记表）：

| 字段 | dream-7b（PRESETS） | llada-8b（PRESETS） |
|------|---------------------|---------------------|
| num_layers / hidden_dim | 28 / 3584 | 32 / 4096 |
| num_heads : num_kv_heads | 28 : 4 | 32 : 8 |
| head_dim / intermediate | 128 / 18944 | 128 / 14336 |
| vocab_size / mask_token_id | 151936 / 151669 | 126464 / 126336 |
| max_position_embeddings | 131072 | 8192 |
| rope_theta / rms_norm_eps | 1e6 / 1e-6 | 5e5 / 1e-5 |
| attention_bias / tie_word_embeddings | true / false | false / false |

任何出入：**在 `specs/archive/AR018-real-dllm-weights/design.md` 落 erratum 并
修订 PRESETS**，然后重跑加载冒烟；代码路径本身 config 驱动、不受影响。
`serve_arm.py` 会在 `execution.mask_token_id` 显式给出且与 checkpoint 不符时
拒绝运行（fail-fast）。

## 3. 环境要求

| 项 | 要求 |
|----|------|
| GPU | 单卡 >= 24 GB（Dream-7B bf16 权重约 15 GB + fp32 页 KV；主验收口径如改用多卡须在 §8 登记实际配置） |
| CUDA / 驱动 | 与下方 torch 版本匹配（登记实测 `nvidia-smi` 输出） |
| Python | 3.10+ |
| torch | 2.2+（与仓库 CI 口径一致；登记服务器实际版本） |
| 依赖 | `pip install -e .` 或按 `requirements-lock.txt`；额外需要 `pyyaml`、`safetensors` |
| 随机性 | 固定 `--seed`；prompt 由 `2026 + seed` 派生，四臂同 seed 即同 prompt |

```bash
# 环境登记（实测后回填）
python --version        # 待回填
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi --query-gpu=name,memory.total --format=csv   # 待回填
```

## 4. 运行前自检

正式跑前先做两项零成本自检：

1. **fixture 冒烟（可在本地 CPU）**：用 `fixture_gen.py` 生成 tiny 真实格式
   checkpoint，四臂各跑一遍命令形态（与服务器同一路径，仅尺寸不同）：

   ```bash
   python - <<'PY'
   import importlib.util, tempfile
   spec = importlib.util.spec_from_file_location(
       "fixture_gen",
       "specs/archive/AR018-real-dllm-weights/fixture_gen.py")
   fg = importlib.util.module_from_spec(spec); spec.loader.exec_module(fg)
   d = tempfile.mkdtemp(prefix="ar018_smoke_")
   fg.generate_fixture(d, style="llama", num_layers=2, num_heads=8,
       num_kv_heads=2, head_dim=32, hidden_size=256, vocab_size=1000,
       mask_token_id=999, max_position_embeddings=512, seed=0)
   print(d)
   PY
   # 四臂冒烟（<fixture_dir> 替换为上一步打印的目录）：
   for ARM in baseline mask-init draft-verify remasking; do
     python specs/archive/AR018-real-dllm-weights/serve_arm.py \
       --config configs/experiments/s13_real_dllm_gpu.yaml \
       --model <fixture_dir> --arm $ARM --seed 0 --device cpu \
       --dtype float32 --prompt-len 64 --gen-len 128 \
       --out results/ar018_smoke_$ARM
   done
   ```

   预期：四臂全部产出 `artifacts.pt` + `stats.json`；remasking 臂
   `mask_residuals == 0`；块式臂 `kv_store` 计数器
   `appended_tokens == prompt_len + gen_len`。随机权重无语义质量主张，
   冒烟只证明命令形态与代码路径。

2. **真实权重加载冒烟（服务器）**：

   ```bash
   python - <<'PY'
   from shadowinfer.models.diffusion_llama import load_diffusion_llama
   m = load_diffusion_llama("/data/weights/Dream-7B", dtype="auto", device="cuda")
   print(m.num_layers, m.config.hidden_size, m.mask_token_id)
   PY
   ```

   预期：输出与 §2.3 核实后的 PRESETS 一致。

## 5. 四臂运行命令

正式协议：每模型 × 四臂 × seed 0/1/2（共 24 次运行；性能结论取 3 seed 中位，
见 §9）。以 Dream-7B 为例（LLaDA-8B 换 `--model` 与输出前缀）：

```bash
cd <repo_root>
for SEED in 0 1 2; do
  for ARM in baseline mask-init draft-verify remasking; do
    python specs/archive/AR018-real-dllm-weights/serve_arm.py \
      --config configs/experiments/s13_real_dllm_gpu.yaml \
      --model /data/weights/Dream-7B \
      --arm $ARM --seed $SEED --device cuda \
      --out results/ar018_s13_dream_${ARM}_seed${SEED}
  done
done
```

- 参数口径（除非显式覆盖）：`prompt_len=256`、`gen_len=512`、`block_size=32`、
  `max_rounds=8`、`page_size=32`、`page_precision=fp32`、
  `unmask_per_pass=32`、`max_passes=32`、`strategy=confidence`
  （全部来自 `configs/experiments/s13_real_dllm_gpu.yaml`）。
- `--capture-block-logits` 默认关闭（真实词表下逐轮块 logits 达 GB 级）；
  S1-S3 只需 sequences 与 stats。需要逐轮 argmax 轨迹取证时单独开启。
- 每次运行的 wall time 由 runner 记录为 `wall_time_ms`（含 Python 开销）；
  块式三臂另有纯生成期 `stats.total_time_ms`——**S1 以 `total_time_ms` 为准**。

## 6. 取证要求

每次运行落两个文件，禁止手工转写数值（以文件为准）：

| 文件 | 内容 | 用途 |
|------|------|------|
| `stats.json` | arm/seed/model/dtype/model_dims/config/wall_time_ms/kv_store/stats | S1/S2 判定；17 键（块式三臂）或 4 键（remasking 臂）统计 |
| `artifacts.pt` | sequences + unmask_positions_per_round（+ 可选 block_logits） | S3 一致率；逐轮轨迹审计 |

补充取证（每次运行前后）：

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv \
  | tee results/ar018_s13_<arm>_seed<seed>_gpu.txt
```

显存峰值建议以 `torch.cuda.max_memory_allocated` 口径补测（可在复跑时加），
登记进 §8 模板。页统计：块式臂 `stats.json` 的 `kv_store` 计数器
（`appended_tokens`/`rewritten_tokens`/`page_evictions`/`rewritten_pages`，
按层）；预期 `appended_tokens == prompt_len + gen_len`、`rewritten_tokens == 0`
（s13 无淘汰配置）。

## 7. S1-S3 判定标准

| 断言 | 口径 | 判定 |
|------|------|------|
| **S1** | draft-verify vs mask-init 端到端加速比 | 3 seed 中位的 `total_time_ms(mask-init) / total_time_ms(draft-verify) >= 1.5`，同时报告 3 seed 区间 |
| **S2** | 每块轮数削减 | `rounds_per_block`（mask-init）vs `rounds_to_fixpoint`（draft-verify）分布对比：逐块中位轮数比 `>= 2`，报告分布（不是只报均值） |
| **S3** | draft-verify vs remasking token 一致率 | 响应区（去掉 prompt 的 gen_len 位）逐位一致率，3 seed 中位 `>= 0.90`；同时报告 mask-init vs remasking 作参考 |

S3 计算脚本（服务器直接可跑）：

```bash
python - <<'PY'
import torch, json, glob
def resp(path, prompt_len):
    seq = torch.load(path)["sequences"]
    return seq[0, prompt_len:]
def agree(a, b):
    return (a == b).float().mean().item()
P = json.load(open(glob.glob("results/ar018_s13_dream_mask-init_seed0/stats.json")[0]))["prompt_len"]
for seed in (0, 1, 2):
    dv = resp(f"results/ar018_s13_dream_draft-verify_seed{seed}/artifacts.pt", P)
    rm = resp(f"results/ar018_s13_dream_remasking_seed{seed}/artifacts.pt", P)
    mi = resp(f"results/ar018_s13_dream_mask-init_seed{seed}/artifacts.pt", P)
    print(seed, "dv-vs-remasking", agree(dv, rm), "maskinit-vs-remasking", agree(mi, rm))
PY
```

诚实注记：S1-S3 是**场地断言**——依赖真实权重的语义质量（随机权重下
draft-verify 与 mask-init 预期持平，AR017 bench-results.md §1/§3.5 已登记）。
实测不达标时如实登记进 §8 并在 README 标注，**禁止挑最好 seed 或改口径凑数**。

## 8. 结果落档

> 以下模板**待服务器实测后回填**；未回填前 README 基准表只允许引用本节
> 「未实测」状态。

### 8.1 环境登记（待回填）

| 项 | 值 |
|----|----|
| GPU 型号 / 显存 | 待回填 |
| torch / CUDA | 待回填 |
| dtype / 权重校验（§2.3 结论） | 待回填 |

### 8.2 S1（端到端加速比，Dream-7B；待回填）

| seed | mask-init total_time_ms | draft-verify total_time_ms | 比值 |
|------|--------------------------|-----------------------------|------|
| 0 | 待回填 | 待回填 | 待回填 |
| 1 | 待回填 | 待回填 | 待回填 |
| 2 | 待回填 | 待回填 | 待回填 |
| 中位 | - | - | 待回填（判定 >= 1.5） |

### 8.3 S2（轮数削减，Dream-7B；待回填）

| seed | mask-init 逐块轮数中位 | draft-verify 逐块轮数中位 | 比值 |
|------|--------------------------|-----------------------------|------|
| 0 | 待回填 | 待回填 | 待回填 |
| ... | 待回填 | 待回填 | 待回填 |

### 8.4 S3（token 一致率，Dream-7B；待回填）

| seed | dv vs remasking | mask-init vs remasking |
|------|------------------|------------------------|
| 0 | 待回填 | 待回填 |
| ... | 待回填 | 待回填 |

### 8.5 LLaDA-8B（次验收；待回填）

同 8.2-8.4 表结构。

### 8.6 显存与页统计（待回填）

| 臂 | 峰值显存 | appended/rewritten/evictions（每层） |
|----|----------|----------------------------------------|
| 待回填 | 待回填 | 待回填 |

## 9. 诚实纪律

1. **中位 + 区间**：性能结论一律 3 seed 中位并报告区间；单次运行不得定论。
2. **不达标如实标注**：S1-S3 任一未达标，§8 如实登记并在 README 标注失败
   项与数值；禁止沉默、禁止选择性报告。
3. **设备口径**：所有数字必须标明 GPU 型号与 dtype；本机 CPU 冒烟数字禁止
   出现在任何性能结论里。
4. **等价性口径**：块式臂是半 AR 近似；S3 一致率是「近似质量」度量，任何
   场合不得表述为「与 dLLM 原生生成的等价性验证」。
5. **回填纪律**：README 基准表的 s13 数字只能来自本节回填后的表格；回填前
   README 只允许出现「未实测，见 EXPERIMENTS.md」占位。
