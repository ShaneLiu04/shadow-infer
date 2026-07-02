"""ShadowInfer 精度计算工具模块。

对应 PROFILER_AGENT.md 中的 accuracy 维度。
"""

__version__ = "3.0"

import math
from typing import Dict, List

import torch
import torch.nn.functional as F


class Metrics:
    """精度计算工具。对应 PROFILER_AGENT.md 中的 accuracy 维度。"""

    @staticmethod
    def compute_perplexity(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        计算 perplexity。

        公式：PPL = exp( -mean(log_prob[labels]) )

        Args:
            logits: [batch, seq_len, vocab_size] 或 [seq_len, vocab_size]。
            labels: [batch, seq_len] 或 [seq_len]，目标 token ids。

        Returns:
            perplexity 值（float）。
        """
        if logits.numel() == 0 or labels.numel() == 0:
            return 0.0

        # 确保 logits 和 labels 形状匹配
        if logits.dim() == 3 and labels.dim() == 2:
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            labels_flat = labels.reshape(-1)
        elif logits.dim() == 2 and labels.dim() == 1:
            logits_flat = logits
            labels_flat = labels
        else:
            raise ValueError(f"Shape mismatch: logits {logits.shape}, labels {labels.shape}")

        # 忽略 pad token（假设 pad_id = -100）
        mask = labels_flat != -100
        if mask.sum() == 0:
            return 0.0

        log_probs = F.log_softmax(logits_flat, dim=-1)
        nll = F.nll_loss(
            log_probs,
            labels_flat,
            reduction="sum",
            ignore_index=-100,
        )
        avg_nll = nll / mask.sum().item()
        perplexity = math.exp(avg_nll)
        return float(perplexity)

    @staticmethod
    def compute_bleu_score(
        reference: str,
        candidate: str,
    ) -> float:
        """
        计算 BLEU 分数（简化版，使用 n-gram overlap）。

        只计算 1-gram 和 2-gram 的精确匹配，使用 brevity penalty。

        Args:
            reference: 参考文本。
            candidate: 生成文本。

        Returns:
            BLEU 分数（0.0 - 1.0）。
        """
        if not reference or not candidate:
            return 0.0

        ref_tokens = reference.split()
        cand_tokens = candidate.split()

        ref_len = len(ref_tokens)
        cand_len = len(cand_tokens)

        if cand_len == 0:
            return 0.0

        # Brevity penalty
        if cand_len < ref_len:
            bp = math.exp(1 - ref_len / cand_len)
        else:
            bp = 1.0

        # 1-gram 精确匹配
        ref_1grams = set(ref_tokens)
        cand_1grams = set(cand_tokens)
        match_1 = len(ref_1grams & cand_1grams)
        precision_1 = match_1 / len(cand_1grams) if cand_1grams else 0.0

        # 2-gram 精确匹配
        ref_2grams = set(tuple(ref_tokens[i : i + 2]) for i in range(len(ref_tokens) - 1))
        cand_2grams = set(tuple(cand_tokens[i : i + 2]) for i in range(len(cand_tokens) - 1))
        match_2 = len(ref_2grams & cand_2grams)
        precision_2 = match_2 / len(cand_2grams) if cand_2grams else 0.0

        # 几何平均
        if precision_1 > 0 and precision_2 > 0:
            geo_mean = math.exp(0.5 * (math.log(precision_1) + math.log(precision_2)))
        else:
            geo_mean = 0.0

        bleu = bp * geo_mean
        return float(bleu)

    @staticmethod
    def compute_relative_error(
        baseline: torch.Tensor,
        optimized: torch.Tensor,
    ) -> float:
        """
        计算相对误差。

        公式：||baseline - optimized|| / ||baseline||

        Args:
            baseline: 基线张量。
            optimized: 优化后张量。

        Returns:
            相对误差（float）。
        """
        if baseline.numel() == 0:
            return 0.0

        baseline_norm = torch.norm(baseline)
        if baseline_norm.item() == 0:
            return 0.0

        diff_norm = torch.norm(baseline - optimized)
        return float(diff_norm / baseline_norm)

    @staticmethod
    def compute_cosine_similarity(
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> float:
        """
        计算余弦相似度。

        Args:
            a: 张量 A。
            b: 张量 B。

        Returns:
            余弦相似度（float），范围 [-1, 1]。
        """
        if a.numel() == 0 or b.numel() == 0:
            return 0.0

        a_flat = a.flatten()
        b_flat = b.flatten()

        a_norm = torch.norm(a_flat)
        b_norm = torch.norm(b_flat)

        if a_norm.item() == 0 or b_norm.item() == 0:
            return 0.0

        cos_sim = torch.dot(a_flat, b_flat) / (a_norm * b_norm)
        return float(cos_sim)

    @staticmethod
    def compute_kl_divergence(
        p: torch.Tensor,
        q: torch.Tensor,
    ) -> float:
        """
        计算 KL 散度。

        公式：KL(p || q) = sum(p * log(p / q))

        Args:
            p: 目标分布。
            q: 近似分布。

        Returns:
            KL 散度值（float，非负）。
        """
        if p.numel() == 0 or q.numel() == 0:
            return 0.0

        # 展平并确保非负
        p_flat = p.flatten().clamp(min=1e-10)
        q_flat = q.flatten().clamp(min=1e-10)

        # 归一化
        p_flat = p_flat / p_flat.sum()
        q_flat = q_flat / q_flat.sum()

        kl = torch.sum(p_flat * torch.log(p_flat / q_flat))
        return float(kl)

    @staticmethod
    def compute_entropy(
        distribution: torch.Tensor,
    ) -> float:
        """
        计算 entropy。

        公式：H = -sum(p * log(p))

        Args:
            distribution: 概率分布张量。

        Returns:
            entropy 值（float，非负）。
        """
        if distribution.numel() == 0:
            return 0.0

        p = distribution.flatten().clamp(min=1e-10)
        p = p / p.sum()

        entropy = -torch.sum(p * torch.log(p))
        return float(entropy)

    @staticmethod
    def compute_latency_stats(
        latencies_ms: List[float],
    ) -> Dict[str, float]:
        """
        计算延迟统计。

        返回：mean, median, p95, p99, min, max, std

        Args:
            latencies_ms: 延迟列表，单位毫秒。

        Returns:
            统计字典。
        """
        if not latencies_ms:
            return {
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
            }

        import statistics

        sorted_latencies = sorted(latencies_ms)
        n = len(sorted_latencies)

        mean = statistics.mean(latencies_ms)
        median = statistics.median(latencies_ms)
        p95_idx = int(math.ceil(0.95 * n)) - 1
        p99_idx = int(math.ceil(0.99 * n)) - 1
        p95 = sorted_latencies[max(0, p95_idx)]
        p99 = sorted_latencies[max(0, p99_idx)]
        min_val = min(latencies_ms)
        max_val = max(latencies_ms)
        std = statistics.stdev(latencies_ms) if n > 1 else 0.0

        return {
            "mean": float(mean),
            "median": float(median),
            "p95": float(p95),
            "p99": float(p99),
            "min": float(min_val),
            "max": float(max_val),
            "std": float(std),
        }

    @staticmethod
    def compute_compression_ratio(
        original_bytes: int,
        compressed_bytes: int,
    ) -> float:
        """
        计算压缩比。

        公式：1 - compressed / original

        Args:
            original_bytes: 原始字节数。
            compressed_bytes: 压缩后字节数。

        Returns:
            压缩比（float），0.0 表示无压缩，接近 1.0 表示高压缩。
        """
        if original_bytes <= 0:
            return 0.0

        ratio = 1.0 - compressed_bytes / original_bytes
        return float(ratio)

    @staticmethod
    def compute_flops_macs(
        input_dim: int,
        output_dim: int,
        batch_size: int,
        seq_len: int,
    ) -> int:
        """
        计算 FLOPs / MACs for linear layer。

        公式：MACs = batch_size * seq_len * input_dim * output_dim
        FLOPs ≈ 2 * MACs（乘加各算一次）

        本函数返回 MACs（乘加次数）。

        Args:
            input_dim: 输入维度。
            output_dim: 输出维度。
            batch_size: 批次大小。
            seq_len: 序列长度。

        Returns:
            MACs 次数（int）。
        """
        if input_dim <= 0 or output_dim <= 0 or batch_size <= 0 or seq_len <= 0:
            return 0
        return batch_size * seq_len * input_dim * output_dim

    @staticmethod
    def compute_accuracy_drop(
        baseline_metric: float,
        optimized_metric: float,
    ) -> float:
        """
        计算 accuracy drop（注意方向和符号）。

        公式：drop = baseline - optimized
        正值表示优化后精度下降，负值表示精度提升。

        Args:
            baseline_metric: 基线精度指标（如 accuracy, BLEU 等）。
            optimized_metric: 优化后精度指标。

        Returns:
            accuracy drop（float）。
        """
        return float(baseline_metric - optimized_metric)
