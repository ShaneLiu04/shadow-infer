"""AR016 server runner: one arm of the s11 blockwise-decoding protocol.

Runs a single arm (baseline / fp32 / fp16 / int8) on the 7B-dimension
reference model (SimpleDiffusionLLM at model-config dimensions) and saves
sequences, per-round logits and stats to the output directory. Real
forwards only; no mocks. Used by the GPU reproduction protocol in
server-experiment.md (section 4); the CPU toy-model evidence lives in
bench-results.md.

Usage (server, CUDA):
  python specs/archive/AR016-blockwise-parallel-decoding/serve_arm.py \
    --config configs/experiments/s11_blockwise_decoding_gpu.yaml \
    --model-config configs/experiments/model_7b_longctx.yaml \
    --arm fp32 --seed 0 --device cuda --out results/ar016_s11_fp32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml

from shadowinfer.core.backends.pytorch_backend import PyTorchModelBackend
from shadowinfer.execution.block_decoding import (
    BlockDecodingConfig,
    blockwise_generate_baseline,
)

ARMS = ("baseline", "fp32", "fp16", "int8")


def _load_yaml(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _model_dimensions(model_cfg_path: str) -> Dict[str, int]:
    cfg = _load_yaml(model_cfg_path)
    arch = cfg["architecture"]
    dims = {
        "num_layers": int(arch["num_layers"]),
        "num_heads": int(arch["num_heads"]),
        "head_dim": int(arch["head_dim"]),
        "hidden_dim": int(arch["hidden_dim"]),
        "vocab_size": int(arch["vocab_size"]),
        "max_seq_len": int(arch["max_seq_len"]),
    }
    return dims


def _build_config(exp_cfg: Dict[str, object], precision: str) -> BlockDecodingConfig:
    execution = exp_cfg.get("execution", {})
    return BlockDecodingConfig(
        block_size=int(execution.get("block_size", 32)),
        max_rounds=int(execution.get("max_rounds", 8)),
        confidence_threshold=float(execution.get("confidence_threshold", 0.9)),
        mask_token_id=int(execution.get("mask_token_id", 0)),
        page_size=int(execution.get("page_size", 32)),
        page_precision=precision,
    )


def _stats_to_json(stats: Dict[str, object]) -> Dict[str, object]:
    serializable: Dict[str, object] = {}
    for key, value in stats.items():
        if isinstance(value, torch.Tensor):
            continue
        if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            continue
        if isinstance(value, list) and value and isinstance(value[0], list):
            if value[0] and isinstance(value[0][0], torch.Tensor):
                continue
            serializable[key] = value
            continue
        serializable[key] = value
    return serializable


def main() -> None:
    parser = argparse.ArgumentParser(description="AR016 s11 protocol single-arm runner")
    parser.add_argument("--config", required=True, help="s11 experiment yaml")
    parser.add_argument("--model-config", required=True, help="model dims yaml")
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--prompt-len", type=int, default=None, help="override execution.prompt_len"
    )
    parser.add_argument(
        "--gen-len", type=int, default=None, help="override execution.gen_len"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    exp_cfg = _load_yaml(args.config)
    dims = _model_dimensions(args.model_config)
    execution = exp_cfg.get("execution", {})
    prompt_len = int(args.prompt_len or execution.get("prompt_len", 256))
    gen_len = int(args.gen_len or execution.get("gen_len", 512))
    if prompt_len + gen_len > dims["max_seq_len"]:
        raise SystemExit(
            f"prompt_len({prompt_len}) + gen_len({gen_len}) exceeds "
            f"max_seq_len({dims['max_seq_len']})"
        )

    torch.manual_seed(args.seed)
    prompt = torch.randint(
        1, dims["vocab_size"], (1, prompt_len), generator=torch.Generator().manual_seed(2026 + args.seed)
    )

    backend = PyTorchModelBackend(model_config=dims, seed=args.seed)
    backend.load("ar016-s11", device=args.device)
    config = _build_config(exp_cfg, "fp32" if args.arm == "baseline" else args.arm)

    if args.arm == "baseline":
        model = backend._model
        result = blockwise_generate_baseline(
            model, prompt, gen_len, config, capture_block_logits=True
        )
    else:
        result = backend.generate(
            prompt, gen_len, config, capture_block_logits=True
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sequences": result.sequences,
            "block_logits_per_round": result.stats.get("block_logits_per_round"),
            "unmask_positions_per_round": result.stats.get(
                "unmask_positions_per_round"
            ),
        },
        out_dir / "artifacts.pt",
    )
    with open(out_dir / "stats.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "arm": args.arm,
                "seed": args.seed,
                "prompt_len": prompt_len,
                "gen_len": gen_len,
                "config": {
                    "block_size": config.block_size,
                    "max_rounds": config.max_rounds,
                    "confidence_threshold": config.confidence_threshold,
                    "page_size": config.page_size,
                    "page_precision": config.page_precision,
                },
                "stats": _stats_to_json(result.stats),
            },
            handle,
            indent=2,
        )
    print(f"[ar016-s11] arm={args.arm} seed={args.seed} -> {out_dir}")
    print(f"[ar016-s11] total_time_ms={result.stats['total_time_ms']:.1f}")


if __name__ == "__main__":
    main()
