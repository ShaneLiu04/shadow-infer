"""AR018 server runner: one arm of the s13 real-dLLM protocol.

Runs a single arm (baseline / mask-init / draft-verify / remasking) on a
REAL pretrained diffusion-LLM checkpoint (Dream-7B / LLaDA-8B HF format:
``config.json`` + safetensors, single or sharded) loaded via
``load_diffusion_llama`` and saves sequences, per-round stats and KV-page
counters to the output directory. Real forwards only; no mocks; the
CPU fixture smoke path runs the SAME code path with a tiny generated
checkpoint (``fixture_gen.py``), so the command shape proven locally is
the command shape used on the server (design D12).

Arms (design D14, four values):
  baseline      ``blockwise_generate_baseline``: full recompute per round
                under the block-causal mask (AR016 exactness reference)
  mask-init     ``blockwise_generate``: paged KV, draft_mode=False
  draft-verify  ``blockwise_generate``: paged KV, draft_mode=True
                (measured arm; page_precision=fp32 keeps S1/S2 a
                scheduling comparison, not a precision comparison)
  remasking     ``remasking_generate``: dLLM-native full bidirectional
                recompute + confidence unmasking (LLaDA/Dream official
                semantics; the true baseline arm)

Server-venue assertions (real pretrained weights only; full protocol in
``configs/experiments/s13_real_dllm_gpu.yaml`` and EXPERIMENTS.md):
  S1: draft-verify vs mask-init end-to-end >= 1.5x
  S2: draft-verify vs mask-init per-block round reduction >= 2x
  S3: draft-verify vs remasking token agreement >= 0.90

Usage (server, CUDA; bash):
  python specs/changes/AR018-real-dllm-weights/serve_arm.py \
    --config configs/experiments/s13_real_dllm_gpu.yaml \
    --model /data/weights/Dream-7B \
    --arm draft-verify --seed 0 --device cuda \
    --out results/ar018_s13_dv_seed0

Usage (local fixture smoke, CPU; proves the command shape end to end):
  python specs/changes/AR018-real-dllm-weights/serve_arm.py \
    --config configs/experiments/s13_real_dllm_gpu.yaml \
    --model <fixture_dir> --arm remasking --seed 0 --device cpu \
    --dtype float32 --prompt-len 64 --gen-len 128 \
    --out results/ar018_s13_smoke_remasking
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml

from shadowinfer.execution.block_decoding import BlockDecodingConfig, blockwise_generate_baseline
from shadowinfer.execution.blockwise_generate import blockwise_generate
from shadowinfer.execution.remasking import RemaskingConfig, remasking_generate
from shadowinfer.models.diffusion_llama import load_diffusion_llama

ARMS = ("baseline", "mask-init", "draft-verify", "remasking")


def _load_yaml(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_blockwise_config(
    exp_cfg: Dict[str, object],
    mode: str,
    mask_token_id: int,
) -> BlockDecodingConfig:
    execution = exp_cfg.get("execution", {})
    return BlockDecodingConfig(
        block_size=int(execution.get("block_size", 32)),
        max_rounds=int(execution.get("max_rounds", 8)),
        confidence_threshold=float(execution.get("confidence_threshold", 0.9)),
        mask_token_id=mask_token_id,
        page_size=int(execution.get("page_size", 32)),
        page_precision=str(execution.get("page_precision", "fp32")),
        draft_mode=(mode == "draft-verify"),
    )


def _build_remasking_config(
    exp_cfg: Dict[str, object],
    mask_token_id: int,
) -> RemaskingConfig:
    remasking = exp_cfg.get("remasking", {})
    return RemaskingConfig(
        mask_token_id=mask_token_id,
        unmask_per_pass=int(remasking.get("unmask_per_pass", 32)),
        max_passes=int(remasking.get("max_passes", 32)),
        strategy=str(remasking.get("strategy", "confidence")),
    )


def _resolve_mask_token_id(
    exp_cfg: Dict[str, object],
    model_mask_token_id: Optional[int],
) -> int:
    yaml_value = exp_cfg.get("execution", {}).get("mask_token_id")
    if yaml_value is not None:
        explicit = int(yaml_value)
        if model_mask_token_id is not None and explicit != model_mask_token_id:
            raise SystemExit(
                f"execution.mask_token_id ({explicit}) does not match the "
                f"checkpoint config.json mask_token_id "
                f"({model_mask_token_id}); refusing to run"
            )
        return explicit
    if model_mask_token_id is None:
        raise SystemExit(
            "checkpoint config.json carries no mask_token_id; set "
            "execution.mask_token_id in the experiment yaml explicitly"
        )
    return model_mask_token_id


def _make_prompt(
    vocab_size: int,
    prompt_len: int,
    mask_token_id: int,
    seed: int,
) -> torch.Tensor:
    prompt = torch.randint(
        1,
        vocab_size,
        (1, prompt_len),
        generator=torch.Generator().manual_seed(2026 + seed),
    )
    collision = prompt == mask_token_id
    if bool(collision.any()):
        replacement = (mask_token_id % (vocab_size - 1)) + 1
        prompt = prompt.masked_fill(collision, replacement)
    return prompt


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


def _kv_store_counters(stores: Dict[int, object]) -> Dict[str, Dict[str, int]]:
    counters: Dict[str, Dict[str, int]] = {}
    for layer in sorted(stores):
        store = stores[layer]
        counters[str(layer)] = {
            "appended_tokens": int(store.appended_tokens),
            "rewritten_tokens": int(store.rewritten_tokens),
            "page_evictions": int(store.page_evictions),
            "rewritten_pages": int(store.rewritten_pages),
        }
    return counters


def main() -> None:
    parser = argparse.ArgumentParser(description="AR018 s13 real-dLLM protocol single-arm runner")
    parser.add_argument("--config", required=True, help="s13 experiment yaml")
    parser.add_argument(
        "--model",
        default=None,
        help="checkpoint directory (config.json + safetensors); defaults to "
        "model_weights in the experiment yaml",
    )
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "auto"),
        default="auto",
        help="weight dtype: auto keeps on-disk dtype (bf16, GPU venue "
        "default); float32 casts (numerical anchor convention)",
    )
    parser.add_argument(
        "--prompt-len", type=int, default=None, help="override execution.prompt_len"
    )
    parser.add_argument("--gen-len", type=int, default=None, help="override execution.gen_len")
    parser.add_argument(
        "--capture-block-logits",
        action="store_true",
        help="store per-round block logits in artifacts.pt (large at real "
        "vocab sizes; off by default -- S1-S3 only need sequences/stats)",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    exp_cfg = _load_yaml(args.config)
    model_path = args.model or exp_cfg.get("model_weights")
    if not model_path:
        raise SystemExit(
            "no checkpoint directory: pass --model or set model_weights "
            "in the experiment yaml (this protocol has no random-weight "
            "fallback)"
        )

    model = load_diffusion_llama(model_path, dtype=args.dtype, device=args.device)
    model.eval()

    execution = exp_cfg.get("execution", {})
    prompt_len = int(args.prompt_len or execution.get("prompt_len", 256))
    gen_len = int(args.gen_len or execution.get("gen_len", 512))
    if prompt_len + gen_len > model.max_seq_len:
        raise SystemExit(
            f"prompt_len({prompt_len}) + gen_len({gen_len}) exceeds "
            f"max_seq_len({model.max_seq_len})"
        )
    mask_token_id = _resolve_mask_token_id(exp_cfg, model.mask_token_id)

    torch.manual_seed(args.seed)
    prompt = _make_prompt(int(model.config.vocab_size), prompt_len, mask_token_id, args.seed)

    capture = bool(args.capture_block_logits)
    wall_start = time.perf_counter()
    kv_counters: Optional[Dict[str, Dict[str, int]]] = None
    if args.arm == "remasking":
        remasking_config = _build_remasking_config(exp_cfg, mask_token_id)
        generator = None
        if remasking_config.strategy == "random":
            generator = torch.Generator().manual_seed(2026 + args.seed)
        with torch.inference_mode():
            result = remasking_generate(
                model, prompt, gen_len, remasking_config, generator=generator
            )
        arm_config: Dict[str, object] = {
            "mask_token_id": remasking_config.mask_token_id,
            "unmask_per_pass": remasking_config.unmask_per_pass,
            "max_passes": remasking_config.max_passes,
            "strategy": remasking_config.strategy,
        }
    else:
        blockwise_config = _build_blockwise_config(
            exp_cfg, "draft-verify" if args.arm == "draft-verify" else "", mask_token_id
        )
        if (
            args.arm in ("mask-init", "draft-verify")
            and gen_len % int(blockwise_config.block_size) != 0
        ):
            raise SystemExit(
                f"gen_len({gen_len}) must be a multiple of "
                f"block_size({blockwise_config.block_size}) for arm "
                f"{args.arm!r}"
            )
        with torch.inference_mode():
            if args.arm == "baseline":
                result = blockwise_generate_baseline(
                    model,
                    prompt,
                    gen_len,
                    blockwise_config,
                    capture_block_logits=capture,
                )
            else:
                result, stores = blockwise_generate(
                    model,
                    prompt,
                    gen_len,
                    blockwise_config,
                    capture_block_logits=capture,
                )
                kv_counters = _kv_store_counters(stores)
        arm_config = {
            "block_size": blockwise_config.block_size,
            "max_rounds": blockwise_config.max_rounds,
            "confidence_threshold": blockwise_config.confidence_threshold,
            "mask_token_id": blockwise_config.mask_token_id,
            "page_size": blockwise_config.page_size,
            "page_precision": blockwise_config.page_precision,
            "draft_mode": blockwise_config.draft_mode,
        }
    wall_time_ms = (time.perf_counter() - wall_start) * 1000.0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sequences": result.sequences,
            "block_logits_per_round": result.stats.get("block_logits_per_round"),
            "unmask_positions_per_round": result.stats.get("unmask_positions_per_round"),
        },
        out_dir / "artifacts.pt",
    )
    with open(out_dir / "stats.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "arm": args.arm,
                "seed": args.seed,
                "model": str(model_path),
                "dtype": args.dtype,
                "device": args.device,
                "prompt_len": prompt_len,
                "gen_len": gen_len,
                "model_dims": {
                    "num_layers": int(model.num_layers),
                    "hidden_dim": int(model.config.hidden_size),
                    "num_heads": int(model.num_heads),
                    "num_kv_heads": int(model.num_kv_heads),
                    "head_dim": int(model.head_dim),
                    "vocab_size": int(model.vocab_size),
                    "max_seq_len": int(model.max_seq_len),
                    "mask_token_id": int(mask_token_id),
                },
                "config": arm_config,
                "capture_block_logits": capture,
                "wall_time_ms": round(wall_time_ms, 3),
                "kv_store": kv_counters,
                "stats": _stats_to_json(result.stats),
            },
            handle,
            indent=2,
        )
    print(f"[ar018-s13] arm={args.arm} seed={args.seed} model={model_path}" f" -> {out_dir}")
    total_time_ms = result.stats.get("total_time_ms")
    if total_time_ms is not None:
        print(f"[ar018-s13] total_time_ms={total_time_ms:.1f}")
    print(f"[ar018-s13] wall_time_ms={wall_time_ms:.1f}")


if __name__ == "__main__":
    main()
