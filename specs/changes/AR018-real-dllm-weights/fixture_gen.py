"""AR018 fixture checkpoint generator (test infrastructure, FR18.5/T001).

Generates tiny real-format diffusion-LLM checkpoints (``config.json`` +
safetensors weights, single-file or sharded-with-index layouts) that
exercise the exact loading code path used for real Dream-7B / LLaDA-8B
weights. Zero-mock discipline: every artifact is a real HF-format file
on disk written and read through real IO; only the dimensions are
shrunk so tests stay fast on CPU.

The weight naming follows the Llama/Qwen convention shared by both
target checkpoints (design.md D9):

    model.embed_tokens.weight
    model.layers.N.self_attn.{q,k,v,o}_proj.weight   (+q/k/v bias when
    model.layers.N.mlp.{gate,up,down}_proj.weight     attention_bias)
    model.layers.N.{input,post_attention}_layernorm.weight
    model.norm.weight
    lm_head.weight                                    (unless tied)

PRESET_TEMPLATES records the literature-scope architecture parameters
for the two real targets (design.md 4.2.2 table). These values are a
documentation and assertion object only; production code stays fully
config-driven. They must be verified against the real ``config.json``
files on the GPU server (s13); any mismatch is recorded as an erratum
in design.md (see srs.md section 5 constraint).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from safetensors.torch import save_file

# Literature-scope architecture parameters (2026-10-24, UNVERIFIED on
# this machine: the corporate gateway blocks huggingface.co / arxiv /
# github, so these follow the published initialization-lineage claims
# documented in design.md 4.2.2). Verify against the real config.json
# on the GPU server before s13; record any mismatch as a design erratum.
PRESET_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "dream-7b": {
        "model_type": "dream",
        "num_hidden_layers": 28,
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 18944,
        "vocab_size": 151936,
        "mask_token_id": 151669,
        "max_position_embeddings": 131072,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "attention_bias": True,
        "tie_word_embeddings": False,
    },
    "llada-8b": {
        "model_type": "llada",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 14336,
        "vocab_size": 126464,
        "mask_token_id": 126336,
        "max_position_embeddings": 8192,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "attention_bias": False,
        "tie_word_embeddings": False,
    },
}

_DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def _expected_weight_names(
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    intermediate_size: int,
    vocab_size: int,
    attention_bias: bool,
    tie_word_embeddings: bool,
) -> Dict[str, List[int]]:
    """Build the ordered {weight_name: shape} map for a checkpoint."""
    qkv_out = num_heads * head_dim
    kv_out = num_kv_heads * head_dim
    names: Dict[str, List[int]] = {
        "model.embed_tokens.weight": [vocab_size, hidden_size],
    }
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"
        names[f"{prefix}.self_attn.q_proj.weight"] = [qkv_out, hidden_size]
        names[f"{prefix}.self_attn.k_proj.weight"] = [kv_out, hidden_size]
        names[f"{prefix}.self_attn.v_proj.weight"] = [kv_out, hidden_size]
        names[f"{prefix}.self_attn.o_proj.weight"] = [qkv_out, qkv_out]
        if attention_bias:
            names[f"{prefix}.self_attn.q_proj.bias"] = [qkv_out]
            names[f"{prefix}.self_attn.k_proj.bias"] = [kv_out]
            names[f"{prefix}.self_attn.v_proj.bias"] = [kv_out]
        names[f"{prefix}.mlp.gate_proj.weight"] = [intermediate_size, hidden_size]
        names[f"{prefix}.mlp.up_proj.weight"] = [intermediate_size, hidden_size]
        names[f"{prefix}.mlp.down_proj.weight"] = [hidden_size, intermediate_size]
        names[f"{prefix}.input_layernorm.weight"] = [hidden_size]
        names[f"{prefix}.post_attention_layernorm.weight"] = [hidden_size]
    names["model.norm.weight"] = [hidden_size]
    if not tie_word_embeddings:
        names["lm_head.weight"] = [vocab_size, hidden_size]
    return names


def _validate_params(params: Dict[str, Any]) -> None:
    """Fail-fast on structurally impossible fixture parameters."""
    required = [
        "num_layers",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "vocab_size",
        "mask_token_id",
        "max_position_embeddings",
        "intermediate_size",
    ]
    for key in required:
        if int(params[key]) < 1:
            raise ValueError(f"{key} must be >= 1, got {params[key]}")
    if int(params["num_kv_heads"]) > int(params["num_heads"]):
        raise ValueError(
            f"num_kv_heads ({params['num_kv_heads']}) must not exceed "
            f"num_heads ({params['num_heads']})"
        )
    if int(params["num_heads"]) % int(params["num_kv_heads"]) != 0:
        raise ValueError(
            f"num_heads ({params['num_heads']}) must be a multiple of "
            f"num_kv_heads ({params['num_kv_heads']}) for GQA repeat"
        )
    hidden = int(params["num_heads"]) * int(params["head_dim"])
    if hidden != int(params["hidden_size"]):
        raise ValueError(
            f"hidden_size ({params['hidden_size']}) must equal num_heads * "
            f"head_dim ({int(params['num_heads'])} * {int(params['head_dim'])} "
            f"= {hidden}) for the AR018 loader contract"
        )
    mask_token_id = int(params["mask_token_id"])
    vocab_size = int(params["vocab_size"])
    if not 0 <= mask_token_id < vocab_size:
        raise ValueError(
            f"mask_token_id ({mask_token_id}) must be within "
            f"[0, vocab_size={vocab_size})"
        )
    if int(params["shard_size"]) < 1:
        raise ValueError(f"shard_size must be >= 1, got {params['shard_size']}")


def generate_fixture(
    out_dir: str | Path,
    *,
    style: str,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    vocab_size: int,
    mask_token_id: int,
    max_position_embeddings: int,
    rope_theta: float = 10000.0,
    rms_norm_eps: float = 1e-5,
    intermediate_size: int | None = None,
    attention_bias: bool = False,
    tie_word_embeddings: bool = False,
    sharded: bool = False,
    shard_size: int = 8,
    dtype: str = "float32",
    seed: int = 0,
) -> Dict[str, Any]:
    """Generate a tiny real-format checkpoint directory.

    Args:
        out_dir: Destination directory (created if missing).
        style: "llama" or "qwen" -- controls config.json field set and
            model_type (both styles share the same weight naming, D9).
        num_layers/num_heads/num_kv_heads/head_dim/hidden_size: Core
            transformer dimensions (hidden == num_heads * head_dim).
        vocab_size/mask_token_id: Vocabulary extent and the diffusion
            mask token id.
        max_position_embeddings: RoPE extent; maps to model max_seq_len.
        rope_theta/rms_norm_eps: Numeric configuration passthrough.
        intermediate_size: SwiGLU FFN width; defaults to 4 * hidden.
        attention_bias: When True, q/k/v projections carry bias weights
            (Qwen style); o_proj never carries bias.
        tie_word_embeddings: When True, lm_head is omitted and tied to
            the embedding at load time.
        sharded: When True, write ``model.safetensors.index.json`` plus
            sharded ``model-XXXXX-of-XXXXX.safetensors`` files; when
            False, write a single ``model.safetensors``.
        shard_size: Maximum number of tensors per shard.
        dtype: "float32" or "bfloat16" storage dtype.
        seed: Torch seed; identical seeds produce bitwise-identical
            checkpoints.

    Returns:
        Manifest dict with the resolved parameters, written file paths,
        and the weight-name list.

    Raises:
        ValueError: On structurally impossible parameters (kv heads,
            hidden mismatch, mask token out of range, bad dtype/style).
    """
    if style not in ("llama", "qwen"):
        raise ValueError(f"style must be 'llama' or 'qwen', got {style!r}")
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"dtype must be one of {sorted(_DTYPE_MAP)}, got {dtype!r}")
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size

    params = {
        "style": style,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "mask_token_id": mask_token_id,
        "max_position_embeddings": max_position_embeddings,
        "rope_theta": float(rope_theta),
        "rms_norm_eps": float(rms_norm_eps),
        "intermediate_size": intermediate_size,
        "attention_bias": bool(attention_bias),
        "tie_word_embeddings": bool(tie_word_embeddings),
        "sharded": bool(sharded),
        "shard_size": shard_size,
        "dtype": dtype,
        "seed": seed,
    }
    _validate_params(params)

    shapes = _expected_weight_names(
        num_layers,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        attention_bias,
        tie_word_embeddings,
    )

    # Deterministic generation: fixed seed, fixed (sorted) key order.
    torch.manual_seed(seed)
    storage_dtype = _DTYPE_MAP[dtype]
    tensors: Dict[str, torch.Tensor] = {}
    for name in sorted(shapes):
        shape = shapes[name]
        if "layernorm" in name or name == "model.norm.weight":
            tensor = 1.0 + 0.01 * torch.randn(shape)
        else:
            tensor = 0.02 * torch.randn(shape)
        tensors[name] = tensor.to(storage_dtype)

    config: Dict[str, Any] = {
        "architectures": ["DiffusionLlamaForCausalLM"],
        "model_type": "llada" if style == "llama" else "dream",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": max_position_embeddings,
        "rope_theta": float(rope_theta),
        "rms_norm_eps": float(rms_norm_eps),
        "tie_word_embeddings": bool(tie_word_embeddings),
        "mask_token_id": mask_token_id,
        "torch_dtype": dtype,
    }
    # Qwen-style configs carry attention_bias explicitly; Llama-style
    # configs omit the field (loader defaults it to False).
    if style == "qwen":
        config["attention_bias"] = bool(attention_bias)
    elif attention_bias:
        raise ValueError(
            "attention_bias=True requires style='qwen' (Llama-style "
            "checkpoints never carry qkv biases)"
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    written: List[str] = []
    if sharded:
        names = sorted(tensors)
        shards: List[Dict[str, torch.Tensor]] = [{}]
        for name in names:
            if len(shards[-1]) >= shard_size:
                shards.append({})
            shards[-1][name] = tensors[name]
        total = len(shards)
        weight_map: Dict[str, str] = {}
        total_size = 0
        for index, shard in enumerate(shards, start=1):
            filename = f"model-{index:05d}-of-{total:05d}.safetensors"
            save_file(shard, str(out_path / filename))
            written.append(filename)
            for name, tensor in shard.items():
                weight_map[name] = filename
                total_size += tensor.numel() * tensor.element_size()
        index_doc = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        index_name = "model.safetensors.index.json"
        (out_path / index_name).write_text(
            json.dumps(index_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written.append(index_name)
    else:
        save_file(tensors, str(out_path / "model.safetensors"))
        written.append("model.safetensors")

    return {
        "params": params,
        "config": config,
        "weight_names": sorted(tensors),
        "files": written,
        "out_dir": str(out_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a tiny real-format diffusion-LLM checkpoint."
    )
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--style", choices=["llama", "qwen"], default="qwen",
        help="Config style (weight naming is shared, design D9)",
    )
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--vocab", type=int, default=1000)
    parser.add_argument(
        "--mask-token", type=int, default=999, help="Diffusion mask token id"
    )
    parser.add_argument("--max-position-embeddings", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-5)
    parser.add_argument(
        "--attention-bias", action="store_true",
        help="Qwen-style qkv biases (requires --style qwen)",
    )
    parser.add_argument("--tie-embeddings", action="store_true")
    parser.add_argument("--sharded", action="store_true")
    parser.add_argument("--shard-size", type=int, default=8)
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16"], default="float32"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--preset", choices=sorted(PRESET_TEMPLATES), default=None,
        help="Fill style-specific numeric fields (rope_theta, rms_norm_eps, "
        "attention_bias, tie_word_embeddings) from a literature-scope "
        "template; dimensions stay tiny unless overridden explicitly",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.preset is not None:
        template = PRESET_TEMPLATES[args.preset]
        args.style = "qwen" if args.preset == "dream-7b" else "llama"
        # Preset fills style-specific numeric fields; explicit CLI flags
        # for dimensions stay authoritative (fixtures stay tiny).
        args.rope_theta = template["rope_theta"]
        args.rms_norm_eps = template["rms_norm_eps"]
        args.attention_bias = template["attention_bias"]
        args.tie_embeddings = template["tie_word_embeddings"]
    manifest = generate_fixture(
        args.out,
        style=args.style,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        hidden_size=args.heads * args.head_dim,
        vocab_size=args.vocab,
        mask_token_id=args.mask_token,
        max_position_embeddings=args.max_position_embeddings,
        intermediate_size=args.intermediate,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.rms_norm_eps,
        attention_bias=args.attention_bias,
        tie_word_embeddings=args.tie_embeddings,
        sharded=args.sharded,
        shard_size=args.shard_size,
        dtype=args.dtype,
        seed=args.seed,
    )
    print(json.dumps({"out_dir": manifest["out_dir"], "files": manifest["files"]}, indent=2))


if __name__ == "__main__":
    main()
