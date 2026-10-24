"""Tests for the real-checkpoint diffusion LLaMA loader (AR018 T002, FR18.1).

Red-phase contract tests for ``shadowinfer/models/diffusion_llama.py``
(``load_diffusion_llama`` / ``DiffusionLlamaConfig`` / ``PRESETS``), plus
the P5 fixture/preset partition (FR18.5 test surface).

Zero-mock discipline: every checkpoint is a real HF-format directory on
disk produced by ``specs/changes/AR018-real-dllm-weights/fixture_gen.py``
(single-file and sharded-with-index layouts), and every fault is injected
through real file IO (delete file / rewrite json / rewrite safetensors).
No production module exists yet; this file is expected to fail collection
until the Green-phase implementation lands.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest
import torch
from safetensors.torch import load_file, save_file

from shadowinfer.execution.block_decoding import segment_causal_mask
from shadowinfer.models.diffusion_llama import (
    PRESETS,
    DiffusionLlama,
    DiffusionLlamaConfig,
    load_diffusion_llama,
)

# Load the AR018 fixture generator (test infrastructure, T001) directly
# from the spec workspace without touching sys.path.
_FIXTURE_GEN_PATH = (
    Path(__file__).resolve().parents[1]
    / "specs"
    / "changes"
    / "AR018-real-dllm-weights"
    / "fixture_gen.py"
)
_spec = importlib.util.spec_from_file_location("ar018_fixture_gen", _FIXTURE_GEN_PATH)
fixture_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture_gen)

# Tiny dimensions (fast on CPU): 21 tensors single-layout, 3 shards when
# sharded with shard_size=8.
_TINY: Dict[str, Any] = {
    "style": "llama",
    "num_layers": 2,
    "num_heads": 8,
    "num_kv_heads": 2,
    "head_dim": 32,
    "hidden_size": 256,
    "vocab_size": 1000,
    "mask_token_id": 999,
    "max_position_embeddings": 512,
}

# A regex that any HF-convention weight name satisfies, but which plain
# file names like "model.safetensors" do not. Used for shape-mismatch
# messages where the design only requires "contains a tensor name".
_TENSOR_NAME_PATTERN = (
    r"(model\.layers\.\d+\.|model\.embed_tokens\.weight|" r"model\.norm\.weight|lm_head\.weight)"
)


def _gen_fixture(out_dir: Path, **overrides: Any) -> Dict[str, Any]:
    """Generate a tiny real-format fixture into ``out_dir``."""
    params = dict(_TINY)
    params.update(overrides)
    return fixture_gen.generate_fixture(str(out_dir), **params)


def _reference_tensors(fixture_dir: str | Path, sharded: bool) -> Dict[str, torch.Tensor]:
    """Read the raw on-disk checkpoint tensors back as the reference set."""
    fixture_dir = Path(fixture_dir)
    merged: Dict[str, torch.Tensor] = {}
    if sharded:
        index = json.loads(
            (fixture_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        for filename in sorted(set(index["weight_map"].values())):
            merged.update(load_file(str(fixture_dir / filename)))
    else:
        merged.update(load_file(str(fixture_dir / "model.safetensors")))
    return merged


def _assert_params_bitwise_match_reference(
    model: torch.nn.Module, reference: Dict[str, torch.Tensor]
) -> None:
    """Assert the model's exposed parameter multiset is bitwise identical
    to the on-disk reference tensor multiset (count, total numel, and
    per-tensor torch.equal after multiset matching)."""
    params = [p.detach().clone() for _, p in model.named_parameters()]
    refs = list(reference.values())

    assert len(params) == len(refs), (
        f"parameter count {len(params)} != reference tensor count {len(refs)}; "
        "tied lm_head must share the embedding parameter (deduplicated by "
        "named_parameters), and no extra non-checkpoint parameters may be "
        "registered"
    )
    assert sum(p.numel() for p in params) == sum(t.numel() for t in refs)

    remaining = list(params)
    for ref in refs:
        for idx, candidate in enumerate(remaining):
            if (
                candidate.shape == ref.shape
                and candidate.dtype == ref.dtype
                and torch.equal(candidate, ref)
            ):
                remaining.pop(idx)
                break
        else:
            pytest.fail(
                "no model parameter is bitwise-equal to reference tensor "
                f"of shape {tuple(ref.shape)} / dtype {ref.dtype}"
            )
    assert not remaining


def _edit_config(fixture_dir: str | Path, mutate) -> None:
    """Rewrite config.json after applying ``mutate(config_dict)``."""
    path = Path(fixture_dir) / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    mutate(config)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_index(fixture_dir: str | Path) -> Dict[str, Any]:
    return json.loads(
        (Path(fixture_dir) / "model.safetensors.index.json").read_text(encoding="utf-8")
    )


def _edit_index(fixture_dir: str | Path, mutate) -> None:
    """Rewrite model.safetensors.index.json after ``mutate(index_dict)``."""
    path = Path(fixture_dir) / "model.safetensors.index.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    mutate(index)
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class TestLoaderSingleAndSharded:
    """Happy-path loading of single-file and sharded-index layouts."""

    def test_load_single_file_layout_returns_module(self, tmp_path):
        manifest = _gen_fixture(tmp_path / "single")
        model = load_diffusion_llama(manifest["out_dir"])

        assert isinstance(model, DiffusionLlama)
        assert isinstance(model, torch.nn.Module)
        assert model.num_layers == 2

    def test_load_sharded_layout_merges_all_shards(self, tmp_path):
        manifest = _gen_fixture(tmp_path / "sharded", sharded=True, shard_size=8)
        index = _read_index(manifest["out_dir"])
        # Sanity: the fixture is genuinely multi-shard.
        assert len(set(index["weight_map"].values())) >= 2

        model = load_diffusion_llama(manifest["out_dir"], dtype="float32", device="cpu")

        assert isinstance(model, DiffusionLlama)
        assert model.num_layers == 2

    def test_sharded_layout_bitwise_identical_to_single(self, tmp_path):
        """Same seed, same params, only the layout differs: both loads must
        produce bitwise-identical parameter multisets (transitively via the
        same deterministic on-disk reference)."""
        single = _gen_fixture(tmp_path / "single")
        sharded = _gen_fixture(tmp_path / "sharded", sharded=True, shard_size=8)

        model_single = load_diffusion_llama(single["out_dir"])
        model_sharded = load_diffusion_llama(sharded["out_dir"])

        reference = _reference_tensors(single["out_dir"], sharded=False)
        _assert_params_bitwise_match_reference(model_single, reference)
        _assert_params_bitwise_match_reference(model_sharded, reference)

    def test_loader_exported_via_models_package(self):
        """The loading interface is re-exported from shadowinfer.models."""
        import shadowinfer.models as models_pkg

        assert models_pkg.load_diffusion_llama is load_diffusion_llama
        assert models_pkg.DiffusionLlamaConfig is DiffusionLlamaConfig


class TestLoaderStylesAndTie:
    """Config-style coverage (LLaDA-8B / Dream-7B naming) and tied embeddings."""

    @pytest.mark.parametrize(
        ("style", "attention_bias"),
        [("llama", False), ("qwen", True)],
        ids=["llada-8b-llama-style", "dream-7b-qwen-style"],
    )
    def test_load_covers_both_checkpoint_naming_styles(self, tmp_path, style, attention_bias):
        """LLaDA-8B (llama-style config, no bias field) and Dream-7B
        (qwen-style config with qkv biases) share one weight-naming map;
        the same loader mapping must cover both."""
        manifest = _gen_fixture(
            tmp_path / f"fx_{style}", style=style, attention_bias=attention_bias
        )
        model = load_diffusion_llama(manifest["out_dir"])

        assert isinstance(model, DiffusionLlama)
        assert model.config.attention_bias is attention_bias

        reference = _reference_tensors(manifest["out_dir"], sharded=False)
        _assert_params_bitwise_match_reference(model, reference)

    def test_load_preset_flavored_numeric_fields(self, tmp_path):
        """Dream-7B literature-scope numeric passthrough (rope/eps/bias)."""
        dream = fixture_gen.PRESET_TEMPLATES["dream-7b"]
        manifest = _gen_fixture(
            tmp_path / "dream_flavored",
            style="qwen",
            attention_bias=dream["attention_bias"],
            rope_theta=dream["rope_theta"],
            rms_norm_eps=dream["rms_norm_eps"],
            tie_word_embeddings=dream["tie_word_embeddings"],
        )
        model = load_diffusion_llama(manifest["out_dir"])

        assert model.config.rope_theta == dream["rope_theta"]
        assert model.config.rms_norm_eps == dream["rms_norm_eps"]
        assert model.config.attention_bias is True

        reference = _reference_tensors(manifest["out_dir"], sharded=False)
        _assert_params_bitwise_match_reference(model, reference)

    def test_load_tied_embeddings_shares_embedding_for_lm_head(self, tmp_path):
        """tie_word_embeddings=True: lm_head is absent from the checkpoint
        and must be shared with embed_tokens at load time, so the exposed
        parameter multiset still equals the on-disk tensor multiset."""
        manifest = _gen_fixture(tmp_path / "tied", tie_word_embeddings=True)
        assert "lm_head.weight" not in manifest["weight_names"]

        model = load_diffusion_llama(manifest["out_dir"])

        assert isinstance(model, DiffusionLlama)
        assert model.config.tie_word_embeddings is True

        reference = _reference_tensors(manifest["out_dir"], sharded=False)
        _assert_params_bitwise_match_reference(model, reference)


class TestLoaderAttributes:
    """The DiffusionLlama attribute contract (assertable subset)."""

    def test_attribute_contract_matches_fixture_config(self, tmp_path):
        manifest = _gen_fixture(tmp_path / "fx", rope_theta=500000.0, rms_norm_eps=1e-6)
        model = load_diffusion_llama(manifest["out_dir"])

        assert model.num_layers == 2
        assert model.num_heads == 8
        assert model.num_kv_heads == 2
        assert model.head_dim == 32
        assert model.max_seq_len == 512
        assert model.vocab_size == 1000
        assert model.mask_token_id == 999
        assert torch.device(model.device).type == "cpu"

        assert isinstance(model.config, DiffusionLlamaConfig)
        cfg = model.config
        assert cfg.num_hidden_layers == 2
        assert cfg.num_attention_heads == 8
        assert cfg.num_key_value_heads == 2
        assert cfg.hidden_size == 256
        assert cfg.head_dim == 32
        assert cfg.intermediate_size == 4 * 256
        assert cfg.vocab_size == 1000
        assert cfg.mask_token_id == 999
        assert cfg.max_position_embeddings == 512
        assert cfg.rope_theta == 500000.0
        assert cfg.rms_norm_eps == 1e-6
        assert cfg.attention_bias is False
        assert cfg.tie_word_embeddings is False

        # Derived / passthrough relations.
        assert model.max_seq_len == cfg.max_position_embeddings
        assert model.num_layers == cfg.num_hidden_layers
        assert model.mask_token_id == cfg.mask_token_id

    def test_head_dim_derived_from_hidden_over_heads_when_missing(self, tmp_path):
        """head_dim absent from config.json -> derived as hidden // heads."""
        manifest = _gen_fixture(tmp_path / "fx")
        _edit_config(manifest["out_dir"], lambda c: c.pop("head_dim"))

        model = load_diffusion_llama(manifest["out_dir"])

        assert model.head_dim == 256 // 8

    def test_config_dataclass_defaults_and_frozen(self):
        cfg = DiffusionLlamaConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=256,
        )
        assert cfg.rope_theta == 10000.0
        assert cfg.rms_norm_eps == 1e-5
        assert cfg.head_dim is None
        assert cfg.attention_bias is False
        assert cfg.mask_token_id is None
        assert cfg.tie_word_embeddings is False

        with pytest.raises(AttributeError):
            cfg.vocab_size = 101


class TestLoaderWeightFidelity:
    """Bitwise weight fidelity against the raw on-disk checkpoint."""

    def test_single_layout_weights_bitwise_match_reference(self, tmp_path):
        manifest = _gen_fixture(tmp_path / "single")
        model = load_diffusion_llama(manifest["out_dir"])

        reference = _reference_tensors(manifest["out_dir"], sharded=False)
        _assert_params_bitwise_match_reference(model, reference)
        for _, p in model.named_parameters():
            assert p.dtype == torch.float32

    def test_sharded_layout_weights_bitwise_match_reference(self, tmp_path):
        manifest = _gen_fixture(tmp_path / "sharded", sharded=True, shard_size=8)
        model = load_diffusion_llama(manifest["out_dir"])

        reference = _reference_tensors(manifest["out_dir"], sharded=True)
        _assert_params_bitwise_match_reference(model, reference)

    def test_default_dtype_casts_bfloat16_checkpoint_to_float32(self, tmp_path):
        manifest = _gen_fixture(tmp_path / "bf16", dtype="bfloat16")
        model = load_diffusion_llama(manifest["out_dir"])  # default "float32"

        for _, p in model.named_parameters():
            assert p.dtype == torch.float32
        # bf16 -> fp32 casting is lossless, so bitwise equality holds
        # against the fp32-cast reference.
        reference = _reference_tensors(manifest["out_dir"], sharded=False)
        reference32 = {k: t.to(torch.float32) for k, t in reference.items()}
        _assert_params_bitwise_match_reference(model, reference32)

    def test_dtype_auto_preserves_bfloat16(self, tmp_path):
        manifest = _gen_fixture(tmp_path / "bf16", dtype="bfloat16")
        model = load_diffusion_llama(manifest["out_dir"], dtype="auto")

        for _, p in model.named_parameters():
            assert p.dtype == torch.bfloat16
        reference = _reference_tensors(manifest["out_dir"], sharded=False)
        _assert_params_bitwise_match_reference(model, reference)

    def test_dtype_auto_preserves_float32(self, tmp_path):
        manifest = _gen_fixture(tmp_path / "fp32")
        model = load_diffusion_llama(manifest["out_dir"], dtype="auto")

        for _, p in model.named_parameters():
            assert p.dtype == torch.float32
        reference = _reference_tensors(manifest["out_dir"], sharded=False)
        _assert_params_bitwise_match_reference(model, reference)


class TestLoaderErrors:
    """Fault injection on real fixture directories -> located ValueErrors.

    All mutations are real file-level corruption (delete file, rewrite
    json, rewrite safetensors); zero mocks.
    """

    def test_nonexistent_directory_raises(self, tmp_path):
        with pytest.raises(ValueError):
            load_diffusion_llama(str(tmp_path / "does_not_exist"))

    def test_missing_config_json_raises(self, tmp_path):
        manifest = _gen_fixture(tmp_path / "fx")
        (Path(manifest["out_dir"]) / "config.json").unlink()

        with pytest.raises(ValueError, match="config"):
            load_diffusion_llama(manifest["out_dir"])

    def test_missing_required_config_field_raises(self, tmp_path):
        manifest = _gen_fixture(tmp_path / "fx")

        def mutate(config):
            del config["num_key_value_heads"]

        _edit_config(manifest["out_dir"], mutate)

        with pytest.raises(ValueError, match="num_key_value_heads"):
            load_diffusion_llama(manifest["out_dir"])

    def test_missing_max_position_embeddings_raises_fail_fast(self, tmp_path):
        """max_position_embeddings is fail-fast: its absence must be
        reported with the field name (it feeds max_seq_len)."""
        manifest = _gen_fixture(tmp_path / "fx")

        def mutate(config):
            del config["max_position_embeddings"]

        _edit_config(manifest["out_dir"], mutate)

        with pytest.raises(ValueError, match="max_position_embeddings"):
            load_diffusion_llama(manifest["out_dir"])

    def test_mask_token_id_out_of_range_raises(self, tmp_path):
        manifest = _gen_fixture(tmp_path / "fx")

        def mutate(config):
            config["mask_token_id"] = config["vocab_size"]

        _edit_config(manifest["out_dir"], mutate)

        with pytest.raises(ValueError, match="mask_token_id"):
            load_diffusion_llama(manifest["out_dir"])

    def test_kv_heads_exceed_heads_raises(self, tmp_path):
        manifest = _gen_fixture(tmp_path / "fx")

        def mutate(config):
            config["num_key_value_heads"] = 2 * config["num_attention_heads"]

        _edit_config(manifest["out_dir"], mutate)

        with pytest.raises(ValueError, match="num_key_value_heads"):
            load_diffusion_llama(manifest["out_dir"])

    def test_missing_single_weight_file_raises(self, tmp_path):
        manifest = _gen_fixture(tmp_path / "fx")
        (Path(manifest["out_dir"]) / "model.safetensors").unlink()

        with pytest.raises(ValueError, match=re.escape("model.safetensors")):
            load_diffusion_llama(manifest["out_dir"])

    def test_single_file_missing_weight_raises(self, tmp_path):
        """A checkpoint file lacking one expected tensor -> ValueError
        locating the missing weight name (real safetensors rewrite)."""
        manifest = _gen_fixture(tmp_path / "fx")
        removed = "model.layers.1.mlp.down_proj.weight"

        weights = load_file(str(Path(manifest["out_dir"]) / "model.safetensors"))
        del weights[removed]
        save_file(weights, str(Path(manifest["out_dir"]) / "model.safetensors"))

        with pytest.raises(ValueError, match=re.escape(removed)):
            load_diffusion_llama(manifest["out_dir"])

    def test_single_file_extra_weight_raises(self, tmp_path):
        """A checkpoint file carrying an unexpected tensor -> ValueError
        locating the extra (orphan) weight name."""
        manifest = _gen_fixture(tmp_path / "fx")
        extra = "model.layers.9.mlp.up_proj.weight"

        weights = load_file(str(Path(manifest["out_dir"]) / "model.safetensors"))
        weights[extra] = torch.randn(16, 16)
        save_file(weights, str(Path(manifest["out_dir"]) / "model.safetensors"))

        with pytest.raises(ValueError, match=re.escape(extra)):
            load_diffusion_llama(manifest["out_dir"])

    def test_index_missing_weight_entry_raises(self, tmp_path):
        """Removing a weight_map entry (tensor still on disk) makes the
        index/filename-set inconsistent -> ValueError locating the weight."""
        manifest = _gen_fixture(tmp_path / "fx", sharded=True, shard_size=8)
        removed = "model.layers.0.self_attn.q_proj.weight"

        def mutate(index):
            del index["weight_map"][removed]

        _edit_index(manifest["out_dir"], mutate)

        with pytest.raises(ValueError, match=re.escape(removed)):
            load_diffusion_llama(manifest["out_dir"])

    def test_index_extra_weight_entry_raises(self, tmp_path):
        """Adding a weight_map entry for a nonexistent tensor -> ValueError
        locating the orphan entry name."""
        manifest = _gen_fixture(tmp_path / "fx", sharded=True, shard_size=8)
        extra = "model.layers.9.mlp.up_proj.weight"

        def mutate(index):
            target = sorted(set(index["weight_map"].values()))[0]
            index["weight_map"][extra] = target

        _edit_index(manifest["out_dir"], mutate)

        with pytest.raises(ValueError, match=re.escape(extra)):
            load_diffusion_llama(manifest["out_dir"])

    def test_shape_mismatch_raises_with_tensor_name(self, tmp_path):
        """Rewriting hidden_size (and head_dim consistently) makes every
        derived shape mismatch the actual weights -> ValueError naming the
        offending tensor. Config stays internally legal (heads*head_dim ==
        hidden) so the failure is purely a shape check."""
        manifest = _gen_fixture(tmp_path / "fx")

        def mutate(config):
            config["hidden_size"] = 512
            config["head_dim"] = 64

        _edit_config(manifest["out_dir"], mutate)

        with pytest.raises(ValueError, match=_TENSOR_NAME_PATTERN):
            load_diffusion_llama(manifest["out_dir"])

    def test_missing_shard_file_raises_with_filename(self, tmp_path):
        """Deleting one shard file referenced by the index -> ValueError
        locating the missing shard file name."""
        manifest = _gen_fixture(tmp_path / "fx", sharded=True, shard_size=8)
        index = _read_index(manifest["out_dir"])
        victim = sorted(set(index["weight_map"].values()))[-1]
        (Path(manifest["out_dir"]) / victim).unlink()

        with pytest.raises(ValueError, match=re.escape(victim)):
            load_diffusion_llama(manifest["out_dir"])

    def test_invalid_dtype_argument_raises(self, tmp_path):
        manifest = _gen_fixture(tmp_path / "fx")

        with pytest.raises(ValueError, match="fp16"):
            load_diffusion_llama(manifest["out_dir"], dtype="fp16")


class TestFixtureGenAndPresets:
    """P5: fixture generator determinism / index coverage and PRESETS."""

    def test_same_seed_generates_bitwise_identical_files(self, tmp_path):
        """Fixed seed + fixed params -> byte-identical files, for both
        the single-file and the sharded layout."""
        layouts = ({"sharded": False}, {"sharded": True, "shard_size": 8})
        for layout in layouts:
            first = _gen_fixture(tmp_path / "a", **layout)
            second = _gen_fixture(tmp_path / "b", **layout)

            assert first["files"] == second["files"]
            for name in first["files"]:
                bytes_first = (Path(first["out_dir"]) / name).read_bytes()
                bytes_second = (Path(second["out_dir"]) / name).read_bytes()
                assert bytes_first == bytes_second, f"file {name} differs"

    def test_sharded_index_weight_map_covers_all_weights(self, tmp_path):
        manifest = _gen_fixture(tmp_path / "fx", sharded=True, shard_size=8)
        index = _read_index(manifest["out_dir"])

        assert set(index["weight_map"]) == set(manifest["weight_names"])
        for filename in set(index["weight_map"].values()):
            assert (Path(manifest["out_dir"]) / filename).is_file()

    def test_presets_match_fixture_gen_preset_templates(self):
        """PRESETS must equal fixture_gen.PRESET_TEMPLATES field by field."""
        templates = fixture_gen.PRESET_TEMPLATES

        assert isinstance(PRESETS, dict)
        assert set(PRESETS) == {"dream-7b", "llada-8b"}
        assert set(PRESETS) == set(templates)
        for name in sorted(templates):
            assert set(PRESETS[name]) == set(templates[name])
            for field, value in templates[name].items():
                assert PRESETS[name][field] == value, (
                    f"preset {name}.{field}: {PRESETS[name][field]!r} " f"!= template {value!r}"
                )

    def test_presets_internally_consistent(self):
        for name, preset in PRESETS.items():
            heads = preset["num_attention_heads"]
            kv_heads = preset["num_key_value_heads"]
            assert heads * preset["head_dim"] == preset["hidden_size"], name
            assert heads % kv_heads == 0, name
            assert kv_heads <= heads, name
            assert 0 <= preset["mask_token_id"] < preset["vocab_size"], name
            assert preset["num_hidden_layers"] >= 1, name


# ---------------------------------------------------------------------------
# P2: modeling-layer numeric comparison (AR018 T003, FR18.2)
#
# A hand-written INDEPENDENT reference implementation (test-inline golden,
# explicit per-operator slow version, no production internals) compared
# against forward_step/forward_block on real fixture weights (zero-mock).
# The production modeling layer is currently NotImplementedError stubs;
# every test below is expected to FAIL with NotImplementedError (Red).
# ---------------------------------------------------------------------------

# Registered golden tolerance (srs FR18.2 acceptance 1: fp32 logits vs the
# hand-written reference, explicitly registered atol/rtol).
GOLD_TOL_ATOL = 1e-5
GOLD_TOL_RTOL = 1e-5

# Registered rel-L2 tolerance for decomposition anchors (design D7 tier
# (ii); mirrors the FB_TOL = 1e-6 precedent in tests/test_block_decoding.py).
FB_TOL2 = 1e-6

# Registered rel-L2 tolerance for the KV tier of decomposition anchors
# (design D7 tier (i), revised by AR018 erratum: KV projections are
# themselves GEMM outputs, so changing the GEMM M dimension (full
# forward S rows vs block forward W rows) can switch the CPU
# microkernel and its K-accumulator order -- measured 2e-7..1e-6
# divergence for M=12 vs M=4, while M=9 vs M=4 stayed bitwise. The
# same applies to the hand-written reference (vectorized SLEEF cos vs
# scalar libm cos differ by ~1 ulp). Bitwise is kernel luck, not a
# contract; a semantic decomposition error (wrong rows / wrong RoPE
# offset) produces O(1) divergence, cleanly separated by this
# tolerance).
KV_TOL = 1e-5

# Tiny fp tolerance for closed-form RoPE angle checks (cos/sin of the
# per-pair inverse frequency).
ROPE_ANGLE_TOL = 1e-6

# Fixed P2 geometry: S = P + W + W2 (prompt / block under test / trailing
# future segment that the segment-causal mask must genuinely hide).
_P2_PREFIX = 5  # P
_P2_WIDTH = 4  # W
_P2_TAIL = 3  # W2 (future segment hidden from block rows)
_P2_SEQ = _P2_PREFIX + _P2_WIDTH + _P2_TAIL  # S = 12

# Fixed seeds for deterministic inputs (torch.manual_seed discipline).
_P2_SEED = 4303
_P2_BLOCK_SEED = 4307
_P2_ANCHOR_SEED = 4311
_P2_ROPE_VEC_SEED = 4313
_P2_ROPE_FWD_SEED = 4317

# Golden matrix head configurations: GQA (kv 2 < q 8) and MHA (kv 8 == q 8),
# both as _TINY overrides.
_P2_KV_HEADS = [2, 8]


def _p2_model(tmp_path: Path, name: str, **overrides: Any) -> DiffusionLlama:
    """Generate a real fixture and load it (zero-mock, D12)."""
    manifest = _gen_fixture(tmp_path / name, **overrides)
    return load_diffusion_llama(manifest["out_dir"])


def _p2_ids(model: DiffusionLlama, seq_len: int, seed: int) -> torch.Tensor:
    """Deterministic [1, seq_len] token ids in [0, vocab_size)."""
    torch.manual_seed(seed)
    return torch.randint(0, model.vocab_size, (1, seq_len))


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 distance (mirrors tests/test_block_decoding.py)."""
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


# --- hand-written reference implementation (no production internals) ---


def _ref_inv_freq(theta: float, head_dim: int) -> torch.Tensor:
    """inv_freq = theta ** (-arange(0, D, 2) / D) (design D6)."""
    return theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)


def _ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """x / sqrt(mean(x^2) + eps) * weight."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(variance + eps) * weight


def _ref_rope(x: torch.Tensor, inv_freq: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
    """HF-style rotate-half RoPE (design D6), explicit per-pair loop.

    x: [B, H, T, D]; pair j couples (x[..., j], x[..., j + D/2]) and
    rotates by angle (position_offset + t) * inv_freq[j]:
    out[..., j] = x0*cos - x1*sin, out[..., j + D/2] = x0*sin + x1*cos.
    This is the standard cat(freqs, freqs) rotate-half pairing used by
    HF Llama/Qwen (and thus by the real LLaDA-8B / Dream-7B training
    code); the pairing is part of the convention and must match the
    production implementation for real-checkpoint fidelity.
    """
    batch, heads, seq_len, head_dim = x.shape
    half = head_dim // 2
    out = torch.empty_like(x)
    for t in range(seq_len):
        pos = position_offset + t
        for j in range(half):
            angle = pos * inv_freq[j]
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            x0 = x[:, :, t, j]
            x1 = x[:, :, t, j + half]
            out[:, :, t, j] = x0 * cos_a - x1 * sin_a
            out[:, :, t, j + half] = x0 * sin_a + x1 * cos_a
    return out


def _ref_repeat_kv(t: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA expand [B, H_kv, T, D] -> [B, H_kv*n_rep, T, D]: each kv head is
    repeated n_rep consecutive times (identity when n_rep == 1)."""
    if n_rep == 1:
        return t
    heads = []
    for h in range(t.shape[1]):
        for _ in range(n_rep):
            heads.append(t[:, h : h + 1])
    return torch.cat(heads, dim=1)


def _ref_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """softmax(q @ k^T / sqrt(D) + mask) @ v per head, explicit head loop.

    q: [B, H, Tq, D]; k/v: [B, H, Tk, D] (already GQA-expanded); mask:
    additive [Tq, Tk] (0 keep / -inf hide, mirroring segment_causal_mask)
    or None for fully bidirectional attention.
    """
    batch, heads, tgt_len, head_dim = q.shape
    out = torch.empty(batch, heads, tgt_len, head_dim)
    scale = math.sqrt(head_dim)
    for b in range(batch):
        for h in range(heads):
            scores = q[b, h] @ k[b, h].t() / scale
            if mask is not None:
                scores = scores + mask
            probs = torch.softmax(scores, dim=-1)
            out[b, h] = probs @ v[b, h]
    return out


def _ref_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """y = x @ W^T (no bias; fixtures in the P2 matrix are bias-free)."""
    return x @ weight.t()


def _ref_silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU = x * sigmoid(x)."""
    return x * torch.sigmoid(x)


def _ref_forward(
    model: DiffusionLlama,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    prefix_kv: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    prefix_len: int = 0,
) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
    """Hand-written golden forward (design 6.1 P2; NOT production code).

    Modes:
      - prefix_kv None: full-sequence forward over input_ids [B, S] with
        attention_mask None (D5(a) bidirectional) or an additive [S, S]
        mask (D5(b) segment-causal).
      - prefix_kv given: D5(c) block-rows conditional forward over
        input_ids [B, W]; rows sit at absolute positions
        [prefix_len, prefix_len + W) and attend to cat(prefix, block)
        with no mask.

    Returns (logits [B, T, V], kv {layer: (k, v)}) where k/v are this
    call's own rows, KV-head-native [B, num_kv_heads, T, D] (design D2).
    """
    params = dict(model.named_parameters())
    cfg = model.config
    num_heads = model.num_heads
    num_kv_heads = model.num_kv_heads
    head_dim = model.head_dim
    n_rep = num_heads // num_kv_heads
    eps = cfg.rms_norm_eps
    inv_freq = _ref_inv_freq(cfg.rope_theta, head_dim)

    h = params["model.embed_tokens.weight"][input_ids]
    batch, seq_len, _ = h.shape
    offset = prefix_len if prefix_kv is not None else 0
    mask = None if prefix_kv is not None else attention_mask

    kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in range(model.num_layers):
        prefix = f"model.layers.{layer}"
        normed = _ref_rmsnorm(h, params[f"{prefix}.input_layernorm.weight"], eps)
        q = _ref_linear(normed, params[f"{prefix}.self_attn.q_proj.weight"])
        k = _ref_linear(normed, params[f"{prefix}.self_attn.k_proj.weight"])
        v = _ref_linear(normed, params[f"{prefix}.self_attn.v_proj.weight"])
        q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        q = _ref_rope(q, inv_freq, offset)
        k = _ref_rope(k, inv_freq, offset)
        if prefix_kv is not None:
            past_k, past_v = prefix_kv[layer]
            k_all = torch.cat([past_k, k], dim=2)
            v_all = torch.cat([past_v, v], dim=2)
        else:
            k_all, v_all = k, v
        attn = _ref_attention(q, _ref_repeat_kv(k_all, n_rep), _ref_repeat_kv(v_all, n_rep), mask)
        attn = attn.transpose(1, 2).reshape(batch, seq_len, num_heads * head_dim)
        h = h + _ref_linear(attn, params[f"{prefix}.self_attn.o_proj.weight"])
        normed_post = _ref_rmsnorm(h, params[f"{prefix}.post_attention_layernorm.weight"], eps)
        gate = _ref_linear(normed_post, params[f"{prefix}.mlp.gate_proj.weight"])
        up = _ref_linear(normed_post, params[f"{prefix}.mlp.up_proj.weight"])
        h = h + _ref_linear(_ref_silu(gate) * up, params[f"{prefix}.mlp.down_proj.weight"])
        kv[layer] = (k, v)

    h = _ref_rmsnorm(h, params["model.norm.weight"], eps)
    logits = _ref_linear(h, params["lm_head.weight"])
    return logits, kv


class TestModelingGoldenComparison:
    """P2 golden matrix (design 6.1 P2, srs FR18.2 acceptance 1):
    {mask state (a)/(b)} x {GQA, MHA} x {RoPE offset 0 via full forward,
    nonzero via mode (c) block positions} against the hand-written
    reference, on real fixture weights."""

    @pytest.mark.parametrize("mask_state", ["bidirectional", "segment-causal"])
    @pytest.mark.parametrize("num_kv_heads", _P2_KV_HEADS, ids=["gqa-kv2", "mha-kv8"])
    def test_golden_full_forward(self, tmp_path, num_kv_heads, mask_state):
        model = _p2_model(tmp_path, "fx_golden_full", num_kv_heads=num_kv_heads)
        ids = _p2_ids(model, _P2_SEQ, _P2_SEED)
        if mask_state == "bidirectional":
            mask = None  # D5(a): fully bidirectional, no mask.
        else:
            # D5(b): the repo's segment-causal additive mask, zero-change
            # reuse; prompt / block / trailing-future segment split.
            mask = segment_causal_mask(_P2_SEQ, _P2_PREFIX, _P2_WIDTH, torch.device("cpu"))

        logits, kv, third = model.forward_step(ids, 0, 0, attention_mask=mask)
        ref_logits, _ = _ref_forward(model, ids, attention_mask=mask)

        assert third is None
        assert logits.shape == (1, _P2_SEQ, model.vocab_size)
        assert torch.allclose(logits, ref_logits, atol=GOLD_TOL_ATOL, rtol=GOLD_TOL_RTOL)
        # KV contract: KV-head-native width (design D2), one entry per layer.
        assert set(kv) == set(range(model.num_layers))
        for layer in range(model.num_layers):
            k, v = kv[layer]
            assert k.shape == (1, model.num_kv_heads, _P2_SEQ, model.head_dim)
            assert v.shape == (1, model.num_kv_heads, _P2_SEQ, model.head_dim)

    @pytest.mark.parametrize("num_kv_heads", _P2_KV_HEADS, ids=["gqa-kv2", "mha-kv8"])
    def test_golden_block_conditional(self, tmp_path, num_kv_heads):
        model = _p2_model(tmp_path, "fx_golden_block", num_kv_heads=num_kv_heads)
        seq = _P2_PREFIX + _P2_WIDTH
        ids = _p2_ids(model, seq, _P2_BLOCK_SEED)
        prompt_ids = ids[:, :_P2_PREFIX]
        block_ids = ids[:, _P2_PREFIX:]

        # Model path: bidirectional prefill (D5(a)) feeds the model's own
        # kv contract into forward_block (D5(c)).
        _, prefix_kv, _ = model.forward_step(prompt_ids, 0, 0)
        block_logits, block_kv = model.forward_block(
            block_ids, prefix_kv=prefix_kv, prefix_len=_P2_PREFIX
        )

        # Reference path: its own bidirectional prefix forward, then only
        # the block rows at absolute positions [P, P+W) with no mask.
        _, ref_prefix_kv = _ref_forward(model, prompt_ids)
        ref_block_logits, ref_block_kv = _ref_forward(
            model, block_ids, prefix_kv=ref_prefix_kv, prefix_len=_P2_PREFIX
        )

        assert block_logits.shape == (1, _P2_WIDTH, model.vocab_size)
        assert torch.allclose(
            block_logits, ref_block_logits, atol=GOLD_TOL_ATOL, rtol=GOLD_TOL_RTOL
        )
        assert set(block_kv) == set(range(model.num_layers))
        for layer in range(model.num_layers):
            ref_k, ref_v = ref_block_kv[layer]
            block_k, block_v = block_kv[layer]
            assert block_k.shape == (1, model.num_kv_heads, _P2_WIDTH, model.head_dim)
            # KV tier (D7 tier (i), tolerance-revised per erratum): the
            # block k/v must match the reference's block-row k/v
            # projections within the registered rounding-noise tolerance.
            assert _rel_l2(block_k, ref_k) <= KV_TOL
            assert _rel_l2(block_v, ref_v) <= KV_TOL


class TestSegmentCausalDecompositionAnchor:
    """D7 two-tier anchor (srs FR18.2 acceptance 2): segment-causal full
    forward block rows vs bidirectional prefill + forward_block, with a
    trailing future segment W2 the mask genuinely hides."""

    @pytest.mark.parametrize("num_kv_heads", _P2_KV_HEADS, ids=["gqa-kv2", "mha-kv8"])
    def test_two_tier_anchor(self, tmp_path, num_kv_heads):
        model = _p2_model(tmp_path, "fx_anchor", num_kv_heads=num_kv_heads)
        full_ids = _p2_ids(model, _P2_SEQ, _P2_ANCHOR_SEED)
        prompt_ids = full_ids[:, :_P2_PREFIX]
        block_ids = full_ids[:, _P2_PREFIX : _P2_PREFIX + _P2_WIDTH]
        # Block-1 rows see prompt + block 0 + block 1 but NOT block 2.
        mask = segment_causal_mask(_P2_SEQ, _P2_PREFIX, _P2_WIDTH, torch.device("cpu"))

        # (i) masked full-sequence forward; (ii) bidirectional prefill;
        # (iii) block-rows conditional forward.
        full_logits, full_kv, _ = model.forward_step(full_ids, 0, 0, attention_mask=mask)
        _, prefix_kv, _ = model.forward_step(prompt_ids, 0, 0)
        block_logits, block_kv = model.forward_block(
            block_ids, prefix_kv=prefix_kv, prefix_len=_P2_PREFIX
        )

        assert full_logits.shape == (1, _P2_SEQ, model.vocab_size)
        assert block_logits.shape == (1, _P2_WIDTH, model.vocab_size)
        assert set(block_kv) == set(range(model.num_layers))

        # KV tier (D7 (i), tolerance-revised per erratum): KV projections
        # are mask-independent row-local computations up to GEMM kernel
        # selection, so rows [P, P+W) of the full forward's kv match
        # block_kv within the registered rounding-noise tolerance.
        for layer in range(model.num_layers):
            full_k, full_v = full_kv[layer]
            block_k, block_v = block_kv[layer]
            assert block_k.shape == (1, model.num_kv_heads, _P2_WIDTH, model.head_dim)
            assert _rel_l2(full_k[:, :, _P2_PREFIX : _P2_PREFIX + _P2_WIDTH, :], block_k) <= KV_TOL
            assert _rel_l2(full_v[:, :, _P2_PREFIX : _P2_PREFIX + _P2_WIDTH, :], block_v) <= KV_TOL

        # Logits tier (D7 (ii)): rel-L2 within the registered tolerance
        # (not bitwise: attention matmul shape difference, repo fp32
        # precedent "not bitwise").
        full_rows = full_logits[:, _P2_PREFIX : _P2_PREFIX + _P2_WIDTH, :]
        assert _rel_l2(full_rows, block_logits) <= FB_TOL2
        # Argmax tier: per-position decoding decisions identical.
        assert torch.equal(full_rows.argmax(-1), block_logits.argmax(-1))


class TestRopeMathProperties:
    """RoPE math properties (design D6): the inv_freq formula, the
    adjacent-position rotation angle increment, and absolute-position
    offset correctness via the model's own decomposition."""

    @pytest.mark.parametrize("rope_theta", [10000.0, 500000.0])
    def test_rope_properties(self, tmp_path, rope_theta):
        model = _p2_model(tmp_path, "fx_rope", rope_theta=rope_theta)
        head_dim = model.head_dim
        expected = _ref_inv_freq(rope_theta, head_dim)

        # (1) The public inv_freq buffer matches theta ** (-2i/d).
        assert model.inv_freq.shape == (head_dim // 2,)
        assert torch.allclose(model.inv_freq, expected, rtol=ROPE_ANGLE_TOL, atol=1e-12)

        # (2) Adjacent positions differ by exactly the per-pair angle:
        # rotating a fixed vector at position p vs p+1 turns each (even,
        # odd) pair by exactly inv_freq[i] = theta^(-2i/d).
        torch.manual_seed(_P2_ROPE_VEC_SEED)
        vec = torch.randn(head_dim)
        half = head_dim // 2
        for pos in (0, 3, _P2_PREFIX):
            rot_p = _ref_rope(vec.view(1, 1, 1, head_dim), expected, pos)[0, 0, 0]
            rot_p1 = _ref_rope(vec.view(1, 1, 1, head_dim), expected, pos + 1)[0, 0, 0]
            for j in range(half):
                # Rotate-half pairs couple (j, j + D/2) (HF Llama/Qwen
                # convention, design D6); treat each pair as a complex
                # number and check the adjacent-position angle increment.
                z_p = torch.complex(rot_p[j], rot_p[j + half])
                z_p1 = torch.complex(rot_p1[j], rot_p1[j + half])
                angle_diff = torch.angle(z_p1 * torch.conj(z_p))
                assert abs(float(angle_diff) - float(expected[j])) <= ROPE_ANGLE_TOL

        # (3) Position-offset correctness: with no mask anywhere and the
        # block as the tail (S = P + W), the full bidirectional forward's
        # block rows have exactly the receptive field of prefix + block
        # conditional rows, so the decomposition must match within the
        # registered rel-L2 tiers, agree on argmax, and match KV rows
        # within the registered KV tolerance (RoPE at absolute positions
        # [P, P+W) makes the KV rows match up to kernel rounding; a wrong
        # offset produces O(1) divergence, far above KV_TOL).
        # The prefix kv is taken from the full bidirectional forward's own
        # prefix rows so both paths condition on identical prefix keys.
        seq = _P2_PREFIX + _P2_WIDTH
        torch.manual_seed(_P2_ROPE_FWD_SEED)
        ids = torch.randint(0, model.vocab_size, (1, seq))
        full_logits, full_kv, _ = model.forward_step(ids, 0, 0)
        prefix_kv = {
            layer: (k[:, :, :_P2_PREFIX, :], v[:, :, :_P2_PREFIX, :])
            for layer, (k, v) in full_kv.items()
        }
        block_logits, block_kv = model.forward_block(
            ids[:, _P2_PREFIX:], prefix_kv=prefix_kv, prefix_len=_P2_PREFIX
        )

        full_rows = full_logits[:, _P2_PREFIX:, :]
        assert _rel_l2(full_rows, block_logits) <= FB_TOL2
        assert torch.equal(full_rows.argmax(-1), block_logits.argmax(-1))
        for layer in range(model.num_layers):
            full_k, full_v = full_kv[layer]
            block_k, block_v = block_kv[layer]
            assert _rel_l2(full_k[:, :, _P2_PREFIX:, :], block_k) <= KV_TOL
            assert _rel_l2(full_v[:, :, _P2_PREFIX:, :], block_v) <= KV_TOL


class TestMhaZeroSpecialCase:
    """FR18.2 acceptance 3: MHA (num_kv_heads == num_heads) loads and
    forwards through the golden comparison path with zero special-casing
    (GQA repeat factor 1); independent of SimpleDiffusionLLM."""

    def test_mha_golden_forward_step_and_forward_block(self, tmp_path):
        model = _p2_model(tmp_path, "fx_mha", num_kv_heads=8)
        assert model.num_kv_heads == model.num_heads

        # Full-sequence bidirectional golden.
        ids = _p2_ids(model, _P2_SEQ, _P2_SEED)
        logits, kv, _ = model.forward_step(ids, 0, 0)
        ref_logits, _ = _ref_forward(model, ids)

        assert logits.shape == (1, _P2_SEQ, model.vocab_size)
        assert torch.allclose(logits, ref_logits, atol=GOLD_TOL_ATOL, rtol=GOLD_TOL_RTOL)
        for layer in range(model.num_layers):
            k, v = kv[layer]
            assert k.shape == (1, model.num_kv_heads, _P2_SEQ, model.head_dim)
            assert v.shape == (1, model.num_kv_heads, _P2_SEQ, model.head_dim)

        # Block-rows conditional golden (repeat factor 1).
        seq = _P2_PREFIX + _P2_WIDTH
        ids_block = _p2_ids(model, seq, _P2_BLOCK_SEED)
        _, prefix_kv, _ = model.forward_step(ids_block[:, :_P2_PREFIX], 0, 0)
        block_logits, block_kv = model.forward_block(
            ids_block[:, _P2_PREFIX:], prefix_kv=prefix_kv, prefix_len=_P2_PREFIX
        )
        _, ref_prefix_kv = _ref_forward(model, ids_block[:, :_P2_PREFIX])
        ref_block_logits, _ = _ref_forward(
            model,
            ids_block[:, _P2_PREFIX:],
            prefix_kv=ref_prefix_kv,
            prefix_len=_P2_PREFIX,
        )

        assert block_logits.shape == (1, _P2_WIDTH, model.vocab_size)
        assert torch.allclose(
            block_logits, ref_block_logits, atol=GOLD_TOL_ATOL, rtol=GOLD_TOL_RTOL
        )
        for layer in range(model.num_layers):
            block_k, block_v = block_kv[layer]
            assert block_k.shape == (1, model.num_kv_heads, _P2_WIDTH, model.head_dim)
            assert block_v.shape == (1, model.num_kv_heads, _P2_WIDTH, model.head_dim)


# ---------------------------------------------------------------------------
# P3: blockwise pipeline contract + generate integration (AR018 T004, FR18.3)
#
# Red-phase tests for ``shadowinfer.execution.blockwise_generate`` (the
# D3 extraction target) plus T003 forward-contract error pins. The pure
# function does not exist yet (Green creates the module), so every import
# of it happens INSIDE test bodies: collection of the 45 existing tests
# must keep working, and the blockwise_generate tests fail with
# ModuleNotFoundError in Red. Tests pinning already-implemented T003
# behavior (forward_step/forward_block contracts) may pass in Red.
# ---------------------------------------------------------------------------

# Fixed P3 geometry: tiny GQA fixture (num_kv_heads=2 < num_heads=8 from
# _TINY), prompt P=5, gen G=8 as two blocks of width W=4.
_P3_PROMPT = 5  # P
_P3_WIDTH = 4  # W
_P3_GEN = 8  # G = 2 * W
_P3_SEED = 4403
_P3_MASK_TOKEN = _TINY["mask_token_id"]  # 999 (< vocab_size 1000)

# MHA delegation-regression geometry (SimpleDiffusionLLM; mirrors the
# GP_* constants and backend idiom in tests/test_block_decoding.py).
_TD_LAYERS = 2
_TD_HEADS = 2
_TD_HEAD_DIM = 16
_TD_HIDDEN = 32
_TD_VOCAB = 64
_TD_MAX_SEQ = 64
_TD_BATCH = 2
_TD_PROMPT = 6  # P
_TD_WIDTH = 4  # W
_TD_GEN = 8  # G = 2 * W
_TD_SEED = 4503

# The exact 17-key stats contract assembled by
# PyTorchModelBackend.generate (pytorch_backend.py) and preserved
# verbatim by the D3 cut-paste extraction. All 17 keys are always
# present; "draft_acceptance_rate" / "rounds_to_fixpoint" /
# "changed_per_round" carry None outside draft_mode. An 18th key
# "block_logits_per_round" appears only when capture_block_logits=True.
_GENERATE_STATS_KEYS = frozenset(
    {
        "blocks",
        "total_rounds",
        "computed_rows_total",
        "kv_write_rows_total",
        "forced_completions",
        "total_time_ms",
        "page_precision",
        "rounds_per_block",
        "unmasked_per_round",
        "fallback_per_round",
        "mean_confidence_per_round",
        "rows_saved_vs_baseline",
        "unmask_positions_per_round",
        "block_time_ms",
        "draft_acceptance_rate",
        "rounds_to_fixpoint",
        "changed_per_round",
    }
)
_DRAFT_ONLY_STATS_KEYS = ("draft_acceptance_rate", "rounds_to_fixpoint", "changed_per_round")

# Wall-clock keys: identical logic still produces different timings per
# call, so per-key equality is only asserted off the timing keys.
_TIMING_STATS_KEYS = frozenset({"total_time_ms", "block_time_ms"})


def _import_blockwise_generate():
    """Lazily import the AR018 T004 pure function (Green-phase module).

    Never called at module import time: the module does not exist in
    Red, and the 45 pre-existing tests must keep collecting.
    """
    from shadowinfer.execution.blockwise_generate import blockwise_generate

    return blockwise_generate


def _p3_config(draft_mode: bool = False, **overrides: Any):
    """Small BlockDecodingConfig matching the P3 geometry."""
    from shadowinfer.execution.block_decoding import BlockDecodingConfig

    settings = dict(
        block_size=_P3_WIDTH,
        max_rounds=8,
        confidence_threshold=0.9,
        mask_token_id=_P3_MASK_TOKEN,
        page_size=_P3_WIDTH,
        page_precision="fp32",
        draft_mode=draft_mode,
    )
    settings.update(overrides)
    return BlockDecodingConfig(**settings)


def _p3_prompt(model: DiffusionLlama, seed: int, batch: int = 1) -> torch.Tensor:
    """Deterministic [batch, P] ids in [0, vocab_size - 1): never the
    mask token 999 (excluded via the exclusive upper bound)."""
    torch.manual_seed(seed)
    return torch.randint(0, model.vocab_size - 1, (batch, _P3_PROMPT))


def _td_model_config() -> Dict[str, Any]:
    return {
        "num_layers": _TD_LAYERS,
        "num_heads": _TD_HEADS,
        "head_dim": _TD_HEAD_DIM,
        "hidden_dim": _TD_HIDDEN,
        "vocab_size": _TD_VOCAB,
        "max_seq_len": _TD_MAX_SEQ,
    }


def _td_backend(seed: int):
    """Backend built exactly like tests/test_block_decoding.py::_gp_backend."""
    from shadowinfer.core.backends.pytorch_backend import PyTorchModelBackend

    backend = PyTorchModelBackend(model_config=_td_model_config(), seed=seed)
    backend.load(model_name="ar018-td", device="cpu")
    return backend


def _td_config(draft_mode: bool = False, **overrides: Any):
    from shadowinfer.execution.block_decoding import BlockDecodingConfig

    settings = dict(
        block_size=_TD_WIDTH,
        max_rounds=8,
        confidence_threshold=0.9,
        mask_token_id=0,
        page_size=_TD_WIDTH,
        page_precision="fp32",
        draft_mode=draft_mode,
    )
    settings.update(overrides)
    return BlockDecodingConfig(**settings)


def _td_prompt(seed: int, batch: int = _TD_BATCH) -> torch.Tensor:
    """Random prompt ids in [1, vocab): never the mask token id 0."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(1, _TD_VOCAB, (batch, _TD_PROMPT), generator=gen)


def _assert_stats_values_equal(key: str, left: Any, right: Any) -> None:
    """Recursive per-key stats equality: torch.equal for tensors,
    element-wise for lists, == for scalars (ints/floats/bools/None)."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), f"stats[{key!r}]: tensor vs {type(right)}"
        assert torch.equal(left, right), f"stats[{key!r}]: tensors differ"
    elif isinstance(left, list):
        assert isinstance(right, list), f"stats[{key!r}]: list vs {type(right)}"
        assert len(left) == len(right), f"stats[{key!r}]: lengths {len(left)} != {len(right)}"
        for idx, (item_l, item_r) in enumerate(zip(left, right)):
            _assert_stats_values_equal(f"{key}[{idx}]", item_l, item_r)
    else:
        assert left == right, f"stats[{key!r}]: {left!r} != {right!r}"


class TestForwardContractErrors:
    """T003 forward-contract error pins (already implemented; may pass in
    Red). Message fragments are copied from the production messages in
    shadowinfer/models/diffusion_llama.py."""

    def test_forward_step_rejects_kv_cache(self, tmp_path):
        model = _p2_model(tmp_path, "fx_p3_kvcache")
        ids = _p2_ids(model, _P2_SEQ, _P2_SEED)

        with pytest.raises(ValueError, match="does not accept kv_cache"):
            model.forward_step(ids, 0, 0, kv_cache={0: "dummy"})
        with pytest.raises(ValueError, match="does not accept kv_cache"):
            model.forward_step(ids, 0, 0, kv_cache=torch.zeros(1))

    def test_forward_step_rejects_1d_x(self, tmp_path):
        model = _p2_model(tmp_path, "fx_p3_1d")
        ids = _p2_ids(model, _P2_SEQ, _P2_SEED)

        with pytest.raises(ValueError, match=r"x must be \[batch, seq_len\]"):
            model.forward_step(ids[0], 0, 0)

    def test_forward_step_rejects_float_dtype_x(self, tmp_path):
        model = _p2_model(tmp_path, "fx_p3_float")
        ids = _p2_ids(model, _P2_SEQ, _P2_SEED)

        with pytest.raises(ValueError, match="must be an integer tensor of token ids"):
            model.forward_step(ids.float(), 0, 0)

    def test_forward_step_rejects_empty_seq(self, tmp_path):
        model = _p2_model(tmp_path, "fx_p3_empty")
        empty = torch.zeros(1, 0, dtype=torch.long)

        with pytest.raises(ValueError, match="seq_len must be >= 1"):
            model.forward_step(empty, 0, 0)

    def test_attention_mask_2d_and_4d_forms_bitwise_identical(self, tmp_path):
        """[S, S] and [1, 1, S, S] are the same additive mask up to
        broadcast; both accepted forms must produce bitwise-identical
        logits on the same fixture and ids."""
        model = _p2_model(tmp_path, "fx_p3_mask_forms")
        ids = _p2_ids(model, _P2_SEQ, _P2_SEED)
        mask2 = segment_causal_mask(_P2_SEQ, _P2_PREFIX, _P2_WIDTH, torch.device("cpu"))

        logits_2d, _, _ = model.forward_step(ids, 0, 0, attention_mask=mask2)
        logits_4d, _, _ = model.forward_step(ids, 0, 0, attention_mask=mask2[None, None])

        assert logits_2d.shape == (1, _P2_SEQ, model.vocab_size)
        assert torch.equal(logits_2d, logits_4d)

    def test_attention_mask_wrong_last_dims_raises(self, tmp_path):
        model = _p2_model(tmp_path, "fx_p3_mask_cover")
        ids = _p2_ids(model, _P2_SEQ, _P2_SEED)
        bad = torch.zeros(_P2_SEQ, _P2_SEQ + 1)

        with pytest.raises(ValueError, match="attention_mask must cover"):
            model.forward_step(ids, 0, 0, attention_mask=bad)

    def test_attention_mask_3d_raises(self, tmp_path):
        model = _p2_model(tmp_path, "fx_p3_mask_3d")
        ids = _p2_ids(model, _P2_SEQ, _P2_SEED)
        bad = torch.zeros(1, _P2_SEQ, _P2_SEQ)

        with pytest.raises(ValueError, match=r"attention_mask must be \[S, S\] or \[1, 1, S, S\]"):
            model.forward_step(ids, 0, 0, attention_mask=bad)

    def test_forward_block_negative_prefix_len_raises(self, tmp_path):
        model = _p2_model(tmp_path, "fx_p3_neglen")
        ids = _p2_ids(model, _P2_PREFIX + _P2_WIDTH, _P2_SEED)
        _, prefix_kv, _ = model.forward_step(ids[:, :_P2_PREFIX], 0, 0)

        with pytest.raises(ValueError, match="prefix_len must be >= 0"):
            model.forward_block(ids[:, _P2_PREFIX:], prefix_kv, -1)

    def test_forward_block_missing_prefix_kv_raises(self, tmp_path):
        model = _p2_model(tmp_path, "fx_p3_noprefix")
        ids = _p2_ids(model, _P2_PREFIX + _P2_WIDTH, _P2_SEED)

        with pytest.raises(ValueError, match="prefix_kv is required when prefix_len"):
            model.forward_block(ids[:, _P2_PREFIX:], None, _P2_PREFIX)

    def test_forward_block_incomplete_layer_set_raises(self, tmp_path):
        model = _p2_model(tmp_path, "fx_p3_layerset")
        ids = _p2_ids(model, _P2_PREFIX + _P2_WIDTH, _P2_SEED)
        _, prefix_kv, _ = model.forward_step(ids[:, :_P2_PREFIX], 0, 0)
        missing = {layer: kv for layer, kv in prefix_kv.items() if layer != 0}

        with pytest.raises(ValueError, match="prefix_kv must cover exactly layers"):
            model.forward_block(ids[:, _P2_PREFIX:], missing, _P2_PREFIX)

    def test_forward_block_wrong_shape_k_raises(self, tmp_path):
        model = _p2_model(tmp_path, "fx_p3_badshape")
        ids = _p2_ids(model, _P2_PREFIX + _P2_WIDTH, _P2_SEED)
        _, prefix_kv, _ = model.forward_step(ids[:, :_P2_PREFIX], 0, 0)
        k, v = prefix_kv[0]

        transposed = dict(prefix_kv)
        transposed[0] = (k.transpose(1, 2).contiguous(), v)
        with pytest.raises(ValueError, match="must have shape"):
            model.forward_block(ids[:, _P2_PREFIX:], transposed, _P2_PREFIX)

        truncated = dict(prefix_kv)
        truncated[0] = (k[:, :, :-1, :], v[:, :, :-1, :])
        with pytest.raises(ValueError, match="must have shape"):
            model.forward_block(ids[:, _P2_PREFIX:], truncated, _P2_PREFIX)


class TestBlockwiseGenerateDiffusionLlama:
    """Core D3/D4 behavior: blockwise_generate on a GQA DiffusionLlama
    fixture in both modes. Fails in Red with ModuleNotFoundError."""

    @pytest.mark.parametrize("batch", [1, 2], ids=["batch1", "batch2"])
    @pytest.mark.parametrize("draft_mode", [False, True], ids=["mask-init", "draft-verify"])
    def test_generate_contract_both_modes(self, tmp_path, draft_mode, batch):
        blockwise_generate = _import_blockwise_generate()
        from shadowinfer.execution.block_decoding import GenerateResult
        from shadowinfer.shadowkv.paged_kv_store import PagedKVStore

        model = _p2_model(tmp_path, "fx_p3_main")
        assert model.num_kv_heads < model.num_heads  # genuine GQA fixture
        prompt = _p3_prompt(model, _P3_SEED, batch=batch)
        config = _p3_config(draft_mode=draft_mode)

        result, stores = blockwise_generate(model, prompt, _P3_GEN, config)

        # Return contract: (GenerateResult, per-layer store dict).
        assert isinstance(result, GenerateResult)
        assert isinstance(stores, dict)

        # Sequences contract: [B, P + G] long, prompt bit-for-bit.
        assert result.sequences.shape == (batch, _P3_PROMPT + _P3_GEN)
        assert result.sequences.dtype == torch.long
        assert torch.equal(result.sequences[:, :_P3_PROMPT], prompt)
        generated = result.sequences[:, _P3_PROMPT:]
        assert bool(((generated >= 0) & (generated < model.vocab_size)).all())

        # Stats contract: the exact 17-key set, nothing more.
        assert set(result.stats) == _GENERATE_STATS_KEYS
        blocks = _P3_GEN // _P3_WIDTH
        assert result.stats["blocks"] == blocks
        assert result.stats["kv_write_rows_total"] == _P3_PROMPT + _P3_GEN
        assert result.stats["page_precision"] == "fp32"

        rounds = result.stats["rounds_per_block"]
        assert isinstance(rounds, list) and len(rounds) == blocks
        for rounds_j in rounds:
            assert isinstance(rounds_j, int) and rounds_j >= 1
        for key in (
            "unmasked_per_round",
            "fallback_per_round",
            "mean_confidence_per_round",
            "unmask_positions_per_round",
        ):
            per_block = result.stats[key]
            assert isinstance(per_block, list) and len(per_block) == blocks
            for j, per_round in enumerate(per_block):
                assert isinstance(per_round, list) and len(per_round) == rounds[j]
        assert isinstance(result.stats["block_time_ms"], list)
        assert len(result.stats["block_time_ms"]) == blocks

        # Draft-mode-specific keys: populated lists/float in draft mode,
        # None outside it (mirroring the current backend assembly).
        if draft_mode:
            assert isinstance(result.stats["draft_acceptance_rate"], float)
            assert isinstance(result.stats["rounds_to_fixpoint"], list)
            assert len(result.stats["rounds_to_fixpoint"]) == blocks
            assert isinstance(result.stats["changed_per_round"], list)
            assert len(result.stats["changed_per_round"]) == blocks
        else:
            for key in _DRAFT_ONLY_STATS_KEYS:
                assert result.stats[key] is None

        # Page residency: append exactly twice per row lifetime
        # (prefill P + one seal W per block = P + G); rounds only gather.
        assert set(stores) == set(range(model.num_layers))
        for layer_id in range(model.num_layers):
            store = stores[layer_id]
            assert isinstance(store, PagedKVStore)
            assert store.appended_tokens == _P3_PROMPT + _P3_GEN
            assert store.rewritten_tokens == 0
            assert store.page_evictions == 0
            assert store.max_seq == _P3_PROMPT + _P3_GEN
            # GQA-native construction (D4/D1): KV-head count, not Q heads.
            assert store.num_heads == model.num_kv_heads
            assert store.batch_size == batch
            assert store.head_dim == model.head_dim
            store.validate_against(batch, model.num_kv_heads, model.head_dim)
            k_all, v_all = store.gather()
            assert k_all.shape == (batch, model.num_kv_heads, _P3_PROMPT + _P3_GEN, model.head_dim)
            assert v_all.shape == (batch, model.num_kv_heads, _P3_PROMPT + _P3_GEN, model.head_dim)

        # Store width ties to the model's own KV-native forward width.
        _, prefill_kv, _ = model.forward_step(prompt, 0, 0)
        k0, _ = prefill_kv[0]
        assert k0.shape[1] == stores[0].num_heads


class TestBlockwiseGenerateFailFast:
    """D3 fail-fast reproduction (same messages, same order as
    validate_generate_inputs). Fails in Red with ModuleNotFoundError."""

    def test_over_length_raises(self, tmp_path):
        blockwise_generate = _import_blockwise_generate()
        model = _p2_model(tmp_path, "fx_p3_overlen")
        prompt = _p3_prompt(model, _P3_SEED)
        gen_len = model.max_seq_len  # P + max_seq_len > max_seq_len; % W == 0

        with pytest.raises(ValueError, match="exceeds model max_seq_len"):
            blockwise_generate(model, prompt, gen_len, _p3_config())

    def test_prompt_contains_mask_token_raises(self, tmp_path):
        blockwise_generate = _import_blockwise_generate()
        model = _p2_model(tmp_path, "fx_p3_maskin")
        prompt = _p3_prompt(model, _P3_SEED)
        prompt[0, 2] = _P3_MASK_TOKEN

        with pytest.raises(ValueError, match="must not contain mask_token_id"):
            blockwise_generate(model, prompt, _P3_GEN, _p3_config())

    def test_mask_token_id_out_of_vocab_raises(self, tmp_path):
        blockwise_generate = _import_blockwise_generate()
        model = _p2_model(tmp_path, "fx_p3_maskrange")
        prompt = _p3_prompt(model, _P3_SEED)

        with pytest.raises(ValueError, match="must be within the model vocabulary"):
            blockwise_generate(model, prompt, _P3_GEN, _p3_config(mask_token_id=model.vocab_size))

    def test_gen_len_not_multiple_of_block_size_raises(self, tmp_path):
        blockwise_generate = _import_blockwise_generate()
        model = _p2_model(tmp_path, "fx_p3_divis")
        prompt = _p3_prompt(model, _P3_SEED)

        with pytest.raises(ValueError, match="must be a multiple of block_size"):
            blockwise_generate(model, prompt, _P3_GEN + 2, _p3_config())

    def test_prompt_1d_raises(self, tmp_path):
        blockwise_generate = _import_blockwise_generate()
        model = _p2_model(tmp_path, "fx_p3_p1d")
        prompt = _p3_prompt(model, _P3_SEED)[0]  # [P]: not 2-D

        with pytest.raises(ValueError, match="prompt_ids must be 2-D"):
            blockwise_generate(model, prompt, _P3_GEN, _p3_config())

    def test_prompt_float_dtype_raises(self, tmp_path):
        blockwise_generate = _import_blockwise_generate()
        model = _p2_model(tmp_path, "fx_p3_pfloat")
        prompt = _p3_prompt(model, _P3_SEED).float()

        with pytest.raises(ValueError, match="prompt_ids must be integer token ids"):
            blockwise_generate(model, prompt, _P3_GEN, _p3_config())


class TestThinDelegationMhaRegression:
    """D3 refactoring anchor: backend.generate must stay bitwise-equal to
    the pure function on SimpleDiffusionLLM (MHA, getattr-default path),
    with _last_generate_stores semantics unchanged. The delegation tests
    fail in Red on import; the fail-fast stores test pins existing
    behavior and may pass."""

    @pytest.mark.parametrize("draft_mode", [False, True], ids=["mask-init", "draft-verify"])
    def test_backend_generate_equals_pure_function(self, draft_mode):
        blockwise_generate = _import_blockwise_generate()

        backend = _td_backend(_TD_SEED)
        prompt = _td_prompt(_TD_SEED + 1)
        config = _td_config(draft_mode=draft_mode)

        backend_result = backend.generate(prompt, _TD_GEN, config)
        pure_result, stores = blockwise_generate(backend._model, prompt, _TD_GEN, config)

        # Sequences bitwise-identical (identical model instance + logic).
        assert torch.equal(backend_result.sequences, pure_result.sequences)
        assert pure_result.sequences.shape == (_TD_BATCH, _TD_PROMPT + _TD_GEN)
        assert torch.equal(pure_result.sequences[:, :_TD_PROMPT], prompt)

        # Stats: same key set; per-key equality off the timing keys.
        assert set(backend_result.stats) == _GENERATE_STATS_KEYS
        assert set(pure_result.stats) == _GENERATE_STATS_KEYS
        for key in backend_result.stats:
            if key in _TIMING_STATS_KEYS:
                continue
            _assert_stats_values_equal(key, backend_result.stats[key], pure_result.stats[key])

        # _last_generate_stores semantics: populated, all layers, MHA
        # getattr-default width (num_kv_heads absent -> num_heads).
        assert set(stores) == set(range(_TD_LAYERS))
        for store in stores.values():
            assert store.num_heads == _TD_HEADS
        backend_stores = backend.last_generate_stores
        assert backend_stores is not None
        assert set(backend_stores) == set(range(_TD_LAYERS))

    def test_failed_generate_leaves_stores_none(self):
        """Fail-fast validation creates no stores: a failed generate on a
        fresh backend leaves last_generate_stores None (semantics the
        thin delegation must preserve)."""
        backend = _td_backend(_TD_SEED + 10)
        assert backend.last_generate_stores is None

        prompt = _td_prompt(_TD_SEED + 11)
        prompt[0, 2] = 0  # == config.mask_token_id
        with pytest.raises(ValueError, match="must not contain mask_token_id"):
            backend.generate(prompt, _TD_GEN, _td_config())
        assert backend.last_generate_stores is None


class TestBlockwiseModelProtocol:
    """D4 BlockwiseModel protocol surface (structural; no import of the
    Green-phase module, so these pass in Red)."""

    def test_diffusion_llama_satisfies_protocol_surface(self, tmp_path):
        model = _p2_model(tmp_path, "fx_p3_protocol")
        for attr in (
            "num_layers",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "max_seq_len",
            "vocab_size",
            "device",
            "mask_token_id",
            "config",
        ):
            assert hasattr(model, attr), f"missing protocol attribute {attr!r}"
        assert callable(model.forward_step)
        assert callable(model.forward_block)

    def test_simple_diffusion_llm_has_no_num_kv_heads(self):
        """D4 getattr-default premise: SimpleDiffusionLLM exposes no
        num_kv_heads attribute, so store assembly falls back to
        num_heads (MHA zero-diff)."""
        from shadowinfer.models.simple_diffusion_llm import SimpleDiffusionLLM

        torch.manual_seed(_TD_SEED + 30)
        model = SimpleDiffusionLLM(
            num_layers=1,
            num_heads=2,
            head_dim=8,
            hidden_dim=16,
            vocab_size=32,
            max_seq_len=32,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert not hasattr(model, "num_kv_heads")
        assert model.num_heads == 2


# ---------------------------------------------------------------------------
# P4: remasking baseline arm (AR018 T005, FR18.4)
#
# Red-phase tests for ``shadowinfer.execution.remasking`` (design D8:
# independent module, independent fail-fast, independent stats key set).
# The module does not exist yet (Green creates it), so every import of
# remasking_generate / RemaskingConfig happens INSIDE test bodies or
# lazy helpers: collection of the 71 existing tests must keep working,
# and every test below fails with ModuleNotFoundError in Red.
# ---------------------------------------------------------------------------

# P4 geometry reuses the P3 constants: prompt P=5, gen G=8, GQA fixture
# (num_kv_heads=2 < num_heads=8 from _TINY), mask token 999 < vocab 1000.
_P4_SEED = 4603
_P4_RANDOM_SEED = 4607
_P4_UNMASK_PER_PASS = 2
_P4_MAX_PASSES = 8  # >= G / unmask_per_pass: never binding on happy paths

# The exact four-key independent stats contract (design D8): explicitly
# NOT the 17-key blockwise contract (_GENERATE_STATS_KEYS above).
_REMASKING_STATS_KEYS = frozenset(
    {"passes_used", "unmasked_per_pass", "final_confidences", "mask_residuals"}
)


def _import_remasking():
    """Lazily import the AR018 T005 remasking arm (Green-phase module).

    Never called at module import time: the module does not exist in
    Red, and the 71 pre-existing tests must keep collecting.
    """
    from shadowinfer.execution.remasking import RemaskingConfig, remasking_generate

    return RemaskingConfig, remasking_generate


def _p4_config(**overrides: Any):
    """Small RemaskingConfig matching the P4 geometry (lazy import)."""
    RemaskingConfig = _import_remasking()[0]
    settings = dict(
        mask_token_id=_P3_MASK_TOKEN,
        unmask_per_pass=_P4_UNMASK_PER_PASS,
        max_passes=_P4_MAX_PASSES,
        strategy="confidence",
    )
    settings.update(overrides)
    return RemaskingConfig(**settings)


class TestRemaskingConfidenceConvergence:
    """confidence strategy: finite-pass full unmask with closed-form
    per-pass bookkeeping, structural monotonicity, and the sequence
    contract. Fails in Red with ModuleNotFoundError."""

    def test_finite_pass_full_unmask_closed_form_stats(self, tmp_path):
        remasking_generate = _import_remasking()[1]
        from shadowinfer.execution.block_decoding import GenerateResult

        model = _p2_model(tmp_path, "fx_p4_conv")
        assert model.num_kv_heads < model.num_heads  # genuine GQA fixture
        prompt = _p3_prompt(model, _P4_SEED)
        gen_len = _P3_GEN
        unmask = _P4_UNMASK_PER_PASS

        result = remasking_generate(model, prompt, gen_len, _p4_config())

        # Independent D8 stats key set: exactly the four keys, and none
        # of them collides with the 17-key blockwise contract.
        assert isinstance(result, GenerateResult)
        assert set(result.stats) == _REMASKING_STATS_KEYS
        assert not (_REMASKING_STATS_KEYS & _GENERATE_STATS_KEYS)

        # Finite-pass full unmask: no mask token left in the response.
        assert result.sequences.shape == (1, _P3_PROMPT + gen_len)
        assert result.sequences.dtype == torch.long
        response = result.sequences[:, _P3_PROMPT:]
        assert not bool((response == _P3_MASK_TOKEN).any())
        assert result.stats["mask_residuals"] == 0

        # Closed forms (batch 1): pass count and per-pass unmask totals.
        passes_used = gen_len // unmask
        assert result.stats["passes_used"] == passes_used
        assert result.stats["unmasked_per_pass"] == [
            min(unmask, gen_len - i * unmask) for i in range(passes_used)
        ]
        confidences = result.stats["final_confidences"]
        assert isinstance(confidences, list)
        assert len(confidences) == passes_used
        for entry in confidences:
            assert isinstance(entry, float)
            assert 0.0 <= entry <= 1.0

    def test_monotonicity_already_unmasked_positions_never_change(self, tmp_path):
        """Structural monotonicity via prefix runs: with the deterministic
        confidence strategy and a deterministic model, the run with
        max_passes=k follows the exact same trajectory as the first k
        passes of the run with max_passes=k+1, so every position already
        unmasked after k passes must keep its token after k+1 passes."""
        remasking_generate = _import_remasking()[1]

        model = _p2_model(tmp_path, "fx_p4_mono")
        prompt = _p3_prompt(model, _P4_SEED)

        runs = {}
        for max_passes in (1, 2, 3, 4):
            runs[max_passes] = remasking_generate(
                model, prompt, _P3_GEN, _p4_config(max_passes=max_passes)
            )

        for k in (1, 2, 3):
            resp_k = runs[k].sequences[:, _P3_PROMPT:]
            resp_k1 = runs[k + 1].sequences[:, _P3_PROMPT:]
            unmasked_k = resp_k != _P3_MASK_TOKEN
            # Run k unmasks exactly min(k * unmask_per_pass, G) positions.
            assert int(unmasked_k.sum()) == min(k * _P4_UNMASK_PER_PASS, _P3_GEN)
            # Monotonicity: those positions are bitwise identical in the
            # (k+1)-pass run (already-unmasked positions never change).
            assert torch.equal(resp_k[unmasked_k], resp_k1[unmasked_k])

    @pytest.mark.parametrize("batch", [1, 2], ids=["batch1", "batch2"])
    def test_sequence_contract_prompt_preserved(self, tmp_path, batch):
        remasking_generate = _import_remasking()[1]

        model = _p2_model(tmp_path, f"fx_p4_seq{batch}")
        prompt = _p3_prompt(model, _P4_SEED, batch=batch)

        result = remasking_generate(model, prompt, _P3_GEN, _p4_config())

        assert result.sequences.shape == (batch, _P3_PROMPT + _P3_GEN)
        assert result.sequences.dtype == torch.long
        assert torch.equal(result.sequences[:, :_P3_PROMPT], prompt)
        assert result.stats["mask_residuals"] == 0
        assert not bool((result.sequences[:, _P3_PROMPT:] == _P3_MASK_TOKEN).any())


class TestRemaskingRandomStrategy:
    """strategy="random": seeded-generator reproducibility and structural
    validity. Fails in Red with ModuleNotFoundError."""

    def test_seeded_generator_bitwise_reproducible(self, tmp_path):
        remasking_generate = _import_remasking()[1]

        model = _p2_model(tmp_path, "fx_p4_rand")
        prompt = _p3_prompt(model, _P4_SEED)
        config = _p4_config(strategy="random")

        gen_a = torch.Generator().manual_seed(_P4_RANDOM_SEED)
        gen_b = torch.Generator().manual_seed(_P4_RANDOM_SEED)
        result_a = remasking_generate(model, prompt, _P3_GEN, config, generator=gen_a)
        result_b = remasking_generate(model, prompt, _P3_GEN, config, generator=gen_b)

        # Identical seeds -> identical trajectories: bitwise-identical
        # sequences and identical stats (exact equality, not a tolerance:
        # both runs are the same deterministic computation).
        assert torch.equal(result_a.sequences, result_b.sequences)
        assert set(result_a.stats) == _REMASKING_STATS_KEYS
        for key in sorted(_REMASKING_STATS_KEYS):
            _assert_stats_values_equal(key, result_a.stats[key], result_b.stats[key])

    def test_different_seed_still_structurally_valid(self, tmp_path):
        """A different seed may pick different positions (not asserted:
        it is a probabilistic outcome); the result must still be a fully
        unmasked, structurally valid generation."""
        remasking_generate = _import_remasking()[1]

        model = _p2_model(tmp_path, "fx_p4_rand2")
        prompt = _p3_prompt(model, _P4_SEED)

        result = remasking_generate(
            model,
            prompt,
            _P3_GEN,
            _p4_config(strategy="random"),
            generator=torch.Generator().manual_seed(_P4_RANDOM_SEED + 1),
        )

        assert result.sequences.shape == (1, _P3_PROMPT + _P3_GEN)
        assert result.sequences.dtype == torch.long
        assert torch.equal(result.sequences[:, :_P3_PROMPT], prompt)
        assert result.stats["mask_residuals"] == 0
        assert not bool((result.sequences[:, _P3_PROMPT:] == _P3_MASK_TOKEN).any())

    def test_no_generator_falls_back_to_global_rng(self, tmp_path):
        """strategy="random" without a generator must still complete
        (global RNG) and fully unmask."""
        remasking_generate = _import_remasking()[1]

        model = _p2_model(tmp_path, "fx_p4_rand3")
        prompt = _p3_prompt(model, _P4_SEED)

        torch.manual_seed(_P4_RANDOM_SEED + 2)
        result = remasking_generate(model, prompt, _P3_GEN, _p4_config(strategy="random"))

        assert result.stats["mask_residuals"] == 0
        assert not bool((result.sequences[:, _P3_PROMPT:] == _P3_MASK_TOKEN).any())


class TestRemaskingMaxPassesExhaustion:
    """max_passes exhaustion: honest residual bookkeeping, per element.
    Fails in Red with ModuleNotFoundError."""

    def test_exhaustion_residual_bookkeeping(self, tmp_path):
        remasking_generate = _import_remasking()[1]

        model = _p2_model(tmp_path, "fx_p4_exhaust")
        prompt = _p3_prompt(model, _P4_SEED)
        gen_len = _P3_GEN
        unmask = 1
        max_passes = 3

        result = remasking_generate(
            model,
            prompt,
            gen_len,
            _p4_config(unmask_per_pass=unmask, max_passes=max_passes),
        )

        response = result.sequences[:, _P3_PROMPT:]
        masked = response == _P3_MASK_TOKEN

        # Closed form (batch 1): the loop stops at max_passes with
        # gen_len - max_passes * unmask_per_pass masks left, honestly
        # reported rather than silently truncated.
        assert result.stats["passes_used"] == max_passes
        assert result.stats["mask_residuals"] == gen_len - max_passes * unmask
        assert int(masked.sum()) == gen_len - max_passes * unmask
        assert int((~masked).sum()) == max_passes * unmask
        assert result.stats["unmasked_per_pass"] == [unmask] * max_passes
        assert len(result.stats["final_confidences"]) == max_passes

    def test_exhaustion_batch2_independent_elements(self, tmp_path):
        """Batch 2, same config on both elements: each element follows
        its own trajectory, unmasking exactly unmask_per_pass positions
        per pass until max_passes; residuals sum across the batch."""
        remasking_generate = _import_remasking()[1]

        model = _p2_model(tmp_path, "fx_p4_exhaust2")
        prompt = _p3_prompt(model, _P4_SEED, batch=2)
        gen_len = _P3_GEN
        unmask = 1
        max_passes = 3

        result = remasking_generate(
            model,
            prompt,
            gen_len,
            _p4_config(unmask_per_pass=unmask, max_passes=max_passes),
        )

        response = result.sequences[:, _P3_PROMPT:]
        masked = response == _P3_MASK_TOKEN

        assert result.stats["passes_used"] == max_passes
        # Cross-batch closed form: 2 * (G - passes * unmask_per_pass).
        assert result.stats["mask_residuals"] == 2 * (gen_len - max_passes * unmask)
        assert int(masked.sum()) == 2 * (gen_len - max_passes * unmask)
        # Per element: exactly max_passes * unmask_per_pass unmasked.
        assert torch.equal(
            (~masked).sum(dim=1), torch.full((2,), max_passes * unmask, dtype=torch.long)
        )
        # Prompt preserved bit-for-bit on both elements.
        assert torch.equal(result.sequences[:, :_P3_PROMPT], prompt)


class TestRemaskingFailFast:
    """D8 independent fail-fast: shared checks mirror
    validate_generate_inputs wording; remasking-only checks (no block
    divisibility; instead gen_len / unmask_per_pass / max_passes /
    strategy) follow the repo's "X must be >= 1" precedent. Configs are
    built inside the raises block so validation living either in
    RemaskingConfig or in remasking_generate satisfies the contract.
    Fails in Red with ModuleNotFoundError."""

    def test_prompt_1d_raises(self, tmp_path):
        _, remasking_generate = _import_remasking()
        model = _p2_model(tmp_path, "fx_p4_ff_1d")
        prompt = _p3_prompt(model, _P4_SEED)[0]  # [P]: not 2-D

        with pytest.raises(ValueError, match="prompt_ids must be 2-D"):
            remasking_generate(model, prompt, _P3_GEN, _p4_config())

    def test_prompt_float_dtype_raises(self, tmp_path):
        _, remasking_generate = _import_remasking()
        model = _p2_model(tmp_path, "fx_p4_ff_float")
        prompt = _p3_prompt(model, _P4_SEED).float()

        with pytest.raises(ValueError, match="prompt_ids must be integer token ids"):
            remasking_generate(model, prompt, _P3_GEN, _p4_config())

    def test_gen_len_zero_raises(self, tmp_path):
        _, remasking_generate = _import_remasking()
        model = _p2_model(tmp_path, "fx_p4_ff_genlen")
        prompt = _p3_prompt(model, _P4_SEED)

        with pytest.raises(ValueError, match="gen_len must be >= 1"):
            remasking_generate(model, prompt, 0, _p4_config())

    def test_over_length_raises(self, tmp_path):
        _, remasking_generate = _import_remasking()
        model = _p2_model(tmp_path, "fx_p4_ff_overlen")
        prompt = _p3_prompt(model, _P4_SEED)

        with pytest.raises(ValueError, match="exceeds model max_seq_len"):
            remasking_generate(model, prompt, model.max_seq_len, _p4_config())

    def test_mask_token_id_out_of_vocab_raises(self, tmp_path):
        _, remasking_generate = _import_remasking()
        model = _p2_model(tmp_path, "fx_p4_ff_maskrange")
        prompt = _p3_prompt(model, _P4_SEED)

        with pytest.raises(ValueError, match="must be within the model vocabulary"):
            remasking_generate(model, prompt, _P3_GEN, _p4_config(mask_token_id=model.vocab_size))

    def test_prompt_contains_mask_token_raises(self, tmp_path):
        _, remasking_generate = _import_remasking()
        model = _p2_model(tmp_path, "fx_p4_ff_maskin")
        prompt = _p3_prompt(model, _P4_SEED)
        prompt[0, 2] = _P3_MASK_TOKEN

        with pytest.raises(ValueError, match="must not contain mask_token_id"):
            remasking_generate(model, prompt, _P3_GEN, _p4_config())

    def test_unmask_per_pass_zero_raises(self, tmp_path):
        _, remasking_generate = _import_remasking()
        model = _p2_model(tmp_path, "fx_p4_ff_upp")
        prompt = _p3_prompt(model, _P4_SEED)

        with pytest.raises(ValueError, match="unmask_per_pass must be >= 1"):
            remasking_generate(model, prompt, _P3_GEN, _p4_config(unmask_per_pass=0))

    def test_max_passes_zero_raises(self, tmp_path):
        _, remasking_generate = _import_remasking()
        model = _p2_model(tmp_path, "fx_p4_ff_mp")
        prompt = _p3_prompt(model, _P4_SEED)

        with pytest.raises(ValueError, match="max_passes must be >= 1"):
            remasking_generate(model, prompt, _P3_GEN, _p4_config(max_passes=0))

    def test_bogus_strategy_raises(self, tmp_path):
        _, remasking_generate = _import_remasking()
        model = _p2_model(tmp_path, "fx_p4_ff_strategy")
        prompt = _p3_prompt(model, _P4_SEED)

        with pytest.raises(ValueError, match="strategy must be"):
            remasking_generate(model, prompt, _P3_GEN, _p4_config(strategy="bogus"))


class TestRemaskingVsBlockwiseComparability:
    """srs FR18.4 acceptance 2: remasking (fully bidirectional, D5(a))
    and the blockwise arm (segment-causal, D5(b)/(c)) both complete on
    the same fixture + prompt and the cross-arm token agreement rate is
    a computable metric. Fails in Red with ModuleNotFoundError (the
    remasking import is in the test body)."""

    def test_token_agreement_metric_computable_no_threshold(self, tmp_path):
        remasking_generate = _import_remasking()[1]
        blockwise_generate = _import_blockwise_generate()

        model = _p2_model(tmp_path, "fx_p4_compare")
        prompt = _p3_prompt(model, _P4_SEED)

        # Remasking arm: confidence strategy, full bidirectional passes.
        remasking_result = remasking_generate(model, prompt, _P3_GEN, _p4_config())
        # Blockwise arm: mask-init mode, same mask token, block_size=4
        # divides gen_len=8 (reuses the P3 config helper).
        blockwise_result, _ = blockwise_generate(model, prompt, _P3_GEN, _p3_config())

        remasking_resp = remasking_result.sequences[:, _P3_PROMPT:]
        blockwise_resp = blockwise_result.sequences[:, _P3_PROMPT:]

        # Both arms fully unmask on this finite geometry (blockwise via
        # its forced completion; remasking by max_passes headroom).
        assert not bool((remasking_resp == _P3_MASK_TOKEN).any())
        assert not bool((blockwise_resp == _P3_MASK_TOKEN).any())

        # The comparability metric is computable: a float in [0, 1].
        # Random fixture weights carry no semantic claim, so there is
        # deliberately NO numeric threshold on the agreement (the s13
        # S3 threshold is a server-side assertion on real weights).
        value = float((remasking_resp == blockwise_resp).float().mean())
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0
