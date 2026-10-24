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
import re
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from safetensors.torch import load_file, save_file

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
