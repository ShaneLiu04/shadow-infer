"""AR017 T001: draft-verify scheduling primitive tests.

Exercises the not-yet-implemented draft/verify round primitives of
``shadowinfer.execution.block_decoding`` (FR17.1): ``draft_round`` fills
every position of an all-mask block with its argmax token along with the
max softmax probability, ``verify_round`` flags positions whose argmax
disagrees with the current block tokens (self-consistency replacement),
and ``BlockDecodingConfig`` gains a boolean ``draft_mode`` field validated
as a strict bool. Argmax ties resolve to the smallest token id (torch
default), confidences are the full-position softmax maximum, batch
elements are judged independently, and degenerate inputs (wrong rank,
zero-width blocks, shape mismatch, non-integer tokens, non-bool
draft_mode) are rejected with ``ValueError``.

The T002 partition appended below exercises the draft_mode pipeline
branch of ``PyTorchModelBackend.generate`` and
``blockwise_generate_baseline`` (FR17.2): full draft-verify completion
and page-residency ledgers at two scales, the max_rounds=1 draft-only
forced degeneration, forced-completions accounting bounds, mask-init
regression anchors for draft_mode=False, and the fail-fast contract
shared by both draft_mode values.

All tests use real tensors; nothing is mocked.
"""

import math

import pytest
import torch

from shadowinfer.execution import BlockDecodingConfig, draft_round, verify_round
from tests.test_block_decoding import (
    GP_BATCH,
    GP_GEN,
    GP_LAYERS,
    GP_PROMPT,
    GP_VOCAB,
    GP_WIDTH,
    _gp_backend,
    _gp_config,
    _gp_model_config,
    _gp_prompt,
    _t5_direct_model,
)

DV_SEED = 2034
DV_VOCAB = 8
DV_WIDTH = 4


def _conf_row(conf, argmax_token, vocab):
    """[vocab] logits row whose max softmax probability is ``conf`` (up to
    fp rounding), attained at ``argmax_token`` uniquely."""
    gap = math.log((1.0 - conf) / (conf * (vocab - 1)))
    row = torch.full((vocab,), float(gap))
    row[int(argmax_token)] = 0.0
    return row


def _block_from_spec(confs, argmax_tokens, vocab=DV_VOCAB):
    """[1, W, V] logits block; position w has confidence ``confs[w]``
    attained at token ``argmax_tokens[w]``."""
    rows = [_conf_row(conf, token, vocab) for conf, token in zip(confs, argmax_tokens)]
    return torch.stack(rows).unsqueeze(0)


def _softmax_conf(logits_block):
    """[B, W] max softmax probability and argmax token per position."""
    probs = torch.softmax(logits_block, dim=-1)
    return probs.max(dim=-1).values, probs.max(dim=-1).indices


# ---------------------------------------------------------------------------
# T001: draft_round / verify_round / BlockDecodingConfig.draft_mode
# ---------------------------------------------------------------------------


class TestDraftRound:
    def test_all_masked_positions_filled_with_argmax(self):
        # FR17.1: given the first-round logits of a fully masked block,
        # drafting must fill every one of the W positions with argmax.
        tokens_spec = [3, 1, 6, 0]
        logits = _block_from_spec([0.95, 0.5, 0.97, 0.4], tokens_spec)

        tokens, confidences = draft_round(logits)

        assert torch.equal(tokens, torch.tensor([tokens_spec]))
        assert confidences.shape == (1, DV_WIDTH)

    def test_confidences_match_full_softmax_max(self):
        # Draft confidences are the full-position softmax maximum at every
        # position, not a threshold-filtered subset.
        confs = [0.95, 0.5, 0.97, 0.4]
        logits = _block_from_spec(confs, [3, 1, 6, 0])

        tokens, confidences = draft_round(logits)

        ref_conf, _ = _softmax_conf(logits)
        assert torch.allclose(confidences, ref_conf, atol=1e-6)
        for w, conf in enumerate(confs):
            assert abs(float(confidences[0, w]) - conf) < 1e-6

    def test_output_shapes_and_dtypes(self):
        gen = torch.Generator().manual_seed(DV_SEED)
        logits = torch.randn(2, DV_WIDTH, DV_VOCAB, generator=gen)

        tokens, confidences = draft_round(logits)

        assert tokens.shape == (2, DV_WIDTH)
        assert tokens.dtype == torch.int64
        assert confidences.shape == (2, DV_WIDTH)
        assert confidences.dtype.is_floating_point

    def test_tie_breaks_to_smallest_token_id(self):
        # Tokens 2 and 6 share the exact maximum logit; torch.argmax
        # semantics must keep the smaller id.
        row = torch.zeros(DV_VOCAB)
        row[2] = 5.0
        row[6] = 5.0
        logits = row.unsqueeze(0).unsqueeze(0)

        tokens, _ = draft_round(logits)

        assert torch.equal(tokens, torch.tensor([[2]]))

    def test_reject_logits_not_3d(self):
        with pytest.raises(ValueError):
            draft_round(torch.randn(DV_WIDTH, DV_VOCAB))
        with pytest.raises(ValueError):
            draft_round(torch.randn(1, DV_WIDTH, 2, DV_VOCAB))

    def test_reject_zero_width_block(self):
        logits = torch.randn(1, 0, DV_VOCAB)
        with pytest.raises(ValueError):
            draft_round(logits)


class TestVerifyRound:
    def test_self_consistent_block_yields_zero_changes(self):
        # FR17.1 two-round convergence trajectory: round 1 drafts the
        # block, round 2 re-checks the drafted tokens against unchanged
        # argmax logits -- every position is self-consistent, so the
        # verify pass must report zero replacements.
        tokens_spec = [3, 1, 6, 0]
        logits = _block_from_spec([0.95, 0.97, 0.99, 0.96], tokens_spec)

        drafted, _ = draft_round(logits)
        changed, tokens, confidences = verify_round(logits, drafted)

        assert torch.equal(changed, torch.zeros(1, DV_WIDTH, dtype=torch.bool))
        assert torch.equal(tokens, drafted)
        assert confidences.shape == (1, DV_WIDTH)

    def test_mismatched_positions_flagged_with_argmax_tokens(self):
        # FR17.1 first-round draft with some positions wrong: exactly the
        # disagreeing positions are flagged, and the replacement tokens
        # are the full-position argmax at those positions.
        tokens_spec = [3, 1, 6, 0]
        logits = _block_from_spec([0.95, 0.97, 0.99, 0.96], tokens_spec)
        block_tokens = torch.tensor([[3, 2, 6, 5]])  # positions 1 and 3 wrong

        changed, tokens, confidences = verify_round(logits, block_tokens)

        expected_changed = torch.tensor([[False, True, False, True]])
        assert torch.equal(changed, expected_changed)
        assert torch.equal(tokens, torch.tensor([tokens_spec]))
        assert int(tokens[0, 1]) == 1
        assert int(tokens[0, 3]) == 0

    def test_confidences_are_full_softmax_max(self):
        # verify_round confidences cover every position (not only the
        # changed ones), matching the full softmax maximum.
        logits = _block_from_spec([0.95, 0.5, 0.97, 0.4], [3, 1, 6, 0])
        block_tokens = torch.tensor([[3, 2, 6, 0]])  # position 1 changed

        changed, tokens, confidences = verify_round(logits, block_tokens)

        ref_conf, _ = _softmax_conf(logits)
        assert changed.any()
        assert torch.allclose(confidences, ref_conf, atol=1e-6)
        # An unchanged position still carries its confidence.
        assert abs(float(confidences[0, 0]) - 0.95) < 1e-6

    def test_tie_breaks_to_smallest_token_id(self):
        row = torch.zeros(DV_VOCAB)
        row[2] = 5.0
        row[6] = 5.0
        logits = row.unsqueeze(0).unsqueeze(0)
        block_tokens = torch.tensor([[0]])  # disagrees with the argmax tie

        changed, tokens, _ = verify_round(logits, block_tokens)

        assert torch.equal(changed, torch.tensor([[True]]))
        assert torch.equal(tokens, torch.tensor([[2]]))

    def test_batch_elements_judged_independently(self):
        # FR17.1 batch independence: element A carries wrong draft tokens
        # (still converging) while element B is already self-consistent;
        # A's replacements must not perturb B's all-False verdict, and B's
        # tokens stay its own argmax.
        elem_a = _block_from_spec([0.95, 0.5, 0.97, 0.4], [3, 1, 6, 0])
        elem_b = _block_from_spec([0.99, 0.98, 0.97, 0.96], [2, 4, 5, 7])
        logits = torch.cat([elem_a, elem_b], dim=0)
        block_tokens = torch.tensor(
            [
                [3, 2, 6, 5],  # A: positions 1 and 3 wrong
                [2, 4, 5, 7],  # B: fully self-consistent
            ]
        )

        changed, tokens, confidences = verify_round(logits, block_tokens)

        expected_changed = torch.tensor(
            [
                [False, True, False, True],
                [False, False, False, False],
            ]
        )
        assert torch.equal(changed, expected_changed)
        assert torch.equal(tokens[0], torch.tensor([3, 1, 6, 0]))
        assert torch.equal(tokens[1], torch.tensor([2, 4, 5, 7]))
        assert confidences.shape == (2, DV_WIDTH)

    def test_output_shapes_and_dtypes(self):
        gen = torch.Generator().manual_seed(DV_SEED + 1)
        logits = torch.randn(2, DV_WIDTH, DV_VOCAB, generator=gen)
        gen2 = torch.Generator().manual_seed(DV_SEED + 2)
        block_tokens = torch.randint(0, DV_VOCAB, (2, DV_WIDTH), generator=gen2)

        changed, tokens, confidences = verify_round(logits, block_tokens)

        assert changed.shape == (2, DV_WIDTH)
        assert changed.dtype == torch.bool
        assert tokens.shape == (2, DV_WIDTH)
        assert tokens.dtype == torch.int64
        assert confidences.shape == (2, DV_WIDTH)
        assert confidences.dtype.is_floating_point

    def test_reject_logits_not_3d(self):
        block_tokens = torch.zeros(1, DV_WIDTH, dtype=torch.long)
        with pytest.raises(ValueError):
            verify_round(torch.randn(DV_WIDTH, DV_VOCAB), block_tokens)
        with pytest.raises(ValueError):
            verify_round(torch.randn(1, DV_WIDTH, 2, DV_VOCAB), block_tokens)

    def test_reject_block_tokens_not_2d(self):
        logits = torch.randn(1, DV_WIDTH, DV_VOCAB)
        with pytest.raises(ValueError):
            verify_round(logits, torch.zeros(DV_WIDTH, dtype=torch.long))
        with pytest.raises(ValueError):
            verify_round(logits, torch.zeros(1, DV_WIDTH, 1, dtype=torch.long))

    @pytest.mark.parametrize(
        "shape",
        [
            (1, DV_WIDTH + 1),  # width mismatch
            (2, DV_WIDTH),  # batch mismatch
        ],
    )
    def test_reject_block_tokens_shape_mismatch(self, shape):
        logits = torch.randn(1, DV_WIDTH, DV_VOCAB)
        block_tokens = torch.zeros(shape, dtype=torch.long)
        with pytest.raises(ValueError):
            verify_round(logits, block_tokens)

    def test_reject_non_integer_block_tokens(self):
        logits = torch.randn(1, DV_WIDTH, DV_VOCAB)
        with pytest.raises(ValueError):
            verify_round(logits, torch.zeros(1, DV_WIDTH, dtype=torch.float32))
        with pytest.raises(ValueError):
            verify_round(logits, torch.zeros(1, DV_WIDTH, dtype=torch.float64))

    def test_reject_zero_width_block(self):
        logits = torch.randn(1, 0, DV_VOCAB)
        block_tokens = torch.zeros(1, 0, dtype=torch.long)
        with pytest.raises(ValueError):
            verify_round(logits, block_tokens)


class TestDraftModeConfig:
    def test_draft_mode_defaults_false(self):
        config = BlockDecodingConfig()
        assert config.draft_mode is False

    def test_draft_mode_true_is_valid(self):
        config = BlockDecodingConfig(draft_mode=True)
        assert config.draft_mode is True
        # The new field leaves the six existing defaults untouched.
        assert config.block_size == 32
        assert config.max_rounds == 8
        assert config.confidence_threshold == 0.9
        assert config.mask_token_id == 0
        assert config.page_size == 32
        assert config.page_precision == "fp32"

    def test_draft_mode_true_with_explicit_settings_is_valid(self):
        config = BlockDecodingConfig(
            block_size=4,
            max_rounds=2,
            confidence_threshold=0.5,
            mask_token_id=0,
            page_size=4,
            page_precision="fp16",
            draft_mode=True,
        )
        assert config.draft_mode is True
        assert config.block_size == 4

    @pytest.mark.parametrize(
        "draft_mode",
        [
            1,
            0,
            "true",
            "False",
            None,
            1.0,
            0.0,
            [True],
        ],
    )
    def test_non_bool_draft_mode_raises_value_error(self, draft_mode):
        with pytest.raises(ValueError):
            BlockDecodingConfig(draft_mode=draft_mode)

    def test_validate_accepts_bool_draft_mode(self):
        config = BlockDecodingConfig(draft_mode=False)
        config.validate()
        config_true = BlockDecodingConfig(draft_mode=True)
        config_true.validate()


# ---------------------------------------------------------------------------
# T002: generate / blockwise_generate_baseline draft_mode branch (FR17.2)
# ---------------------------------------------------------------------------

DV_PIPE_SEED = 2601
DV_PIPE_SCALES = (
    (DV_PIPE_SEED, GP_PROMPT, GP_GEN),  # (P=6, G=8): two blocks
    (DV_PIPE_SEED + 2, 10, 16),  # (P=10, G=16): four blocks
)


def _dv_baseline(model, prompt, gen_len, config):
    """Full-recompute reference loop (blockwise_generate_baseline)."""
    from shadowinfer.execution.block_decoding import blockwise_generate_baseline

    return blockwise_generate_baseline(model, prompt, gen_len, config)


class TestGenerateDraftVerifyPipeline:
    @pytest.mark.parametrize("seed,prompt_len,gen_len", DV_PIPE_SCALES)
    def test_draft_mode_completes_with_no_mask_residue(self, seed, prompt_len, gen_len):
        # FR17.2(a): draft-verify completes at both scales: [B, P + G]
        # long sequences, prompt preserved bit-for-bit, every generated
        # token in vocabulary, and no mask_token_id in the generated
        # region -- the draft round fills the whole block in one shot, so
        # no MASK position can survive into the returned sequences. The
        # seeds are known (repo convention) to also keep legitimate
        # argmax tokens off the mask id, which doubles as vocab token 0.
        backend = _gp_backend(seed)
        prompt = _gp_prompt(seed + 1, length=prompt_len)
        config = _gp_config(draft_mode=True)

        result = backend.generate(prompt, gen_len, config)

        assert result.sequences.shape == (GP_BATCH, prompt_len + gen_len)
        assert result.sequences.dtype == torch.long
        assert torch.equal(result.sequences[:, :prompt_len], prompt)
        generated = result.sequences[:, prompt_len:]
        assert bool(((generated >= 0) & (generated < GP_VOCAB)).all())
        assert not bool((generated == config.mask_token_id).any())
        assert result.stats["blocks"] == gen_len // GP_WIDTH

    @pytest.mark.parametrize("seed,prompt_len,gen_len", DV_PIPE_SCALES)
    def test_draft_mode_page_residency_ledger(self, seed, prompt_len, gen_len):
        # FR17.2(b): prefill appends P rows once and each block's sealing
        # pass appends W rows exactly once (G total); draft/verify rounds
        # only gather from the pages, so the per-layer ledger must read
        # appended == P + G with zero rewrites and zero evictions.
        from shadowinfer.shadowkv.paged_kv_store import PagedKVStore

        backend = _gp_backend(seed)
        prompt = _gp_prompt(seed + 1, length=prompt_len)

        backend.generate(prompt, gen_len, _gp_config(draft_mode=True))

        stores = backend.last_generate_stores
        assert stores is not None
        assert set(stores.keys()) == set(range(GP_LAYERS))
        for layer_id in range(GP_LAYERS):
            store = stores[layer_id]
            assert isinstance(store, PagedKVStore)
            assert store.appended_tokens == prompt_len + gen_len
            assert store.rewritten_tokens == 0
            assert store.page_evictions == 0
            assert store.max_seq == prompt_len + gen_len

    def test_max_rounds_one_degrades_to_draft_only_completion(self):
        # FR17.2(c): max_rounds=1 caps the block loop at the draft round
        # (no verify round exists); every block still completes through
        # the forced path with a fully filled, mask-free generated region.
        seed = DV_PIPE_SEED + 4
        backend = _gp_backend(seed)
        prompt = _gp_prompt(seed + 1)
        config = _gp_config(draft_mode=True, max_rounds=1)

        result = backend.generate(prompt, GP_GEN, config)

        assert result.sequences.shape == (GP_BATCH, GP_PROMPT + GP_GEN)
        assert torch.equal(result.sequences[:, :GP_PROMPT], prompt)
        generated = result.sequences[:, GP_PROMPT:]
        assert not bool((generated == config.mask_token_id).any())
        assert result.stats["blocks"] == GP_GEN // GP_WIDTH

    def test_max_rounds_one_forces_every_block(self):
        # FR17.2(d): with max_rounds=1 the single round is the draft, so
        # no block reaches a verified fixpoint and every block is
        # force-sealed: forced_completions == blocks exactly.
        seed = DV_PIPE_SEED + 6
        backend = _gp_backend(seed)
        prompt = _gp_prompt(seed + 1)
        config = _gp_config(draft_mode=True, max_rounds=1)

        result = backend.generate(prompt, GP_GEN, config)

        assert result.stats["forced_completions"] == GP_GEN // GP_WIDTH

    def test_default_max_rounds_forced_completions_bounded_by_blocks(self):
        # FR17.2(d): with max_rounds=8 a block can be force-sealed at
        # most once, so forced_completions stays within [0, blocks].
        seed = DV_PIPE_SEED + 8
        backend = _gp_backend(seed)
        prompt = _gp_prompt(seed + 1)

        result = backend.generate(prompt, GP_GEN, _gp_config(draft_mode=True))

        assert 0 <= result.stats["forced_completions"] <= GP_GEN // GP_WIDTH

    def test_draft_mode_output_diverges_from_mask_init_path(self):
        # FR17.2 Red anchor: draft-verify is a different decoding
        # algorithm from threshold-gated mask-init scheduling -- the
        # draft round fills the whole block from the all-MASK argmax in
        # one shot, while mask-init unmasks gradually (top-1 fallback per
        # round on this model), so the returned sequences cannot coincide
        # bit-for-bit. Equality means the draft_mode branch is not wired
        # into generate -- the pre-T002 state where draft_mode is
        # silently ignored.
        seed = DV_PIPE_SEED
        backend = _gp_backend(seed)
        prompt = _gp_prompt(seed + 1)

        result_true = backend.generate(prompt, GP_GEN, _gp_config(draft_mode=True))
        result_false = backend.generate(prompt, GP_GEN, _gp_config(draft_mode=False))

        assert not torch.equal(result_true.sequences, result_false.sequences)

    def test_baseline_draft_mode_completes_with_no_mask_residue(self):
        # FR17.2: the baseline arm shares the draft-verify completion
        # contract (shape, prompt preservation, mask-free generation);
        # generate-vs-baseline bitwise equality is pinned by T004.
        seed = DV_PIPE_SEED + 2
        model = _t5_direct_model(seed)
        prompt = _gp_prompt(seed + 1)
        config = _gp_config(draft_mode=True)

        base_result = _dv_baseline(model, prompt, GP_GEN, config)

        assert base_result.sequences.shape == (GP_BATCH, GP_PROMPT + GP_GEN)
        assert torch.equal(base_result.sequences[:, :GP_PROMPT], prompt)
        generated = base_result.sequences[:, GP_PROMPT:]
        assert not bool((generated == config.mask_token_id).any())

    def test_baseline_draft_mode_diverges_from_mask_init_path(self):
        # FR17.2 Red anchor (baseline arm): the same algorithm-level
        # argument as the generate-arm anchor -- the baseline loop must
        # not return identical sequences with draft_mode on and off,
        # otherwise the draft branch is not wired into the baseline.
        seed = DV_PIPE_SEED + 2
        model = _t5_direct_model(seed)
        prompt = _gp_prompt(seed + 1)

        result_true = _dv_baseline(model, prompt, GP_GEN, _gp_config(draft_mode=True))
        result_false = _dv_baseline(model, prompt, GP_GEN, _gp_config(draft_mode=False))

        assert not torch.equal(result_true.sequences, result_false.sequences)


class TestMaskInitRegressionAnchor:
    def test_explicit_draft_mode_false_matches_default_path(self):
        # FR17.2(6): draft_mode=False is the explicit default -- the same
        # backend and prompt must produce bit-identical sequences and
        # identical round accounting with the field omitted or explicitly
        # False; adding the field must not perturb the mask-init path.
        seed = DV_PIPE_SEED + 10
        backend = _gp_backend(seed)
        prompt = _gp_prompt(seed + 1)

        result_default = backend.generate(prompt, GP_GEN, _gp_config())
        result_false = backend.generate(prompt, GP_GEN, _gp_config(draft_mode=False))

        assert torch.equal(result_false.sequences, result_default.sequences)
        assert result_false.stats["rounds_per_block"] == result_default.stats["rounds_per_block"]
        assert result_false.stats["total_rounds"] == result_default.stats["total_rounds"]

    @pytest.mark.parametrize("seed,prompt_len,gen_len", DV_PIPE_SCALES)
    def test_draft_mode_false_generate_matches_baseline_bitwise(self, seed, prompt_len, gen_len):
        # FR17.2(6): the AR016 exactness property survives the field
        # addition -- with draft_mode=False the store-backed generate arm
        # and the full-recompute baseline arm agree bit-for-bit.
        backend = _gp_backend(seed)
        prompt = _gp_prompt(seed + 1, length=prompt_len)
        config = _gp_config(draft_mode=False)

        gen_result = backend.generate(prompt, gen_len, config)
        base_result = _dv_baseline(_t5_direct_model(seed), prompt, gen_len, config)

        assert gen_result.sequences.shape == (GP_BATCH, prompt_len + gen_len)
        assert torch.equal(gen_result.sequences[:, :prompt_len], prompt)
        assert torch.equal(gen_result.sequences, base_result.sequences)


class TestDraftModeFailFast:
    """FR17.2(7): fail-fast validation is fully shared with the mask-init
    path for both draft_mode values; a rejected call raises before any
    page store is created (last_generate_stores stays None)."""

    @pytest.mark.parametrize("draft_mode", [False, True])
    def test_prompt_with_mask_token_raises(self, draft_mode):
        backend = _gp_backend(DV_PIPE_SEED + 30)
        prompt = _gp_prompt(DV_PIPE_SEED + 31)
        prompt[0, 2] = 0  # == config.mask_token_id

        with pytest.raises(ValueError):
            backend.generate(prompt, GP_GEN, _gp_config(draft_mode=draft_mode))
        assert backend.last_generate_stores is None

    @pytest.mark.parametrize("draft_mode", [False, True])
    def test_prompt_plus_gen_len_over_max_seq_len_raises(self, draft_mode):
        backend = _gp_backend(DV_PIPE_SEED + 32)
        prompt = _gp_prompt(DV_PIPE_SEED + 33)

        with pytest.raises(ValueError):
            backend.generate(prompt, 60, _gp_config(draft_mode=draft_mode))
        assert backend.last_generate_stores is None

    @pytest.mark.parametrize("draft_mode", [False, True])
    def test_gen_len_not_multiple_of_block_size_raises(self, draft_mode):
        backend = _gp_backend(DV_PIPE_SEED + 34)
        prompt = _gp_prompt(DV_PIPE_SEED + 35)

        with pytest.raises(ValueError):
            backend.generate(prompt, 6, _gp_config(draft_mode=draft_mode))
        assert backend.last_generate_stores is None

    @pytest.mark.parametrize("draft_mode", [False, True])
    def test_prompt_1d_raises(self, draft_mode):
        backend = _gp_backend(DV_PIPE_SEED + 36)
        prompt = _gp_prompt(DV_PIPE_SEED + 37)[0]  # [P]: not 2-D

        with pytest.raises(ValueError):
            backend.generate(prompt, GP_GEN, _gp_config(draft_mode=draft_mode))
        assert backend.last_generate_stores is None

    @pytest.mark.parametrize("draft_mode", [False, True])
    def test_prompt_float_dtype_raises(self, draft_mode):
        backend = _gp_backend(DV_PIPE_SEED + 38)
        gen = torch.Generator().manual_seed(DV_PIPE_SEED + 39)
        prompt = torch.rand(GP_BATCH, GP_PROMPT, generator=gen)

        with pytest.raises(ValueError):
            backend.generate(prompt, GP_GEN, _gp_config(draft_mode=draft_mode))
        assert backend.last_generate_stores is None

    @pytest.mark.parametrize("bad_mask_token_id", [GP_VOCAB, -1])
    @pytest.mark.parametrize("draft_mode", [False, True])
    def test_mask_token_id_out_of_vocab_raises(self, draft_mode, bad_mask_token_id):
        backend = _gp_backend(DV_PIPE_SEED + 40)
        prompt = _gp_prompt(DV_PIPE_SEED + 41)

        with pytest.raises(ValueError):
            backend.generate(
                prompt,
                GP_GEN,
                _gp_config(draft_mode=draft_mode, mask_token_id=bad_mask_token_id),
            )
        assert backend.last_generate_stores is None

    @pytest.mark.parametrize("draft_mode", [False, True])
    def test_unloaded_backend_raises_runtime_error(self, draft_mode):
        from shadowinfer.core.backends.pytorch_backend import PyTorchModelBackend

        backend = PyTorchModelBackend(model_config=_gp_model_config(), seed=DV_PIPE_SEED + 42)
        prompt = _gp_prompt(DV_PIPE_SEED + 43)

        with pytest.raises(RuntimeError):
            backend.generate(prompt, GP_GEN, _gp_config(draft_mode=draft_mode))
        assert backend.last_generate_stores is None
