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

The T003 partition appended below pins the draft-verify statistics
contract (FR17.3): one shared 17-key stats set across both modes and
both arms -- the fourteen v3.18.0 keys plus three new draft-verify keys
(draft_acceptance_rate, rounds_to_fixpoint, changed_per_round) -- the
closed-form accounting that parses computed_rows_total, total_rounds,
forced_completions and kv_write_rows_total back from the per-block
round-and-fixpoint ledgers, the max_rounds=1 / max_rounds=2 structural
anchors, the mask-init None-pinning of the new keys, and the baseline
arm's recompute-semantics variants of the same contract.

The T004 partition appended below pins the end-to-end draft-verify
behavior (design 6.1 T004, FR17.1/17.2 acceptance): the fp32
causal-time-invariance anchor -- the store-backed generate arm and the
full-recompute baseline arm produce bit-identical sequences with
identical per-block scheduling ledgers at two scales -- the convergence
dichotomy of real forwards (every block either confirms a fixpoint with
a zero-replacement final round or is force-sealed at the round budget
with replacements still happening), the mixed-prompt batch completion
with structural-only statistics, and the capture_block_logits round
isomorphism in draft-verify mode.

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
    _gp_generate,
    _gp_in_vocab,
    _gp_is_int,
    _gp_model_config,
    _gp_prompt,
    _t5_assert_bitwise_equal,
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


# ---------------------------------------------------------------------------
# T003: draft-verify statistics contract (FR17.3)
# ---------------------------------------------------------------------------

DV_STATS_SEED = 2701

# The fourteen v3.18.0 statistics keys every generate call reports.
DV_STATS_EXISTING_KEYS = (
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
)
# The three AR017 draft-verify keys joining the contract.
DV_STATS_NEW_KEYS = (
    "draft_acceptance_rate",
    "rounds_to_fixpoint",
    "changed_per_round",
)
DV_STATS_ALL_KEYS = DV_STATS_EXISTING_KEYS + DV_STATS_NEW_KEYS

DV_STATS_SCALES = (
    (GP_PROMPT, GP_GEN),  # (P=6, G=8): two blocks
    (10, 16),  # (P=10, G=16): four blocks
)


def _dv_generate(seed, gen_len=GP_GEN, prompt_len=GP_PROMPT, **overrides):
    """One real draft-verify generate call on a fresh seeded backend."""
    backend = _gp_backend(seed)
    prompt = _gp_prompt(seed + 1, length=prompt_len)
    result = backend.generate(prompt, gen_len, _gp_config(draft_mode=True, **overrides))
    return backend, result


def _dv_assert_keyset(stats, label):
    """The 17-key contract set is present exactly: the fourteen v3.18.0
    statistics keys plus the three AR017 draft-verify keys and nothing
    else (``block_logits_per_round`` is not a contract key; it only
    appears under ``capture_block_logits``)."""
    expected = set(DV_STATS_ALL_KEYS)
    actual = set(stats.keys())
    assert actual == expected, (
        f"{label}: stats key set mismatch: "
        f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
    )


def _dv_assert_new_key_structure(stats, max_rounds, label):
    """Structural contract of the three draft-verify statistics keys.

    ``draft_acceptance_rate`` is a float in [0, 1];
    ``rounds_to_fixpoint`` is a per-block list holding -1 for forced
    blocks (whose round count must then equal max_rounds) or the
    fixpoint-confirming scheduled-forward count in [2, max_rounds] equal
    to that block's ``rounds_per_block`` entry; ``changed_per_round`` is
    a per-block list isomorphic to ``rounds_per_block`` whose first item
    is the B*W draft fill and whose remaining items are per-verify-round
    replaced position counts within [0, B*W].

    Returns (rounds_to_fixpoint, changed_per_round) for closed-form
    cross-checks by the caller.
    """
    rate = stats["draft_acceptance_rate"]
    assert isinstance(
        rate, float
    ), f"{label}: draft_acceptance_rate must be a float, got {type(rate).__name__}"
    assert 0.0 <= rate <= 1.0, f"{label}: draft_acceptance_rate {rate} outside [0, 1]"

    blocks = stats["blocks"]
    rounds = stats["rounds_per_block"]
    positions = GP_BATCH * GP_WIDTH

    to_fixpoint = stats["rounds_to_fixpoint"]
    assert isinstance(to_fixpoint, list), f"{label}: rounds_to_fixpoint must be a list"
    assert (
        len(to_fixpoint) == blocks
    ), f"{label}: rounds_to_fixpoint length {len(to_fixpoint)} != blocks {blocks}"
    for b, value in enumerate(to_fixpoint):
        assert _gp_is_int(value), f"{label}: rounds_to_fixpoint[{b}] must be an int, got {value!r}"
        if value == -1:
            assert rounds[b] == max_rounds, (
                f"{label}: forced block {b} ran {rounds[b]} rounds " f"!= max_rounds {max_rounds}"
            )
        else:
            assert 2 <= value <= max_rounds, (
                f"{label}: rounds_to_fixpoint[{b}]={value} outside " f"[2, max_rounds={max_rounds}]"
            )
            assert value == rounds[b], (
                f"{label}: fixpoint block {b} confirms at round {value} "
                f"but ran {rounds[b]} rounds"
            )

    changed = stats["changed_per_round"]
    assert isinstance(changed, list), f"{label}: changed_per_round must be a list"
    assert (
        len(changed) == blocks
    ), f"{label}: changed_per_round length {len(changed)} != blocks {blocks}"
    for b, per_round in enumerate(changed):
        assert isinstance(per_round, list), f"{label}: changed_per_round[{b}] must be a list"
        assert len(per_round) == rounds[b], (
            f"{label}: changed_per_round[{b}] length {len(per_round)} "
            f"!= rounds_per_block[{b}]={rounds[b]}"
        )
        assert per_round[0] == positions, (
            f"{label}: changed_per_round[{b}][0] must be the B*W draft "
            f"fill {positions}, got {per_round[0]}"
        )
        for r, count in enumerate(per_round[1:], start=1):
            assert _gp_is_int(count) and 0 <= count <= positions, (
                f"{label}: changed_per_round[{b}][{r}]={count} " f"outside [0, {positions}]"
            )
    return to_fixpoint, changed


class TestDraftVerifyStatsContract:
    def test_draft_verify_stats_key_set_is_seventeen_keys(self):
        # FR17.3(a): the draft-verify generate arm always surfaces exactly
        # the shared 17-key contract set -- the fourteen v3.18.0 keys plus
        # draft_acceptance_rate / rounds_to_fixpoint / changed_per_round;
        # without capture_block_logits no extra key may appear.
        _, result = _dv_generate(DV_STATS_SEED)
        _dv_assert_keyset(result.stats, "generate-draft-verify")

    def test_mask_init_stats_key_set_matches_draft_verify(self):
        # FR17.3(a): the mask-init arm carries the very same 17-key set
        # (its three new keys are pinned to None), so the two modes can
        # never drift apart in reported stats keys.
        backend = _gp_backend(DV_STATS_SEED + 2)
        prompt = _gp_prompt(DV_STATS_SEED + 3)

        result_dv = backend.generate(prompt, GP_GEN, _gp_config(draft_mode=True))
        result_mi = backend.generate(prompt, GP_GEN, _gp_config(draft_mode=False))

        _dv_assert_keyset(result_dv.stats, "generate-draft-verify")
        _dv_assert_keyset(result_mi.stats, "generate-mask-init")
        assert set(result_dv.stats.keys()) == set(result_mi.stats.keys())

    def test_generate_and_baseline_draft_verify_key_sets_match(self):
        # FR17.3(a): the store-backed generate arm and the full-recompute
        # baseline arm share one draft-verify statistics contract.
        seed = DV_STATS_SEED + 4
        backend = _gp_backend(seed)
        prompt = _gp_prompt(seed + 1)
        config = _gp_config(draft_mode=True)

        gen_result = backend.generate(prompt, GP_GEN, config)
        base_result = _dv_baseline(_t5_direct_model(seed), prompt, GP_GEN, config)

        _dv_assert_keyset(gen_result.stats, "generate-draft-verify")
        _dv_assert_keyset(base_result.stats, "baseline-draft-verify")
        assert set(gen_result.stats.keys()) == set(base_result.stats.keys())

    @pytest.mark.parametrize("max_rounds", [1, 2, 8])
    def test_new_keys_structure_across_round_budgets(self, max_rounds):
        # FR17.3(a): the three new keys keep their structural contract at
        # every round budget -- the draft-only degeneration (max_rounds=1),
        # the single-verify budget (max_rounds=2) and the default budget
        # (max_rounds=8). Toy-model convergence is not predictable, so
        # only structure, value domains and cross-key relations are
        # asserted, never concrete round counts.
        _, result = _dv_generate(DV_STATS_SEED + 6, max_rounds=max_rounds)
        _dv_assert_new_key_structure(result.stats, max_rounds, "generate-draft-verify")

    def test_capture_block_logits_appends_only_the_capture_key(self):
        # FR17.3(a): capture_block_logits is observation-only on the stats
        # contract too: the 17 contract keys stay unchanged and the capture
        # key appears on top of them.
        backend = _gp_backend(DV_STATS_SEED + 8)
        prompt = _gp_prompt(DV_STATS_SEED + 9)

        result = backend.generate(
            prompt, GP_GEN, _gp_config(draft_mode=True), capture_block_logits=True
        )

        expected = set(DV_STATS_ALL_KEYS) | {"block_logits_per_round"}
        assert set(result.stats.keys()) == expected


class TestDraftVerifyStatsClosedForm:
    def test_computed_rows_total_matches_seal_closed_form(self):
        # FR17.3(b): computed_rows_total parses back from the per-block
        # round counts and the fixpoint ledger: a fixpoint block
        # (rounds_to_fixpoint != -1) seals with the KV of its confirming
        # verify forward (seal 0 extra rows); a forced block
        # (rounds_to_fixpoint == -1) pays one extra W-row sealing forward.
        _, result = _dv_generate(DV_STATS_SEED + 10)
        stats = result.stats
        to_fixpoint, _ = _dv_assert_new_key_structure(stats, 8, "generate-draft-verify")
        expected = sum(
            GP_WIDTH * (rounds + (1 if to_fixpoint[b] == -1 else 0))
            for b, rounds in enumerate(stats["rounds_per_block"])
        )
        assert stats["computed_rows_total"] == expected

    def test_total_rounds_and_forced_completions_parse_back(self):
        # FR17.3(b): total_rounds is the sum of the per-block round counts,
        # and forced_completions counts exactly the blocks whose
        # rounds_to_fixpoint entry is -1.
        _, result = _dv_generate(DV_STATS_SEED + 12)
        stats = result.stats
        to_fixpoint, _ = _dv_assert_new_key_structure(stats, 8, "generate-draft-verify")
        assert stats["total_rounds"] == sum(stats["rounds_per_block"])
        assert stats["forced_completions"] == sum(1 for value in to_fixpoint if value == -1)

    def test_kv_write_rows_total_is_prompt_plus_gen(self):
        # FR17.3(b): draft-verify writes pages only at prefill (P rows)
        # and at each block's sealing pass (W rows per block, G in total);
        # draft and verify rounds only gather. Cross-checked against the
        # page store append ledger.
        backend, result = _dv_generate(DV_STATS_SEED + 14)
        _dv_assert_new_key_structure(result.stats, 8, "generate-draft-verify")
        assert result.stats["kv_write_rows_total"] == GP_PROMPT + GP_GEN
        assert (
            result.stats["kv_write_rows_total"] == backend.last_generate_stores[0].appended_tokens
        )

    def test_max_rounds_one_draft_only_anchor(self):
        # FR17.3(b) structural anchor: with max_rounds=1 only the draft
        # round exists, so every block is forced (rounds_to_fixpoint all
        # -1), the final block equals its draft (nothing can replace it),
        # the acceptance rate is exactly 1.0, each block's changed ledger
        # is the single B*W draft fill, and every block pays one forced
        # sealing forward (computed_rows == 2 * W * blocks).
        _, result = _dv_generate(DV_STATS_SEED + 16, max_rounds=1)
        stats = result.stats
        blocks = stats["blocks"]
        _dv_assert_new_key_structure(stats, 1, "generate-draft-verify")

        assert stats["rounds_to_fixpoint"] == [-1] * blocks
        assert stats["changed_per_round"] == [[GP_BATCH * GP_WIDTH]] * blocks
        assert stats["draft_acceptance_rate"] == 1.0
        assert stats["forced_completions"] == blocks
        assert stats["total_rounds"] == blocks
        assert stats["computed_rows_total"] == 2 * GP_WIDTH * blocks

    @pytest.mark.parametrize("prompt_len,gen_len", DV_STATS_SCALES)
    def test_max_rounds_two_acceptance_closed_form(self, prompt_len, gen_len):
        # FR17.3(b) structural anchor: with max_rounds=2 every block runs
        # draft + exactly one verify round, so the final block equals the
        # draft except at the verify-replaced positions and
        # acceptance == 1 - sum(changed_per_round[b][1]) / (B*W*blocks).
        # A zero-replacement verify confirms the fixpoint
        # (rounds_to_fixpoint == 2); any replacement leaves the block
        # forced (-1). Toy-model convergence is not predicted: whichever
        # branch each block takes, the relations must hold.
        seed = DV_STATS_SEED + 18 + prompt_len
        _, result = _dv_generate(seed, gen_len=gen_len, prompt_len=prompt_len, max_rounds=2)
        stats = result.stats
        blocks = stats["blocks"]
        to_fixpoint, changed = _dv_assert_new_key_structure(stats, 2, "generate-draft-verify")

        assert stats["rounds_per_block"] == [2] * blocks
        total_positions = GP_BATCH * GP_WIDTH * blocks
        expected_rate = 1.0 - sum(per_round[1] for per_round in changed) / total_positions
        assert abs(stats["draft_acceptance_rate"] - expected_rate) <= 1e-9
        for b in range(blocks):
            if changed[b][1] == 0:
                assert to_fixpoint[b] == 2
            else:
                assert to_fixpoint[b] == -1


class TestMaskInitStatsNewKeysNone:
    def test_mask_init_new_keys_present_and_none(self):
        # FR17.3(c): the mask-init arm keeps the 17-key contract with the
        # three draft-verify keys pinned to None -- threshold-gated
        # scheduling has no draft, no verify round and no fixpoint notion.
        _, result = _gp_generate(DV_STATS_SEED + 20)
        _dv_assert_keyset(result.stats, "generate-mask-init")
        for key in DV_STATS_NEW_KEYS:
            assert (
                result.stats[key] is None
            ), f"generate-mask-init: stats[{key!r}] must be None in mask-init mode"

    def test_mask_init_computed_rows_keeps_v318_closed_form(self):
        # FR17.3(c): adding the three keys leaves the existing mask-init
        # accounting untouched: every block still seals with one extra
        # W-row forward, so computed_rows_total == sum(W * (rounds + 1)).
        _, result = _gp_generate(DV_STATS_SEED + 22)
        _dv_assert_keyset(result.stats, "generate-mask-init")
        rounds = result.stats["rounds_per_block"]
        expected = sum(GP_WIDTH * (rounds_j + 1) for rounds_j in rounds)
        assert result.stats["computed_rows_total"] == expected


class TestBaselineDraftVerifyStats:
    def test_baseline_draft_verify_key_set_and_new_key_structure(self):
        # FR17.3(d): the baseline arm shares the draft-verify statistics
        # contract: the 17-key set with the three new keys populated and
        # structurally identical to the generate arm's.
        seed = DV_STATS_SEED + 24
        prompt = _gp_prompt(seed + 1)
        base_result = _dv_baseline(
            _t5_direct_model(seed), prompt, GP_GEN, _gp_config(draft_mode=True)
        )

        _dv_assert_keyset(base_result.stats, "baseline-draft-verify")
        _dv_assert_new_key_structure(base_result.stats, 8, "baseline-draft-verify")

    def test_baseline_draft_verify_computed_rows_follow_recompute_formula(self):
        # FR17.3(d): the recompute loop has no sealing concept: every round
        # pays the full L_j + W rows, and no page store exists, so
        # kv_write_rows_total stays 0 even in draft-verify mode.
        seed = DV_STATS_SEED + 26
        prompt = _gp_prompt(seed + 1)
        base_result = _dv_baseline(
            _t5_direct_model(seed), prompt, GP_GEN, _gp_config(draft_mode=True)
        )
        stats = base_result.stats
        _dv_assert_new_key_structure(stats, 8, "baseline-draft-verify")

        assert stats["kv_write_rows_total"] == 0
        expected = sum(
            rounds * (GP_PROMPT + b * GP_WIDTH + GP_WIDTH)
            for b, rounds in enumerate(stats["rounds_per_block"])
        )
        assert stats["computed_rows_total"] == expected

    def test_baseline_mask_init_new_keys_none(self):
        # FR17.3(d): the baseline mask-init arm pins the three new keys to
        # None exactly like the generate mask-init arm.
        seed = DV_STATS_SEED + 28
        prompt = _gp_prompt(seed + 1)
        base_result = _dv_baseline(
            _t5_direct_model(seed), prompt, GP_GEN, _gp_config(draft_mode=False)
        )

        _dv_assert_keyset(base_result.stats, "baseline-mask-init")
        for key in DV_STATS_NEW_KEYS:
            assert (
                base_result.stats[key] is None
            ), f"baseline-mask-init: stats[{key!r}] must be None in mask-init mode"


# ---------------------------------------------------------------------------
# T004: end-to-end draft-verify behavior (design 6.1 T004, FR17.1/17.2)
# ---------------------------------------------------------------------------

DV_E2E_SEED = 2801

DV_E2E_SCALES = (
    (DV_E2E_SEED, GP_PROMPT, GP_GEN),  # (P=6, G=8): two blocks
    (DV_E2E_SEED + 6, 10, 16),  # (P=10, G=16): four blocks
)


class TestDraftVerifyTwoArmBitwiseAnchor:
    @pytest.mark.parametrize("seed,prompt_len,gen_len", DV_E2E_SCALES)
    def test_draft_verify_two_arm_bitwise_and_scheduling_anchor(self, seed, prompt_len, gen_len):
        # Core FR17.1/17.2 acceptance: on the fp32 plane the store-backed
        # generate arm and the full-recompute baseline arm are the same
        # causal computation carried through different time layouts --
        # same seeded model (_t5_direct_model mirrors backend.load), same
        # prompt, draft_mode=True -- so the sequences must agree
        # bit-for-bit and every scheduling decision (fixpoint round,
        # per-round replacements, forced seal, round count) must be
        # identical block by block.
        backend = _gp_backend(seed)
        prompt = _gp_prompt(seed + 1, length=prompt_len)
        config = _gp_config(draft_mode=True)

        gen_result = backend.generate(prompt, gen_len, config)
        base_result = _dv_baseline(_t5_direct_model(seed), prompt, gen_len, config)

        assert gen_result.sequences.shape == (GP_BATCH, prompt_len + gen_len)
        assert torch.equal(gen_result.sequences[:, :prompt_len], prompt)
        _t5_assert_bitwise_equal(gen_result.sequences, base_result.sequences, "generate-dv")

        gen_stats = gen_result.stats
        base_stats = base_result.stats
        blocks = gen_len // GP_WIDTH
        assert gen_stats["blocks"] == blocks == base_stats["blocks"]

        assert gen_stats["rounds_to_fixpoint"] == base_stats["rounds_to_fixpoint"]
        assert gen_stats["changed_per_round"] == base_stats["changed_per_round"]
        assert gen_stats["forced_completions"] == base_stats["forced_completions"]
        assert gen_stats["rounds_per_block"] == base_stats["rounds_per_block"]
        assert gen_stats["draft_acceptance_rate"] == base_stats["draft_acceptance_rate"], (
            f"draft_acceptance_rate generate={gen_stats['draft_acceptance_rate']} "
            f"vs baseline={base_stats['draft_acceptance_rate']}"
        )

        for b in range(blocks):
            assert gen_stats["rounds_to_fixpoint"][b] == base_stats["rounds_to_fixpoint"][b], (
                f"block {b}: rounds_to_fixpoint generate="
                f"{gen_stats['rounds_to_fixpoint'][b]} vs baseline="
                f"{base_stats['rounds_to_fixpoint'][b]}"
            )
            assert gen_stats["changed_per_round"][b] == base_stats["changed_per_round"][b], (
                f"block {b}: changed_per_round generate="
                f"{gen_stats['changed_per_round'][b]} vs baseline="
                f"{base_stats['changed_per_round'][b]}"
            )


class TestDraftVerifyConvergenceDichotomy:
    @pytest.mark.parametrize("max_rounds", [2, 3])
    @pytest.mark.parametrize("seed,prompt_len,gen_len", DV_E2E_SCALES)
    def test_real_forwards_convergence_dichotomy(self, seed, prompt_len, gen_len, max_rounds):
        # FR17.1/17.2 on real multi-block forwards: every block lands on
        # exactly one side of the dichotomy -- either it confirms a
        # fixpoint (rounds_to_fixpoint in [2, max_rounds], the confirming
        # round replaces nothing) or the round budget force-seals it
        # (rounds_to_fixpoint == -1, the final round still replaced
        # something, forced_completions counts it). No fabricated
        # trajectories: the toy model's concrete convergence counts are
        # never pinned, only the dichotomy relations.
        backend = _gp_backend(seed)
        prompt = _gp_prompt(seed + 1, length=prompt_len)
        config = _gp_config(draft_mode=True, max_rounds=max_rounds)

        result = backend.generate(prompt, gen_len, config)

        stats = result.stats
        blocks = gen_len // GP_WIDTH
        assert stats["blocks"] == blocks
        to_fixpoint, changed = _dv_assert_new_key_structure(
            stats, max_rounds, "generate-draft-verify"
        )
        rounds = stats["rounds_per_block"]
        assert stats["forced_completions"] == sum(1 for value in to_fixpoint if value == -1)

        for b in range(blocks):
            assert (
                rounds[b] <= max_rounds
            ), f"block {b}: rounds_per_block {rounds[b]} exceeds budget {max_rounds}"
            assert len(changed[b]) == rounds[b]
            if to_fixpoint[b] != -1:
                assert changed[b][-1] == 0, (
                    f"block {b}: fixpoint confirmed at round {to_fixpoint[b]} "
                    f"but the confirming round replaced {changed[b][-1]} positions"
                )
            else:
                assert changed[b][-1] > 0, (
                    f"block {b}: force-sealed at budget {max_rounds} but the "
                    f"final round replaced {changed[b][-1]} positions"
                )

        generated = result.sequences[:, prompt_len:]
        assert bool(((generated >= 0) & (generated < GP_VOCAB)).all())
        assert not bool((generated == config.mask_token_id).any())

    def test_default_budget_structure_only(self):
        # max_rounds=8 default budget: only the structural invariants are
        # asserted (rounds_to_fixpoint value domain, changed ledger
        # lengths, mask-free generated region) -- never the toy model's
        # concrete convergence round counts.
        seed = DV_E2E_SEED + 10
        backend = _gp_backend(seed)
        prompt = _gp_prompt(seed + 1, length=10)
        config = _gp_config(draft_mode=True)

        result = backend.generate(prompt, 16, config)

        stats = result.stats
        assert stats["blocks"] == 16 // GP_WIDTH
        to_fixpoint, changed = _dv_assert_new_key_structure(stats, 8, "generate-draft-verify")
        rounds = stats["rounds_per_block"]
        for b in range(stats["blocks"]):
            assert to_fixpoint[b] == -1 or 2 <= to_fixpoint[b] <= 8
            assert len(changed[b]) == rounds[b]
        generated = result.sequences[:, 10:]
        assert bool(((generated >= 0) & (generated < GP_VOCAB)).all())
        assert not bool((generated == config.mask_token_id).any())


class TestDraftVerifyBatchMixedPrompts:
    def test_mixed_prompt_batch_completes_with_structural_stats(self):
        # B=2 with two independently seeded prompt rows (clearly different
        # content): draft-verify completes the whole batch with both rows
        # in vocabulary and no mask residue. Batch-aggregated round counts
        # are block-synchronized across elements, so per-element
        # convergence trajectories are deliberately not asserted -- only
        # the structural statistics contract.
        seed = DV_E2E_SEED + 50
        backend = _gp_backend(seed)
        row_a = _gp_prompt(seed + 1, batch=1)
        row_b = _gp_prompt(seed + 81, batch=1)
        prompt = torch.cat([row_a, row_b], dim=0)
        assert not torch.equal(prompt[0], prompt[1])
        config = _gp_config(draft_mode=True)

        result = backend.generate(prompt, GP_GEN, config)

        assert result.sequences.shape == (GP_BATCH, GP_PROMPT + GP_GEN)
        assert torch.equal(result.sequences[:, :GP_PROMPT], prompt)
        assert _gp_in_vocab(result.sequences)
        assert not bool((result.sequences[:, GP_PROMPT:] == config.mask_token_id).any())

        stats = result.stats
        _dv_assert_new_key_structure(stats, 8, "generate-draft-verify")
        assert 0.0 <= stats["draft_acceptance_rate"] <= 1.0


class TestDraftVerifyCaptureRounds:
    def test_capture_block_logits_round_isomorphism(self):
        # capture_block_logits in draft-verify mode: one [B, W, V] logits
        # tensor per scheduled round per block, isomorphic to
        # rounds_per_block (draft + verify rounds; the forced sealing
        # forward is not a scheduled round and must not appear).
        seed = DV_E2E_SEED + 60
        backend = _gp_backend(seed)
        prompt = _gp_prompt(seed + 1)
        config = _gp_config(draft_mode=True)

        result = backend.generate(prompt, GP_GEN, config, capture_block_logits=True)

        stats = result.stats
        blocks = stats["blocks"]
        rounds = stats["rounds_per_block"]
        captured = stats["block_logits_per_round"]
        assert isinstance(captured, list)
        assert len(captured) == blocks
        for b in range(blocks):
            assert isinstance(captured[b], list)
            assert len(captured[b]) == rounds[b], (
                f"block {b}: captured {len(captured[b])} logits tensors "
                f"for {rounds[b]} scheduled rounds"
            )
            for logits in captured[b]:
                assert isinstance(logits, torch.Tensor)
                assert logits.shape == (GP_BATCH, GP_WIDTH, GP_VOCAB)
