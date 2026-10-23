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

All tests use real tensors; nothing is mocked.
"""

import math

import pytest
import torch

from shadowinfer.execution import BlockDecodingConfig, draft_round, verify_round

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
