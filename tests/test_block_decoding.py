"""AR016 T001: confidence-aware parallel unmask scheduling tests.

Exercises the not-yet-implemented ``shadowinfer.execution.block_decoding``
module (FR16.1): ``BlockDecodingConfig`` validation (frozen dataclass whose
``__post_init__`` raises ``ValueError`` on illegal values) and
``schedule_round`` -- per-batch-element confidence-aware parallel unmasking
(every masked position whose max softmax probability reaches
``confidence_threshold`` is unmasked with its argmax token) plus a
guaranteed-progress top-1 fallback for elements whose masked positions are
all below the threshold. Already-unmasked positions are never re-unmasked,
and degenerate inputs (wrong logits rank, mask/logits shape mismatch) are
rejected with ``ValueError``.

All tests use real tensors; nothing is mocked.
"""

import dataclasses
import math

import pytest
import torch

BD_SEED = 2033
BD_VOCAB = 8
BD_WIDTH = 4
BD_THRESHOLD = 0.9


def _bd_config(**overrides):
    """Valid BlockDecodingConfig; confidence threshold 0.9 by default."""
    from shadowinfer.execution.block_decoding import BlockDecodingConfig

    settings = dict(
        block_size=32,
        max_rounds=8,
        confidence_threshold=BD_THRESHOLD,
        mask_token_id=0,
        page_size=32,
        page_precision="fp32",
    )
    settings.update(overrides)
    return BlockDecodingConfig(**settings)


def _bd_schedule(logits_block, masked, config):
    from shadowinfer.execution.block_decoding import schedule_round

    return schedule_round(logits_block, masked, config)


def _conf_row(conf, argmax_token, vocab):
    """[vocab] logits row whose max softmax probability is ``conf`` (up to
    fp rounding), attained at ``argmax_token`` uniquely."""
    gap = math.log((1.0 - conf) / (conf * (vocab - 1)))
    row = torch.full((vocab,), float(gap))
    row[int(argmax_token)] = 0.0
    return row


def _block_from_spec(confs, argmax_tokens, vocab=BD_VOCAB):
    """[1, W, V] logits block; position w has confidence ``confs[w]``
    attained at token ``argmax_tokens[w]``."""
    rows = [_conf_row(conf, token, vocab) for conf, token in zip(confs, argmax_tokens)]
    return torch.stack(rows).unsqueeze(0)


def _softmax_conf(logits_block):
    """[B, W] max softmax probability and argmax token per position."""
    probs = torch.softmax(logits_block, dim=-1)
    return probs.max(dim=-1).values, probs.max(dim=-1).indices


def _reference_schedule(logits_block, masked, config):
    """Slow per-element oracle: parallel unmask above threshold, otherwise
    top-1 fallback with smallest-index tie-breaking."""
    batch, width, _ = logits_block.shape
    conf, _ = _softmax_conf(logits_block)
    unmask = torch.zeros(batch, width, dtype=torch.bool)
    fallback = torch.zeros(batch, dtype=torch.bool)
    for b in range(batch):
        masked_positions = [w for w in range(width) if bool(masked[b, w])]
        if not masked_positions:
            continue
        above = [w for w in masked_positions if float(conf[b, w]) >= config.confidence_threshold]
        if above:
            for w in above:
                unmask[b, w] = True
        else:
            best = max(masked_positions, key=lambda w: (float(conf[b, w]), -w))
            unmask[b, best] = True
            fallback[b] = True
    return unmask, fallback


# ---------------------------------------------------------------------------
# T001: ConfidenceSchedule + BlockDecodingConfig
# ---------------------------------------------------------------------------


class TestBlockDecodingConfigValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"block_size": 0},
            {"block_size": -4},
            {"max_rounds": 0},
            {"max_rounds": -1},
            {"confidence_threshold": 0.0},
            {"confidence_threshold": -0.05},
            {"confidence_threshold": 1.001},
            {"page_size": 0},
            {"page_size": -8},
            {"page_precision": "bf16"},
            {"page_precision": "int4"},
            {"page_precision": "FP32"},
        ],
    )
    def test_invalid_values_raise_value_error(self, overrides):
        with pytest.raises(ValueError):
            _bd_config(**overrides)

    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {"block_size": 1},
            {"max_rounds": 1},
            {"confidence_threshold": 1.0},
            {"confidence_threshold": 0.05},
            {"page_size": 1},
            {"page_precision": "fp16"},
            {"page_precision": "int8"},
        ],
    )
    def test_valid_boundary_values_pass(self, overrides):
        config = _bd_config(**overrides)
        assert 0.0 < config.confidence_threshold <= 1.0

    def test_defaults(self):
        from shadowinfer.execution.block_decoding import BlockDecodingConfig

        config = BlockDecodingConfig()
        assert config.block_size == 32
        assert config.max_rounds == 8
        assert config.confidence_threshold == 0.9
        assert config.mask_token_id == 0
        assert config.page_size == 32
        assert config.page_precision == "fp32"

    def test_config_is_frozen(self):
        config = _bd_config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.block_size = 64


class TestScheduleRoundParallelUnmasking:
    def test_above_threshold_positions_all_unmasked_with_argmax_tokens(self):
        confs = [0.95, 0.5, 0.97, 0.4]
        tokens = [3, 1, 6, 0]
        logits = _block_from_spec(confs, tokens)
        masked = torch.ones(1, BD_WIDTH, dtype=torch.bool)

        decision = _bd_schedule(logits, masked, _bd_config())

        expected_unmask = torch.tensor([[True, False, True, False]])
        assert torch.equal(decision.unmask, expected_unmask)
        assert decision.progress_fallback.tolist() == [False]
        for w in (0, 2):
            assert int(decision.tokens[0, w]) == tokens[w]
        # Positions that stay masked keep their logits-derived confidence.
        conf, _ = _softmax_conf(logits)
        still = masked & ~decision.unmask
        assert torch.allclose(decision.confidences[still], conf[still], atol=1e-6)

    def test_all_positions_above_threshold_unmask_everything(self):
        tokens = [2, 0, 7, 4]
        logits = _block_from_spec([0.95, 0.97, 0.99, 0.96], tokens)
        masked = torch.ones(1, BD_WIDTH, dtype=torch.bool)

        decision = _bd_schedule(logits, masked, _bd_config())

        assert torch.equal(decision.unmask, torch.ones(1, BD_WIDTH, dtype=torch.bool))
        assert decision.progress_fallback.tolist() == [False]
        for w, token in enumerate(tokens):
            assert int(decision.tokens[0, w]) == token


class TestScheduleRoundFallbackTop1:
    def test_all_below_threshold_unmasks_exactly_top1(self):
        confs = [0.5, 0.6, 0.7, 0.4]
        logits = _block_from_spec(confs, [0, 2, 5, 1])
        masked = torch.ones(1, BD_WIDTH, dtype=torch.bool)

        decision = _bd_schedule(logits, masked, _bd_config())

        expected = torch.zeros(1, BD_WIDTH, dtype=torch.bool)
        expected[0, 2] = True  # highest confidence 0.7
        assert torch.equal(decision.unmask, expected)
        assert decision.progress_fallback.tolist() == [True]
        assert int(decision.tokens[0, 2]) == 5

    def test_fallback_tie_breaks_to_smallest_position(self):
        # Positions 1 and 2 hold bit-identical logit rows (same argmax
        # token), so their confidences tie exactly; the fallback must pick
        # the smaller index.
        tie_row = _conf_row(0.7, 4, BD_VOCAB)
        rows = [_conf_row(0.6, 1, BD_VOCAB), tie_row, tie_row.clone(), _conf_row(0.5, 7, BD_VOCAB)]
        logits = torch.stack(rows).unsqueeze(0)
        masked = torch.ones(1, BD_WIDTH, dtype=torch.bool)

        decision = _bd_schedule(logits, masked, _bd_config())

        expected = torch.zeros(1, BD_WIDTH, dtype=torch.bool)
        expected[0, 1] = True
        assert torch.equal(decision.unmask, expected)
        assert decision.progress_fallback.tolist() == [True]
        assert int(decision.tokens[0, 1]) == 4

    def test_fallback_considers_only_masked_positions(self):
        # Position 0 is already unmasked and holds the highest confidence of
        # all positions; the fallback must ignore it.
        logits = _block_from_spec([0.99, 0.6, 0.55], [7, 3, 1])
        masked = torch.tensor([[False, True, True]])

        decision = _bd_schedule(logits, masked, _bd_config())

        expected = torch.tensor([[False, True, False]])
        assert torch.equal(decision.unmask, expected)
        assert decision.progress_fallback.tolist() == [True]
        assert int(decision.tokens[0, 1]) == 3


class TestScheduleRoundBatchIndependence:
    def test_mixed_batch_parallel_and_fallback_elements(self):
        elem_a = _block_from_spec([0.95, 0.5, 0.97, 0.4], [1, 2, 3, 4])
        elem_b = _block_from_spec([0.5, 0.6, 0.45, 0.55], [5, 6, 7, 0])
        logits = torch.cat([elem_a, elem_b], dim=0)
        masked = torch.ones(2, BD_WIDTH, dtype=torch.bool)

        decision = _bd_schedule(logits, masked, _bd_config())

        expected = torch.tensor(
            [
                [True, False, True, False],  # A: parallel unmask of 0 and 2
                [False, True, False, False],  # B: top-1 fallback at 1
            ]
        )
        assert torch.equal(decision.unmask, expected)
        assert decision.progress_fallback.tolist() == [False, True]
        assert int(decision.tokens[0, 0]) == 1
        assert int(decision.tokens[0, 2]) == 3
        assert int(decision.tokens[1, 1]) == 6


class TestScheduleRoundThresholdBoundary:
    def test_confidence_exactly_at_threshold_is_unmasked(self):
        # Uniform logits over V=4 give a max softmax probability of exactly
        # 1/4, so a 0.25 threshold must unmask them via ``>=``. A strict
        # ``>`` comparison would leave every position masked and trigger
        # the top-1 fallback instead.
        logits = torch.zeros(1, 3, 4)
        masked = torch.ones(1, 3, dtype=torch.bool)

        decision = _bd_schedule(logits, masked, _bd_config(confidence_threshold=0.25))

        expected = torch.ones(1, 3, dtype=torch.bool)
        assert torch.equal(decision.unmask, expected)
        assert decision.progress_fallback.tolist() == [False]


class TestScheduleRoundAlreadyUnmasked:
    def test_unmasked_positions_are_never_reunmasked(self):
        # Positions 0 and 3 are already unmasked with very high confidence;
        # they must stay masked=False in the decision regardless.
        logits = _block_from_spec([0.99, 0.95, 0.5, 0.99], [0, 2, 1, 5])
        masked = torch.tensor([[False, True, True, False]])

        decision = _bd_schedule(logits, masked, _bd_config())

        expected = torch.tensor([[False, True, False, False]])
        assert torch.equal(decision.unmask, expected)
        assert decision.progress_fallback.tolist() == [False]
        assert int(decision.tokens[0, 1]) == 2

    def test_empty_masked_set_returns_all_false(self):
        logits = _block_from_spec([0.99, 0.99], [1, 2])
        masked = torch.zeros(1, 2, dtype=torch.bool)

        decision = _bd_schedule(logits, masked, _bd_config())

        assert torch.equal(decision.unmask, torch.zeros(1, 2, dtype=torch.bool))
        assert decision.progress_fallback.tolist() == [False]

    def test_empty_masked_element_in_mixed_batch_stays_idle(self):
        elem_a = _block_from_spec([0.95, 0.4], [3, 1])
        elem_b = _block_from_spec([0.99, 0.99], [2, 4])
        logits = torch.cat([elem_a, elem_b], dim=0)
        masked = torch.tensor([[True, True], [False, False]])

        decision = _bd_schedule(logits, masked, _bd_config())

        expected = torch.tensor([[True, False], [False, False]])
        assert torch.equal(decision.unmask, expected)
        assert decision.progress_fallback.tolist() == [False, False]


class TestScheduleRoundOutputContract:
    def test_output_shapes_and_dtypes(self):
        gen = torch.Generator().manual_seed(BD_SEED)
        logits = torch.randn(2, BD_WIDTH, BD_VOCAB, generator=gen)
        masked = torch.ones(2, BD_WIDTH, dtype=torch.bool)

        decision = _bd_schedule(logits, masked, _bd_config())

        assert decision.unmask.shape == (2, BD_WIDTH)
        assert decision.unmask.dtype == torch.bool
        assert decision.tokens.shape == (2, BD_WIDTH)
        assert decision.tokens.dtype == torch.int64
        assert decision.confidences.shape == (2, BD_WIDTH)
        assert decision.confidences.dtype.is_floating_point
        assert decision.progress_fallback.shape == (2,)
        assert decision.progress_fallback.dtype == torch.bool

    def test_confidences_of_still_masked_positions_match_softmax_max(self):
        gen = torch.Generator().manual_seed(BD_SEED + 1)
        logits = torch.randn(2, 6, BD_VOCAB, generator=gen)
        masked = torch.rand(2, 6, generator=gen) < 0.5

        decision = _bd_schedule(logits, masked, _bd_config())

        conf, _ = _softmax_conf(logits)
        still = masked & ~decision.unmask
        assert still.any()
        assert torch.allclose(decision.confidences[still], conf[still], atol=1e-6)


class TestScheduleRoundValidation:
    def test_reject_logits_not_3d(self):
        config = _bd_config()
        masked = torch.ones(1, BD_WIDTH, dtype=torch.bool)
        with pytest.raises(ValueError):
            _bd_schedule(torch.randn(BD_WIDTH, BD_VOCAB), masked, config)
        with pytest.raises(ValueError):
            _bd_schedule(torch.randn(1, BD_WIDTH, 2, BD_VOCAB), masked, config)

    @pytest.mark.parametrize("shape", [(1, BD_WIDTH), (2, 5), (BD_WIDTH,), (2, 4, 1), ()])
    def test_reject_masked_shape_mismatch(self, shape):
        logits = torch.randn(2, 4, BD_VOCAB)
        masked = torch.ones(shape, dtype=torch.bool)
        with pytest.raises(ValueError):
            _bd_schedule(logits, masked, _bd_config())


class TestScheduleRoundRandomReference:
    @pytest.mark.parametrize(
        "batch,width,vocab",
        [
            (1, 1, 2),
            (2, 5, 4),
            (3, 8, 16),
        ],
    )
    def test_matches_reference_schedule(self, batch, width, vocab):
        gen = torch.Generator().manual_seed(BD_SEED + batch * 100 + width * 7 + vocab)
        logits = torch.randn(batch, width, vocab, generator=gen)
        masked = torch.rand(batch, width, generator=gen) < 0.6
        config = _bd_config()

        decision = _bd_schedule(logits, masked, config)

        ref_unmask, ref_fallback = _reference_schedule(logits, masked, config)
        assert torch.equal(decision.unmask, ref_unmask)
        assert torch.equal(decision.progress_fallback, ref_fallback)

        conf, argmax_tokens = _softmax_conf(logits)
        unmasked = decision.unmask
        if unmasked.any():
            assert torch.equal(decision.tokens[unmasked], argmax_tokens[unmasked])
        still = masked & ~unmasked
        if still.any():
            assert torch.allclose(decision.confidences[still], conf[still], atol=1e-6)
