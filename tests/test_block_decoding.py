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


# ---------------------------------------------------------------------------
# T002: forward_block + block-causal attention mask
# ---------------------------------------------------------------------------

FB_SEED = 2101
FB_LAYERS = 2
FB_HEADS = 2
FB_HEAD_DIM = 16
FB_HIDDEN = 32
FB_VOCAB = 64
FB_MAX_SEQ = 64
FB_BATCH = 2
FB_PREFIX = 6  # L
FB_WIDTH = 4  # W
FB_FULL = FB_PREFIX + FB_WIDTH  # S = L + W
FB_TOL = 1e-6


def _fb_model():
    """Deterministic tiny two-layer model (CPU, fp32)."""
    from shadowinfer.models.simple_diffusion_llm import SimpleDiffusionLLM

    torch.manual_seed(FB_SEED)
    return SimpleDiffusionLLM(
        num_layers=FB_LAYERS,
        num_heads=FB_HEADS,
        head_dim=FB_HEAD_DIM,
        hidden_dim=FB_HIDDEN,
        vocab_size=FB_VOCAB,
        max_seq_len=FB_MAX_SEQ,
    )


def _fb_ids(seed, batch, seq):
    """Random token ids in [0, vocab), fixed per-call generator seed."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, FB_VOCAB, (batch, seq), generator=gen)


def _fb_block_causal_mask(seq_len, prefix_len, device):
    from shadowinfer.execution.block_decoding import block_causal_mask

    return block_causal_mask(seq_len, prefix_len, device)


def _rel_l2(a, b):
    """Relative L2 distance: ||a - b|| / clamp_min(||b||, 1e-12)."""
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def _fb_prefix_kv(model, seed=FB_SEED + 1):
    """Per-layer prefix KV from a prefix-only forward_step (total_steps=0)."""
    prefix_ids = _fb_ids(seed, FB_BATCH, FB_PREFIX)
    _, prefix_kv, _ = model.forward_step(prefix_ids, 0, 0)
    return prefix_kv


class TestForwardBlockMatchesMaskedForwardStep:
    def test_block_rows_match_masked_full_forward(self):
        # Prefix rows of the masked full forward see only the prefix (block
        # keys are -inf-masked), block rows see everything -- the exact
        # receptive field of forward_block's W active rows.
        model = _fb_model()
        full_ids = _fb_ids(FB_SEED + 1, FB_BATCH, FB_FULL)
        prefix_ids = full_ids[:, :FB_PREFIX]
        block_ids = full_ids[:, FB_PREFIX:]

        _, prefix_kv, _ = model.forward_step(prefix_ids, 0, 0)
        mask = _fb_block_causal_mask(FB_FULL, FB_PREFIX, torch.device("cpu"))
        full_logits, full_kv, _ = model.forward_step(full_ids, 0, 0, attention_mask=mask)
        block_logits, block_kv = model.forward_block(
            block_ids, prefix_kv, FB_PREFIX, step_t=0, total_steps=0
        )

        assert block_logits.shape == (FB_BATCH, FB_WIDTH, FB_VOCAB)
        # Attention matmul reduction-order noise only; not bitwise.
        assert _rel_l2(full_logits[:, FB_PREFIX:, :], block_logits) <= FB_TOL
        # K/V projections are row-local and attention-independent: bitwise.
        for layer_id in range(FB_LAYERS):
            full_k, full_v = full_kv[layer_id]
            block_k, block_v = block_kv[layer_id]
            assert full_k.shape == (FB_BATCH, FB_HEADS, FB_FULL, FB_HEAD_DIM)
            assert block_k.shape == (FB_BATCH, FB_HEADS, FB_WIDTH, FB_HEAD_DIM)
            assert torch.equal(full_k[:, :, FB_PREFIX:, :], block_k)
            assert torch.equal(full_v[:, :, FB_PREFIX:, :], block_v)

    def test_prefix_len_zero_matches_unmasked_forward_step(self):
        # No prefix and no mask: forward_block degenerates to a plain
        # full-sequence forward_step over W rows.
        model = _fb_model()
        ids = _fb_ids(FB_SEED + 2, FB_BATCH, FB_WIDTH)

        ref_logits, ref_kv, _ = model.forward_step(ids, 0, 0)
        blk_logits, blk_kv = model.forward_block(ids, None, 0, step_t=0, total_steps=0)

        assert _rel_l2(ref_logits, blk_logits) <= FB_TOL
        for layer_id in range(FB_LAYERS):
            ref_k, ref_v = ref_kv[layer_id]
            blk_k, blk_v = blk_kv[layer_id]
            assert torch.equal(ref_k, blk_k)
            assert torch.equal(ref_v, blk_v)


class TestAttentionMaskNoneRegression:
    def test_none_mask_bitwise_same_without_cache(self):
        model = _fb_model()
        ids = _fb_ids(FB_SEED + 3, FB_BATCH, FB_FULL)

        ref_logits, ref_kv, _ = model.forward_step(ids, 0, 0)
        got_logits, got_kv, _ = model.forward_step(ids, 0, 0, attention_mask=None)

        assert torch.equal(got_logits, ref_logits)
        for layer_id in range(FB_LAYERS):
            assert torch.equal(got_kv[layer_id][0], ref_kv[layer_id][0])
            assert torch.equal(got_kv[layer_id][1], ref_kv[layer_id][1])

    def test_none_mask_bitwise_same_with_cache(self):
        model = _fb_model()
        full_ids = _fb_ids(FB_SEED + 4, FB_BATCH, FB_FULL)
        prefix_ids = full_ids[:, :FB_PREFIX]
        block_ids = full_ids[:, FB_PREFIX:]
        _, prefix_kv, _ = model.forward_step(prefix_ids, 0, 0)

        ref_logits, ref_kv, _ = model.forward_step(block_ids, 0, 0, kv_cache=prefix_kv)
        got_logits, got_kv, _ = model.forward_step(
            block_ids, 0, 0, kv_cache=prefix_kv, attention_mask=None
        )

        assert torch.equal(got_logits, ref_logits)
        for layer_id in range(FB_LAYERS):
            assert torch.equal(got_kv[layer_id][0], ref_kv[layer_id][0])
            assert torch.equal(got_kv[layer_id][1], ref_kv[layer_id][1])


class TestBlockCausalMaskForwardSemantics:
    def test_masked_prefix_rows_equal_prefix_only_forward(self):
        # Block-causal masking removes the block keys from every prefix
        # row's softmax (exp(-inf) = 0 exactly), so prefix-row logits must
        # be bitwise identical to an isolated prefix-only forward.
        model = _fb_model()
        full_ids = _fb_ids(FB_SEED + 5, FB_BATCH, FB_FULL)
        prefix_ids = full_ids[:, :FB_PREFIX]

        prefix_logits, _, _ = model.forward_step(prefix_ids, 0, 0)
        mask = _fb_block_causal_mask(FB_FULL, FB_PREFIX, torch.device("cpu"))
        full_logits, _, _ = model.forward_step(full_ids, 0, 0, attention_mask=mask)

        assert torch.equal(full_logits[:, :FB_PREFIX, :], prefix_logits)

    def test_two_mask_shapes_broadcast_identically(self):
        # [S, S] and [1, 1, S, S] additive masks must be interchangeable.
        model = _fb_model()
        ids = _fb_ids(FB_SEED + 6, FB_BATCH, FB_FULL)
        mask2d = _fb_block_causal_mask(FB_FULL, FB_PREFIX, torch.device("cpu"))
        mask4d = mask2d.unsqueeze(0).unsqueeze(0)

        logits2d, _, _ = model.forward_step(ids, 0, 0, attention_mask=mask2d)
        logits4d, _, _ = model.forward_step(ids, 0, 0, attention_mask=mask4d)

        assert torch.equal(logits2d, logits4d)


class TestBlockCausalMaskValues:
    def test_small_scale_pattern(self):
        seq_len, prefix_len = 5, 2
        device = torch.device("cpu")
        mask = _fb_block_causal_mask(seq_len, prefix_len, device)

        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype.is_floating_point
        assert mask.device == device
        expected = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(seq_len):
                if i < prefix_len and j >= prefix_len:
                    expected[i, j] = float("-inf")
        assert torch.equal(mask, expected)

    @pytest.mark.parametrize("prefix_len", [0, 5])
    def test_degenerate_boundaries_are_all_zero(self, prefix_len):
        # L=0: no prefix rows; L=S: no block columns; neither yields -inf.
        mask = _fb_block_causal_mask(5, prefix_len, torch.device("cpu"))
        assert torch.equal(mask, torch.zeros(5, 5))

    @pytest.mark.parametrize(
        "seq_len,prefix_len",
        [
            (0, 0),
            (-1, 0),
            (5, -1),
            (5, 6),
        ],
    )
    def test_invalid_args_raise_value_error(self, seq_len, prefix_len):
        with pytest.raises(ValueError):
            _fb_block_causal_mask(seq_len, prefix_len, torch.device("cpu"))


class TestForwardBlockInputForms:
    def test_token_ids_and_prembedded_hidden_bitwise_identical(self):
        model = _fb_model()
        full_ids = _fb_ids(FB_SEED + 7, FB_BATCH, FB_FULL)
        prefix_ids = full_ids[:, :FB_PREFIX]
        block_ids = full_ids[:, FB_PREFIX:]
        _, prefix_kv, _ = model.forward_step(prefix_ids, 0, 0)

        logits_ids, kv_ids = model.forward_block(
            block_ids, prefix_kv, FB_PREFIX, step_t=0, total_steps=0
        )
        hidden_block = model.embed(block_ids.long())
        logits_hidden, kv_hidden = model.forward_block(
            hidden_block, prefix_kv, FB_PREFIX, step_t=0, total_steps=0
        )

        assert torch.equal(logits_hidden, logits_ids)
        for layer_id in range(FB_LAYERS):
            assert torch.equal(kv_hidden[layer_id][0], kv_ids[layer_id][0])
            assert torch.equal(kv_hidden[layer_id][1], kv_ids[layer_id][1])


class TestForwardBlockValidation:
    def test_reject_x_block_not_2d_or_3d(self):
        model = _fb_model()
        prefix_kv = _fb_prefix_kv(model)
        with pytest.raises(ValueError):
            model.forward_block(torch.randn(FB_WIDTH, FB_HIDDEN), prefix_kv, FB_PREFIX)
        with pytest.raises(ValueError):
            model.forward_block(torch.randn(1, FB_BATCH, FB_WIDTH, FB_HIDDEN), prefix_kv, FB_PREFIX)

    def test_reject_hidden_dim_mismatch(self):
        model = _fb_model()
        prefix_kv = _fb_prefix_kv(model)
        bad_hidden = torch.randn(FB_BATCH, FB_WIDTH, FB_HIDDEN - 1)
        with pytest.raises(ValueError):
            model.forward_block(bad_hidden, prefix_kv, FB_PREFIX)

    def test_reject_negative_prefix_len(self):
        model = _fb_model()
        prefix_kv = _fb_prefix_kv(model)
        block_ids = _fb_ids(FB_SEED + 8, FB_BATCH, FB_WIDTH)
        with pytest.raises(ValueError):
            model.forward_block(block_ids, prefix_kv, -1)

    def test_reject_prefix_kv_layer_set_mismatch(self):
        model = _fb_model()
        prefix_kv = _fb_prefix_kv(model)
        block_ids = _fb_ids(FB_SEED + 9, FB_BATCH, FB_WIDTH)

        missing = {layer_id: prefix_kv[layer_id] for layer_id in range(1, FB_LAYERS)}
        with pytest.raises(ValueError):
            model.forward_block(block_ids, missing, FB_PREFIX)

        extra = dict(prefix_kv)
        extra[FB_LAYERS] = prefix_kv[0]
        with pytest.raises(ValueError):
            model.forward_block(block_ids, extra, FB_PREFIX)

    @pytest.mark.parametrize(
        "batch,heads,length,head_dim",
        [
            (FB_BATCH + 1, FB_HEADS, FB_PREFIX, FB_HEAD_DIM),
            (FB_BATCH, FB_HEADS + 1, FB_PREFIX, FB_HEAD_DIM),
            (FB_BATCH, FB_HEADS, FB_PREFIX, FB_HEAD_DIM - 1),
            (FB_BATCH, FB_HEADS, FB_PREFIX + 1, FB_HEAD_DIM),
        ],
    )
    def test_reject_prefix_kv_shape_mismatch(self, batch, heads, length, head_dim):
        model = _fb_model()
        block_ids = _fb_ids(FB_SEED + 10, FB_BATCH, FB_WIDTH)
        gen = torch.Generator().manual_seed(FB_SEED + 11)
        k = torch.randn(batch, heads, length, head_dim, generator=gen)
        v = torch.randn(batch, heads, length, head_dim, generator=gen)
        bad_kv = {layer_id: (k, v) for layer_id in range(FB_LAYERS)}
        with pytest.raises(ValueError):
            model.forward_block(block_ids, bad_kv, FB_PREFIX)


class TestForwardStepMaskValidation:
    @pytest.mark.parametrize(
        "shape",
        [
            (FB_FULL, FB_FULL + 1),
            (1, FB_FULL),
            (FB_FULL, FB_FULL, FB_FULL),
            (),
        ],
    )
    def test_bad_mask_shape_raises_value_error(self, shape):
        # Neither [S, S] nor [1, 1, S, S]: must be rejected up front.
        model = _fb_model()
        ids = _fb_ids(FB_SEED + 12, FB_BATCH, FB_FULL)
        mask = torch.zeros(shape)
        with pytest.raises(ValueError):
            model.forward_step(ids, 0, 0, attention_mask=mask)


# ---------------------------------------------------------------------------
# T003: backend generate pipeline
# ---------------------------------------------------------------------------

GP_SEED = 2301
GP_LAYERS = 2
GP_HEADS = 2
GP_HEAD_DIM = 16
GP_HIDDEN = 32
GP_VOCAB = 64
GP_MAX_SEQ = 64
GP_BATCH = 2
GP_PROMPT = 6  # P
GP_WIDTH = 4  # W (block size)
GP_GEN = 8  # G: two blocks of W
GP_THRESHOLD = 0.9


def _gp_result_class():
    from shadowinfer.execution.block_decoding import GenerateResult

    return GenerateResult


def _gp_model_config():
    return {
        "num_layers": GP_LAYERS,
        "num_heads": GP_HEADS,
        "head_dim": GP_HEAD_DIM,
        "hidden_dim": GP_HIDDEN,
        "vocab_size": GP_VOCAB,
        "max_seq_len": GP_MAX_SEQ,
    }


def _gp_backend(seed):
    from shadowinfer.core.backends.pytorch_backend import PyTorchModelBackend

    backend = PyTorchModelBackend(model_config=_gp_model_config(), seed=seed)
    backend.load(model_name="ar016-tiny", device="cpu")
    return backend


def _gp_prompt(seed, batch=GP_BATCH, length=GP_PROMPT):
    """Random prompt ids in [1, vocab): never the mask token id (0)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(1, GP_VOCAB, (batch, length), generator=gen)


def _gp_config(**overrides):
    from shadowinfer.execution.block_decoding import BlockDecodingConfig

    settings = dict(
        block_size=GP_WIDTH,
        max_rounds=8,
        confidence_threshold=GP_THRESHOLD,
        mask_token_id=0,
        page_size=GP_WIDTH,
        page_precision="fp32",
    )
    settings.update(overrides)
    return BlockDecodingConfig(**settings)


def _gp_in_vocab(sequences):
    """Every generated token lies in [0, vocab)."""
    generated = sequences[:, GP_PROMPT:]
    return bool(((generated >= 0) & (generated < GP_VOCAB)).all())


class TestGenerateNormalCompletion:
    def test_fp32_generate_returns_complete_result_and_stats(self):
        GenerateResult = _gp_result_class()
        backend = _gp_backend(GP_SEED)
        prompt = _gp_prompt(GP_SEED + 1)

        result = backend.generate(prompt, GP_GEN, _gp_config())

        assert isinstance(result, GenerateResult)
        assert isinstance(result.sequences, torch.Tensor)
        assert result.sequences.shape == (GP_BATCH, GP_PROMPT + GP_GEN)
        assert result.sequences.dtype == torch.long
        assert torch.equal(result.sequences[:, :GP_PROMPT], prompt)
        assert _gp_in_vocab(result.sequences)
        assert isinstance(result.stats, dict)
        assert result.stats["blocks"] == GP_GEN // GP_WIDTH
        assert "forced_completions" in result.stats
        assert result.stats["page_precision"] == "fp32"


class TestGeneratePageResidency:
    def test_stores_none_before_any_generate(self):
        backend = _gp_backend(GP_SEED + 10)
        assert backend.last_generate_stores is None

    def test_stores_cover_all_layers_with_exact_append_ledger(self):
        from shadowinfer.shadowkv.paged_kv_store import PagedKVStore

        backend = _gp_backend(GP_SEED + 11)
        prompt = _gp_prompt(GP_SEED + 12)

        backend.generate(prompt, GP_GEN, _gp_config())

        stores = backend.last_generate_stores
        assert stores is not None
        assert set(stores.keys()) == set(range(GP_LAYERS))
        for layer_id in range(GP_LAYERS):
            store = stores[layer_id]
            assert isinstance(store, PagedKVStore)
            # Prefill appends P once; each block's sealing pass appends W
            # once (G total). Rounds only gather, so the append ledger
            # equals P + G iff zero page writes happened between rounds.
            assert store.appended_tokens == GP_PROMPT + GP_GEN
            assert store.rewritten_tokens == 0
            assert store.page_evictions == 0
            assert store.max_seq == GP_PROMPT + GP_GEN

    def test_stores_reflect_the_most_recent_generate(self):
        backend = _gp_backend(GP_SEED + 13)
        prompt = _gp_prompt(GP_SEED + 14)
        backend.generate(prompt, GP_GEN, _gp_config())

        backend.generate(prompt, GP_GEN, _gp_config())

        stores = backend.last_generate_stores
        assert stores is not None
        for layer_id in range(GP_LAYERS):
            # The second run must expose fresh per-run ledgers, not an
            # accumulation across generate calls.
            assert stores[layer_id].appended_tokens == GP_PROMPT + GP_GEN
            assert stores[layer_id].max_seq == GP_PROMPT + GP_GEN


class TestGenerateForcedCompletion:
    def test_max_rounds_one_completes_via_forced_completions(self):
        backend = _gp_backend(GP_SEED + 20)
        prompt = _gp_prompt(GP_SEED + 21)
        config = _gp_config(max_rounds=1)

        result = backend.generate(prompt, GP_GEN, config)

        assert result.sequences.shape == (GP_BATCH, GP_PROMPT + GP_GEN)
        assert torch.equal(result.sequences[:, :GP_PROMPT], prompt)
        assert _gp_in_vocab(result.sequences)
        assert result.stats["forced_completions"] >= 1
        assert result.stats["blocks"] == GP_GEN // GP_WIDTH


class TestGeneratePrecisionPlanes:
    @pytest.mark.parametrize("precision", ["fp32", "fp16", "int8"])
    def test_all_precisions_complete_and_preserve_prompt(self, precision):
        backend = _gp_backend(GP_SEED + 30)
        prompt = _gp_prompt(GP_SEED + 31)
        config = _gp_config(page_precision=precision)

        result = backend.generate(prompt, GP_GEN, config)

        assert result.sequences.shape == (GP_BATCH, GP_PROMPT + GP_GEN)
        assert result.sequences.dtype == torch.long
        assert torch.equal(result.sequences[:, :GP_PROMPT], prompt)
        assert _gp_in_vocab(result.sequences)
        assert result.stats["page_precision"] == precision
        assert result.stats["blocks"] == GP_GEN // GP_WIDTH


class TestGenerateFailFast:
    def test_prompt_with_mask_token_raises(self):
        backend = _gp_backend(GP_SEED + 40)
        prompt = _gp_prompt(GP_SEED + 41)
        prompt[0, 2] = 0  # == config.mask_token_id

        with pytest.raises(ValueError):
            backend.generate(prompt, GP_GEN, _gp_config())
        assert backend.last_generate_stores is None

    def test_prompt_plus_gen_len_over_max_seq_len_raises(self):
        backend = _gp_backend(GP_SEED + 42)
        prompt = _gp_prompt(GP_SEED + 43)
        gen_len = 60  # 60 % W == 0 and P + 60 > max_seq_len

        with pytest.raises(ValueError):
            backend.generate(prompt, gen_len, _gp_config())
        assert backend.last_generate_stores is None

    def test_gen_len_not_multiple_of_block_size_raises(self):
        backend = _gp_backend(GP_SEED + 44)
        prompt = _gp_prompt(GP_SEED + 45)

        with pytest.raises(ValueError):
            backend.generate(prompt, 6, _gp_config())  # 6 % 4 != 0
        assert backend.last_generate_stores is None

    def test_prompt_1d_raises(self):
        backend = _gp_backend(GP_SEED + 46)
        prompt = _gp_prompt(GP_SEED + 47)[0]  # [P]: not 2-D

        with pytest.raises(ValueError):
            backend.generate(prompt, GP_GEN, _gp_config())
        assert backend.last_generate_stores is None

    def test_prompt_float_dtype_raises(self):
        backend = _gp_backend(GP_SEED + 48)
        gen = torch.Generator().manual_seed(GP_SEED + 49)
        prompt = torch.rand(GP_BATCH, GP_PROMPT, generator=gen)

        with pytest.raises(ValueError):
            backend.generate(prompt, GP_GEN, _gp_config())
        assert backend.last_generate_stores is None

    def test_unloaded_backend_raises_runtime_error(self):
        from shadowinfer.core.backends.pytorch_backend import PyTorchModelBackend

        backend = PyTorchModelBackend(model_config=_gp_model_config(), seed=GP_SEED + 50)
        prompt = _gp_prompt(GP_SEED + 51)

        with pytest.raises(RuntimeError):
            backend.generate(prompt, GP_GEN, _gp_config())
        assert backend.last_generate_stores is None
