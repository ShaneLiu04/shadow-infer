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
        # row's softmax (exp(-inf) = 0 exactly), so prefix-row logits match
        # an isolated prefix-only forward. The match is up to last-bit
        # matmul noise: the linear layers run at M=FB_FULL vs M=FB_PREFIX,
        # which reorders the reduction in the last bit (measured relL2
        # ~1.4e-7), so this uses the repo's quantified tolerance instead of
        # torch.equal (same rationale as the AR015 design 6.3(a) note).
        model = _fb_model()
        full_ids = _fb_ids(FB_SEED + 5, FB_BATCH, FB_FULL)
        prefix_ids = full_ids[:, :FB_PREFIX]

        prefix_logits, _, _ = model.forward_step(prefix_ids, 0, 0)
        mask = _fb_block_causal_mask(FB_FULL, FB_PREFIX, torch.device("cpu"))
        full_logits, _, _ = model.forward_step(full_ids, 0, 0, attention_mask=mask)

        prefix_window = full_logits[:, :FB_PREFIX, :]
        rel_l2 = (prefix_window - prefix_logits).norm() / prefix_logits.norm()
        assert rel_l2.item() <= 1e-6

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

    @pytest.mark.parametrize("bad_mask_token_id", [GP_VOCAB, -1])
    def test_mask_token_id_out_of_vocab_raises(self, bad_mask_token_id):
        backend = _gp_backend(GP_SEED + 52)
        prompt = _gp_prompt(GP_SEED + 53)

        with pytest.raises(ValueError):
            backend.generate(prompt, GP_GEN, _gp_config(mask_token_id=bad_mask_token_id))
        assert backend.last_generate_stores is None

    def test_unloaded_backend_raises_runtime_error(self):
        from shadowinfer.core.backends.pytorch_backend import PyTorchModelBackend

        backend = PyTorchModelBackend(model_config=_gp_model_config(), seed=GP_SEED + 50)
        prompt = _gp_prompt(GP_SEED + 51)

        with pytest.raises(RuntimeError):
            backend.generate(prompt, GP_GEN, _gp_config())
        assert backend.last_generate_stores is None


# ---------------------------------------------------------------------------
# T004: generation statistics
# ---------------------------------------------------------------------------

GP_STATS_TOTAL_KEYS = (
    "blocks",
    "total_rounds",
    "computed_rows_total",
    "kv_write_rows_total",
    "forced_completions",
    "total_time_ms",
    "page_precision",
)
GP_STATS_PER_BLOCK_KEYS = (
    "rounds_per_block",
    "unmasked_per_round",
    "fallback_per_round",
    "mean_confidence_per_round",
)
GP_STATS_DERIVED_KEYS = ("rows_saved_vs_baseline",)
GP_STATS_ALL_KEYS = GP_STATS_TOTAL_KEYS + GP_STATS_PER_BLOCK_KEYS + GP_STATS_DERIVED_KEYS


def _gp_is_int(value):
    """True ints only: bool is an int subclass but never a valid counter."""
    return isinstance(value, int) and not isinstance(value, bool)


def _gp_generate(seed, gen_len=GP_GEN, **overrides):
    """One real generate call on a fresh seeded backend; returns (backend,
    result) so callers can cross-check stats against the page ledgers."""
    backend = _gp_backend(seed)
    prompt = _gp_prompt(seed + 1)
    result = backend.generate(prompt, gen_len, _gp_config(**overrides))
    return backend, result


def _gp_assert_stats_structure(stats, gen_len=GP_GEN):
    """FR16.4 key-presence / type / shape contract (srs 3.4, design 6.1).

    Every scalar key exists with the right type, and every per-block key is
    a list of per-round lists whose inner lengths match rounds_per_block.
    Returns rounds_per_block for the caller's consistency checks.
    """
    for key in GP_STATS_ALL_KEYS:
        assert key in stats, f"stats is missing required key {key!r}"
    blocks = gen_len // GP_WIDTH
    assert _gp_is_int(stats["blocks"]) and stats["blocks"] == blocks
    assert _gp_is_int(stats["total_rounds"])
    assert _gp_is_int(stats["computed_rows_total"])
    assert _gp_is_int(stats["kv_write_rows_total"])
    assert _gp_is_int(stats["forced_completions"])
    assert stats["forced_completions"] >= 0
    assert _gp_is_int(stats["rows_saved_vs_baseline"])
    assert isinstance(stats["total_time_ms"], float)
    assert isinstance(stats["page_precision"], str)
    rounds = stats["rounds_per_block"]
    assert isinstance(rounds, list) and len(rounds) == blocks
    for rounds_j in rounds:
        assert _gp_is_int(rounds_j) and rounds_j >= 1
    for key in GP_STATS_PER_BLOCK_KEYS[1:]:
        per_block = stats[key]
        assert isinstance(per_block, list) and len(per_block) == blocks
        for j, per_round in enumerate(per_block):
            assert isinstance(per_round, list)
            assert len(per_round) == rounds[j]
    for per_round in stats["unmasked_per_round"]:
        for count in per_round:
            assert _gp_is_int(count) and count >= 0
    for per_round in stats["fallback_per_round"]:
        for flag in per_round:
            assert isinstance(flag, bool)
    for per_round in stats["mean_confidence_per_round"]:
        for confidence in per_round:
            assert isinstance(confidence, float)
            assert math.isfinite(confidence)
            assert 0.0 < confidence <= 1.0
    return rounds


def _gp_expected_rows_saved(rounds_per_block):
    """Closed-form rows_saved derived from stats' own rounds plus the known
    P/W constants (baseline rows per round: L_j + W; actual: W per round
    plus one W-row sealing pass per block)."""
    saved = 0
    for j, rounds_j in enumerate(rounds_per_block):
        prefix_len = GP_PROMPT + j * GP_WIDTH
        baseline_rows = rounds_j * (prefix_len + GP_WIDTH)
        actual_rows = GP_WIDTH * (rounds_j + 1)
        saved += baseline_rows - actual_rows
    return saved


class TestGenerateStatsContract:
    def test_stats_contain_full_key_set_with_correct_types(self):
        # (a) Key presence: the full eight-key set always exists with the
        # documented scalar types and per-block list isomorphism.
        _, result = _gp_generate(GP_SEED + 60)
        _gp_assert_stats_structure(result.stats)
        assert result.stats["page_precision"] == "fp32"


class TestGenerateStatsConsistency:
    def test_total_rounds_equals_sum_of_rounds_per_block(self):
        # (b) total_rounds parses back to the per-block round counts.
        _, result = _gp_generate(GP_SEED + 61)
        rounds = _gp_assert_stats_structure(result.stats)
        assert result.stats["total_rounds"] == sum(rounds)

    def test_computed_rows_total_follows_rounds_formula(self):
        # (b) Each round computes W active rows and the sealing pass adds W
        # more rows per block; no batch factor. Derived purely from stats.
        _, result = _gp_generate(GP_SEED + 62)
        rounds = _gp_assert_stats_structure(result.stats)
        expected = sum(GP_WIDTH * (rounds_j + 1) for rounds_j in rounds)
        assert result.stats["computed_rows_total"] == expected

    def test_kv_write_rows_total_matches_prompt_plus_gen_and_page_ledger(self):
        # (b) Prefill appends P rows and each block's sealing pass appends W
        # rows (G total); cross-checked against the store append ledger.
        backend, result = _gp_generate(GP_SEED + 63)
        _gp_assert_stats_structure(result.stats)
        assert result.stats["kv_write_rows_total"] == GP_PROMPT + GP_GEN
        assert result.stats["kv_write_rows_total"] == (
            backend.last_generate_stores[0].appended_tokens
        )

    def test_unmasked_counts_fill_every_block_without_forced_completion(self):
        # (b) With max_rounds=8 > W=4 the guaranteed-progress fallback
        # finishes every block inside the round loop, so nothing is
        # force-filled and each block's unmask counts sum to exactly B*W.
        _, result = _gp_generate(GP_SEED + 64)
        _gp_assert_stats_structure(result.stats)
        assert result.stats["forced_completions"] == 0
        for per_round in result.stats["unmasked_per_round"]:
            assert sum(per_round) == GP_BATCH * GP_WIDTH

    def test_rows_saved_vs_baseline_matches_closed_form(self):
        # (b) rows_saved parses back to the closed form computed from
        # stats' own rounds_per_block and the known P/W constants.
        _, result = _gp_generate(GP_SEED + 65)
        rounds = _gp_assert_stats_structure(result.stats)
        assert result.stats["rows_saved_vs_baseline"] == _gp_expected_rows_saved(rounds)


class TestGenerateStatsRowsSavedScale:
    def test_rows_saved_positive_at_default_scale(self):
        # (c) Each block contributes rounds_j * L_j - W >= L_j - W
        # = P - W + j*W = 2 + 4j > 0 saved rows, so the total is positive.
        _, result = _gp_generate(GP_SEED + 66)
        _gp_assert_stats_structure(result.stats)
        assert result.stats["rows_saved_vs_baseline"] > 0

    def test_rows_saved_does_not_decrease_with_more_blocks(self):
        # (c) Same backend/prompt seeds make the first two blocks'
        # trajectories identical across both runs, and every extra block
        # adds a positive contribution (>= L_j - W = P + j*W - W > 0), so
        # G=16 saves at least as many rows as G=8.
        backend = _gp_backend(GP_SEED + 67)
        prompt = _gp_prompt(GP_SEED + 68)
        result_g8 = backend.generate(prompt, GP_GEN, _gp_config())
        result_g16 = backend.generate(prompt, 2 * GP_GEN, _gp_config())
        saved_g8 = result_g8.stats["rows_saved_vs_baseline"]
        saved_g16 = result_g16.stats["rows_saved_vs_baseline"]
        assert saved_g8 > 0
        assert saved_g16 > 0
        assert saved_g16 >= saved_g8


class TestGenerateStatsForcedCompletionAccounting:
    def test_max_rounds_one_accounts_for_forced_path(self):
        # (d) max_rounds=1: every block runs exactly one scheduling round
        # and the still-masked positions are force-filled, so per-block
        # unmask totals may fall below B*W (the gap is the forced fill)
        # but stay >= 1 (the top-1 fallback unmasks at least one element
        # per round). forced_completions is only asserted >= 1 and the
        # fallback flags stay structurally isomorphic: a block whose
        # single round unmasks everything above threshold never triggers
        # forced completion, so tighter claims would over-constrain.
        _, result = _gp_generate(GP_SEED + 69, max_rounds=1)
        rounds = _gp_assert_stats_structure(result.stats)
        assert rounds == [1] * (GP_GEN // GP_WIDTH)
        assert result.stats["forced_completions"] >= 1
        for per_round in result.stats["unmasked_per_round"]:
            block_total = sum(per_round)
            assert 1 <= block_total <= GP_BATCH * GP_WIDTH


class TestGenerateStatsTiming:
    def test_total_time_ms_is_nonnegative_float(self):
        # (e) Wall-clock accounting: a finite float, never negative; no
        # upper bound is imposed.
        _, result = _gp_generate(GP_SEED + 70)
        _gp_assert_stats_structure(result.stats)
        assert result.stats["total_time_ms"] >= 0.0


class TestGenerateStatsPrecisionPlanes:
    @pytest.mark.parametrize("precision", ["fp16", "int8"])
    def test_all_precisions_report_full_stats(self, precision):
        # (f) Both reduced-precision page planes still surface the full
        # key set and the correct page_precision identifier.
        _, result = _gp_generate(GP_SEED + 71, page_precision=precision)
        _gp_assert_stats_structure(result.stats)
        assert result.stats["page_precision"] == precision


# ---------------------------------------------------------------------------
# T005: end-to-end exactness vs baseline recompute loop
# ---------------------------------------------------------------------------

T5_SEED = 2401
T5_RELL2_TOL = 1e-6


def _t5_direct_model(seed):
    """SimpleDiffusionLLM built exactly like PyTorchModelBackend.load():
    the same ``torch.manual_seed(seed)`` call followed by the same
    constructor kwargs (cpu device, fp32 dtype), so the weights are
    bit-identical to a backend loaded with the same seed."""
    from shadowinfer.models.simple_diffusion_llm import SimpleDiffusionLLM

    torch.manual_seed(seed)
    model = SimpleDiffusionLLM(
        num_layers=GP_LAYERS,
        num_heads=GP_HEADS,
        head_dim=GP_HEAD_DIM,
        hidden_dim=GP_HIDDEN,
        vocab_size=GP_VOCAB,
        max_seq_len=GP_MAX_SEQ,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    model.eval()
    return model


def _t5_baseline(model, prompt, gen_len, config, capture=False):
    """Full-recompute reference loop (blockwise_generate_baseline)."""
    from shadowinfer.execution.block_decoding import blockwise_generate_baseline

    return blockwise_generate_baseline(model, prompt, gen_len, config, capture_block_logits=capture)


def _t5_assert_bitwise_equal(actual, expected, label):
    """Bit-for-bit sequence equality with a diagnostic first-divergence
    report (row, col, both tokens, total mismatch count). The assertion is
    never relaxed: any divergence is a hard failure of the core exactness
    acceptance (FR16.3)."""
    assert (
        actual.shape == expected.shape
    ), f"{label}: shape {tuple(actual.shape)} != baseline {tuple(expected.shape)}"
    assert (
        actual.dtype == expected.dtype
    ), f"{label}: dtype {actual.dtype} != baseline {expected.dtype}"
    mismatch = actual != expected
    if bool(mismatch.any()):
        row, col = mismatch.nonzero()[0].tolist()
        raise AssertionError(
            f"{label}: {int(mismatch.sum())} token(s) differ; first divergence at "
            f"(row={row}, col={col}): {label}={int(actual[row, col])} vs "
            f"baseline={int(expected[row, col])}"
        )


def _t5_assert_block_logits_structure(captured, rounds_per_block, blocks, label):
    """block_logits_per_round isomorphism: outer length == blocks, inner
    length == rounds_per_block[j], each tensor [B, W, V]."""
    assert isinstance(captured, list), f"{label}: block_logits_per_round must be a list"
    assert len(captured) == blocks
    for j, per_round in enumerate(captured):
        assert isinstance(per_round, list)
        assert len(per_round) == rounds_per_block[j]
        for logits in per_round:
            assert isinstance(logits, torch.Tensor)
            assert logits.shape == (GP_BATCH, GP_WIDTH, GP_VOCAB)


class TestT005Fp32BitwiseMatch:
    @pytest.mark.parametrize(
        "prompt_len,gen_len",
        [
            (GP_PROMPT, GP_GEN),  # (P=6, G=8, W=4)
            (10, 16),  # (P=10, G=16, W=4)
        ],
    )
    def test_generate_matches_baseline_bitwise(self, prompt_len, gen_len):
        # (a) fp32 exactness at two scales: the store-backed generate arm
        # and the full-recompute baseline arm must agree bit-for-bit.
        seed = T5_SEED + prompt_len
        backend = _gp_backend(seed)
        prompt = _gp_prompt(seed + 1, length=prompt_len)
        config = _gp_config()

        gen_result = backend.generate(prompt, gen_len, config)
        base_result = _t5_baseline(_t5_direct_model(seed), prompt, gen_len, config)

        assert gen_result.sequences.shape == (GP_BATCH, prompt_len + gen_len)
        assert torch.equal(gen_result.sequences[:, :prompt_len], prompt)
        assert torch.equal(base_result.sequences[:, :prompt_len], prompt)
        _t5_assert_bitwise_equal(gen_result.sequences, base_result.sequences, "generate")


class TestT005ForcedCompletionBitwiseMatch:
    def test_max_rounds_one_both_arms_bitwise_equal(self):
        # (b) Forced-completion path: with max_rounds=1 every block runs a
        # single scheduling round and the still-masked positions are
        # force-filled from the last block-row logits; both arms share the
        # semantics, so the sequences stay bit-identical. Seeds GP_SEED+20
        # /+21 are known (T003) to exercise the forced path on this model.
        seed = GP_SEED + 20
        backend = _gp_backend(seed)
        prompt = _gp_prompt(seed + 1)
        config = _gp_config(max_rounds=1)

        gen_result = backend.generate(prompt, GP_GEN, config)
        base_result = _t5_baseline(_t5_direct_model(seed), prompt, GP_GEN, config)

        _t5_assert_bitwise_equal(gen_result.sequences, base_result.sequences, "generate")
        assert gen_result.stats["forced_completions"] >= 1
        assert base_result.stats["forced_completions"] == gen_result.stats["forced_completions"]


class TestT005StepwiseBlockLogitsRelL2:
    def test_per_round_block_logits_within_tolerance(self):
        # (c) Stepwise evidence behind the bitwise match: with both arms
        # capturing block-row logits, the per-block round counts agree and
        # every captured [B, W, V] logits tensor is within relL2 <= 1e-6.
        seed = T5_SEED + 50
        backend = _gp_backend(seed)
        prompt = _gp_prompt(seed + 1)
        config = _gp_config()
        blocks = GP_GEN // GP_WIDTH

        gen_result = backend.generate(prompt, GP_GEN, config, capture_block_logits=True)
        base_result = _t5_baseline(_t5_direct_model(seed), prompt, GP_GEN, config, capture=True)

        assert gen_result.stats["rounds_per_block"] == base_result.stats["rounds_per_block"]
        rounds = gen_result.stats["rounds_per_block"]
        gen_logits = gen_result.stats["block_logits_per_round"]
        base_logits = base_result.stats["block_logits_per_round"]
        _t5_assert_block_logits_structure(gen_logits, rounds, blocks, "generate")
        _t5_assert_block_logits_structure(base_logits, rounds, blocks, "baseline")
        for j in range(blocks):
            for r in range(rounds[j]):
                rel = _rel_l2(gen_logits[j][r], base_logits[j][r])
                assert rel <= T5_RELL2_TOL, (
                    f"block {j} round {r}: relL2 {rel:.3e} exceeds " f"{T5_RELL2_TOL:.0e}"
                )


class TestT005CaptureBlockLogitsNoSideEffect:
    def test_capture_flag_leaves_sequences_bitwise_unchanged(self):
        # (d) capture_block_logits is observation-only: the same backend
        # and prompt produce bit-identical sequences with and without it,
        # the default call surfaces no capture key, and the capture call
        # still reports the full stats key set.
        backend = _gp_backend(T5_SEED + 60)
        prompt = _gp_prompt(T5_SEED + 61)
        config = _gp_config()

        plain_result = backend.generate(prompt, GP_GEN, config)
        captured_result = backend.generate(prompt, GP_GEN, config, capture_block_logits=True)

        _t5_assert_bitwise_equal(captured_result.sequences, plain_result.sequences, "capture")
        assert "block_logits_per_round" not in plain_result.stats
        rounds = _gp_assert_stats_structure(captured_result.stats)
        _t5_assert_block_logits_structure(
            captured_result.stats["block_logits_per_round"],
            rounds,
            GP_GEN // GP_WIDTH,
            "generate",
        )


class TestT005BaselineDeterminism:
    def test_same_model_same_prompt_two_runs_bitwise_equal(self):
        # (e) The baseline recompute loop is deterministic: two runs on the
        # same model instance with the same prompt/config agree bit-for-bit.
        model = _t5_direct_model(T5_SEED + 70)
        prompt = _gp_prompt(T5_SEED + 71)
        config = _gp_config()

        first = _t5_baseline(model, prompt, GP_GEN, config)
        second = _t5_baseline(model, prompt, GP_GEN, config)

        _t5_assert_bitwise_equal(second.sequences, first.sequences, "baseline rerun")


class TestT005Fp16ArmCompletes:
    def test_fp16_generate_completes_and_preserves_prompt(self):
        # (f) The fp16 quantized-KV arm is approximate, so only completion,
        # prompt preservation and the page_precision identifier are
        # asserted -- never bitwise agreement with the fp32 arm.
        backend = _gp_backend(T5_SEED + 80)
        prompt = _gp_prompt(T5_SEED + 81)
        config = _gp_config(page_precision="fp16")

        result = backend.generate(prompt, GP_GEN, config)

        assert result.sequences.shape == (GP_BATCH, GP_PROMPT + GP_GEN)
        assert result.sequences.dtype == torch.long
        assert torch.equal(result.sequences[:, :GP_PROMPT], prompt)
        assert _gp_in_vocab(result.sequences)
        assert result.stats["page_precision"] == "fp16"


class TestT005BaselineStatsContract:
    def test_baseline_stats_follow_recompute_semantics(self):
        # The baseline recomputes the full sequence every round and never
        # touches a page store: computed_rows_total follows the
        # rounds_j * (L_j + W) full-row formula, no KV rows are written,
        # nothing is saved versus itself, and the key set matches generate.
        seed = T5_SEED + 90
        prompt = _gp_prompt(seed + 1)
        config = _gp_config()

        base_result = _t5_baseline(_t5_direct_model(seed), prompt, GP_GEN, config)

        rounds = _gp_assert_stats_structure(base_result.stats)
        assert base_result.stats["page_precision"] == "fp32"
        assert base_result.stats["kv_write_rows_total"] == 0
        assert base_result.stats["rows_saved_vs_baseline"] == 0
        expected_rows = sum(
            rounds_j * (GP_PROMPT + j * GP_WIDTH + GP_WIDTH) for j, rounds_j in enumerate(rounds)
        )
        assert base_result.stats["computed_rows_total"] == expected_rows


class TestT005BaselineFailFast:
    def test_prompt_with_mask_token_raises(self):
        model = _t5_direct_model(T5_SEED + 100)
        prompt = _gp_prompt(T5_SEED + 101)
        prompt[0, 2] = 0  # == config.mask_token_id

        with pytest.raises(ValueError):
            _t5_baseline(model, prompt, GP_GEN, _gp_config())

    def test_prompt_plus_gen_len_over_max_seq_len_raises(self):
        model = _t5_direct_model(T5_SEED + 102)
        prompt = _gp_prompt(T5_SEED + 103)
        gen_len = 60  # 60 % W == 0 and P + 60 > max_seq_len

        with pytest.raises(ValueError):
            _t5_baseline(model, prompt, gen_len, _gp_config())

    def test_gen_len_not_multiple_of_block_size_raises(self):
        model = _t5_direct_model(T5_SEED + 104)
        prompt = _gp_prompt(T5_SEED + 105)

        with pytest.raises(ValueError):
            _t5_baseline(model, prompt, 6, _gp_config())  # 6 % 4 != 0

    def test_prompt_1d_raises(self):
        model = _t5_direct_model(T5_SEED + 106)
        prompt = _gp_prompt(T5_SEED + 107)[0]  # [P]: not 2-D

        with pytest.raises(ValueError):
            _t5_baseline(model, prompt, GP_GEN, _gp_config())

    def test_prompt_float_dtype_raises(self):
        model = _t5_direct_model(T5_SEED + 108)
        gen = torch.Generator().manual_seed(T5_SEED + 109)
        prompt = torch.rand(GP_BATCH, GP_PROMPT, generator=gen)

        with pytest.raises(ValueError):
            _t5_baseline(model, prompt, GP_GEN, _gp_config())

    @pytest.mark.parametrize("bad_mask_token_id", [GP_VOCAB, -1])
    def test_mask_token_id_out_of_vocab_raises(self, bad_mask_token_id):
        model = _t5_direct_model(T5_SEED + 110)
        prompt = _gp_prompt(T5_SEED + 111)

        with pytest.raises(ValueError):
            _t5_baseline(model, prompt, GP_GEN, _gp_config(mask_token_id=bad_mask_token_id))
