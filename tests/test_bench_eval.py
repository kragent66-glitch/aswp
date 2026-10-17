"""Tests for the reference-benchmark scoring core.

These pin the *arithmetic* of `bench/reference_eval.py` against hand-computed
values. The scoring core is pure numpy precisely so it is testable in the same CI
as the rest of the suite; the one test that needs a model is skipped when torch
is absent rather than being deleted, because the shift convention is exactly what
a bug would break.

Convention under test (one sentence, so a mutant is unambiguous): the stream is cut
into blocks of `ctx`; position j of a block holds the prediction of that block's
token j+1; the block's first token is therefore never scored, and the final partial
block is scored for its remaining tokens.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from reference_eval import aggregate, block_lengths, score_mask  # noqa: E402
from _strict import STRICT, import_torch_or_skip, require  # noqa: E402


def frozen_pair() -> tuple[dict, Path]:
    """The frozen stream sidecar and array, or a skip/failure if unavailable."""
    frozen = Path(__file__).resolve().parents[1] / "bench" / "frozen"
    metas = sorted(frozen.glob("tinystories-valid-*.json"))
    require(bool(metas), "no frozen stream sidecar in this checkout")
    meta = json.loads(metas[0].read_text())
    npy = frozen / f"{meta['stream_id']}.npy"
    require(npy.exists(), "frozen array not present (it is gitignored; run bench/freeze_stream.py)")
    return meta, npy


# ---------------------------------------------------------------- block_lengths
def test_block_lengths_exact_multiple():
    assert block_lengths(1536, 512) == [512, 512, 512]


def test_block_lengths_keeps_partial_tail():
    # 2000 = 512 + 512 + 512 + 464; the tail must be scored, not dropped
    assert block_lengths(2000, 512) == [512, 512, 512, 464]
    assert sum(block_lengths(2000, 512)) == 2000


def test_block_lengths_drops_a_one_token_tail():
    # a 1-token block predicts nothing, so it contributes no block at all
    assert block_lengths(1025, 512) == [512, 512]
    assert block_lengths(1, 512) == []
    assert block_lengths(0, 512) == []


def test_block_lengths_rejects_degenerate_ctx():
    with pytest.raises(ValueError):
        block_lengths(100, 1)


# ------------------------------------------------------------------ score_mask
def test_scored_position_count_is_length_minus_one_per_block():
    """The shift, stated as a count: a block of L tokens yields L-1 predictions.

    Column j holds the prediction for token j+1, so no column corresponds to the
    block's first token and the mask must have max_len-1 columns, not max_len. A
    mutant that scored every token (including the first) would produce L and a
    mask of width max_len -- this assertion fails on both counts.
    """
    lengths = [512, 512, 464]
    mask = score_mask(lengths)
    assert mask.shape == (3, 511)
    assert mask.sum() == 511 + 511 + 463          # = sum(L) - n_blocks
    for row, length in zip(mask, lengths):
        assert int(row.sum()) == length - 1       # every in-range position scored
    assert mask[2, :463].all() and not mask[2, 463:].any()


def test_score_mask_masks_padding_of_short_blocks():
    # block of 3 -> predicts tokens 1,2 -> columns 0,1; block of 5 -> columns 0..3
    mask = score_mask([3, 5])
    assert mask.shape == (2, 4)
    assert list(mask[0]) == [True, True, False, False]
    assert list(mask[1]) == [True, True, True, True]


def test_score_mask_of_no_blocks_is_empty():
    mask = score_mask([])
    assert mask.shape == (0, 0)


# ------------------------------------------------------------------- aggregate
def test_aggregate_is_token_weighted_not_a_mean_of_block_means():
    """The failure this guards: averaging per-block means over unequal blocks.

    Two blocks of lengths 3 and 5 contribute 2 and 4 scored tokens. The summed
    values are 2.0 and 8.0 nats, so the token-weighted mean (10/6) and the mean of
    block means (1.5) differ by 1/6 nats/token -- a visible change in the second
    decimal of the published table, not something rounding hides.
    """
    lengths = [3, 5]
    mask = score_mask(lengths)
    gathered = np.zeros(mask.shape, dtype=np.float64)
    gathered[0, :2] = -1.0                 # 2 scored tokens x 1.0 nat
    gathered[1, :4] = -2.0                 # 4 scored tokens x 2.0 nats
    nats, n_scored = aggregate(gathered, mask)
    assert n_scored == 6
    assert nats == pytest.approx(10.0)
    token_weighted = nats / n_scored
    mean_of_block_means = (1.0 + 2.0) / 2
    assert token_weighted == pytest.approx(10 / 6)
    assert mean_of_block_means == pytest.approx(1.5)
    assert token_weighted - mean_of_block_means == pytest.approx(1 / 6)


def test_aggregate_ignores_padding_and_counts_exactly():
    lengths = [512, 464]
    mask = score_mask(lengths)
    rng = np.random.default_rng(0)
    gathered = rng.normal(-3.0, 1.0, size=mask.shape)
    poisoned = gathered.copy()
    poisoned[1, 463:] = 1e6                 # garbage in the padding tail
    assert aggregate(gathered, mask) == aggregate(poisoned, mask)
    assert aggregate(gathered, mask)[1] == 511 + 463


def test_aggregate_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        aggregate(np.zeros((2, 3)), np.zeros((2, 4), dtype=bool))


def test_aggregate_of_nothing_is_zero():
    assert aggregate(np.zeros((0, 0)), np.zeros((0, 0), dtype=bool)) == (0.0, 0)


# ------------------------------------------------------- end-to-end convention
def test_uniform_predictions_normalise_by_scored_tokens_only():
    """Uniform logits must give exactly ln V -- in particular the denominator is
    the number of scored tokens, not blocks x ctx. Those differ here (18 vs 24),
    so a mutant that normalised by the block grid would report 1.333 x ln V.
    """
    vocab, ctx, n_tokens = 50257, 8, 21
    lengths = block_lengths(n_tokens, ctx)
    mask = score_mask(lengths)
    gathered = np.full(mask.shape, -math.log(vocab))
    nats, n_scored = aggregate(gathered, mask)
    assert n_scored == sum(l - 1 for l in lengths) == 18
    assert nats / n_scored == pytest.approx(math.log(vocab))
    grid_denominator = len(lengths) * ctx
    assert grid_denominator == 24
    assert (nats / grid_denominator) == pytest.approx(math.log(vocab) * 18 / 24)


def test_block_nll_scores_each_token_against_the_preceding_context():
    """End-to-end shift check with a stub model: logits spike for the *next* token.

    If the implementation scored a token against its own position instead of the
    preceding one, the stub's logits would put probability mass on a different
    id and the per-token NLL would jump from ~0 to ~ln V. Requires torch, and is
    skipped (not deleted) where torch is unavailable.
    """
    torch = import_torch_or_skip()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
    from reference_eval import block_nll

    V, ctx = 32, 8
    rng = np.random.default_rng(0)
    ids = rng.integers(0, V, size=40).astype(np.uint16)

    class NextTokenStub(torch.nn.Module):
        """At position j it puts all mass on the token that actually follows."""

        def forward(self, input_ids):
            B, L = input_ids.shape
            logits = torch.zeros(B, L, V)
            nxt = torch.cat([input_ids[:, 1:], torch.zeros(B, 1, dtype=input_ids.dtype)], dim=1)
            logits.scatter_(2, nxt.unsqueeze(-1), 20.0)
            return type("Out", (), {"logits": logits})()

    nats, scored = block_nll(NextTokenStub(), ids, ctx, batch_blocks=4, device="cpu")
    assert scored == sum(l - 1 for l in block_lengths(ids.size, ctx)) == 35  # 5 blocks x 7
    per_token = nats / scored
    assert per_token < 1e-3, f"expected ~0 nats when the shift is right, got {per_token}"

    # The mutant this test rejects: score position j against the token at j
    # (no shift at all). Same stub, same data, targets shifted the wrong way --
    # the spike sits on a different id, so the cost per token explodes to ~ln V
    # plus the spike. Without this, a shift bug that happened to be consistent
    # could still produce a plausible-looking small number.
    with torch.no_grad():
        inp = torch.from_numpy(ids[:ctx].astype(np.int64)).unsqueeze(0)
        logits = NextTokenStub()(inp).logits.float()
        wrong = torch.log_softmax(logits[:, :-1, :], dim=-1).gather(
            -1, inp[:, :-1].unsqueeze(-1)
        ).squeeze(-1)
    wrong_per_token = float(-wrong.mean())
    assert wrong_per_token > 10.0, f"wrong-shift convention should be very costly, got {wrong_per_token}"
    assert wrong_per_token > per_token * 1e3


# ------------------------------------------------------- the frozen record itself
def test_frozen_metadata_hash_matches_the_array_bytes():
    """The recorded hash must be of the token array's bytes, and must be checkable.

    This guards a mistake that actually happened: `token_stream_sha256` was compared
    against a hash of the `.npy` *file*, which includes a numpy header, so the two can
    never be equal and a correct stream looked corrupt on the second machine that
    verified it. The stream id must also derive from the hash, so a changed stream
    cannot keep a frozen name.
    """
    meta, npy = frozen_pair()

    ids = np.load(npy)
    assert ids.dtype == np.uint16, "stream dtype is part of the frozen record"
    assert ids.size == meta["n_tokens"]
    assert hashlib.sha256(ids.tobytes()).hexdigest() == meta["token_stream_sha256"]
    assert meta["stream_id"] == f"tinystories-valid-{meta['token_stream_sha256'][:12]}"
    assert int(np.unique(ids).size) == meta["distinct_token_ids"]
    assert meta["bytes_per_token"] == pytest.approx(meta["raw_bytes"] / meta["n_tokens"])


def test_frozen_metadata_pins_the_protocol_choices():
    """The sidecar must name the dataset revision and tokenizer it was built from."""
    meta, _npy = frozen_pair()
    assert meta["dataset_repo"] == "roneneldan/TinyStories"
    assert len(meta["dataset_revision"]) == 40, "a revision must be a full commit sha"
    assert meta["tokenizer_vocab_size"] == 50257
    assert meta["tokenizer"] == "gpt2"
