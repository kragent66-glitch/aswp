"""Tests for the data pipeline.

The sampler half is numpy-only and runs in both CI jobs. The tokenizer half needs
`transformers`, so it uses the shared skip-vs-fail helper: a clean skip locally, a failure
in the job that is supposed to exercise it.

What is pinned here is the two things that would silently invalidate a comparison: that
chunked tokenization equals whole-file tokenization (or the training stream is not the
corpus everyone thinks it is), and that the sampled order is deterministic and resumable
(or the paired arms did not see the same data).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _strict import require  # noqa: E402

from harness.data import (  # noqa: E402
    EOT_MARKER,
    EOT_TOKEN_ID,
    BlockSampler,
    iter_pieces,
    tokenize_chunked,
)

FIXTURE = (
    "Once upon a time there was a little girl named Lily.\n"
    f"{EOT_MARKER}\n"
    "One day a small robot learned to count to three.\n"
    f"{EOT_MARKER}\n"
    "The end of the third story, and a rather long closing sentence to force a boundary.\n"
    f"{EOT_MARKER}\n"
)


def write_fixture(tmp_path: Path, text: str = FIXTURE, name: str = "fixture.txt") -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def get_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        require(False, "transformers is not installed")
        return None
    return AutoTokenizer.from_pretrained("gpt2", use_fast=True)


# ------------------------------------------------------------------ chunked tokenizing
def test_chunked_tokenization_equals_whole_file_tokenization(tmp_path):
    """The load-bearing equivalence: a 467M-token stream is only buildable in pieces.

    The stream is assembled from marker-aligned pieces; if a subword or a marker could span
    a boundary the concatenation would differ from the canonical whole-file tokenization and
    every run would be training on a slightly different corpus. Tested at a chunk size far
    smaller than a story, so boundaries land inside sentences and inside markers.
    """
    tok = get_tokenizer()
    path = write_fixture(tmp_path)
    whole = np.asarray(
        tok(FIXTURE, add_special_tokens=False, truncation=False)["input_ids"], dtype=np.uint16
    )
    for chunk_bytes in (1, 7, 13, 64, 4096):
        chunked = tokenize_chunked(path, tok, chunk_bytes=chunk_bytes)
        assert chunked.size == whole.size, f"chunk_bytes={chunk_bytes} size"
        assert np.array_equal(chunked, whole), f"chunk_bytes={chunk_bytes} content"


def test_every_marker_becomes_exactly_one_token(tmp_path):
    """The marker invariant, which is what makes piece boundaries safe."""
    tok = get_tokenizer()
    path = write_fixture(tmp_path)
    ids = tokenize_chunked(path, tok, chunk_bytes=9)
    assert int((ids == EOT_TOKEN_ID).sum()) == FIXTURE.count(EOT_MARKER) == 3


def test_pieces_end_exactly_after_a_marker(tmp_path):
    """Boundaries are structural, not arbitrary: every piece but the last ends at a marker."""
    path = write_fixture(tmp_path)
    pieces = list(iter_pieces(path, chunk_bytes=5))
    assert len(pieces) > 1, "the fixture must span several pieces for this to mean anything"
    assert all(p.endswith(EOT_MARKER) for p in pieces[:-1])
    assert "".join(pieces) == FIXTURE, "pieces must reassemble into the original text exactly"
    # the final piece is the text after the last marker -- here a bare newline, which is a
    # real token. Flushing only non-whitespace remainders would drop it.
    assert pieces[-1] == "\n"


def test_a_marker_split_across_reads_is_reassembled(tmp_path):
    """A one-byte read size forces every marker to straddle a boundary."""
    tok = get_tokenizer()
    path = write_fixture(tmp_path)
    ids = tokenize_chunked(path, tok, chunk_bytes=1)
    assert int((ids == EOT_TOKEN_ID).sum()) == 3
    assert ids.size == int(
        len(tok(FIXTURE, add_special_tokens=False)["input_ids"])
    )


def test_trailing_text_without_a_final_marker_is_kept(tmp_path):
    tok = get_tokenizer()
    text = FIXTURE + "a trailing story with no closing marker"
    path = write_fixture(tmp_path, text=text)
    ids = tokenize_chunked(path, tok, chunk_bytes=32)
    assert ids.size == int(len(tok(text, add_special_tokens=False)["input_ids"]))


def test_iter_pieces_refuses_a_document_with_no_marker(tmp_path):
    """Better to fail than to split a document and silently break canonicality."""
    path = write_fixture(tmp_path, text="no markers anywhere in this text at all", name="nom.txt")
    with pytest.raises(RuntimeError, match="refusing to split"):
        list(iter_pieces(path, chunk_bytes=4, max_carry_bytes=16))


# -------------------------------------------------------------------------- sampling
def make_ids(n_tokens: int, seed: int = 0) -> np.ndarray:
    """Random ids whose block-start tokens are unique.

    Block identity in these tests is read off the first token of a block, so the starts are
    made distinct deliberately: with 1000 random ids and 8 blocks a collision is a ~3% flake,
    and a flaky determinism test is worse than no determinism test.
    """
    rng = np.random.default_rng(seed)
    ids = rng.integers(1, 1000, size=n_tokens).astype(np.uint16)
    starts = np.arange(len(range(0, n_tokens, 512)))
    ids[::512] = starts.astype(np.uint16)
    return ids


def test_sampler_yields_blocks_of_exactly_ctx_tokens():
    ids = make_ids(512 * 10)
    s = BlockSampler(ids, ctx=512, seed=1)
    assert len(s) == 10
    block = s.next_block()
    assert block.shape == (512,) and block.dtype == np.int64
    batch = s.next_batch(4)
    assert batch.shape == (4, 512)


def test_an_epoch_is_a_permutation_with_no_repeats_or_gaps():
    """Every block exactly once per epoch: no block is seen twice, none is dropped."""
    ids = make_ids(512 * 8)
    s = BlockSampler(ids, ctx=512, seed=3)
    first = [int(s.next_block()[0]) for _ in range(8)]   # first token identifies the block
    assert len(set(first)) == 8, "a block was repeated within one epoch"
    expected = {int(ids[i * 512]) for i in range(8)}
    assert set(first) == expected, "some blocks were never sampled"
    assert len(s) == 0, "every block has been handed out exactly once"
    assert s.epoch == 0, "the epoch rolls over lazily, on the next draw"
    s.next_block()
    assert s.epoch == 1


def test_same_seed_gives_the_same_order_and_a_different_seed_does_not():
    ids = make_ids(512 * 12)
    a = BlockSampler(ids, ctx=512, seed=7)
    b = BlockSampler(ids, ctx=512, seed=7)
    c = BlockSampler(ids, ctx=512, seed=8)
    ab = [int(a.next_batch(2)[0, 0]) for _ in range(4)]
    bb = [int(b.next_batch(2)[0, 0]) for _ in range(4)]
    cb = [int(c.next_batch(2)[0, 0]) for _ in range(4)]
    assert ab == bb
    assert ab != cb


def test_resume_continues_the_exact_order():
    """Phase 0 item 7 for data: a resumed run must not see different tokens."""
    ids = make_ids(512 * 20)
    a = BlockSampler(ids, ctx=512, seed=11)
    for _ in range(5):
        a.next_batch(2)                        # 10 of 20 blocks consumed
    state = a.state_dict()                     # the resume point
    expected = [int(a.next_batch(2)[0, 0]) for _ in range(3)]   # what a does next

    b = BlockSampler(ids, ctx=512, seed=999)   # deliberately wrong seed
    b.load_state_dict(json.loads(json.dumps(state.__dict__)))   # as it would round-trip
    assert [int(b.next_batch(2)[0, 0]) for _ in range(3)] == expected


def test_resume_across_an_epoch_boundary():
    ids = make_ids(512 * 6)
    a = BlockSampler(ids, ctx=512, seed=5)
    a.next_batch(6)                     # exactly exhausts epoch 0
    state = a.state_dict()
    b = BlockSampler(ids, ctx=512, seed=1234)
    b.load_state_dict(state)
    assert [int(b.next_batch(1)[0, 0]) for _ in range(3)] == \
           [int(a.next_batch(1)[0, 0]) for _ in range(3)]


def test_resume_rejects_a_different_stream_length():
    ids = make_ids(512 * 10)
    s = BlockSampler(ids, ctx=512, seed=1)
    state = s.state_dict()
    other = BlockSampler(make_ids(512 * 9), ctx=512, seed=1)
    with pytest.raises(ValueError, match="cannot be resumed"):
        other.load_state_dict(state)


def test_a_malformed_sampler_checkpoint_is_refused():
    """Three ways a same-length state can be wrong, and one that must stay legal.

    The length check alone accepted an order that repeated some blocks and omitted others: the
    load succeeded and the run continued while seeing different data than the run it resumed.
    """
    import dataclasses

    ids = make_ids(512 * 8)
    sampler = BlockSampler(ids, ctx=512, seed=11)
    sampler.next_block()
    good = sampler.state_dict()

    duplicated = dataclasses.replace(good, order=[good.order[0]] * len(good.order))
    with pytest.raises(ValueError, match="repeats or omits"):
        sampler.load_state_dict(duplicated)

    out_of_range = dataclasses.replace(good, order=[sampler.n_blocks] + list(good.order[1:]))
    with pytest.raises(ValueError, match="out-of-range"):
        sampler.load_state_dict(out_of_range)

    with pytest.raises(ValueError, match="outside 0"):
        sampler.load_state_dict(dataclasses.replace(good, position=sampler.n_blocks + 1))
    with pytest.raises(ValueError, match="outside 0"):
        sampler.load_state_dict(dataclasses.replace(good, position=-1))

    # An exhausted epoch is a real state: the next draw rolls over into the next epoch.
    sampler.load_state_dict(dataclasses.replace(good, position=sampler.n_blocks))
    assert sampler.next_block().shape == (512,)
    sampler.load_state_dict(good)
    assert sampler.position == 1


def test_a_refused_sampler_state_leaves_the_sampler_unchanged():
    """Validation happens before assignment, so a refusal cannot half-apply.

    The first version of these checks assigned the epoch, position and generator and only then
    validated the permutation -- the exception was raised with the sampler already mutated.
    """
    import dataclasses

    ids = make_ids(512 * 4)
    sampler = BlockSampler(ids, ctx=512, seed=5)
    sampler.next_block()
    at_refusal = sampler.state_dict()

    # A reference restored to exactly this state defines what the sampler must do next. (Taking
    # `next_block()` directly would answer a different question: it would give the block *after*
    # the state captured, which is not what the next call on `sampler` returns.)
    reference = BlockSampler(ids, ctx=512, seed=999)
    reference.load_state_dict(at_refusal)
    expected_next = reference.next_block()

    bad = dataclasses.replace(at_refusal, position=at_refusal.n_blocks + 3,
                              order=list(reversed(at_refusal.order)))
    with pytest.raises(ValueError):
        sampler.load_state_dict(bad)

    after = sampler.state_dict()
    assert after.position == at_refusal.position
    assert after.epoch == at_refusal.epoch
    assert list(after.order) == list(at_refusal.order)
    assert after.generator_state == at_refusal.generator_state
    assert np.array_equal(sampler.next_block(), expected_next)
    assert np.array_equal(sampler.next_block(), reference.next_block())


def test_sampler_rejects_degenerate_configurations():
    with pytest.raises(ValueError):
        BlockSampler(make_ids(100), ctx=1)
    with pytest.raises(ValueError, match="cannot fill one block"):
        BlockSampler(make_ids(100), ctx=512)
    s = BlockSampler(make_ids(512 * 4), ctx=512, seed=0)
    with pytest.raises(ValueError):
        s.next_batch(0)


def test_drop_last_false_exposes_the_partial_tail_but_cannot_batch_it():
    """The tail option exists so nothing is silently unreachable; it is not the default.

    A short block has fewer than `ctx` columns, so it cannot be stacked into a batch
    without padding, and padding a causal-LM block would inject tokens the model is then
    scored on. Hence `drop_last=True` by default and no silent padding anywhere.
    """
    ids = make_ids(512 * 3 + 100)
    assert BlockSampler(ids, ctx=512, seed=0).n_blocks == 3
    padded = BlockSampler(ids, ctx=512, seed=0, drop_last=False)
    assert padded.n_blocks == 4

    blocks = [padded.next_block() for _ in range(4)]
    shapes = sorted(b.shape[0] for b in blocks)
    assert shapes == [100, 512, 512, 512], shapes
    with pytest.raises(ValueError):
        np.stack(blocks)

    # and the default never produces a short block at all
    s = BlockSampler(ids, ctx=512, seed=0)
    assert all(b.shape[0] == 512 for b in (s.next_block() for _ in range(3)))
