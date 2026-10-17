#!/usr/bin/env python3
"""Convention-sensitivity check for the reference benchmark.

The danger in publishing a cross-model NLL table is that the number depends on the
scoring convention, not only on the model. This script measures that dependence
directly: it scores the same checkpoint, on the same frozen tokens, under

  (a) block-wise        -- non-overlapping ctx-token blocks, first token of each
                           block not scored. This is the predeclared convention
                           (bench/reference_eval.py) and the one HF Trainer's
                           causal-LM eval uses, so it is the convention the
                           reference checkpoints' own published validation losses
                           were produced under.
  (b) carry-over window -- each scored block is preceded by the previous ctx
                           tokens, so every token except the stream's very first
                           is scored with a full left context. This is the
                           lm-eval-harness rolling-window idea and is the
                           "correct" causal NLL.

If (a) and (b) agree to within a few thousandths of a nat, the convention choice is
not load-bearing for the comparison and the table can say so in one line. If they
disagree materially, the table must report both and say why.

Run on CPU for a small checkpoint; not part of the published table.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
from reference_eval import block_lengths, block_nll, find_frozen_stream  # noqa: E402


def window_nll(model, ids: np.ndarray, ctx: int, device: str = "cpu") -> tuple[float, int]:
    """Score every token except the first with a full ctx of left context.

    Block s is fed as [s-ctx, s+ctx) and the trailing ctx tokens are scored, so the
    scored region is identical to the block-wise convention; only the available
    context differs.
    """
    n = int(ids.size)
    starts = list(range(0, n, ctx))
    total_nats, total_scored = 0.0, 0
    with torch.no_grad():
        for s in starts:
            lo = max(0, s - ctx)
            hi = min(n, s + ctx)
            window = torch.from_numpy(ids[lo:hi].astype(np.int64)).unsqueeze(0).to(device)
            logits = model(input_ids=window).logits.float()
            # positions that predict stream indices [s, hi): the last `hi - s` ones
            scored_preds = logits[:, -(hi - s) - 1: -1, :] if hi - s < window.shape[1] else logits[:, :-1, :]
            tgt = window[:, -scored_preds.shape[1]:]
            lp = torch.log_softmax(scored_preds, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            keep = 1 if s == 0 else scored_preds.shape[1]   # skip the stream's first token
            total_nats += float(-lp[0, -keep:].sum())
            total_scored += int(keep)
    return total_nats, total_scored


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="roneneldan/TinyStories-1M")
    ap.add_argument("--limit-tokens", type=int, default=100_000)
    ap.add_argument("--ctx", type=int, default=512)
    args = ap.parse_args()

    npy, meta = find_frozen_stream(BENCH / "frozen")
    ids = np.load(npy)[: args.limit_tokens]
    print(f"stream {meta['stream_id']} | scoring {ids.size:,} tokens with {args.repo}")

    model = AutoModelForCausalLM.from_pretrained(args.repo, torch_dtype=torch.float32)
    model.eval()

    nats_a, n_a = block_nll(model, ids, args.ctx, batch_blocks=8, device="cpu")
    ll_a = nats_a / n_a
    nats_b, n_b = window_nll(model, ids, args.ctx)
    ll_b = nats_b / n_b

    bpt = meta["bytes_per_token"]
    print(f"\n(a) block-wise    scored={n_a:,}  NLL={ll_a:.6f}  ppl={math.exp(ll_a):.4f}  "
          f"bpb={ll_a/(math.log(2)*bpt):.6f}")
    print(f"(b) carry-over    scored={n_b:,}  NLL={ll_b:.6f}  ppl={math.exp(ll_b):.4f}  "
          f"bpb={ll_b/(math.log(2)*bpt):.6f}")
    print(f"\nabsolute difference = {abs(ll_a-ll_b):.6f} nats/token "
          f"({abs(ll_a-ll_b)/ll_a*100:.3f}% of the value)")
    print(f"perplexity difference = {abs(math.exp(ll_a)-math.exp(ll_b)):.4f}")
    print(f"scored-token difference = {abs(n_a-n_b):,} "
          f"(expected: {len(block_lengths(ids.size, args.ctx))} = one per block)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
