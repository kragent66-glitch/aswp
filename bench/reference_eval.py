#!/usr/bin/env python3
"""Evaluate public TinyStories checkpoints on *our* frozen validation stream.

Why this exists
---------------
`ASWP_Actual_Training_and_Benchmark_Execution_Plan.pdf` section 11 is explicit that
a public checkpoint is contextual reference, not causal evidence, unless
architecture, data, tokenizer, budget and evaluation all match. Rather than quote
numbers other people published under protocols we cannot inspect, we re-measure
every checkpoint ourselves on one frozen token stream with one evaluation routine,
and we report what cannot be known (tokens seen by the reference models) as unknown.

The comparison is legitimate at the token level because the reference family uses
the same token unit we do: gpt2 and EleutherAI/gpt-neo-125M have identical
token->id maps (50257 entries, zero differing ids) and every released
`roneneldan/TinyStories-*` checkpoint ships an identically-keyed vocab. Bits per
byte is reported as well, so third-party models on *other* tokenizers can be added
to the same table without a unit change -- for the reference family itself bpb is
only a linear rescale of NLL.

Scoring convention (predeclared, identical for every model)
----------------------------------------------------------
The frozen stream is cut into non-overlapping blocks of `--ctx` tokens. Within a
block the first token is not scored: it has no left context inside the block, which
is the convention HF Trainer uses for causal-LM evaluation and therefore the
convention the reference checkpoints' own published validation losses were produced
under. The final partial block is included, scoring its remaining tokens. Reported
NLL is the token-weighted mean over every scored token:

    NLL = sum_t -log p(token_t | preceding tokens in its block) / n_scored

Perplexity is exp(NLL); bits-per-byte is NLL / (ln 2 * bytes_per_token), with
bytes_per_token taken from the frozen stream's sidecar metadata.

The scoring arithmetic (`block_lengths`, `score_mask`, `aggregate`) is pure numpy so
that it is testable without torch -- `tests/test_bench_eval.py` pins it against
hand-computed fixtures. Torch only supplies logits and the gather.

Usage
-----
    python bench/reference_eval.py                        # all reference checkpoints
    python bench/reference_eval.py --limit-tokens 200000  # smoke run
    python bench/reference_eval.py --repos gpt2           # any causal LM on the Hub
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
FROZEN_DIR = BENCH_DIR / "frozen"
RESULTS_PATH = BENCH_DIR / "results" / "reference_results.json"

# The published TinyStories reference family. All gpt_neo, window 256,
# max_position_embeddings 2048, intermediate_size = 4 * hidden_size.
REFERENCE_REPOS = [
    "roneneldan/TinyStories-1M",          # hidden 64   x 8 layers (~1M)
    "roneneldan/TinyStories-3M",          # hidden 128  x 8 layers (~3M)
    "roneneldan/TinyStories-8M",          # hidden 256  x 8 layers (~8M)  <- screen-scale match
    "roneneldan/TinyStories-28M",         # hidden 512  x 8 layers (~28M)
    "roneneldan/TinyStories-33M",         # hidden 768  x 4 layers (~33M)
    "roneneldan/TinyStories-1Layer-21M",  # hidden 1024 x 1 layer (~21M, shallow control)
]


# --------------------------------------------------------------------------- #
# scoring core: pure numpy, no torch, unit-tested
# --------------------------------------------------------------------------- #
def block_lengths(n_tokens: int, ctx: int) -> list[int]:
    """Lengths of the non-overlapping blocks the stream is cut into.

    The final block may be shorter than `ctx`. A block shorter than 2 tokens
    predicts nothing and is dropped (it can only occur as a 1-token tail).
    """
    if ctx < 2:
        raise ValueError(f"ctx must be at least 2, got {ctx}")
    if n_tokens <= 0:
        return []
    out = []
    for start in range(0, n_tokens, ctx):
        length = min(ctx, n_tokens - start)
        if length >= 2:
            out.append(length)
    return out


def score_mask(lengths: list[int]) -> np.ndarray:
    """Boolean (n_blocks, max_len-1) mask: True where a token is scored.

    Position j of a block holds the prediction of the block's token j+1, so
    column 0 (the block's first token) is never scored and the trailing columns
    of a short block are padding, also never scored.
    """
    if not lengths:
        return np.zeros((0, 0), dtype=bool)
    max_len = max(lengths)
    mask = np.zeros((len(lengths), max_len - 1), dtype=bool)
    for i, length in enumerate(lengths):
        mask[i, : length - 1] = True
    return mask


def aggregate(gathered: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    """Token-weighted sum of -log p over the scored positions.

    `gathered` is (n_blocks, max_len-1) of log p(true token). The result is a
    *sum* plus the count, never a mean of block means: blocks are not equally
    long, so averaging per-block means would over-weight the short final block.
    """
    gathered = np.asarray(gathered, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if gathered.shape != mask.shape:
        raise ValueError(f"shape mismatch: gathered {gathered.shape} vs mask {mask.shape}")
    n_scored = int(mask.sum())
    if n_scored == 0:
        return 0.0, 0
    return float(-gathered[mask].sum()), n_scored


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def find_frozen_stream(frozen_dir: Path) -> tuple[Path, dict]:
    metas = sorted(frozen_dir.glob("tinystories-valid-*.json"))
    if not metas:
        raise SystemExit(f"no frozen stream in {frozen_dir}; run bench/freeze_stream.py first")
    if len(metas) > 1:
        raise SystemExit(
            f"{len(metas)} frozen streams present ({[m.name for m in metas]}); "
            "exactly one must exist -- remove the stale one deliberately, not silently"
        )
    meta = json.loads(metas[0].read_text())
    npy = frozen_dir / f"{meta['stream_id']}.npy"
    if not npy.exists():
        raise SystemExit(f"metadata {metas[0]} has no matching array {npy}")
    return npy, meta


def model_revision(repo: str) -> dict:
    """The exact commit sha of the weights evaluated, so the number is pinned."""
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=60) as fh:
            meta = json.load(fh)
        return {"model_sha": meta.get("sha"), "last_modified": meta.get("lastModified")}
    except Exception as exc:  # offline / gated / private
        return {"model_sha": None, "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def block_nll(model, ids: np.ndarray, ctx: int, batch_blocks: int, device: str) -> tuple[float, int]:
    """Sum of -log p over scored tokens and the count of scored tokens.

    Log-probabilities are computed explicitly instead of calling the model's loss
    because HF's loss is a mean per block; aggregating those would weight the
    short final block like a full one.
    """
    import torch

    lengths = block_lengths(int(ids.size), ctx)
    total_nats, total_scored = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(lengths), batch_blocks):
            chunk = lengths[start:start + batch_blocks]
            max_len = max(chunk)
            batch = np.zeros((len(chunk), max_len), dtype=np.int64)
            for i, length in enumerate(chunk):
                batch[i, :length] = ids[(start + i) * ctx:(start + i) * ctx + length]
            inp = torch.from_numpy(batch).to(device)
            logits = model(input_ids=inp).logits.float()
            logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            gathered = logprobs.gather(-1, inp[:, 1:].unsqueeze(-1)).squeeze(-1)
            nats, scored = aggregate(gathered.cpu().numpy(), score_mask(chunk))
            total_nats += nats
            total_scored += scored
    return total_nats, total_scored


def evaluate(repo: str, ids: np.ndarray, meta: dict, ctx: int, batch_blocks: int,
             limit_tokens: int | None, device: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    print(f"\n=== {repo} ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    cfg = model.config
    arch = {
        "model_type": getattr(cfg, "model_type", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "num_layers": getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "num_layers", None),
        "num_heads": getattr(cfg, "num_heads", None) or getattr(cfg, "num_attention_heads", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "window_size": getattr(cfg, "window_size", None),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
    }
    print(f"  params={n_params/1e6:.2f}M  {arch}")

    used = ids
    if limit_tokens is not None:
        used = ids[:limit_tokens]
        print(f"  SMOKE: scoring only the first {used.size:,} tokens")
    nats, scored = block_nll(model, used, ctx, batch_blocks, device)
    nll = nats / scored
    bpb = nll / (math.log(2) * meta["bytes_per_token"])
    rec = {
        "repo": repo,
        "n_tokens_in_stream": int(meta["n_tokens"]),
        "ctx": ctx,
        "n_tokens_scored": scored,
        "nll_nats_per_token": nll,
        "perplexity": math.exp(nll),
        "bits_per_byte": bpb,
        "params": n_params,
        "architecture": arch,
        "dtype": "float32",
        "limit_tokens": limit_tokens,
        "wall_seconds": round(time.time() - t0, 1),
        "env": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "host": platform.node(),
            "device": device,
        },
        "stream_id": meta["stream_id"],
        "stream_sha256": meta["token_stream_sha256"],
        "dataset_revision": meta["dataset_revision"],
        **model_revision(repo),
    }
    print(f"  NLL={nll:.6f}  ppl={math.exp(nll):.4f}  bpb={bpb:.6f}  "
          f"scored={scored:,}  {rec['wall_seconds']}s")
    return rec


def merge_results(path: Path, records: list[dict]) -> None:
    """Append results without losing earlier ones (never overwrite a run)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {"runs": []}
    if path.exists():
        existing = json.loads(path.read_text())

    def key(r):
        return (r["repo"], r.get("ctx"), r.get("limit_tokens"))

    seen = {key(r): i for i, r in enumerate(existing.get("runs", []))}
    for rec in records:
        if key(rec) in seen:
            existing["runs"][seen[key(rec)]] = rec
        else:
            existing["runs"].append(rec)
    existing["runs"].sort(key=lambda r: (str(r["repo"]), r.get("limit_tokens") or 0))
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {path} ({len(existing['runs'])} records)")


def print_table(records: list[dict]) -> None:
    print("\n| model | params | scored tokens | val NLL | perplexity | bits/byte |")
    print("|---|---|---|---|---|---|")
    for r in sorted(records, key=lambda r: r["params"]):
        label = r["repo"].split("/")[-1]
        print(f"| `{label}` | {r['params']/1e6:.2f}M | {r['n_tokens_scored']:,} | "
              f"{r['nll_nats_per_token']:.4f} | {r['perplexity']:.3f} | "
              f"{r['bits_per_byte']:.4f} |")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--repos", nargs="*", default=REFERENCE_REPOS)
    ap.add_argument("--ctx", type=int, default=512,
                    help="block length; 512 matches the reference protocol")
    ap.add_argument("--batch-blocks", type=int, default=8,
                    help="blocks per forward pass (lower it if RAM is tight)")
    ap.add_argument("--limit-tokens", type=int, default=None,
                    help="score only this many tokens (smoke test; recorded in the result)")
    ap.add_argument("--frozen-dir", type=Path, default=FROZEN_DIR)
    ap.add_argument("--out", type=Path, default=RESULTS_PATH)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    npy, meta = find_frozen_stream(args.frozen_dir)
    ids = np.load(npy)
    print(f"frozen stream {meta['stream_id']}: {ids.size:,} tokens "
          f"({meta['bytes_per_token']:.4f} bytes/token, dtype={ids.dtype})")

    records = []
    for repo in args.repos:
        try:
            records.append(evaluate(repo, ids, meta, args.ctx, args.batch_blocks,
                                    args.limit_tokens, args.device))
        except Exception as exc:
            print(f"  FAILED {repo}: {type(exc).__name__}: {exc}", file=sys.stderr)
            records.append({"repo": repo, "error": f"{type(exc).__name__}: {exc}",
                            "limit_tokens": args.limit_tokens})

    ok = [r for r in records if "nll_nats_per_token" in r]
    merge_results(args.out, records)
    if ok:
        print_table(ok)
    return 0 if len(ok) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
