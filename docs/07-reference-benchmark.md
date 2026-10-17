# ASWP — Reference Benchmark: public TinyStories models on our frozen stream

**Status:** pre-results. This document reports *other people's models*, measured by us.
It says nothing about ASWP, and it is not evidence that ASWP helps or hurts.

## 1. Why these numbers were re-measured

`ASWP_Actual_Training_and_Benchmark_Execution_Plan.pdf` §11 is explicit: a public
checkpoint is contextual reference, not causal evidence, unless architecture, data,
tokenizer, budget and evaluation pipeline all match. The published validation losses
of the `roneneldan/TinyStories-*` family therefore cannot be dropped into a table next
to our own model — we cannot see the protocol that produced them (tokenizer handling,
sequence packing, evaluation split boundaries, whether model selection touched the
validation file).

So every model here was re-measured by this repository, on one frozen token stream,
with one evaluation routine, on one machine type, and every row records the checkpoint
commit sha it was measured from. The one thing that cannot be recovered — how many
tokens each reference model was trained on — is reported as unknown rather than
guessed (§6).

## 2. The frozen validation stream

| | |
|---|---|
| Dataset | `roneneldan/TinyStories`, revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` |
| File | `TinyStories-valid.txt`, 19 447 282 bytes, sha256 `94e431816c4cce81ff71e4408ff8d3bdaa9a42e8d2663986697c3954288cb38b4` |
| Tokenizer | GPT-2 BPE, `vocab_size = 50257` (see §3) |
| Stream id | `tinystories-valid-aa85f8db4e23` |
| Token-stream sha256 | `aa85f8db4e2336b441f2b46474032bb35f4bc26967e0f6f6a94bc9042fc64154` — sha256 of the **array bytes**, not of the `.npy` file, which carries a header |
| Tokens | 4 724 503 (4.1163 bytes/token, 12 612 distinct ids) |
| Tokens scored per row | 4 715 275 (= all tokens minus one per 512-token block) |

Reproduce: `python bench/freeze_stream.py` then `python bench/verify_stream.py`.
The `.npy` array is gitignored because it regenerates deterministically; the sidecar
JSON and this table are the record.

## 3. Why this family is comparable to a GPT-2-BPE model of ours

The paper states: *"We use GPT-Neo architecture with window size 256 and context length
512. We use GPT-Neo tokenizer but only keep the top 10K most common tokens."*

The "top 10K" clause is a property of the **corpus**, not the tokenizer:

- `gpt2` and `EleutherAI/gpt-neo-125M` `vocab.json` differ as files (1 042 301 vs
  898 669 bytes) but have **identical token→id maps**: 50 257 entries, zero differing
  ids, `<|endoftext|> = 50256`. `merges.txt` is byte-identical
  (`1ce1664773c50f3e…`, 50 001 lines).
- Every released `roneneldan/TinyStories-*` repo ships its own `vocab.json` and
  `tokenizer.json` — **identical across all of them**, and a full 50 257-entry vocab.
- Measured on the data: **15 029 distinct ids occur in train** and the top 10 000 cover
  **99.92%** of tokens (valid: 12 612 ids, top-10k coverage 99.9436%).

So a model of ours using GPT-2 BPE predicts in exactly the same token unit as every row
below, and **token-level NLL is directly comparable**. Bits-per-byte is reported as well:
for this family it is a linear rescale of NLL (`nll_nats / (ln2 × 4.1163)`), and it is
what would let a model with a *different* tokenizer onto the same table.

## 4. Scoring convention, and the ctx-512 wall

Convention (predeclared, identical for every row): the stream is cut into
non-overlapping blocks of 512 tokens; **the first token of each block is not scored**
(it has no left context inside the block), which is HF Trainer's causal-LM convention
and therefore the convention the reference models' own published losses were produced
under. Reported NLL is the token-weighted mean over scored tokens — never a mean of
block means, since the final block is short (`bench/reference_eval.py`).

The window length is **forced to 512** by the checkpoints themselves. Their learned
absolute position embeddings are trained only for rows 0–511:

| positions | mean row-norm |
|---|---|
| 0–511 | 0.143 |
| 512–2047 | 0.082 (flat across all twelve 128-row bins) |

The break is at exactly row 512, and the untrained region's std (0.0103) equals a
*shrunken* init scale (init 0.02 × 0.514 ≈ 13 000 steps of decoupled weight decay at
`lr 5e-4, wd 0.1`) — i.e. no gradient ever reached those rows. The consequence is
measurable:

| input window | mean NLL over its positions |
|---|---|
| 256 | 1.8744 |
| 512 | 2.0203 |
| 768 | 3.6997 |
| 1024 | 4.4782 |
| 1536 | 5.1697 |
| 2048 | 5.6293 |

Verified identical under `transformers` 4.44.2 and 5.20.1, so this is the checkpoint,
not a library regression (the `attn.attention.bias` / `masked_bias` UNEXPECTED-load
warnings under 5.x are unrelated). **Sliding-window / carry-over evaluation of this
family is therefore invalid beyond one context length**, and every row below is
computed at 512. Run `python bench/convention_check.py` before trusting a new
checkpoint in the table.

## 5. Results — six public checkpoints, our frozen stream, ctx 512

All rows: `roneneldan/TinyStories-*`, fp32, `transformers` 5.20.1, `torch` 2.14.1+cu130,
T4 GPU, 4 715 275 scored tokens, stream `tinystories-valid-aa85f8db4e23`.

| model | d / layers | total params | non-embedding params | val NLL | perplexity | bits/byte | checkpoint sha |
|---|---|---|---|---|---|---|---|
| `TinyStories-1M` | 64 / 8 | 3.75M | 0.53M | 2.2190 | 9.198 | 0.7777 | `77f1b168` |
| `TinyStories-3M` | 128 / 8 | 8.28M | 1.85M | 1.7995 | 6.047 | 0.6307 | `cfaf26ec` |
| `TinyStories-8M` | 256 / 8 | 19.70M | 6.84M | 1.4950 | 4.459 | 0.5240 | `8612e3b1` |
| `TinyStories-28M` | 512 / 8 | 51.99M | 26.26M | 1.3422 | 3.827 | 0.4704 | `52dabea9` |
| `TinyStories-33M` | 768 / 4 | 68.51M | 29.92M | 1.3414 | 3.824 | 0.4702 | `2ad0a164` |
| `TinyStories-1Layer-21M` | 1024 / 1 | 66.16M | 14.69M | 1.8030 | 6.068 | 0.6319 | `0d983335` |

Wall time per model on a T4: 89–289 s (GPU total 1 088 s).

### What the table says

- **The published names count non-embedding parameters**, not totals: "1M" is 3.75M
  parameters of which 0.53M are non-embedding, "8M" is 19.70M total / 6.84M
  non-embedding. Any scale-matching must use the non-embedding column, because at
  50 257 vocab the embedding dominates every model at this size.
- **Depth beats width per parameter at this scale.** `1Layer-21M` carries 2.1× the
  non-embedding parameters of `8M` (14.69M vs 6.84M) and is 0.31 nats *worse*
  (1.8030 vs 1.4950). This is the paper's qualitative claim, quantified on one stream.
- **At matched non-embedding scale, wide-and-shallow ≈ narrow-and-deep**: `33M`
  (768×4, 29.92M) and `28M` (512×8, 26.26M) land within 0.0008 nats of each other
  (1.3414 vs 1.3422).
- Improvement with width at fixed 8 layers is monotone across 0.53M → 1.85M → 6.84M →
  26.26M non-embedding parameters: 2.2190 → 1.7995 → 1.4950 → 1.3422.
- A 100k-token CPU smoke run of `TinyStories-1M` gave 2.0050; that subset is scored on a
  different token set from the full stream here (2.2190) and the two are not comparable.
  Only full-stream rows belong in a comparison.

## 6. Limitations

- **Tokens seen by the reference checkpoints is unknown.** The cards give
  `lr = 5e-4` constant, `wd = 0.1`, betas 0.9/0.95, `context_length = 512`,
  `batch_size = 80` and `gradient_accumulation_steps = 16`, and `TinyStories-1Layer-21M`
  was saved from a path ending `checkpoint-28000`. If `batch_size` counts sequences,
  that implies 28 000 × 80 × 16 × 512 ≈ **18.4B tokens**, which contradicts the paper's
  "≤30 h on a single V100" (≈170k tokens/s). The reconstruction is not reliable, so no
  row here carries a tokens-seen figure. **Our model will almost certainly be compared
  at an unequal budget**; the table labels it rather than hiding it.
- **The reference authors may have used `TinyStories-valid.txt` for model selection.**
  The dataset card nominates that file for validation loss, and their checkpoints were
  chosen per-flop-budget. Our primary endpoint is that file because it is the canonical
  shared stream; `TinyStoriesV2-GPT4-valid.txt` (22.5 MB) is preregistered as a pristine
  robustness endpoint for exactly this reason.
- **Nothing here is causal.** A difference between a row and our model mixes
  architecture, optimizer, recipe and possibly token budget. Only the paired
  baseline-vs-ASWP comparison measures ASWP.
- **Community checkpoints are excluded deliberately.** Of the third-party models
  fine-tuned on TinyStories, `reaperdoesntknow/SharperSwarm` is the only one sharing
  this token unit; `neuralcrew/neutrino-instruct` is 3B-class on a different tokenizer
  (and an instruct fine-tune), and `riddhiraj/Butterfly-12M-TinyStories` /
  `Prannesshkva/QU-SSM-15M` require `trust_remote_code=True`. Executing third-party
  modelling code to add rows is a decision this repository has not taken.
- T4 rather than the cheapest available GPU; fp32 throughout, no TF32/BF16 reduction.
  Rounding differences at the 4th decimal are possible on other hardware.

## 7. Reproduce

```bash
python -m pip install -r bench/requirements.txt   # numpy, torch (cpu is enough)
python bench/freeze_stream.py                     # fetch + hash + tokenize + freeze
python bench/verify_stream.py                     # re-derive and compare to the record
python bench/convention_check.py                  # convention sensitivity + ctx-512 wall

# the six rows above (a Modal workspace with free compute; see bench/modal_reference_eval.py)
MODAL_PROFILE=<workspace> modal run bench/modal_reference_eval.py
```

Results are appended to `bench/results/reference_results.json`, one record per
(model, ctx, limit) with its environment, stream id and checkpoint sha. The evaluator is
never reimplemented for a second runner: the Modal path invokes
`bench/reference_eval.py` verbatim, and its scoring core is unit-tested in
`tests/test_bench_eval.py` without torch.
