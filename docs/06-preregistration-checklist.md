# ASWP — Preregistration Checklist

**Purpose.** This is the gate for confirmatory training. The protocol states that
every bracketed field in the SAP is completed and committed *before* confirmatory
data collection, and that later changes are protocol deviations rather than silent
edits. This document is the operational form of that rule: it lists what is frozen,
what is open, and — for each open item — whether it is blocked on a **decision**
(protocol owner) or on **work** (the harness, Phase 0, Phase 1).

**Status: decision items 1–6 and 9 are resolved and committed; Phase 0 is complete on both
CPU and GPU, and its cost measurement fixed n = 8. The release tag is published
(`v0.1.0-preregistration`). Open before Phase 2: an external signature on the tag, a successor
tag matching the commit the confirmatory runs execute, and the σ0 the screen must choose. An
external review (Codex, 2026-10-17) is recorded in §5 with each finding's resolution.**

Note: the SAP (`docs/03`) carried `<value>` placeholders in its Locked Configuration Fields
until this point. That was an auditability failure in its own right — a normative document that
does not state the values cannot pre-register anything — and the review caught it. Every field
is now filled except σ0, which is the screen's output.

---

## 1. Frozen now

| Item | Value | Source |
|---|---|---|
| Confirmatory hypothesis | one-sided (H1: dᵢ > 0), α = 0.05 | SAP §Hypotheses; 05 §P1 |
| Minimum / target paired seeds | n_min = 6, n_target = 8 | SAP §Sample Size; 05 §P1 |
| Paired seed list | **8 integers, frozen** — `1857649130, 1434172854, 1387619681, 536290471, 1202662852, 1106087492, 1881574181, 1859844390` | `config/manifest.json` (issue #10) |
| Per-role RNG streams | `global`, `init`, `data-loader`, `dropout`, `perturbation` — derived per paired seed; perturbations use a dedicated `torch.Generator`. The loop reads them from the manifest rather than re-deriving them, so the mapping cannot drift | `config/manifest.json`; protocol §Reproducibility |
| Seed derivation | `sha256("ASWP/v1/<role>/<n>")`, first 8 bytes big-endian, mod 2³¹ — recomputable by anyone, re-checked by the test suite | `analysis/seed_manifest.py` |
| Practical threshold δ | 1.0 × SD_baseline, plus its SD_delta equivalent 1.0/r | SAP §Practical Significance; 05 §P5 |
| Unit convention | r = √(1 + q² − 2qρ); predeclared tolerance box q ≤ 1.25, ρ ≥ 0; worst corner r = 1.6008 | SAP §Endpoint; 05 §P5 |
| Interval convention | one-sided 95% randomization bound, inverted from the same one-sided test | SAP §Primary Analysis; 05 §P6 |
| Decision rule | point estimate > δ **and** one-sided 95% bound excludes 0 | SAP §Decision Language |
| Primary test | exact sign-flip, all 2ⁿ patterns enumerated | SAP §Primary Analysis; 05 §P8 |
| Selection rule | ≤ 1 instability event, throughput ≥ 80% of baseline, next-best fallback | SAP §Multiplicity; 05 §P3 |
| Selection-stability check | chosen + runner-up σ0 on 2 fresh seeds; any reversal → exploratory | 05 §P3 |
| Margin pre-filter | top-two gap over all candidates, m = 0.5·σ_screen, pre-filter only | 05 §P3, §P5 |
| C4 arms | 0.6124·σ0 (variance-matched) and σ0 (dose extreme). The factor is **derived** — mean σ(t)²/σ0² → 3/8 — and the finite-run value (0.6121 at T = 1526, −0.044%) is recorded in every run summary rather than assumed equal | 05 §P2; `harness/arms.py` |
| C5/C5b tuning | same screen rule and budget as treatment (symmetric effort) | 05 §P2 |
| Instability trigger | ≥ 30% of pairs unstable → "instability-dominated negative" | 05 §P4 |
| Eligible modules | attention q/k/v/o + MLP up/gate/down; no embeddings, head, biases, norms, tied weights — 56 projections at d256/L8 | protocol §Primary Design Choices |
| Noise spec | Gaussian, RMS-relative scaling, cosine schedule reaching exactly 0 at the final step, one perturbation per matrix per optimizer update shared across microbatches | protocol §Method Definition, §Schedules |
| Precision policy | noise and perturbation constructed in fp32; realized perturbation measured after any bf16 cast | protocol §Primary Design Choices |
| Analysis reproducibility | seeds 42 / 7 / 20260905; 40 000 trials per cell; 2 000 000 screen draws; 256 MiB memory budget | 05 §P7, §P8 |

### Resolved decisions (items 1–4)

| Item | Locked value | Note |
|---|---|---|
| 1. Tokenizer | **GPT-2 BPE**, 50 257 ids, via `transformers` `gpt2` (fast) | Gives token-level comparability with the reference family; the paper's "top 10K" is a corpus property, not a tokenizer one |
| 2. Token / step budget | **250M tokens**; 80 × accum 4 × ctx 512 = 163 840 tokens/step → **1526 steps**, 250 019 840 realized (+0.007%) | 1526 optimizer updates, versus 381 under the reference recipe's effective batch — the update count is the number that matters, because ASWP perturbs once per update |
| 3. Primary checkpoint step | **the final step T = 1526**, where the annealed schedule is *exactly* 0 and the endpoint weights are untouched by the intervention | Evaluation reads the model's own weights, not the perturbed ones |
| 4. Optimizer | **AdamW, lr 5e-4 constant, betas (0.9, 0.95), eps 1e-8, wd 0.1** — the reference recipe; **`max_grad_norm = 1.0`**, which is our addition | The reference card documents no clipping. Clipping is applied identically to every arm, so it cannot bias the contrast; it is recorded here as a deliberate deviation rather than a default |
| 4a. Batch configuration | **80 sequences × accum 4**, ctx 512 | Keeps the reference micro-batch (which governs per-step gradient statistics and memory) while multiplying perturbation events |
| 4b. Dropout | **0.0** in the primary arms | Dropout is comparator C5b, not a default |
| 4c. Model scale | Screen at **d256/L8 = 6.43M non-embedding** (0.94× the reference "8M" row); verification planned at d512/L8 = 25.70M non-embedding (0.98× "28M") | Scale-matching uses the published non-embedding convention |
| 4d. Training precision | **bf16** on a tensor-core GPU (A10G primary, L4 fallback). Noise and perturbation still constructed in fp32, and the realized perturbation is measured *after* the cast, as the SAP requires — verified at scale on hardware: ratio 1.0059 at σ = 0.03 | measured, item 10 |

### Resolved work (items 5–6)

| Item | Frozen value | Where |
|---|---|---|
| 5. Dataset revision | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` (`roneneldan/TinyStories`), CDLA-Sharing-1.0 | `data/train_manifest.json`, `bench/frozen/*.json` |
| 5a. Training stream | `tinystories-train-41272badd4c7` — 469 429 735 tokens, raw sha256 `c5cf5e22ff13…`, 4.0992 bytes/token, 29 362 distinct ids | `data/train_manifest.json` |
| 5b. Confirmatory endpoint | `tinystories-v2-gpt4-valid-fd5d9b500ac0` — 5 532 654 tokens, 4.0672 bytes/token, flagged `pristine` | `bench/frozen/` |
| 5c. Comparability stream | `tinystories-valid-aa85f8db4e23` — 4 724 503 tokens; the stream the six reference rows are measured on | `bench/frozen/` |
| 5d. Packed-sequence behaviour | The stream is cut into non-overlapping 512-token blocks; an epoch is a permutation of those blocks drawn from a dedicated data-loader generator, so both arms of a pair see an identical token order | `harness/data.py` |
| 5e. Tokenization post-hash | Marker-aligned chunking, **proven equal to whole-file tokenization**: 8 MiB pieces reproduce the frozen whole-file hash exactly, and the marker count invariant (2 119 718 eot ids == 2 119 718 markers in the raw bytes) holds for the built stream | `harness/data.py`; `tests/test_harness_data.py` |
| 5f. Tokenizer artifacts pinned by hash | `gpt2` @ revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`; `vocab.json`, `merges.txt`, `tokenizer.json`, `tokenizer_config.json` hashed. All three manifests (train + both endpoints) carry an identical provenance block, so "GPT-2 BPE" is a verifiable claim rather than a description | `harness/data.py::tokenizer_provenance` |
| 6. Model source commit + architecture | `harness/model.py` at the run's `git_commit` (recorded in every checkpoint and summary); pre-norm RMSNorm, RoPE, SwiGLU, no biases, tied embeddings, ctx 512, dropout 0 | `harness/model.py` |

## 2. Open — and what each is waiting on

**Blocked by design (produced by the runs themselves):**

7. **Selected σ0 and its exploratory parent run ID.** By protocol this comes out of
   the Phase 1 screen. Its *rule* is already locked; only the outcome is pending.

**Remaining process and measurement steps:**

8. **Signatures, commit hash, and archive tag** — the release tag is **published**:
   `v0.1.0-preregistration`, with notes stating what it freezes and, explicitly, what it does
   *not* contain (σ0, any result). **Still open: an external signature/timestamp**, which needs
   the protocol owner — the tag commits the code, but only a signed timestamp proves *when* the
   protocol was fixed. Also open: the tag predates the runner commits, so a successor tag must
   be cut from the commit the confirmatory runs actually execute. The external review flagged
   this and it is a real gap: the tag must name the executed code, not the near-miss.
9. ~~Pin the tokenizer artifacts by hash.~~ **Done.** The tokenizer is pinned by
   artifact hash at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, and the
   provenance block is identical across the training manifest and both endpoint
   sidecars because all three are written by one shared helper. Re-freezing both
   validation streams reproduced their stream ids and token-stream hashes exactly,
   which is the check that the schema change did not disturb the data.
10. **Phase 0, GPU half — DONE.** Nine correctness checks pass on CPU; the GPU half is
    measured and it decided n.

    | segment | GPU | precision | tokens/s | $/run | incl. endpoint eval | n |
    |---|---|---|---|---|---|---|
    | A10-bf16 | A10G | bf16 | 126 059 | 0.6060 | **0.6601** | **8** |
    | L4-bf16 | L4 | bf16 | 88 282 | 0.6293 | 0.6687 | 8 |
    | L4-fp32 | L4 | fp32 | 38 479 | 1.4439 | 1.4832 | — |
    | L4-fp32 baseline | L4 | fp32 | 37 321 | 1.4887 | 1.5280 | — |
    | T4-bf16 | T4 | bf16 | 15 792 | 2.5947 | 2.6237 | — |

    **Decision: n = 8 (28 runs = $18.48–18.72 of the $22.50 usable, reserve intact).** Both
    bf16 tensor-core profiles clear the rule's strictest band; fp32 and the T4 do not. A10G is
    chosen: cheapest per run *and* 1.43× faster than L4 in wall-clock.

    Measured along the way, and part of what makes the number trustworthy:
    - **ASWP overhead is not measurable above noise** (−3.0%, 4.258 s vs 4.390 s per step on
      one 30-step segment per arm). The intervention is negligible for the budget; the *sign*
      is noise and is not being claimed as a speedup.
    - **The decided batch shape (80 × 512) cannot run in fp32** on a 16 GB card: 40 960 ×
      50 257 vocab = 8.2 GB for the logits alone, plus an equal gradient buffer. The loss is
      therefore chunked over the vocabulary projection (equivalence pinned by test), and the
      endpoint evaluation costs 176.9 s per run on the full streams.
    - **The endpoint NLL is read from bf16 weights** while the six reference rows are fp32.
      That is a real caveat for the *comparability* row in docs/07 (absolute NLL), and
      irrelevant to the confirmatory contrast, where both arms share the precision.

## 3. Order of operations

```
harness written  →  Phase 0 gate passes  [DONE on CPU: 9/9 for d256/L8]
                 →  decision items 1–4 locked, items 5–6 hashed and committed  [DONE]
                 →  tokenizer artifacts hashed (item 9)  [DONE]
                 →  GPU half of Phase 0 + cost segment (item 10)  →  n fixed  [DONE: n = 8]
                 →  protocol release tagged (item 8)  [DONE: v0.1.0-preregistration; successor
                                                     tag to name the executed commit]
                 →  Phase 1 screen (C0 + C1–C3)  →  item 7 locked from its result
                 →  Phase 2 confirmation (paired seeds from config/manifest.json)
```

Nothing in Phase 2 may begin until items 1–6 and 8 are committed. That ordering is
the whole point of the preregistration and is why this document exists separately
from the SAP: the SAP states the rules, this states which rules still lack values.

Three implementation properties are worth stating here because they are what make
the paired comparison mean anything, and each is asserted rather than intended:

- **Zero-noise equivalence is bitwise.** The baseline arm runs the *same wrapped code
  path* with σ = 0, so the arm contrast cannot come from the code.
- **One scoring path.** Evaluation calls `bench/reference_eval.block_nll`, the same
  function that produced the six reference rows, so our NLL and theirs cannot differ
  in how the mean is aggregated.
- **An unstable run is retained and flagged**, never silently dropped: a non-finite
  loss skips that update and is recorded in the ledger and counted in the summary.

## 4. What is already executable

The analysis is not blocked by any of the above — it runs today:

```bash
python -m pip install numpy scipy pytest
python -m pytest                      # 138 tests, green under ASWP_BENCH_STRICT=1
python analysis/seed_manifest.py      # verify the frozen seeds against their derivation
python analysis/p1_exact_power.py     # resolution, power, δ unit conversion
python analysis/p23_match_curse.py    # variance matching, winner's curse, margin rule
python analysis/p4_delta_criteria.py  # decision-rule fire rates, type-I error, δ under units

python harness/data.py --out-dir data          # build the training stream (deterministic)
python -m harness.phase0 --model screen_d256_l8 # the nine implementation checks
python -m harness.phase0 --model screen_d256_l8 --include-cost  # + the cost measurement
```

---

## 5. External review — findings and resolutions

An independent adversarial review (Codex CLI, `gpt-6-luna`, whole repository at `91bff96`)
asked to prioritise anything that could make a reported number wrong or a claim unsupported.
Its verdict was **"not fit to produce the confirmatory result as it stands"**, and it was
right. Its positive findings are worth as much as its criticisms, since they are things it
actively tried to break.

**Findings, and what happened to each:**

1. **Resume identity — major, fixed.** Resume checked only `total_steps`, so a checkpoint from
   a *different* configuration with the same step count was accepted, and the resumed run was
   then reported under the newly-supplied config's identity. `run_id` cannot cover this: it
   identifies stage, arm, sigma, seed and model width/depth only — so 80 × accum 4 and
   40 × accum 8 (both 1526 steps, both 163,840 tokens/step) share a run id, as do bf16 and
   fp32. Now `trajectory_mismatches()` compares every trajectory-defining field individually
   plus the recorded data hashes, and identity is verified **before** any state is loaded so a
   refusal cannot leave a half-loaded model. Five tests, including that exact batch-shape trap.
2. **Tag vs HEAD, and unfilled SAP fields — major, fixed.** The tag pointed at a commit that
   predates the runner, and `docs/03` still had `<value>` placeholders in its Locked
   Configuration Fields — a normative document that does not state its values pre-registers
   nothing. Every SAP field is now filled with the real hashes and settings; **σ0 alone stays
   open**, because it is the screen's output and cannot be known in advance without peeking.
   A successor tag will name the commit the confirmatory runs actually execute.
3. **Sampler checkpoint validation — minor, fixed.** The permutation's *length* was checked but
   not that it is a permutation, and state was assigned before validating. A same-length order
   that repeated blocks while omitting others loaded happily — a resume that appears to work
   while seeing different data. Now index range, uniqueness and position are validated first,
   then assigned; two tests, one specific to the half-apply case.
4. **"Ten checks" wording — minor, fixed.** `harness/phase0.py` implements nine correctness
   checks; the cost measurement is opt-in via `--include-cost`.

**Attacked and found sound** (each of these was explicitly probed): the two-stage
`chunked_lm_loss` gradient hand-off; the cosine schedule hitting exactly σ = 0; eligible-module
selection excluding embeddings, head, norms and biases; pairing by construction with a dedicated
perturbation generator that does not consume the global RNG stream; √(3/8) as the continuous
variance-matching factor; and the r = √(1 + q² − 2qρ) conversion with its worst corner.

**Not fixed, stated rather than hidden:** the marker-aligned tokenization equivalence holds as
evidence *recorded in the manifest* (exact marker invariant, the chunking rule, and the token
and raw hashes) rather than as a test a reader can re-run — the raw 1.9 GB text was deleted
after hashing to recover disk. The reproduction is available to anyone from the pinned Hub
revision; what cannot be re-run from this repository alone is the whole-file comparison itself.
