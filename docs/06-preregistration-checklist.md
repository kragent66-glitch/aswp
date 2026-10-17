# ASWP — Preregistration Checklist

**Purpose.** This is the gate for confirmatory training. The protocol states that
every bracketed field in the SAP is completed and committed *before* confirmatory
data collection, and that later changes are protocol deviations rather than silent
edits. This document is the operational form of that rule: it lists what is frozen,
what is open, and — for each open item — whether it is blocked on a **decision**
(protocol owner) or on **work** (the harness, Phase 0, Phase 1).

**Status: decision items 1–6 and 9 are resolved and committed; the harness exists and the
Phase 0 gate passes on CPU for the screen model. Two things remain before Phase 2: the GPU
half of Phase 0 (mixed precision at scale + the item-10 cost measurement), and the protocol
release tag (item 8).**

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

8. **Signatures, commit hash, and archive tag** — the open-science plan asks for a
   pre-results protocol release tag before confirmed experimentation begins.
9. ~~Pin the tokenizer artifacts by hash.~~ **Done.** The tokenizer is pinned by
   artifact hash at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, and the
   provenance block is identical across the training manifest and both endpoint
   sidecars because all three are written by one shared helper. Re-freezing both
   validation streams reproduced their stream ids and token-stream hashes exactly,
   which is the check that the schema change did not disturb the data.
10. **Phase 0, GPU half.** The nine correctness checks pass on CPU for the screen
    model. Still to measure on GPU at the real batch shape: bf16 behaviour at scale,
    peak memory, and the item-10 cost segment. **This is what fixes n**: the design
    of record is 22 runs (4-run screen + n = 6 × 3 arms), which fits the $22.50
    usable budget only above ~50k tokens/s on a T4. Predeclared rule: ≤ $0.82/run →
    n = 8 × 3 arms; $0.82–0.94 → n = 6; > $0.94 → the design does not fit the
    reserve and the protocol owner decides.

## 3. Order of operations

```
harness written  →  Phase 0 gate passes  [DONE on CPU: 9/9 for d256/L8]
                 →  decision items 1–4 locked, items 5–6 hashed and committed  [DONE]
                 →  tokenizer artifacts hashed (item 9)  [DONE]
                 →  GPU half of Phase 0 + cost segment (item 10)  →  n fixed
                 →  protocol release tagged (item 8)
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
python -m harness.phase0 --model screen_d256_l8 # the ten implementation checks
```
