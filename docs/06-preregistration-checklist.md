# ASWP — Preregistration Checklist

**Purpose.** This is the gate for confirmatory training. The protocol states that
every bracketed field in the SAP is completed and committed *before* confirmatory
data collection, and that later changes are protocol deviations rather than silent
edits. This document is the operational form of that rule: it lists what is frozen,
what is open, and — for each open item — whether it is blocked on a **decision**
(protocol owner) or on **work** (the harness, Phase 0, Phase 1).

**Status: NOT READY to run confirmatory training.** Two open items are blocked on
decisions, five on the harness/Phase 0, and one is a Phase 1 output by design.

---

## 1. Frozen now

| Item | Value | Source |
|---|---|---|
| Confirmatory hypothesis | one-sided (H1: dᵢ > 0), α = 0.05 | SAP §Hypotheses; 05 §P1 |
| Minimum / target paired seeds | n_min = 6, n_target = 8 | SAP §Sample Size; 05 §P1 |
| Paired seed list | **8 integers, frozen** — `1857649130, 1434172854, 1387619681, 536290471, 1202662852, 1106087492, 1881574181, 1859844390` | `config/manifest.json` (issue #10) |
| Per-role RNG streams | `global`, `init`, `data-loader`, `dropout`, `perturbation` — derived per paired seed; perturbations use a dedicated `torch.Generator` | `config/manifest.json`; protocol §Reproducibility |
| Seed derivation | `sha256("ASWP/v1/<role>/<n>")`, first 8 bytes big-endian, mod 2³¹ — recomputable by anyone, re-checked by the test suite | `analysis/seed_manifest.py` |
| Practical threshold δ | 1.0 × SD_baseline, plus its SD_delta equivalent 1.0/r | SAP §Practical Significance; 05 §P5 |
| Unit convention | r = √(1 + q² − 2qρ); predeclared tolerance box q ≤ 1.25, ρ ≥ 0; worst corner r = 1.6008 | SAP §Endpoint; 05 §P5 |
| Interval convention | one-sided 95% randomization bound, inverted from the same one-sided test | SAP §Primary Analysis; 05 §P6 |
| Decision rule | point estimate > δ **and** one-sided 95% bound excludes 0 | SAP §Decision Language |
| Primary test | exact sign-flip, all 2ⁿ patterns enumerated | SAP §Primary Analysis; 05 §P8 |
| Selection rule | ≤ 1 instability event, throughput ≥ 80% of baseline, next-best fallback | SAP §Multiplicity; 05 §P3 |
| Selection-stability check | chosen + runner-up σ0 on 2 fresh seeds; any reversal → exploratory | 05 §P3 |
| Margin pre-filter | top-two gap over all candidates, m = 0.5·σ_screen, pre-filter only | 05 §P3, §P5 |
| C4 arms | 0.6124·σ0 (variance-matched) and σ0 (dose extreme) | 05 §P2 |
| C5/C5b tuning | same screen rule and budget as treatment (symmetric effort) | 05 §P2 |
| Instability trigger | ≥ 30% of pairs unstable → "instability-dominated negative" | 05 §P4 |
| Eligible modules | attention q/k/v/o + MLP up/gate/down; no embeddings, head, biases, norms, tied weights | protocol §Primary Design Choices |
| Noise spec | Gaussian, RMS-relative scaling, cosine schedule reaching exactly 0 at the final step, one perturbation per matrix per optimizer update shared across microbatches | protocol §Method Definition, §Schedules |
| Precision policy | noise and perturbation constructed in fp32; realized perturbation measured after any bf16 cast | protocol §Primary Design Choices |
| Analysis reproducibility | seeds 42 / 7 / 20260905; 40 000 trials per cell; 2 000 000 screen draws; 256 MiB memory budget | 05 §P7, §P8 |

## 2. Open — and what each is waiting on

**Blocked on a decision (cannot be automated; needs the protocol owner):**

1. **Tokenizer identity/version/hash.** The protocol never names one. GPT-2 BPE
   gives comparability with prior TinyStories work and a vocabulary of 50 257; a
   TinyStories-trained tokenizer gives better tokens-per-parameter at this scale
   but makes the transfer phase (FineWeb-Edu) harder to interpret. This choice
   propagates into every downstream hash, so it is first in the queue.
2. **Total optimizer steps and token budget.** Sets run cost, wall-clock, and the
   MDE. The power tables in 05 §P1 assume nothing about it, but Phase 0's measured
   throughput is what turns the budget into a cost per run.
3. **Primary checkpoint step** (the fixed endpoint at which validation NLL is read).
4. **Optimizer configuration**: LR schedule, warmup, decay, clipping threshold,
   dropout, batch/sequence configuration, weight decay. Defaults should be stated
   as predeclared values, not tuned per arm.

**Blocked on work (the harness, which does not exist yet):**

5. **Dataset revision + SHA-256**, and **validation split + token-stream hash.**
   Mechanical once the tokenizer is chosen and the preprocessing code exists —
   the protocol requires the packed-sequence behaviour and post-tokenization hash
   to be frozen too.
6. **Model source commit + architecture.** Frozen when the model definition is
   written.

**Blocked by design (produced by the runs themselves):**

7. **Selected σ0 and its exploratory parent run ID.** By protocol this comes out of
   the Phase 1 screen. Its *rule* is already locked; only the outcome is pending.
   Phase 0 correct the ordering: it runs before the screen, so nothing here is
   being deferred past the point of usefulness.

**Process step before run 1:**

8. **Signatures, commit hash, and archive tag** — the open-science plan asks for a
   pre-results protocol release tag before confirmed experimentation begins.

## 3. Order of operations

```
harness written  →  Phase 0 gate passes (σ=0 equivalence, deterministic eval,
                    checkpoint/resume, throughput + memory measured)
                 →  decision items 1–4 locked, items 5–6 hashed and committed
                 →  protocol release tagged (item 8)
                 →  Phase 1 screen (C0 + C1–C3, 10M)  →  item 7 locked from its result
                 →  Phase 2 confirmation (paired seeds from config/manifest.json)
```

Nothing in Phase 2 may begin until items 1–6 and 8 are committed. That ordering is
the whole point of the preregistration and is why this document exists separately
from the SAP: the SAP states the rules, this states which rules still lack values.

## 4. What is already executable

The analysis is not blocked by any of the above — it runs today:

```bash
python -m pip install numpy scipy pytest
python -m pytest                      # 64 tests, including the seed-manifest checks
python analysis/seed_manifest.py      # verify the frozen seeds against their derivation
python analysis/p1_exact_power.py     # resolution, power, δ unit conversion
python analysis/p23_match_curse.py    # variance matching, winner's curse, margin rule
python analysis/p4_delta_criteria.py  # decision-rule fire rates, type-I error, δ under units
```
