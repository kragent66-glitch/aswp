# ASWP — Design Review (Red-Team)

**Part of the open experimental record.** Adversarial review of protocol v1.0, published before implementation. Findings are tracked as GitHub issues.

**Document:** Annealed Stochastic Weight Perturbation for Transformer Training
**Reviewed set:** Research Protocol v1.0 · Implementation & Operations Manual v1.0 · Statistical Analysis Plan / Preregistration v1.0
**Reviewer:** adversarial review (Hermes agent) · **Date:** 2026-08-19 · **Review version:** 1.0

---

## 0. Bottom Line

This is an unusually well-built preregistration — locked endpoint, paired seeds, exact test, controls for generic-stochastic-regularization, run ledger, and honest non-novelty positioning. Better than most lab protocols I've red-teamed.

Two **structural flaws must be fixed before the SAP is locked**, one design arm is **not a valid control as specified**, and there is one **unaddressed selection-bias channel** that will quietly inflate the confirmatory estimate. Everything else is tightening, but the tightening list is long enough that you should do it in one pass while the docs are still unfrozen.

---

## 1. Statistical Red-Team

### 1.1 BLOCKER — n=5 × two-sided exact test can never reach α = 0.05

The sign-flip/randomization test enumerates 2ⁿ sign assignments. The smallest achievable **two-sided** p-value is 2/2ⁿ:

| n (pairs) | min two-sided p | min one-sided p | comment |
|---|---|---|---|
| 5 | **0.0625** | 0.0313 | two-sided α=0.05 unreachable |
| 6 | 0.0313 | 0.0156 | two-sided OK |
| 7 | 0.0156 | 0.0078 | comfortable |
| 8 | 0.0078 | 0.0039 | |
| 9 | 0.0039 | 0.0020 | |
| 10 | 0.0020 | 0.0010 | |

The SAP currently says *"The primary test is two-sided unless a one-sided direction and justification are explicitly locked before training."* With the stated minimum of **5 pairs**, a two-sided test at α=0.05 is mathematically incapable of significance — the study would be pre-committed to a null result at its own minimum sample size, which is the worst possible outcome for a compute-limited project (you spend the budget and cannot win).

**Fix (pick one, lock it now):**
- Lock a **one-sided test** with the justification written into the SAP ("only improvement is of scientific/practical interest; a worse-than-baseline result is reported as an estimate regardless"), **or**
- Raise the minimum to **6 paired seeds** (two-sided min p = 0.0313).

Recommendation: do both — one-sided pre-specified (it genuinely is one-sided: the decision ladder in §"Decision Language" has no use for "significantly worse") **and** target 8–10 pairs as the SAP already prefers. At n=8 the exact test has real resolution.

### 1.2 Winner's curse in the screen → confirm handoff (unaddressed)

C1–C3 each run one exploratory seed; the fixed rule selects the best σ0; that σ0 then gets a fresh confirmatory estimate. Selecting the maximum of three correlated screen metrics **inflates the expected confirmatory effect**, and the inflation is not captured by the confirmatory CI. The screen's one-seed-per-condition design makes this worse: σ0 selection noise is on the order of the screen's seed-to-seed σ, so the argmax is partly luck.

**Cheap, pre-registered mitigations (pick at least one):**
- **Selection-stability check:** in Phase 2, run the chosen σ0 *and the runner-up σ0* on 2 fresh seeds each. If the ranking flips, the confirmatory claim is downgraded to exploratory. Cost: 2 extra 10M runs. This is the highest value-per-dollar fix in the entire protocol.
- **Margin rule:** pre-register "chosen σ0 must beat the runner-up by > m NLL" with m set from Phase 0 baseline calibration; if the margin isn't met, declare the screen inconclusive and report C1–C3 as exploratory only.
- At minimum, disclose: the confirmatory estimate is conditional on screen selection and is biased upward; report the raw screen table so readers can see the margin.

### 1.3 Bootstrap CI at n=5 is quantized and miscalibrated — use the randomization CI

With n paired differences, the percentile bootstrap resamples 5 values with replacement: there are only **126 distinct possible resampled means** (compositions of 5 into 5 counts), so CI endpoints are quantized to a coarse grid, and percentile-bootstrap coverage error at n=5 is large (the method's accuracy is O(1/√n) and worse in the tails).

**Fix:** invert the sign-flip test you're already computing — the randomization CI (set of δ such that the test of d−δ fails to reject) is exact under precisely the exchangeability assumption the primary test already makes, costs zero extra machinery, and is the natural companion. Report the percentile bootstrap only as a descriptive sensitivity, or drop it. Also report each dᵢ and the per-pair plot (already in the SAP — good).

### 1.4 δ and "compatible uncertainty" must be operationalized — they currently are not

The SAP requires δ to be set blind to treatment results, but never says **how** δ is set, and the decision ladder's "compatible uncertainty" is a hand-wave that will be argued about after results exist. That is a researcher-degrees-of-freedom hole.

**Fix — lock the exact rule before training:**
- δ = k × SD of baseline NLL across the paired seeds (measured from the Phase 0/1 baseline calibration pilot), with k ∈ [1, 2] pre-chosen. This ties "practical" to measured run-to-run noise, which is exactly what the estimand needs.
- "Compatible uncertainty" = **95% CI lower bound > δ** (strong) or **point estimate > δ and CI excludes 0** (weak). Pick one in the SAP. Recommend the strong version; with n=5 the weak version will trigger on noise alone.

### 1.5 Instability is informative censoring — it needs a primary-analysis rule

The SAP correctly says instability is an experimental outcome, retained and reported. But the primary estimand has no rule for what happens when treatment pairs go unstable: if 2 of 5 treatment runs NaN out, the confirmatory analysis silently runs on 3 pairs — power collapses *and* the surviving pairs are a biased subsample (the most stable noise realizations). The decision ladder's "instability dominates" category exists, but nothing triggers it quantitatively.

**Fix:** pre-register: *if ≥ X% (e.g., 30%) of treatment pairs fail stability criteria, the primary conclusion is "instability-dominated negative" regardless of the surviving pairs' d̄.* And state whether unstable pairs' dᵢ enter as missing or as the worst observed value (recommend: report both).

### 1.6 Paired t-test as sensitivity — fine, but it's an estimate, not a rescue

At n=5–8 the t-test has negligible power to reject and its normality assumption is untestable; the SAP already relegates it to sensitivity. Don't let it be used later to claim "significant by t-test" — its only job is a sanity check against the exact test. Consider adding a **Wilcoxon signed-rank** as a second sensitivity: it's the other standard small-n pairing test and costs nothing.

### 1.7 Set expectations: this design can only confirm large effects

With 5–10 pairs and exact testing, the study is powered for effects that are visible as a consistent sign across nearly all pairs — i.e., effects larger than ~1 baseline-SD. That is a feature (you avoid chasing noise), but the confirmatory write-up must say so. The pre-registration already leans "estimate with CI" rather than "significant or bust" — good; keep that posture in the final paper's framing.

---

## 2. Experimental-Design Red-Team

### 2.1 BLOCKER-ish — C4 constant noise is NOT a matched-exposure control

C4 holds σ = σ0 (the screen winner) constant for all of training. The annealed arm's total noise dose is far smaller: with f(p) = 0.5(1+cos πp), ∫₀¹ f(p)² dp = **3/8**, so the annealed arm's integrated variance equals a constant arm running at **σ_const = σ0·√(3/8) ≈ 0.61·σ0**. The specified C4 therefore delivers ~2.7× the noise variance of the treatment. Constant noise at 1.0·σ0 late in training is known-bad (that's why annealing exists), so "annealing beats constant noise" would be a **trivially true comparison** — it would not support the annealing-specific claim the protocol says it supports.

**Fix:** run C4 at **both** levels: σ0 (as specified, the "too much noise" extreme) and **0.61·σ0 (variance-matched)**. The variance-matched arm is the one that actually separates "the schedule shape matters" from "any noise at this dose is equally good." Cost: one extra run per seed. This is the single most important design change after §1.1.

### 2.2 Missing control: vanilla dropout

C5 is DropConnect (multiplicative weight noise) — the closest relative. But the standard, cheapest, best-understood regularizer is **dropout**, and the practical claim readers will ask is "does this beat just cranking dropout?" Add C5b: activation dropout at variance-matched strength (for a linear layer with unit-ish activations, dropout p and weight noise σ relate roughly via var matching; or just tune it with the same fixed budget as C5). If ASWP can't beat a tuned dropout at equal cost, the paper's value collapses to "a more complicated dropout" — better to know that inside the protocol than outside.

### 2.3 Perturbed parameter mass is small and unquantified

Eligible modules: q/k/v/o + MLP up/gate/down. Excluded: embeddings, **output head (tied)** , biases, norms. In a 10M decoder-only TinyStories model, the tied embedding/LM-head is typically **20–40% of parameter mass** (vocab × d_model, e.g. 4096×384 ≈ 1.6M of ~10M). The intervention therefore acts on only ~60–80% of the weights — and possibly the least critical share (heads are where much of the memorization lives). This is not wrong, but:
- **Log the fraction of perturbed parameter mass per run** (one line in the long-format layer diagnostics).
- Add a Phase 3 ablation: **untied output-head noise** — if the effect is sharpness-driven, the head's curvature is where you'd expect a large signal.

### 2.4 Noise peaks exactly when the model is least able to absorb it

Cosine from step 0 means maximum σ during warmup, when gradients are largest relative to weights and the model is leaving initialization. The noise may simply be re-seeding the init trajectory rather than shaping the converged basin. Add a Phase 3 ablation with **delayed onset** (noise ramps in after warmup, peaks mid-training): schedule shape is already an ablation axis — make "onset delay" an explicit one.

### 2.5 LR × noise endpoint confound — lock the LR tail

Both σ(t) → 0 and (if cosine LR) LR(t) → 0 at the endpoint. The claim "annealing lets the optimizer converge cleanly" is only meaningful relative to the update scale: what matters at the end is the **noise-to-update ratio**, not σ itself. If the LR schedule has a flat tail (constant LR after decay), annealed noise genuinely vanishes relative to updates — clean. If LR → 0 too, the ratio is ambiguous and the C7 (LR-coupled) arm is your only disentangler. **Lock the LR schedule's tail behavior in the SAP config block** and log noise-RMS / update-RMS per step (the logging schema already has both — add the ratio as a derived column).

### 2.6 DropConnect comparator tuning budget is unspecified

"A tuned multiplicative weight-noise comparator" — tuned on what, with how many runs, selected by which rule? If C5 gets 10 tuning runs and ASWP gets a frozen screen, the comparator is advantaged. **Lock C5's tuning budget and selection rule in the SAP** (recommend: same 3-point screen rule, same fixed token budget, selection by the same written rule — symmetry of effort is what makes the comparator honest).

### 2.7 The screen selection rule has a wiggle clause

"…subject to stability and throughput requirements" — unquantified. A post-hoc judgment call at the moment of selection is exactly the degree of freedom the preregistration exists to remove. **Pre-write the numbers:** e.g., "chosen σ0 must complete without > 1 instability event, with throughput ≥ 80% of baseline; otherwise the next-best σ0 is considered, in order." Then the rule is executable by a script.

### 2.8 Distributed semantics must be locked, not documented

The manual says rank-local vs synchronized noise "must be deliberate and documented." It's more than a documentation issue: with DDP, rank-local noise means each rank's forward uses a different ε, and the all-reduced gradient is an average over R noise draws — that *changes the estimator* (it becomes lower-variance noise than the single-draw method; effectively a different method). **Decide now:** single-GPU for all confirmatory runs (cleanest — 10M–50M fits easily) and treat DDP as a Phase 4 engineering question. If DDP is used for the 30–50M transfer runs, note in the paper that noise variance is effectively divided by √R.

### 2.9 Unit test #2 verifies nothing (spec contradiction)

The non-negotiables say *"Disable perturbation unconditionally in eval mode."* Unit test #2 says *"Evaluation determinism: repeated eval-mode calls yield identical outputs with enabled=true and nonzero configured sigma."* Since eval mode disables perturbation, the test passes trivially and asserts nothing about determinism under an active perturbation. **Fix the test:** assert (a) eval-mode outputs are byte-identical to a σ=0 run (i.e., perturbation provably off), and (b) *training-mode* determinism: two training forwards with the same generator state and same σ produce identical logits (this is the property that actually matters for resume reproducibility).

### 2.10 The smoothed-objective story is exact *only because of the `.detach()`* — say so

E[∇L(W+ε)] = ∇E[L(W+ε)] holds for fixed σ — the reference module's `.detach()` on the RMS scale is what makes the Gaussian-smoothing interpretation literally correct rather than approximate. This is a quiet strength; most implementations would accidentally include ∂RMS/∂W terms. Two honest caveats to put in the paper:
- The smoothing is **state-dependent** (scale ∝ RMS(W)), so the "minimizing a smoothed objective" framing is exact *per step*, not globally — the objective itself drifts as weights grow.
- The second-order term is ½σ²·RMS(W)²·tr(H): the curvature pressure is **weighted by per-layer RMS²**, so large-weight layers get pushed toward flatness more than small-weight layers. That's a mechanism claim worth pre-registering as exploratory (it predicts *which layers* should show sensitivity changes post hoc).

### 2.11 Phase 0's 1M–3M pilot is mostly redundant

The pilot's real job is the σ=0 equivalence gate and overhead measurement — those don't need a distinct model scale, and Phase 1's 10M runs would catch any scale-dependent bug anyway. Keep the pilot only if it's cheap (it is) — but don't spend a 1M–3M hyperparameter hunt on it; run the Phase 1 config at 1M–3M, verify equivalence, and go.

### 2.12 Feasibility / token budget (worth a Phase 0 line item)

Rough math for the 10M screen (AdamW, seq 512, no activation checkpointing): ~6·N·D FLOPs/run. At D = 10⁸ tokens → 6×10¹⁵ FLOPs ≈ 2–6 h on a 16-core CPU (300 GFLOP/s optimistic), ~15–30 min on a 4090-class GPU. So Phase 1's "fixed token budget" should be **pinned to a number now** (recommend 50–150M tokens for the 10M screen; enough for TinyStories to show separation, cheap enough to run C0 + C1–C3 + C4×2 in ~a day of GPU or a weekend of CPU). The 30–50M verification phase at the same token count is ~10× the FLOPs — plan the budget before Phase 0 so "fixed token budget" isn't negotiated later. Also: report time-to-threshold on a **tokens-seen** basis, not wall time — throughput overhead is a treatment effect on cost, and wall time confounds it with hardware noise.

---

## 3. Threats-to-Validity Register

| Category | Threat | Severity | Mitigation (status) |
|---|---|---|---|
| Conclusion | Exact test resolution at n=5 | **High** | §1.1 — lock one-sided or n≥6 (not yet fixed) |
| Conclusion | Winner's curse from σ0 screen | **High** | §1.2 — stability check / margin rule (not in protocol) |
| Conclusion | Percentile bootstrap miscalibrated at n=5 | Medium | §1.3 — randomization CI (not in protocol) |
| Conclusion | Undefined δ rule & "compatible uncertainty" | **High** | §1.4 — k·SD rule + CI-lower-bound criterion (not in protocol) |
| Conclusion | Instability censoring biases surviving pairs | Medium | §1.5 — instability-triggered conclusion rule (not in protocol) |
| Internal | C4 not variance-matched → trivial annealing win | **High** | §2.1 — add 0.61·σ0 arm (not in protocol) |
| Internal | DropConnect tuning asymmetry | Medium | §2.6 — symmetric screen budget (not in protocol) |
| Internal | Screen selection wiggle clause | Medium | §2.7 — numeric thresholds pre-written (not in protocol) |
| Internal | RNG coupling across arms | Low | Manual §Non-Negotiables — dedicated generator (handled ✓) |
| Internal | Noise re-sampled on grad-checkpoint recompute | Low | Manual §Common Bugs — flagged (handled ✓) |
| Internal | Eval nondeterminism | Low | Eval disabled in eval mode (handled ✓; test #2 mis-specified, §2.9) |
| Construct | Val-NLL on synthetic child stories is a narrow construct | Medium | Acknowledged in §Known Limitations (handled ✓) |
| Construct | Perturbed parameter mass < total mass | Medium | §2.3 — log fraction; head-noise ablation (not in protocol) |
| External | TinyStories → FineWeb-Edu covers one transfer axis | Medium | Acknowledged (handled ✓); note: no scale >50M test |
| External | Decoder-only LM only — no encoder/vision/RL claim | Low | Keep claims scoped (handled ✓) |
| Measurement | bf16 cast erodes realized σ | Low | Realized-perturbation logging (handled ✓) |
| Measurement | Wall-time comparisons confounded by overhead | Low | §2.12 — tokens-seen basis (not in protocol) |

---

## 4. Lock-List Before Launch (SAP config block)

Everything below must be a concrete value in the SAP before any confirmatory run; today every one is a placeholder or absent:

1. **Hypothesis direction:** one-sided (recommended) or n ≥ 6 minimum (required — §1.1).
2. **Seed list:** ≥ 6 integers, generated once, written into the SAP, never regenerated.
3. **δ rule:** δ = k·SD_baseline with k pre-chosen; "compatible uncertainty" = 95% CI lower bound > δ.
4. **Dataset revision, tokenizer version, validation stream hash** (the template already requires — fill from Phase 0).
5. **Total tokens & endpoint step** for screen AND confirmation AND transfer phases (pin numbers, §2.12).
6. **LR schedule incl. tail behavior** (flat tail recommended; §2.5).
7. **Optimizer settings:** AdamW betas, ε, weight decay, warmup fraction, clip norm — one block.
8. **Dropout rate and module coverage** (needed for the C5b comparator).
9. **C4 arms:** σ0 and 0.61·σ0 variance-matched (required — §2.1).
10. **C5 tuning budget** and selection rule (required — §2.6).
11. **Screen selection rule with numeric thresholds** (required — §2.7).
12. **Selection-stability check:** chosen σ0 + runner-up on 2 fresh seeds (recommended — §1.2).
13. **Instability trigger:** ≥ X% of pairs unstable → "instability-dominated negative" (required — §1.5).
14. **Primary CI:** randomization CI (inverted sign-flip); percentile bootstrap as sensitivity only.
15. **Eval cadence** for trajectory metrics and the tokens-seen basis for time-to-threshold.

---

## 5. Document-Level Amendments

**SAP (highest priority):**
- §Hypotheses: replace the two-sided default with a locked one-sided test + justification, and raise the n minimum to 6.
- §Primary Analysis: replace/append the percentile bootstrap with the randomization CI; state the 126-composition quantization caveat.
- §Practical Significance: insert the k·SD rule and the CI-lower-bound criterion.
- §Failures and Missingness: add the instability-triggered conclusion rule and the surviving-pairs bias note.
- New §Selection: winner's-curse disclosure + stability-check requirement (or margin rule).

**Research Protocol:**
- C4: add the variance-matched constant arm (0.61·σ0); restate the annealing-specific claim to require the matched comparison.
- Add C5b dropout comparator.
- Phase 3: add "noise onset delay" to schedule-shape ablations; add untied-head-noise ablation; add parameter-mass fraction to logging.
- Phase 1: state the numeric selection-rule thresholds; state the screen token budget.
- Statistical Plan: add the sign-flip resolution table (n=5 min two-sided p = 0.0625) so the n choice is visibly motivated.

**Operations Manual:**
- Fix unit test #2 (assert eval provably off; add training-mode determinism under fixed generator state).
- Add noise-RMS / update-RMS ratio to the step log (derived column).
- Add parameter-mass fraction to the layer diagnostics file.
- Note: DDP rank-local noise changes the estimator (√R variance reduction) — decide single-GPU for confirmatory phases.

---

## 6. Quick Wins — Top 10 in Order

1. **Lock one-sided hypothesis (or n ≥ 6)** — unresolvable at n=5, costs nothing.
2. **Add the 0.61·σ0 variance-matched constant arm** — makes the annealing-specific claim honest.
3. **Add the selection-stability check** (chosen + runner-up σ0 on 2 fresh seeds) — kills the winner's curse cheaply.
4. **Replace percentile bootstrap with the randomization CI** — exact, free, consistent with the primary test.
5. **Write the δ rule as k·SD and "compatible uncertainty" as CI-lower > δ** — removes the post-hoc wiggle.
6. **Add the instability-triggered conclusion rule** — prevents silent power collapse.
7. **Add the dropout comparator** — the standard-control gap in the arms.
8. **Quantify the screen rule** ("stability and throughput requirements" → numbers).
9. **Fix unit test #2** — as written it asserts nothing.
10. **Pin token budgets + LR tail now** — prevents budget negotiation after results exist.
