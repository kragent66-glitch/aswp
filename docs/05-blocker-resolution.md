# ASWP — Blocker Resolution

**Session 2026-09-05** resolves GitHub issues #1, #2, #3 (and the δ linkage in #4).
**Session 2026-10-17** resolves issues #12, #13, #14, #15, #16 and regenerates every
value in this document from the fixed scripts.

Every value below is computed, not asserted. Scripts: `analysis/p1_exact_power.py`,
`analysis/p23_match_curse.py`, `analysis/p4_delta_criteria.py`, sharing
`analysis/aswp_core.py`. Tests: `tests/test_aswp_core.py` (`python3 -m pytest`).

| # | Issue | Resolution | Locked value(s) |
|---|---|---|---|
| 1 | n=5 two-sided can't reach α=0.05 | Lock one-sided test **and** raise minimum to n=6; target n=8 | one-sided, α=0.05, n_min=6, n_target=8; seed list written once |
| 2 | C4 not variance-matched | Run C4 at **both** σ0 and 0.6124·σ0 | σ_c = σ0·√(3/8) = **0.6124·σ0** (cosine); dose ratio 2.667× |
| 3 | Winner's curse in screen→confirm | Pre-register selection-stability check (chosen + runner-up, 2 fresh seeds) | flip on any ranking reversal; margin rule m = 0.5·SD_baseline as pre-filter |
| 4* | δ undefined / unconfirmable | Power-consistent rule: **weak criterion, k=1** primary; **strong criterion, k=0.5** sensitivity | δ = 1.0·SD_baseline (point > δ AND one-sided 95% randomization bound > 0) |
| 12 | Margin was `abs(z[:,0] - z[:,1])` | Margin = top-two gap over all candidates (`winner_margin`), permutation-invariant, tested | null margin median **0.691σ** (was 0.955σ) — see §P5 |
| 13 | δ in sd_baseline vs power in sd_delta | Both SDs named; ratio r = √(1+q²−2qρ) with the arm ratio q = sd_treatment/sd_baseline carried explicitly; power recomputed at the actual δ | predeclared box q ≤ 1.25, ρ ≥ 0 → r ≤ **1.6008**; δ = 1.0·sd_baseline = **1.0/r** sd_delta — see §P5 |
| 14 | One-sided test vs two-sided CI | Interval is the **one-sided 95% randomization bound** inverted from the same one-sided test; a two-sided interval (when reported) inverts **both** tails | "interval excludes 0" ≡ "test rejects" (**exact**, not ±1%); type-I ≤ α — see §P6 |
| 15 | p4 unseeded SciPy sampling | Seeded `numpy` Generator (`SEED_DELTA = 20260905`); seed, trials and assumptions recorded and printed | 40,000 trials/cell; reproducibility test in suite — see §P7 |
| 16 | p1 peak ~3.3 GB | Batched enumeration to a declared memory budget, arithmetically identical output | 256 MiB statistic block; 3.4 GiB → 0.6 GiB, 130 s → 4.9 s — see §P8 |
| 10 | Seed list not frozen | Paired + per-role seeds derived from a published rule and frozen; re-checked by the suite | 8 paired seeds in `config/manifest.json`; derivation `sha256("ASWP/v1/<role>/<n>")` — see §P9 |

---

## P1. Exact-test resolution and power (issue #1) — SOLVED

**Minimum achievable p** (sign-flip test enumerates 2ⁿ assignments):

| n | 2ⁿ | min two-sided p | min one-sided p |
|---|---|---|---|
| 5 | 32 | **0.0625** | 0.0313 |
| 6 | 64 | 0.0313 | 0.0156 |
| 8 | 256 | 0.0078 | 0.0039 |
| 10 | 1024 | 0.0020 | 0.0010 |

**Power of the exact one-sided test at α=0.05** (40k trials, seed 42). Effects are in
**sd_delta** units — the SD of the paired differences, which is the unit the test
statistic lives in. (The 2026-09-05 revision printed the same numbers labelled
"baseline-SD units"; see §P5 for why that label was wrong and what it costs.)

| n | eff=0.25 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|
| 5 | 0.076 | 0.156 | 0.279 | 0.421 | 0.712 | 0.893 |
| 6 | 0.125 | 0.265 | 0.450 | 0.647 | 0.910 | 0.989 |
| 8 | 0.154 | 0.340 | 0.589 | 0.802 | 0.982 | 0.999 |
| 10 | 0.182 | 0.424 | 0.704 | 0.897 | 0.996 | 1.000 |

**Numbers unchanged by the §P8 rewrite**: the batched path is arithmetically identical
to the previous broadcast implementation (asserted bit-for-bit in the test suite), so
this table is the *same* computation under the *same* seed, in 0.6 GiB instead of 3.4.

**Two-sided α=0.05 is literally impossible at n=5** (power = 0.000 → the study
cannot win at its own minimum). At n=6 two-sided reaches 80% only at 2.0σ.

**Locked:**
- **Hypothesis: one-sided (H1: dᵢ > 0).** Justification for the SAP: the decision
  ladder has no use for "significantly worse than baseline" — a worse result is
  reported as an estimate regardless. Written once, never revisited.
- **Minimum n = 6** (two-sided α=0.05 becomes *possible* as a fallback), **target
  n = 8**: 80% power at 1.0·sd_delta, 98% at 1.5·sd_delta. n=8 also restores
  two-sided capability (65.6% at 1.0σ) if a reviewer demands it.
- **80%-power honesty line:** the design confirms effects ≥ 1.0·sd_delta (n=8).
  Effects below that are reported as point estimates with CI, per the existing posture.
- Seed list: generate 8 integers once, write into the SAP, never regenerate.

---

## P2. Variance-matched C4 (issue #2) — SOLVED

For f(p) = 0.5(1+cos πp): **∫₀¹ f(p)² dp = 3/8** (verified analytically and by
discrete 2000-step integration = 0.37525). Therefore:

- **Matched constant level: σ_c = σ0·√(3/8) = 0.6124·σ0** (≈ 0.61·σ0 — the
  review's figure, now exact).
- **Constant-at-σ0 delivers 8/3 = 2.667× the integrated noise variance** of the
  annealed arm — "annealing beats constant noise" against that arm is trivially
  true, which is why the matched arm exists.
- Linear schedule (secondary ablation), f(p)=1−p: ∫ = 1/3 → **σ_c = 0.5774·σ0**,
  ratio 3.000×. Use the same formula for any schedule: σ_c = σ0·√(∫f²).

**Locked for C4 (Phase 2, per seed):**
- **C4a: σ_c = 0.6124·σ0** — the variance-matched arm. *This* is the
  annealing-specific comparison.
- **C4b: σ0** — kept as the "too much noise" extreme (documents why annealing
  exists, harmless to include).

**C5/C5b calibration — a new, honest insight from the session:**
Per-weight variance matching gives σ0 = √(r(1−r)) for DropConnect retention r:

| drop rate | 0.1% | 0.5% | 1% | 2% | 5% | 10% | 20% |
|---|---|---|---|---|---|---|---|
| ⇔ σ0 | 0.032 | 0.071 | 0.100 | 0.140 | 0.218 | 0.300 | 0.400 |

The treatment (σ0 = 0.01–0.05) is a **whisper**; standard-strength dropout
(10–20%) is 100–10,000× its per-weight variance. So:
- C5/C5b must be **tuned with the same screen budget and rule as the treatment**
  (symmetric effort), NOT forced to naive per-weight matching — matched-strength
  dropout (<1% drop) is not a meaningful comparator regime.
- The paper must state the dose asymmetry explicitly; "beats tuned dropout at
  equal cost" is the honest, stronger claim.

---

## P3. Winner's curse (issue #3) — SOLVED

**Inflation sized.** Selecting the best of k=3 tied candidates: the screen winner
overstates its true effect by **E[max of 3 N(0,1)] = 3/(2√π) = 0.846·σ_screen**
(2M-draw simulation: 0.8462). With σ_screen ≈ baseline seed-to-seed SD — the very
noise floor the confirmation runs on — the argmax is partly luck and the
confirmatory estimate inherits ~0.85·SD of optimism that no CI captures.

**Selection-stability check is the fix, and it is powerful** (chosen + runner-up,
**2 fresh seeds each**; fresh diff ~ N(true gap g, σ_screen)):

| true gap g (σ) | 0.0 | 0.25 | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|
| P(ranking flips) | 50% | 40% | 31% | 16% | 7% | 2% |

It catches ties (50% flip → downgrade to exploratory) and almost never wrongs a
real gap (7% at 1.5σ, 2% at 2σ). This is the review's "highest value-per-dollar"
fix, now quantified. **Cost: 2 extra 10M-screen runs per confirmed σ0.**

**Margin rule — recomputed after the issue #12 fix.** The margin is now the gap
between the best and second-best score over all three candidates. Simulated value
first, previous two-candidate analytic approximation in parentheses:

P(top-two margin < m | true gap g on candidate 1), three candidates, 2M draws:

| m (σ_screen) | g=0.0 | g=0.5 | g=1.0 | g=1.5 | g=2.0 |
|---|---|---|---|---|---|
| 0.25 | 20% (57%) | 19% (43%) | 17% (30%) | 13% (19%) | 9% (11%) |
| 0.5 | 38% (64%) | 37% (50%) | 32% (36%) | 25% (24%) | 17% (14%) |
| 1.0 | 66% (76%) | 64% (64%) | 57% (50%) | 47% (36%) | 34% (24%) |

The old approximation treats the chosen/runner-up pair as fixed with independent
noise; in reality a third candidate often takes the runner-up slot, which makes
the top-two gap *narrower* than a fixed pair's difference, so the old model
over-credited the margin rule at large gaps.

Under the global null (all candidates tied), 2M draws:

| statistic | top-two margin (correct) | old `abs(z₀−z₁)` |
|---|---|---|
| median | **0.691 σ** | 0.955 σ |
| P(< 0.5 σ) | **38.1%** | 27.6% |
| P(< 1 σ) | **66.0%** | 52.0% |

The reason the old figure was so far off: **the winner is not column 0 in 66.7%
of null draws**, so a fixed-column difference is not the margin in two-thirds of
rows (it agreed with the correct value on only 35.2% of 2000 random draws).

**Locked (combined rule, Phase 1 → Phase 2 handoff):**
1. Screen selects by final deterministic val-NLL, numeric thresholds only:
   chosen σ0 must complete with ≤ 1 instability event and throughput ≥ 80% of
   baseline; else next-best, in order.
2. **Stability check:** chosen + runner-up on 2 fresh seeds. Any ranking reversal
   → confirmatory claim downgraded to exploratory; the screen table is published
   regardless (raw margin visible).
3. Margin rule kept as a cheap pre-filter only, m = 0.5·σ_screen (from Phase 0
   calibration), NOT as the sole arbiter. It is weaker than it looked: with m =
   0.5·σ a genuine 1.0·σ gap still fails it **32%** of the time, and under the null
   a tied screen clears m = 1.0·σ **34%** of the time.

---

## P4. δ and "compatible uncertainty" — made power-consistent (issue #4 link)

Recomputed with the **exact** sign-flip test (no t-quantile approximation) and
interval clauses evaluated by the inversion that defines them — "CI lower > k" is
exactly "the test of d − k rejects". Seed 20260905, 40k trials/cell.

Fire rates, k in **sd_delta** units (μ also in sd_delta units):

| n | k | criterion | μ=0.5 | μ=1.0 | μ=1.5 | μ=2.0 |
|---|---|---|---|---|---|---|
| 6 | 1.0 | one-sided CI_lower > k (strong) | 0.003 | 0.048 | 0.265 | 0.649 |
| 6 | 1.0 | point>k & one-sided bound excl 0 (**primary**) | 0.096 | 0.454 | 0.856 | 0.985 |
| 6 | 1.0 | point>k & two-sided CI excl 0 (former rule) | 0.053 | 0.286 | 0.640 | 0.869 |
| 8 | 1.0 | one-sided CI_lower > k (strong) | 0.002 | 0.046 | 0.344 | 0.799 |
| 8 | 1.0 | point>k & one-sided bound excl 0 (**primary**) | 0.078 | 0.494 | 0.921 | 0.997 |
| 8 | 1.0 | point>k & two-sided CI excl 0 (former rule) | 0.070 | 0.460 | 0.901 | 0.994 |
| 10 | 1.0 | one-sided CI_lower > k (strong) | 0.001 | 0.049 | 0.426 | 0.897 |
| 10 | 1.0 | point>k & one-sided bound excl 0 (**primary**) | 0.056 | 0.498 | 0.943 | 0.999 |
| 10 | 1.0 | point>k & two-sided CI excl 0 (former rule) | 0.055 | 0.493 | 0.941 | 0.999 |

**Type-I error at μ = 0** (the acceptance criterion for issue #14 — the complete
rule must not fire on noise):

| n | one-sided test only | primary (consistent) | two-sided CI variant |
|---|---|---|---|
| 6 | 0.0481 | 0.0065 | 0.0037 |
| 8 | 0.0481 | 0.0022 | 0.0019 |
| 10 | 0.0478 | 0.0009 | 0.0009 |

The test alone holds the nominal level (0.048 ≤ α = 0.05; the small shortfall is the
discreteness of a 2ⁿ enumeration). The composite rules sit far below α because the
point-estimate clause is the binding one, so the rule is conservative rather than
anti-conservative — at the cost of power, quantified in the table above.

The design review warned that at n=5 the weak criterion "will trigger on noise alone".
That is not borne out: the one-sided test's type-I at n=5 is 0.031 (§P1b, eff=0.0
column) and the composite rule's is lower still, since passing the point-estimate
clause on pure noise is the harder condition of the two.

**Power at the actual preregistered δ, with the arm ratio carried** (§P5; exact test,
seed 20260905). "1.5σ" is a fixed true effect of 1.5·sd_delta; "at δ" is a true effect
exactly equal to the threshold:

| q | ρ | r | δ in sd_delta | n=6 @1.5σ | n=8 @1.5σ | n=10 @1.5σ | n=8 at δ |
|---|---|---|---|---|---|---|---|
| 1.00 | 0.75 | 0.7071 | 1.4142 | 0.578 | 0.598 | 0.610 | 0.505 |
| 1.00 | 0.50 | 1.0000 | 1.0000 | 0.857 | 0.917 | 0.942 | 0.492 |
| 1.00 | 0.25 | 1.2247 | 0.8165 | 0.896 | 0.966 | 0.984 | 0.473 |
| 1.00 | 0.00 | 1.4142 | 0.7071 | 0.905 | 0.976 | 0.993 | 0.444 |
| 1.25 | 0.75 | 0.8292 | 1.2060 | 0.749 | 0.797 | 0.824 | 0.495 |
| 1.25 | 0.50 | 1.1456 | 0.8729 | 0.886 | 0.958 | 0.975 | 0.483 |
| 1.25 | 0.25 | 1.3919 | 0.7184 | 0.905 | 0.974 | 0.993 | 0.444 |
| **1.25** | **0.00** | **1.6008** | **0.6247** | 0.908 | 0.979 | 0.996 | **0.408** |

Carrying q changes the headline: the rule's power at a fixed 1.5·sd_delta effect
spans **0.797 (q=1.25, ρ=0.75) to 0.979 (q=1.25, ρ=0)** at n=8, and power at the δ
threshold spans **0.408 to 0.505**. The earlier single-number quote (~91%) sat inside
that range at the most favourable corner.

**Locked:**
- **δ = 1.0 · sd_baseline** (SD from Phase 0/1 baseline calibration across seeds,
  measured before any treatment result is examined), **reported together with its
  sd_delta equivalent 1.0/r**, computed from the observed baseline and paired-difference
  SDs after the runs (a reported quantity, never a design input).
- **Primary "confirmed favorable": point estimate > δ AND one-sided 95%
  randomization bound > 0** (the bound inverted from the same one-sided test, §P6).
  At n=8 this fires 92% of the time at a true 1.5·sd_delta effect when ρ = 0.5 —
  and 60% when ρ = 0.75, because a fixed δ is a larger threshold in sd_delta units
  there. The unit assumption is now stated instead of implied.
- **Sensitivity: strong criterion at k = 0.5**, defined as the **one-sided** 95%
  bound > 0.5·sd_delta (0.801 at n=8, μ=1.5σ). The review's k=1 strong rule is
  still rejected as unconfirmable at this design (0.344 at μ=1.5σ).
- **Instability trigger (from issue #5):** ≥ 30% of treatment pairs unstable →
  primary conclusion "instability-dominated negative" regardless of surviving
  pairs; unstable dᵢ enter as missing AND as worst-observed (both reported).

---

## P5. Effect-size units (issue #13) — SOLVED

Two different SDs circulate in this protocol and the 2026-09-05 revision used one
label for both:

- **sd_baseline** — seed-to-seed SD of the baseline arm's final eval NLL. δ is
  written in this unit.
- **sd_delta** — seed-to-seed SD of the paired difference dᵢ = NLL_base(i) −
  NLL_annealed(i). The sign-flip test and every power number live in this unit.

Bienaymé's identity with equal arm SDs σ and correlation ρ between paired arms gives

    Var(d) = Var(B) + Var(T) − 2·Cov(B,T) = 2σ²(1 − ρ),   r = sd_delta/sd_baseline = √(2(1−ρ)) ∈ [0, √2]

so **δ = 1.0·sd_baseline is an effect of 1.0/r in sd_delta units** — equal to 1.0
only when ρ = 0.5, and 0.707 at ρ = 0. Predeclared handling:

- **Worst case ρ = 0 (r = √2) is the case the design must clear**, because ρ cannot
  be known before the treatment runs. Power is therefore reported across the ρ grid
  {0.75, 0.50, 0.25, 0.00} rather than at a single assumed value.
- **Arm variances are an assumption about the outcome, not an identity.** The
  earlier revision used r = √(2(1−ρ)), which silently assumes
  sd_treatment = sd_baseline. The general form carries the arm ratio
  q = sd_treatment/sd_baseline:

      Var(d) = Var(B) + Var(T) − 2·Cov(B,T)  ⟹  r = √(1 + q² − 2qρ)

  A noise arm can plausibly have a wider seed-to-seed spread than the
  deterministic arm, and pretending q = 1 when q > 1 flatters the design.
- **Predeclared tolerance: q ≤ 1.25, ρ ≥ 0.** A perturbation of at most 5% of RMS,
  applied through the same data order and seed pipeline, is not expected to inflate
  the treatment arm's SD by more than 25%; the realized q is *reported* after the
  runs and, if it exceeds this bound, the power statements are restated as a
  protocol deviation rather than reinterpreted. Worst case over the box is the
  corner (q = 1.25, ρ = 0): **r = √(1 + 1.25²) = 1.6008**.
- **Power at the δ threshold** (p1 §P1f, n=6/8/10):

  | q | ρ | r | δ in sd_delta | n=6 | n=8 | n=10 |
  |---|---|---|---|---|---|---|
  | 1.00 | 0.50 | 1.0000 | 1.0000 | 0.645 | 0.800 | 0.894 |
  | 1.00 | 0.00 | 1.4142 | 0.7071 | 0.415 | 0.542 | 0.660 |
  | 1.25 | 0.50 | 1.1456 | 0.8729 | 0.551 | 0.697 | 0.819 |
  | **1.25** | **0.00** | **1.6008** | **0.6247** | **0.352** | **0.458** | **0.572** |

  So the SAP's "80% power at a 1.0·SD effect" holds at **1.0·sd_delta**, and at the
  δ threshold only in the most favourable corner (q = 1, ρ = 0.5). At the
  predeclared worst case, δ is a 0.625·sd_delta effect and n=8 power at it is
  **0.458** — the design cannot claim 80% power at its own practical threshold
  without either more seeds or an argument that q ≈ 1 and ρ ≈ 0.5.
- **No confirmatory outcome is used to choose r or q**: both enter only through the
  predeclared grid and the post-hoc reporting of the realized values.

## P6. Test and interval on one convention (issue #14) — SOLVED

The SAP specified a one-sided sign-flip test at α = 0.05 while the positive-result
rule asked a **two-sided** 95% interval to exclude zero. A two-sided interval's
lower limit sits at the 2.5% tail, so the direction carrying the decision was held
to α/2 — stricter than the test that motivated it, and inconsistent with it.

**Resolved by construction:** the primary interval is the **one-sided 95%
randomization bound**, obtained by inverting the *same* one-sided test. Under that
convention the two clauses are the same event:

    "one-sided 95% randomization bound > 0"  ≡  "one-sided sign-flip test rejects at α"

This is asserted as **exact equality** over n = 6, 8, 10 × μ = 0, 0.4, 0.8, 1.2, and
separately for the rows whose p-value sits on an achievable level adjacent to α. It
was not exact at first: the interval path re-summed the observed statistic instead of
reading it from the enumeration, which disagreed with the test on 3 of 150 rows at
n=8, μ=0.4 — the kind of disagreement a ±1% tolerance would have hidden. Both paths
now read `t_obs` out of the statistic matrix.

**Two-sided intervals invert both tails.** A two-sided interval is not generally
symmetric about zero, so its upper endpoint cannot be the negation of its lower one;
each is now inverted from its own tail. The earlier version returned `upper = −lower`,
which failed an endpoint test on every row (0/200 valid) — it never affected the
primary one-sided rule, but it would have misled anyone reusing it. Consequences now
locked:

- Type-I error for the complete rule is **≤ α by construction** (measured 0.0022 at
  n=8, §P4 table).
- The former two-sided-CI variant is retained in the script and printed for
  comparison; it fires no more often (0.901 vs 0.921 at n=8, μ=1.5σ) and its
  type-I error is not smaller in any way that matters (0.0019 vs 0.0022).
- The percentile bootstrap stays descriptive-only, as before.

## P7. Reproducibility (issue #15) — SOLVED

`p4_delta_criteria.py` drew from SciPy with no RNG argument, inheriting global state.
Now:

- **Seeds:** `SEED_POWER = 42` (p1), `SEED_MATCH_CURSE = 7` (p23),
  `SEED_DELTA = 20260905` (p4), all in `analysis/aswp_core.py` and printed by each
  script in a "config (reproducibility)" block with trial counts, α, the ρ grid and
  the memory budget.
- **Sampling:** one `numpy.random.Generator` (PCG64) per script; draws use
  `Generator.normal` and `Generator.chisquare` (`aswp_core.chi2_scaled`), so no
  SciPy call can silently fall back to a hidden global stream.
- **Assumptions printed with the numbers:** iid normal paired differences within an
  arm; sign-flip exchangeability under H0; σ_delta = 1 for reported standardized
  effects; equal arm SDs for r = √(2(1−ρ)).
- **Test:** same seed → identical summary values; different seed → agreement within
  three Monte-Carlo standard errors (both asserted in the suite).

## P8. Memory-bounded exact enumeration (issue #16) — SOLVED

The previous implementation materialised a `(TRIALS, 2ⁿ, n)` broadcast — at 40,000
trials and n=10 that is ~3.3 GB in float64 alone. Measured before the change:
**3499 MiB peak, 130 s** at n=10.

`aswp_core.exact_p_batched` now accumulates `d_batch @ signs.T`, whose working set
is one float64 per (trial, sign pattern) pair, and partitions trials into batches
sized from a declared budget (`MEMORY_BUDGET_BYTES = 256 MiB`):

| n | 2ⁿ | batch (trials) | bytes/trial |
|---|---|---|---|
| 5 | 32 | 1,048,576 | 296 |
| 6 | 64 | 524,288 | 560 |
| 8 | 256 | 131,072 | 2,112 |
| 10 | 1024 | 32,768 | 8,272 |

Measured after: **596 MiB peak (p1), 610 MiB (p4), 4.9 s** at the same settings —
same numbers, no approach to the old peak. Every trial still enumerates all 2ⁿ sign
patterns (asserted: p-values fall exactly on the 1/2ⁿ grid), and results are
**bit-identical** to the retained broadcast reference implementation.

One correctness detail found while doing this, which applied to the original code as
well: the observed statistic *is* one of the enumerated statistics (the all-plus sign
row), so it is now read out of the statistic matrix instead of being re-summed from
`d`. An independently summed copy can differ by an ulp and silently move the p-value
by one sign pattern — which is exactly what happened while comparing the two paths
(differences of 2/2ⁿ on n=8 and n=10, with disagreeing decisions at α=0.05).

---

## P9. Frozen seed manifest (issue #10) — SOLVED

The eight paired seeds and the per-role RNG seeds are frozen in
`config/manifest.json`, with the derivation published rather than the numbers
merely drawn:

    seed = int.from_bytes(sha256("ASWP/v1/<role>/<n>")[:8], "big") % 2**31

Deriving them means the list is fully determined by a published string — no clock,
no beacon, no draw that could be re-rolled after seeing something — and anyone can
recompute it: `python analysis/seed_manifest.py` verifies the committed file against
the rule, and `tests/test_seed_manifest.py` does the same on every test run
(including a tampered-file case, so hand-editing a seed fails the suite).

Paired seeds (8): `1857649130, 1434172854, 1387619681, 536290471, 1202662852,
1106087492, 1881574181, 1859844390`. Per-role streams: `global`, `init`,
`data-loader`, `dropout`, `perturbation`. Writing the file is byte-idempotent, so
regenerating it cannot change the frozen list, and the SAP carries the same values.

What this does **not** resolve: the remaining SAP config fields (dataset/tokenizer/
validation hashes, model commit, training configuration, selected σ0) are blocked on
decisions or on the harness — enumerated with owners in
`docs/06-preregistration-checklist.md`.

## Final lock-list — paste into SAP config block

1. Hypothesis: **one-sided**, justification written in (§P1).
2. Seeds: **8 integers, generated once, written into the SAP** (min 6).
3. δ = **1.0·sd_baseline**, reported with its **sd_delta equivalent 1.0/r** under
   the predeclared box **q ≤ 1.25, ρ ≥ 0** (§P5, worst case r = 1.6008);
   "compatible uncertainty" = **point > δ AND one-sided 95% randomization bound > 0**
   (§P4, §P6).
4. Primary CI: **one-sided 95% randomization bound** (inverted sign-flip, the same
   one-sided test as the p-value); percentile bootstrap only as descriptive
   sensitivity (§P6; 126-composition quantization caveat for n=5 stays noted).
5. C4 arms: **0.6124·σ0 and σ0** (§P2).
6. C5/C5b tuning: same screen rule, same fixed token budget as treatment (§P2).
7. Screen rule (numeric): ≤ 1 instability event, throughput ≥ 80% baseline,
   next-best fallback (§P3).
8. Selection-stability check: chosen + runner-up on **2 fresh seeds**, any
   reversal → exploratory (§P3). Margin computed as the **top-two gap over all
   candidates**; m = 0.5·σ_screen is a pre-filter only (§P3).
9. Instability trigger: **≥ 30%** → "instability-dominated negative" (§P4).
10. Token budgets + LR tail (flat tail recommended): pinned per phase before
    Phase 0 (unchanged from review §2.12/§2.5 — this session did not re-derive).
11. Analysis reproducibility: seeds **42 / 7 / 20260905**, 40,000 trials per cell,
    2,000,000 screen draws, memory budget **256 MiB** — all printed by the scripts
    and asserted by `tests/test_aswp_core.py` (§P7, §P8).

## Analysis scripts (reproducible)
- `analysis/aswp_core.py` — sign-flip enumeration, batched exact p-values, test/interval
  inversion, decision rules, top-two margin, unit conversions, seeded RNG helpers
- `analysis/p1_exact_power.py` — resolution table, power curves, 80% thresholds,
  units table (power at δ by ρ)
- `analysis/p23_match_curse.py` — variance-match integrals, dropout calibration,
  winner's-curse inflation, top-two margin rule, stability-check power
- `analysis/p4_delta_criteria.py` — decision-rule fire rates with the exact test,
  type-I error, power at the preregistered δ under the unit conversion
- `analysis/seed_manifest.py` — frozen paired/per-role seed derivation and verification (issue #10)
- `tests/test_aswp_core.py`, `tests/test_seed_manifest.py` — 67 tests pinning all of the above
