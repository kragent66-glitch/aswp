# ASWP — Blocker Resolution (Session 2026-09-05)

**Resolves GitHub issues #1, #2, #3 (and the δ linkage in #4).**
Every value below is computed, not asserted. Scripts: `analysis/p1_exact_power.py`,
`analysis/p23_match_curse.py`, `analysis/p4_delta_criteria.py`.

| # | Issue | Resolution | Locked value(s) |
|---|---|---|---|
| 1 | n=5 two-sided can't reach α=0.05 | Lock one-sided test **and** raise minimum to n=6; target n=8 | one-sided, α=0.05, n_min=6, n_target=8; seed list written once |
| 2 | C4 not variance-matched | Run C4 at **both** σ0 and 0.6124·σ0 | σ_c = σ0·√(3/8) = **0.6124·σ0** (cosine); dose ratio 2.667× |
| 3 | Winner's curse in screen→confirm | Pre-register selection-stability check (chosen + runner-up, 2 fresh seeds) | flip on any ranking reversal; margin rule m = 0.5·SD_baseline as pre-filter |
| 4* | δ undefined / unconfirmable | Power-consistent rule: **weak criterion, k=1** primary; **strong criterion, k=0.5** sensitivity | δ = 1.0·SD_baseline (point > δ AND 95% CI excludes 0) |

---

## P1. Exact-test resolution and power (issue #1) — SOLVED

**Minimum achievable p** (sign-flip test enumerates 2ⁿ assignments):

| n | 2ⁿ | min two-sided p | min one-sided p |
|---|---|---|---|
| 5 | 32 | **0.0625** | 0.0313 |
| 6 | 64 | 0.0313 | 0.0156 |
| 8 | 256 | 0.0078 | 0.0039 |
| 10 | 1024 | 0.0020 | 0.0010 |

**Power of the exact one-sided test at α=0.05** (effect size in baseline-SD units,
40k simulations, seed 42):

| n | eff=0.25 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|
| 5 | 0.076 | 0.156 | 0.279 | 0.421 | 0.712 | 0.893 |
| 6 | 0.125 | 0.265 | 0.450 | 0.647 | 0.910 | 0.989 |
| 8 | 0.154 | 0.340 | 0.589 | 0.802 | 0.982 | 0.999 |
| 10 | 0.182 | 0.424 | 0.704 | 0.897 | 0.996 | 1.000 |

**Two-sided α=0.05 is literally impossible at n=5** (power = 0.000 → the study
cannot win at its own minimum). At n=6 two-sided reaches 80% only at 2.0σ.

**Locked:**
- **Hypothesis: one-sided (H1: dᵢ > 0).** Justification for the SAP: the decision
  ladder has no use for "significantly worse than baseline" — a worse result is
  reported as an estimate regardless. Written once, never revisited.
- **Minimum n = 6** (two-sided α=0.05 becomes *possible* as a fallback), **target
  n = 8**: 80% power at 1.0σ, 98% at 1.5σ. n=8 also restores two-sided capability
  (65.6% at 1.0σ) if a reviewer demands it.
- **80%-power honesty line:** the design confirms effects ≥ 1.0σ (n=8). Effects
  below that are reported as point estimates with CI, per the existing posture.
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

**Margin rule alone is weak on single-seed screens** (P4-detail): with m = 0.5·σ,
a real 1σ gap still fails the margin 36% of the time; with m = 1·σ, tied screens
pass it 48% of the time. Under the global null the winning margin vs runner-up is
< 1σ 52% of the time (median 0.955σ).

**Locked (combined rule, Phase 1 → Phase 2 handoff):**
1. Screen selects by final deterministic val-NLL, numeric thresholds only:
   chosen σ0 must complete with ≤ 1 instability event and throughput ≥ 80% of
   baseline; else next-best, in order.
2. **Stability check:** chosen + runner-up on 2 fresh seeds. Any ranking reversal
   → confirmatory claim downgraded to exploratory; the screen table is published
   regardless (raw margin visible).
3. Margin rule kept as a cheap pre-filter, m = 0.5·SD_baseline (from Phase 0
   calibration), NOT as the sole arbiter.

---

## P4. δ and "compatible uncertainty" — made power-consistent (issue #4 link)

The review recommended the strong criterion (95% CI lower > δ, k∈[1,2]). The
power analysis shows that combination is **unconfirmable** at this design:

| n | k | criterion | P(confirm \| μ=1.5σ) | P(confirm \| μ=2.0σ) |
|---|---|---|---|---|
| 6 | 1.0 | strong (CI_lower>δ) | 0.170 | 0.506 |
| 8 | 1.0 | strong | 0.231 | 0.680 |
| 8 | 1.0 | **weak (point>δ & excl 0)** | **0.905** | 0.996 |
| 8 | 0.5 | strong | 0.680 | 0.897 |
| 10 | 1.0 | strong | 0.293 | 0.802 |

**Locked:**
- **δ = 1.0 · SD_baseline** (SD from Phase 0/1 baseline calibration across seeds,
  measured before any treatment result is examined).
- **Primary "confirmed favorable": point estimate > δ AND 95% randomization CI
  excludes 0.** At n=8 this fires 91% of the time at a true 1.5σ effect and 47%
  at 1.0σ — consistent with the power table in P1. It cannot trigger on noise
  alone (type-I ≈ α).
- **Sensitivity: strong criterion at k = 0.5** (CI lower > 0.5·SD_baseline).
- **Instability trigger (from issue #5):** ≥ 30% of treatment pairs unstable →
  primary conclusion "instability-dominated negative" regardless of surviving
  pairs; unstable dᵢ enter as missing AND as worst-observed (both reported).

---

## Final lock-list — paste into SAP config block

1. Hypothesis: **one-sided**, justification written in (§P1).
2. Seeds: **8 integers, generated once, written into the SAP** (min 6).
3. δ = **1.0·SD_baseline**; "compatible uncertainty" = **point > δ AND 95%
   randomization CI excludes 0** (§P4).
4. Primary CI: **randomization CI** (inverted sign-flip); percentile bootstrap
   only as descriptive sensitivity (§P1; 126-composition quantization caveat for
   n=5 stays noted).
5. C4 arms: **0.6124·σ0 and σ0** (§P2).
6. C5/C5b tuning: same screen rule, same fixed token budget as treatment (§P2).
7. Screen rule (numeric): ≤ 1 instability event, throughput ≥ 80% baseline,
   next-best fallback (§P3).
8. Selection-stability check: chosen + runner-up on **2 fresh seeds**, any
   reversal → exploratory (§P3).
9. Instability trigger: **≥ 30%** → "instability-dominated negative" (§P4).
10. Token budgets + LR tail (flat tail recommended): pinned per phase before
    Phase 0 (unchanged from review §2.12/§2.5 — this session did not re-derive).

## Analysis scripts (reproducible)
- `analysis/p1_exact_power.py` — resolution table, power curves, 80% thresholds
- `analysis/p23_match_curse.py` — variance-match integrals, dropout calibration,
  winner's-curse inflation, stability-check power, margin-rule tradeoffs
- `analysis/p4_delta_criteria.py` — δ criteria reachability (t-CI approximation
  to the flip CI, normal model; 200k-draw Monte Carlo)