#!/usr/bin/env python3
"""ASWP P4 appendix: does the preregistered decision rule match study power?

Model: paired diffs d_i ~ N(mu, sigma_delta). Every criterion below is evaluated
with the EXACT sign-flip randomization test (2^n enumeration, batched to the
memory budget) rather than the paired t-quantile approximation the earlier
revision used. Interval clauses are evaluated by the inversion that defines them:
"CI lower bound > k" is exactly "the test of d - k rejects", so no grid search or
approximation is needed and the numbers are reproducible from the seed alone.

Reproducibility (issue #15)
---------------------------
seed            : SEED_DELTA = 20260905 (analysis/aswp_core.py)
trial count     : TRIALS = 40,000 per cell
sampling        : numpy Generator (PCG64); chi-square/SD draws use
                  Generator.chisquare via aswp_core.chi2_scaled, and normal
                  draws use Generator.normal, so no SciPy call can fall back to
                  a hidden global RNG state
assumptions     : (a) d_i iid N(mu, sigma_delta) within an arm;
                  (b) exchangeability of sign flips under H0 (the primary test's
                  own assumption, so the test is exact under it);
                  (c) sigma_delta == 1 (all effects are reported in sigma_delta
                  units), and the unit conversion carries the arm ratio q =
                  sigma_treatment/sigma_baseline explicitly:
                  r = sqrt(1 + q^2 - 2*q*rho), which is sqrt(2(1-rho)) only at
                  q = 1. Equal arm SDs is an assumption about the outcome, not an
                  identity, so the design predeclares the tolerance box
                  q <= 1.25, rho >= 0 (issue #13)
memory budget   : 256 MiB statistic block (issue #16)

Issue #14, made explicit: the primary test is one-sided at alpha=0.05, so the
interval that goes with it is a ONE-SIDED 95% lower bound. Requesting that a
*two-sided* 95% interval exclude zero holds the positive direction to the
alpha/2 = 0.025 standard that the test did not ask for; both are reported below
so the cost of the mismatch is a number rather than a claim.
"""
import resource

import numpy as np

from aswp_core import (
    MEMORY_BUDGET_BYTES,
    Q_TOLERANCE_MAX,
    RHO_GRID,
    SEED_DELTA,
    TRIALS_RULE,
    delta_in_delta_units,
    exact_p_batched,
    make_rng,
    sd_ratio,
    sign_matrix,
)

RNG = make_rng(SEED_DELTA)
TRIALS = TRIALS_RULE
ALPHA = 0.05
MUS = [0.5, 1.0, 1.5, 2.0]

print("=" * 78)
print("P4 config (reproducibility)")
print("=" * 78)
print(f"  seed = {SEED_DELTA}, trials per cell = {TRIALS:,}, alpha = {ALPHA}")
print(f"  rho grid = {RHO_GRID}, q grid = (1.0, {Q_TOLERANCE_MAX}) "
      f"[q = sigma_treatment/sigma_baseline]")
print(f"  unit conversion: r = sqrt(1 + q^2 - 2*q*rho); r = sqrt(2(1-rho)) only at q = 1")
print(f"  predeclared tolerance box: q <= {Q_TOLERANCE_MAX}, rho >= 0 -> worst r = 1.6008")
print(f"  memory budget = {MEMORY_BUDGET_BYTES / 2**20:.0f} MiB statistic block")
print("  test = exact one-sided sign-flip; interval = one-sided 95% randomization bound")
print("  ('CI lower > k' evaluated as the inverted test of d - k: exact, no gridding)")


def fire_rate(n, signs, mu, delta, trials=TRIALS):
    """P(primary rule fires | true effect mu), one seeded draw.

    The rule is exactly 'one-sided sign-flip test rejects at alpha AND the point
    estimate exceeds delta', which under the consistent convention is the same
    as 'the one-sided 95% randomization interval excludes zero AND point > delta'.
    """
    d = RNG.normal(loc=mu, scale=1.0, size=(trials, n))
    point = d.mean(axis=1)
    p_one = exact_p_batched(d, signs, "one")
    return float(((p_one <= ALPHA) & (point > delta)).mean())


def cells(n, k, one_sided_interval=True):
    """Fire rates for every criterion at (n, k), from one seeded draw per mu."""
    signs = sign_matrix(n)
    rows = {"strong": [], "primary": [], "mismatched": []}
    for mu in MUS:
        d = RNG.normal(loc=mu, scale=1.0, size=(TRIALS, n))
        point = d.mean(axis=1)
        p_one = exact_p_batched(d, signs, "one")
        p_two = exact_p_batched(d, signs, "two")
        # strong criterion: relevant 95% CI lower bound > k
        p_ci = exact_p_batched(d - k, signs, "one" if one_sided_interval else "two")
        rows["strong"].append(float((p_ci <= ALPHA).mean()))
        # primary (consistent): one-sided test rejects AND point estimate > k
        rows["primary"].append(float(((p_one <= ALPHA) & (point > k)).mean()))
        # mismatched variant: one-sided test, but a TWO-SIDED 95% interval must exclude 0
        rows["mismatched"].append(float(((p_one <= ALPHA) & (p_two <= ALPHA) & (point > k)).mean()))
    return rows


print()
print("=" * 78)
print("P4. Thresholds k in SD_DELTA units vs achievable power")
print("     NOTE ON UNITS: this table's k is a threshold in sd_delta -- the units the")
print("     generated paired differences and their estimates are in. It is NOT a")
print("     threshold in sd_baseline; the conversion delta = k * sd_baseline is")
print("     applied in P4b, and is 1.0 only when r = 1. The earlier revision labelled")
print("     these rows 'delta = k * sd_baseline' while comparing them against")
print("     sd_delta-scale estimates, which read one unit as the other.")
print("=" * 78)
print(f"{'n':>3} {'k':>4} {'criterion (k, mu in sd_delta)':>34} "
      + "".join(f"{f'mu={m:.1f}':>10}" for m in MUS))
for n in [6, 8, 10]:
    for k in [1.0, 1.5]:
        r = cells(n, k)
        print(f"{n:>3} {k:>4} {'one-sided CI_lower > k (strong)':>34} "
              + "".join(f"{v:>10.3f}" for v in r["strong"]))
        print(f"{'':>3} {'':>4} {'point>k & one-sided CI excl 0':>34} "
              + "".join(f"{v:>10.3f}" for v in r["primary"]))
        print(f"{'':>3} {'':>4} {'point>k & TWO-sided CI excl 0 (old)':>34} "
              + "".join(f"{v:>10.3f}" for v in r["mismatched"]))

print()
print("=" * 78)
print("P4a. TYPE-I ERROR of each complete rule at mu = 0 (issue #14)")
print("=" * 78)
print("     (fire = rule fires when the true effect is exactly zero; must be <= alpha)")
print(f"{'n':>3} {'one-sided test only':>22} {'primary (consistent)':>22} {'two-sided CI variant':>22}")
for n in [6, 8, 10]:
    signs = sign_matrix(n)
    d = RNG.normal(loc=0.0, scale=1.0, size=(TRIALS, n))
    point = d.mean(axis=1)
    p_one = exact_p_batched(d, signs, "one")
    p_two = exact_p_batched(d, signs, "two")
    delta = 1.0
    p_ci_k = exact_p_batched(d - delta, signs, "one")
    test_only = float((p_one <= ALPHA).mean())
    primary = float(((p_one <= ALPHA) & (point > delta)).mean())
    mismatched = float(((p_one <= ALPHA) & (p_two <= ALPHA) & (point > delta)).mean())
    print(f"{n:>3} {test_only:>22.4f} {primary:>22.4f} {mismatched:>22.4f}")

print()
print("=" * 78)
print("P4b. THE DECISION RULE UNDER THE UNIT CONVERSION (issue #13)")
print("     delta = 1.0*sd_baseline. The rule fires on (point > delta) AND (test")
print("     rejects), so its power depends on delta expressed in sd_delta units:")
print("     delta_sd_delta = 1.0/r, r = sqrt(1 + q^2 - 2*q*rho).")
print("=" * 78)
print("     'at 1.5 sd_delta' is the fixed realistic effect the SAP quotes;")
print("     'at the delta' is a true effect exactly equal to the threshold.")
print(f"     q = sd_treatment/sd_baseline; q = 1 is the equal-variance model, q = {Q_TOLERANCE_MAX}")
print("     is the predeclared tolerance the design's headline claims rest on.")
print(f"{'q':>5} {'rho':>5} {'r':>7} {'delta (sd_delta)':>17} "
      + "".join(f"{f'n={n} 1.5sd':>12}" for n in (6, 8, 10)) + f"{'n=8 at delta':>14}")
for q in (1.0, Q_TOLERANCE_MAX):
    for rho in RHO_GRID:
        r = sd_ratio(rho, q)
        delta_d = delta_in_delta_units(1.0, rho, q)
        pows, at_delta = [], 0.0
        for n in (6, 8, 10):
            signs = sign_matrix(n)
            pows.append(fire_rate(n, signs, mu=1.5, delta=delta_d))
            if n == 8:
                at_delta = fire_rate(n, signs, mu=delta_d, delta=delta_d)
        print(f"{q:>5.2f} {rho:>5.2f} {r:>7.4f} {delta_d:>17.4f} "
              + "".join(f"{v:>12.3f}" for v in pows) + f"{at_delta:>14.3f}")
print()
print("  The SAP's headline ('fires ~91% of the time at a true 1.5*sd effect at n=8')")
print("  is the rho = 0.5 column, i.e. it silently assumed r = 1, so that a")
print("  1.0*sd_baseline threshold and a 1.0*sd_delta effect are the same number.")
print("  They are the same only at r = 1. Across the predeclared grid the SAME rule")
print("  at the SAME fixed 1.5*sd_delta effect spans the 1.5sd column above.")

print()
print("=" * 78)
print("P4c. what delta is reachable? max k such that the strong criterion has >=50% power")
print("     at mu = 1.5 sd_delta (the realistic 'visible consistent sign' effect)")
print("=" * 78)
signs_cache = {}
for n in [6, 8, 10]:
    signs = signs_cache.setdefault(n, sign_matrix(n))
    d15 = RNG.normal(loc=1.5, scale=1.0, size=(TRIALS, n))
    for k in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p_ci = exact_p_batched(d15 - k, signs, "one")
        print(f"  n={n} k={k:>4}: P(one-sided CI_lower > k | mu=1.5) = {(p_ci <= ALPHA).mean():.3f}")

print()
print(f"P4 peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB")
