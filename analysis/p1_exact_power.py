#!/usr/bin/env python3
"""ASWP Problem 1: exact sign-flip test resolution + power at small n.

Model: paired differences d_i ~ N(mu, sigma). Effect size e = mu / sigma, where
sigma is SIGMA_DELTA -- the SD of the *paired differences* -- because the test
statistic is a function of the d_i. See analysis/aswp_core.py for why
sd_delta != sd_baseline (issue #13) and section P1f below for the consequence.

Exact one-sided p = P(T* >= T), two-sided p = P(|T*| >= |T|) over 2^n sign flips.

Reproducibility (issue #15 protocol, applied here too)
-----------------------------------------------------
seed          : SEED_POWER = 42 (analysis/aswp_core.py)
trial count   : TRIALS = 40,000 per (n, effect) cell
sampling      : numpy Generator (PCG64), one stream, consumed in the order the
                tables are printed -- so the tables below are reproducible
                exactly, not just within Monte-Carlo error
memory budget : MEMORY_BUDGET_BYTES = 256 MiB (issue #16). The 2^n enumeration
                is performed in trial batches sized to that budget, so peak
                memory no longer scales with TRIALS * 2^n.
"""
import resource

import numpy as np

from aswp_core import (
    MEMORY_BUDGET_BYTES,
    Q_TOLERANCE_MAX,
    RHO_GRID,
    SEED_POWER,
    TRIALS_POWER,
    batch_size_for,
    delta_in_delta_units,
    exact_p_batched,
    make_rng,
    sd_ratio,
    sign_matrix,
    worst_case_sd_ratio,
)

RNG = make_rng(SEED_POWER)
TRIALS = TRIALS_POWER

print("=" * 78)
print("P1 config (reproducibility)")
print("=" * 78)
print(f"  seed = {SEED_POWER}, trials per cell = {TRIALS:,}")
print(f"  memory budget = {MEMORY_BUDGET_BYTES / 2**20:.0f} MiB")
for n in (5, 6, 8, 10):
    print(f"    n={n:>2}: 2^n={2**n:>5} sign patterns, batch = {batch_size_for(n):>6,} trials,"
          f" {2**n * 8 + n * 8:>6,} bytes/trial")
print("  (the previous unbroadcast-cast implementation peaked at ~3.4 GiB at n=10)")


def exact_p(d, SIGNS, n, side="two"):
    """Exact p-values in bounded memory; identical arithmetic to the reference."""
    return exact_p_batched(d, SIGNS, side)


def power_at(n, effects, side, alpha, trials=TRIALS):
    SIGNS = sign_matrix(n)
    out = {}
    for e in effects:
        d = RNG.normal(loc=e, scale=1.0, size=(trials, n))
        p = exact_p(d, SIGNS, n, side)
        out[e] = float((p <= alpha).mean())
    return out


print()
print("=" * 78)
print("P1a. MINIMUM ACHIEVABLE p (resolution table)")
print("=" * 78)
print(f"{'n':>3} {'2^n':>6} {'min two-sided':>14} {'min one-sided':>14}")
for n in range(4, 11):
    print(f"{n:>3} {2**n:>6} {2/2**n:>14.4f} {1/2**n:>14.4f}")

print()
print("=" * 78)
print("P1b. POWER of exact one-sided test at alpha=0.05 (effect in SIGMA_DELTA units)")
print("=" * 78)
effects = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
print(f"{'n':>3} " + "".join(f"{f'eff={e:>4}':>12}" for e in effects))
for n in [5, 6, 8, 10]:
    p = power_at(n, effects, "one", 0.05)
    print(f"{n:>3} " + "".join(f"{p[e]:>12.3f}" for e in effects))

print()
print("=" * 78)
print("P1c. POWER of exact two-sided test at alpha=0.05 (n>=6 only; n=5 cannot)")
print("=" * 78)
print(f"{'n':>3} " + "".join(f"{f'eff={e:>4}':>12}" for e in effects))
for n in [5, 6, 8, 10]:
    p = power_at(n, effects, "two", 0.05)
    print(f"{n:>3} " + "".join(f"{p[e]:>12.3f}" for e in effects))

print()
print("=" * 78)
print("P1d. What does n=5 two-sided actually give? Power at its ONLY achievable")
print("     significance levels (alpha=0.0625 exact, 0.125 coarse)")
print("=" * 78)
for alpha in [0.0625, 0.125]:
    print(f"  two-sided n=5 alpha={alpha}:")
    p = power_at(5, effects, "two", alpha)
    print("   " + "".join(f"{f'{e:.2f}:{p[e]:.3f}':>12}" for e in effects))

print()
print("=" * 78)
print("P1e. Effect size the study can actually 'confirm' — n needed for 80% power")
print("     (one-sided alpha=0.05, exact test)")
print("=" * 78)
for n in [5, 6, 8, 10]:
    p = power_at(n, effects, "one", 0.05)
    # smallest listed effect >= 0.8
    ok = [e for e in effects if p[e] >= 0.8]
    print(f"  n={n}: 80%+ power for effect >= {min(ok) if ok else '>2.0'} sigma_delta")

print()
print("=" * 78)
print("P1f. UNITS: the preregistered delta is written in sd_baseline, the test")
print("     lives in sd_delta. delta = 1.0*sd_baseline is an effect of 1.0/r in")
print("     sd_delta units, with r = sqrt(1 + q^2 - 2*q*rho), q = sd_treatment/")
print("     sd_baseline (issue #13). q = 1 recovers sqrt(2(1-rho)).")
print("=" * 78)
print(f"{'q':>5} {'rho':>5} {'r':>7} {'delta=1.0 sd_base':>18} {'= in sd_delta':>14} "
      + "".join(f"{f'n={n}':>9}" for n in (6, 8, 10)))
for q in (1.0, Q_TOLERANCE_MAX):
    for rho in (0.5, 0.0):
        r = sd_ratio(rho, q)
        eff = delta_in_delta_units(1.0, rho, q)
        pows = {}
        for n in (6, 8, 10):
            pows[n] = power_at(n, [eff], "one", 0.05)[eff]
        tag = " (equal-variance model)" if q == 1.0 else " (predeclared tolerance)"
        print(f"{q:>5.2f} {rho:>5.2f} {r:>7.4f} {1.0:>18.2f} {eff:>14.4f} "
              + "".join(f"{pows[n]:>9.3f}" for n in (6, 8, 10)) + tag)
print()
print(f"  Arm ratio q = sd_treatment/sd_baseline enters as r = sqrt(1 + q^2 - 2*q*rho);")
print(f"  the grid above spans q = 1.0 to the predeclared tolerance q = {Q_TOLERANCE_MAX},")
print(f"  at rho = 0.5 and the worst-case rho = 0. Worst case over the whole predeclared")
print(f"  box (q <= {Q_TOLERANCE_MAX}, rho >= 0): r = {worst_case_sd_ratio():.4f}, so")
print(f"  delta = 1.0*sd_baseline is a {delta_in_delta_units(1.0, 0.0, Q_TOLERANCE_MAX):.4f}*sd_delta effect.")
print()
print("  Interpretation: the '80% power at a 1.0*sd effect' figure holds only at")
print("  r = 1, i.e. rho about 0.5 with equal arm variances. Power at the delta")
print("  threshold falls to 0.542 at the equal-variance worst case (rho = 0) and to")
print("  0.458 once the predeclared arm-ratio tolerance q <= 1.25 is carried: a 25%")
print("  wider treatment arm makes delta a smaller standardized effect (0.625 vs")
print("  0.707 sd_delta) with correspondingly less power. Both grids are reproduced")
print("  so the difference is visible rather than assumed.")

print()
print(f"P1 peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB"
      f" (budget {MEMORY_BUDGET_BYTES / 2**20:.0f} MiB)")
