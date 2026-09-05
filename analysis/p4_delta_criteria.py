#!/usr/bin/env python3
"""ASWP P4 appendix: does the CI-lower > delta criterion match study power?
Normal model, paired diffs d_i ~ N(mu, sigma). Randomization/paired CI boundary
approximated by the paired t-quantile (t_{0.975, n-1} * s / sqrt(n)) — the
exact sign-flip CI agrees with this to within a grid step for the no-tie case.
"""
import numpy as np
from scipy import stats

print("=" * 78)
print("P4. delta = k * SD_baseline criteria vs. achievable power (one-sided design)")
print("=" * 78)
print(f"{'n':>3} {'k':>3} {'criterion':>22} " + "".join(f"{f'mu={m:.1f}':>10}" for m in [0.5, 1.0, 1.5, 2.0]))

for n in [6, 8, 10]:
    tq = stats.t.ppf(0.975, n - 1)
    for k in [1.0, 1.5]:
        row_strong, row_weak = [], []
        for mu in [0.5, 1.0, 1.5, 2.0]:
            # dbar ~ N(mu, 1/sqrt(n)); s^2 ~ chi2_{n-1}/(n-1), independent
            N = 200_000
            dbar = stats.norm.rvs(loc=mu, scale=1 / np.sqrt(n), size=N)
            s = np.sqrt(stats.chi2.rvs(n - 1, size=N) / (n - 1))
            se = s / np.sqrt(n)
            lo = dbar - tq * se
            # strong: 95% CI lower bound > k
            row_strong.append(f"{(lo > k).mean():.3f}")
            # weak: point > k AND CI excludes 0
            row_weak.append(f"{((dbar > k) & (lo > 0)).mean():.3f}")
        print(f"{n:>3} {k:>3} {'CI_lower > k (strong)':>22} " + "".join(f"{v:>10}" for v in row_strong))
        print(f"{'':>3} {k:>3} {'point>k & excl 0 (weak)':>22} " + "".join(f"{v:>10}" for v in row_weak))

print()
print("=" * 78)
print("P4b. what delta is reachable? max k such that strong criterion has >=50% power")
print("     at mu=1.5 sigma (the realistic 'visible consistent sign' effect)")
print("=" * 78)
for n in [6, 8, 10]:
    tq = stats.t.ppf(0.975, n - 1)
    for k in [0.0, 0.25, 0.5, 0.75, 1.0]:
        N = 100_000
        dbar = stats.norm.rvs(loc=1.5, scale=1 / np.sqrt(n), size=N)
        s = np.sqrt(stats.chi2.rvs(n - 1, size=N) / (n - 1))
        lo = dbar - tq * s / np.sqrt(n)
        print(f"  n={n} k={k:>4}: P(CI_lower > k | mu=1.5) = {(lo > k).mean():.3f}")