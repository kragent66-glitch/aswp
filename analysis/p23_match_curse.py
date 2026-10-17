#!/usr/bin/env python3
"""ASWP Problems 2 & 3: variance matching + winner's-curse sizing.

Reproducibility
---------------
seed       : SEED_MATCH_CURSE = 7 (analysis/aswp_core.py)
trial count: N = 2,000,000 screen draws for the null/selection simulations
sampling   : numpy Generator (PCG64), one stream, consumed top-to-bottom; the
             screen matrix z is drawn once and reused by P3a and P3d so both
             describe the same draws.

Fix (issue #12): the margin is the gap between the best and second-best score
over ALL candidates (`winner_margin`, implemented and tested in aswp_core). The
earlier `np.abs(z[:, 0] - z[:, 1])` measured the absolute difference between two
fixed columns, which is not the winner-versus-runner-up margin: it is wrong in
value whenever the argmax is not column 0, and it is invariant to which candidate
actually won. Both quantities are printed so the size of the error is visible.
"""
import numpy as np
from scipy import integrate, stats

from aswp_core import SEED_MATCH_CURSE, make_rng, winner_margin

RNG = make_rng(SEED_MATCH_CURSE)

print("=" * 78)
print("P2. VARIANCE-MATCHED CONTROLS — exact multipliers")
print("=" * 78)

def cos_sched(p): return 0.5 * (1 + np.cos(np.pi * p))
def lin_sched(p): return 1.0 - p

for name, f in [("cosine (primary)", cos_sched), ("linear (secondary)", lin_sched)]:
    I, _ = integrate.quad(lambda p: f(p) ** 2, 0, 1)
    print(f"\n{name}: integral of f(p)^2 over [0,1] = {I:.6f} = {I.as_integer_ratio()[0]}/{I.as_integer_ratio()[1]}"
          if False else f"\n{name}: integral of f(p)^2 over [0,1] = {I:.6f}")
    print(f"   matched constant level  sigma_c = sigma0 * sqrt(I) = sigma0 * {np.sqrt(I):.6f}")
    print(f"   integrated-variance ratio of constant-at-sigma0 vs annealed = 1/I = {1/I:.4f}x")

# verification with discrete steps (one perturbation per optimizer step)
steps = 2000
p = np.linspace(0, 1, steps, endpoint=False)
disc = np.mean(cos_sched(p) ** 2)
print(f"\ndiscrete-step check (2000 steps, cosine) mean f^2 = {disc:.6f}  ->  sqrt = {np.sqrt(disc):.6f}")

# DropConnect / dropout: what sigma0 corresponds to a given drop rate (per-weight var match)?
print("\nper-weight variance matching: sigma0^2 * RMS^2 == r(1-r) * RMS^2  ->  sigma0 = sqrt(r(1-r))")
for r in [0.999, 0.995, 0.99, 0.98, 0.95, 0.90, 0.80]:
    print(f"   drop rate {(1-r)*100:4.1f}% (retention {r:.3f})  <=>  sigma0 = {np.sqrt(r*(1-r)):.4f}")

print()
print("=" * 78)
print("P3. WINNER'S CURSE — screen->confirm inflation, sized")
print("=" * 78)

# --- 3a: expected inflation when selecting best of k correlated-ish candidates ---
# screen metric for candidate j: mu_j + eps_j, eps ~ N(0, sigma_screen). Tied null => mu all equal.
k = 3
N = 2_000_000
z = RNG.standard_normal((N, k))
maxz = z.max(axis=1)
print(f"\nP3a. expected inflation = sigma_screen * E[max of {k} N(0,1)]")
print(f"   E[max] = {maxz.mean():.4f}  (exact E[max of 3 iid N(0,1)] = 3/(2*sqrt(pi)) = {3/(2*np.sqrt(np.pi)):.4f})")
print(f"   -> tied candidates: screen winner overstates its true effect by ~0.85 * screen-SD")

# --- 3b: stability check power (chosen + runner-up on 2 fresh seeds each) ---
# flip if fresh-seed mean difference < 0;  dbar2 ~ N(g, sigma*sqrt(1/2+1/2)) = N(g, sigma)
print("\nP3b. stability check (chosen vs runner-up, 2 fresh seeds each): P(ranking flips)")
print("   fresh diff ~ N(true gap g, sigma_screen)")
for g in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]:
    flip = stats.norm.cdf(0, loc=g, scale=1.0)
    print(f"   true gap g = {g:>4} sigma -> flip prob = {flip:.1%}   "
          f"({'catches tie' if g < 0.5 else 'will likely confirm'})")

# --- 3c: margin rule tradeoff, single-seed screen (chosen vs runner-up) ---
# Candidates 1..k share the screen; candidate 1 carries a genuine gap g.  The
# margin is the top-two gap of the realised screen scores, so a third candidate
# can take either the winning or the runner-up slot.  The two-candidate
# approximation printed alongside assumes the pair is fixed and its noise
# independent -- that is the model the previous revision used.
print("\nP3c. margin rule (single-seed screen): P(top-two margin < m | true gap g)")
print("   three candidates, gap g on candidate 1; margin = max - second max")
for m in [0.25, 0.5, 1.0]:
    row = []
    for g in [0.0, 0.5, 1.0, 1.5, 2.0]:
        mu = np.array([g, 0.0, 0.0])
        s = mu + RNG.standard_normal((N, k))
        margin = winner_margin(s)
        approx = stats.norm.cdf((m - g) / np.sqrt(2.0))
        row.append(f"g={g}:{(margin < m).mean():.0%}({approx:.0%})")
    print(f"   m = {m} sigma_screen:  " + "  ".join(row))
print("   (simulated value first; two-candidate analytic approximation in parentheses)")

# --- 3d: screen-table disclosure: p(any tie beaten by luck) ---
# under tied null, distribution of winning margin over runner-up, over all k
# candidates (issue #12: was abs(z[:,0] - z[:,1]), two fixed columns)
win_margin = winner_margin(z)
fixed_cols = np.abs(z[:, 0] - z[:, 1])
print("\nP3d. under global null (all candidates tied): screen winning margin vs runner-up")
print(f"   top-two margin      : median = {np.median(win_margin):.3f} sigma, "
      f"P(margin < 0.5 sigma) = {(win_margin < 0.5).mean():.1%},  "
      f"P(margin < 1 sigma) = {(win_margin < 1.0).mean():.1%}")
print(f"   (old fixed-column |z0-z1|: median = {np.median(fixed_cols):.3f} sigma, "
      f"P(<0.5) = {(fixed_cols < 0.5).mean():.1%}, P(<1) = {(fixed_cols < 1.0).mean():.1%}")
print(f"   winner is not column 0 in {(np.argmax(z, axis=1) != 0).mean():.1%} of null draws, "
      f"so a fixed-column difference misstates the margin in most rows")
print(f"   margin rule as pre-filter (m = 0.5 sigma): passes {100 * (win_margin >= 0.5).mean():.1f}% "
      f"of tied screens; at m = 1.0 sigma: {100 * (win_margin >= 1.0).mean():.1f}%")
