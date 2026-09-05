#!/usr/bin/env python3
"""ASWP Problems 2 & 3: variance matching + winner's-curse sizing."""
import numpy as np
from scipy import integrate, stats

RNG = np.random.default_rng(7)

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
# d = chosen - runnerup ~ N(g, sigma*sqrt(2)); fail margin m => declare inconclusive
print("\nP3c. margin rule (single-seed screen): P(screen margin < m | true gap g)")
print("   d ~ N(g, sigma*sqrt(2))")
for m in [0.25, 0.5, 1.0]:
    row = []
    for g in [0.0, 0.5, 1.0, 1.5, 2.0]:
        row.append(f"g={g}:{stats.norm.cdf((m-g)/(np.sqrt(2))):.0%}")
    print(f"   m = {m} sigma_screen:  " + "  ".join(row))

# --- 3d: screen-table disclosure: p(any tie beaten by luck) ---
# under tied null, distribution of winning margin over runner-up
d = z[:, 0:2]  # reuse: winner= c1, runner=c2 when tied
win_margin = np.abs(z[:, 0] - z[:, 1])
print("\nP3d. under global null (all candidates tied): screen winning margin vs runner-up")
print(f"   median margin = {np.median(win_margin):.3f} sigma, P(margin < 0.5 sigma) = "
      f"{(win_margin < 0.5).mean():.1%},  P(margin < 1 sigma) = {(win_margin < 1.0).mean():.1%}")