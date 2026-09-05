#!/usr/bin/env python3
"""ASWP Problem 1: exact sign-flip test resolution + power at small n.
Model: paired differences d_i ~ N(mu, sigma). Effect size e = mu/sigma.
Exact one-sided p = P(T* >= T), two-sided p = P(|T*| >= |T|) over 2^n sign flips.
"""
import numpy as np

RNG = np.random.default_rng(42)
TRIALS = 40_000

def sign_matrix(n):
    """All 2^n sign assignments as rows, shape (2^n, n)."""
    s = np.arange(2**n, dtype=np.int64)
    bits = ((s[:, None] >> np.arange(n)[::-1]) & 1).astype(np.float64)
    return np.where(bits == 0, -1.0, 1.0)

def exact_p(d, SIGNS, n, side="two"):
    """Vector over trials: exact p-value for each row of d (TRIALS, n)."""
    T = d.sum(axis=1)                       # observed stats
    # stats for every sign assignment: broadcast (TRIALS,1,n) x (2^n,n)
    Tstar = (d[:, None, :] * SIGNS[None, :, :]).sum(axis=2)  # (TRIALS, 2^n)
    if side == "two":
        p = (np.abs(Tstar) >= np.abs(T)[:, None]).mean(axis=1)
    else:
        p = (Tstar >= T[:, None]).mean(axis=1)
    return p, T

def power_at(n, effects, side, alpha, trials=TRIALS):
    SIGNS = sign_matrix(n)
    out = {}
    for e in effects:
        d = RNG.normal(loc=e, scale=1.0, size=(trials, n))
        p, _ = exact_p(d, SIGNS, n, side)
        out[e] = float((p <= alpha).mean())
    return out

print("=" * 78)
print("P1a. MINIMUM ACHIEVABLE p (resolution table)")
print("=" * 78)
print(f"{'n':>3} {'2^n':>6} {'min two-sided':>14} {'min one-sided':>14}")
for n in range(4, 11):
    print(f"{n:>3} {2**n:>6} {2/2**n:>14.4f} {1/2**n:>14.4f}")

print()
print("=" * 78)
print("P1b. POWER of exact one-sided test at alpha=0.05 (effect in baseline-SD units)")
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
    print(f"  n={n}: 80%+ power for effect >= {min(ok) if ok else '>2.0'} sigma")