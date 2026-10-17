#!/usr/bin/env python3
"""ASWP shared analysis core.

Everything the analysis scripts and the test suite share lives here, so that a
claim in the documentation can be traced to one implementation that a test
exercises directly.

Conventions used throughout
---------------------------
SIGMA_BASELINE (``sd_baseline``)
    Seed-to-seed standard deviation of the baseline arm's final deterministic
    validation NLL, across the frozen paired seed list.  This is the unit the
    preregistered practical threshold is written in: ``delta = 1.0 * sd_baseline``.

SIGMA_DELTA (``sd_delta``)
    Seed-to-seed standard deviation of the *paired difference* ``d_i =
    NLL_baseline(i) - NLL_annealed(i)``.  This is the unit the sign-flip test and
    the power simulation operate in, because the test is a function of the ``d_i``.

    The two are different units, and conflating them is the error tracked as
    issue #13.  With equal arm variances and correlation ``rho`` between paired
    arms, Bienayme's identity gives

        Var(d) = Var(B) + Var(T) - 2 Cov(B, T) = 2 * sigma^2 * (1 - rho)

    so the ratio ``r = sd_delta / sd_baseline = sqrt(2 (1 - rho))`` lies in
    [0, sqrt(2)].  A threshold written as ``1.0 * sd_baseline`` is therefore an
    effect of ``1.0 / r`` in ``sd_delta`` units -- 1.0 when rho = 0.5, but only
    0.707 when rho = 0.  ``rho`` is not knowable before the treatment runs, so the
    design declares a worst case (``RHO_WORST_CASE``) rather than assuming 0.5.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Reproducibility: one seed per analysis, documented in the docs and here.
# --------------------------------------------------------------------------
SEED_POWER = 42        # p1: exact sign-flip resolution + power
SEED_MATCH_CURSE = 7   # p23: winner's-curse / margin simulations
SEED_DELTA = 20260905  # p4: delta-criteria decision-rule simulations
SEED_MARGIN_REF = 11   # deterministic margin reference case

TRIALS_POWER = 40_000
TRIALS_RULE = 40_000

#: Memory ceiling for the exact sign-flip enumeration, in bytes.  The batched
#: implementation never allocates more than roughly this much for the
#: (batch, 2**n) statistic block, so the default settings run inside it.
MEMORY_BUDGET_BYTES = 256 * 1024 * 1024

#: Worst-case (most conservative) correlation between paired arms.  rho = 0 is
#: the assumption that the noise perturbation is uncorrelated with the baseline
#: trajectory -- wrong in the optimistic direction, which is why it is the bound
#: the design must clear rather than the value it may assume.
RHO_WORST_CASE = 0.0

#: Reference correlation, rho = 0.5, the "no pairing benefit, no pairing harm"
#: case that a paired design with equal arm SDs would show at r = 1.0.
RHO_REFERENCE = 0.5

RHO_GRID = (0.75, 0.5, 0.25, 0.0)

#: Arm-variance ratio grid: q = sd_treatment / sd_baseline.  q = 1 is the
#: equal-variance case the earlier revision assumed silently; the grid shows what
#: the unit conversion costs when the treatment arm is more or less variable.
Q_GRID = (0.75, 1.0, 1.25, 1.5)

#: The tolerance the design's headline power claims are conditional on.  A
#: perturbation of at most 5% of RMS, applied through the same data order and
#: seed pipeline, is not expected to inflate the treatment arm's seed-to-seed SD
#: by more than 25%.  The realized q is reported after the runs; if it exceeds
#: this tolerance the power statements are restated as a protocol deviation
#: rather than silently reinterpreted.
Q_TOLERANCE_MAX = 1.25


# --------------------------------------------------------------------------
# Sign-flip machinery
# --------------------------------------------------------------------------
def sign_matrix(n: int) -> np.ndarray:
    """All 2**n sign assignments as rows, shape (2**n, n).

    Enumerated in full: the exact test's validity rests on every assignment
    being present, so no sampling shortcut is used anywhere in this module.
    """
    s = np.arange(2 ** n, dtype=np.int64)
    bits = ((s[:, None] >> np.arange(n)[::-1]) & 1).astype(np.float64)
    return np.where(bits == 0, -1.0, 1.0)


def batch_size_for(n: int, memory_budget_bytes: int = MEMORY_BUDGET_BYTES) -> int:
    """Trials per batch so the float64 statistic block fits the budget.

    The batched path accumulates ``d @ signs.T``, whose working set is
    ``batch * 2**n * 8`` bytes -- one float64 per (trial, sign pattern) pair,
    with the ``axis=-1`` reduction done by the matrix product rather than by
    materialising a (batch, 2**n, n) intermediate.  Batching on that per-trial
    cost is what actually bounds peak memory: sizing batches on the *statistic*
    count while the implementation still broadcast to (batch, 2**n, n) would
    leave peak memory n times over budget.
    """
    per_trial = 2 ** n * 8
    return max(1, int(memory_budget_bytes // per_trial))


def exact_p_reference(d: np.ndarray, signs: np.ndarray, side: str = "two") -> np.ndarray:
    """Reference exact sign-flip p-values, broadcasting (trials, 2**n, n).

    Kept as the reference implementation the batched path is checked against.
    Memory-hungry by construction -- do not call it with large ``trials``.
    """
    t_star = (d[:, None, :] * signs[None, :, :]).sum(axis=2)
    # The observed statistic IS one of the enumerated statistics: the all-plus
    # sign row (last row of sign_matrix).  Reading it out of t_star instead of
    # re-summing d keeps the identity (and all-minus) comparisons exact -- an
    # independently summed copy can differ by an ulp and silently change the
    # p-value by one sign pattern.
    t_obs = t_star[:, -1]
    if side == "two":
        return (np.abs(t_star) >= np.abs(t_obs)[:, None]).mean(axis=1)
    return (t_star >= t_obs[:, None]).mean(axis=1)


def exact_p_batched(d: np.ndarray, signs: np.ndarray, side: str = "two",
                    batch_size: int | None = None,
                    memory_budget_bytes: int = MEMORY_BUDGET_BYTES) -> np.ndarray:
    """Exact sign-flip p-values, computed in trial batches.

    Identical arithmetic to :func:`exact_p_reference` (same broadcast, same
    accumulation order, same comparison), only partitioned over trials, so
    results are bit-identical while peak memory is bounded by
    ``memory_budget_bytes`` instead of growing with ``trials * 2**n``.

    The full 2**n enumeration happens for every trial.
    """
    trials, n = d.shape
    if batch_size is None:
        batch_size = batch_size_for(n, memory_budget_bytes)
    out = np.empty(trials, dtype=np.float64)
    for start in range(0, trials, batch_size):
        stop = min(start + batch_size, trials)
        out[start:stop] = _p_from_stats(d[start:stop], signs, side)
    return out


def _p_from_stats(d: np.ndarray, signs: np.ndarray, side: str) -> np.ndarray:
    """Exact p-values for a block of trials, reducing over n inside the matmul."""
    t_star = d @ signs.T                      # (batch, 2**n)
    t_obs = t_star[:, -1]                     # see exact_p_reference
    if side == "two":
        return (np.abs(t_star) >= np.abs(t_obs)[:, None]).mean(axis=1)
    return (t_star >= t_obs[:, None]).mean(axis=1)


def _p_at_shift(d: np.ndarray, shift: np.ndarray, signs: np.ndarray,
                side: str, batch_size: int) -> np.ndarray:
    """One-sided/two-sided sign-flip p for per-row shifts (vectorised).

    ``t_obs`` is read from the all-plus column of each block's statistic matrix,
    exactly as in :func:`_p_from_stats`, rather than re-summed from the centred
    data. Re-summing lets a rounding-unit difference move the comparison for the
    identity (and all-minus) patterns by one sign pattern, which would make the
    interval decide differently from the p-value it is supposed to invert.
    """
    centred = d - shift[:, None]
    trials = d.shape[0]
    out = np.empty(trials, dtype=np.float64)
    for start in range(0, trials, batch_size):
        stop = min(start + batch_size, trials)
        t_star = centred[start:stop] @ signs.T
        t_obs = t_star[:, -1]
        if side == "two":
            out[start:stop] = (np.abs(t_star) >= np.abs(t_obs)[:, None]).mean(axis=1)
        else:
            out[start:stop] = (t_star >= t_obs[:, None]).mean(axis=1)
    return out


def randomization_ci(d: np.ndarray, signs: np.ndarray, alpha: float = 0.05,
                     side: str = "one", tol: float = 1e-9, max_iter: int = 80) -> dict:
    """Randomization confidence interval by inverting the sign-flip test.

    ``side="one"`` returns a one-sided lower bound: the smallest shift at which
    the one-sided test at level ``alpha`` stops rejecting.  ``side="two"``
    returns a two-sided interval, whose lower limit answers the same question at
    ``alpha/2`` in the lower tail.

    The inversion is done by vectorised bisection over all rows, so the returned
    bounds are roots of ``p(shift) = alpha`` (one-sided) or ``alpha/2``
    (two-sided lower limit) to within ``tol`` on the shift axis.
    """
    d = np.atleast_2d(d)
    n = d.shape[1]
    batch = batch_size_for(n)
    mean = d.mean(axis=1)
    span = 50.0 * d.std(axis=1) + 10.0

    def bisect(left: np.ndarray, right: np.ndarray, side_: str, level: float,
               rejecting_at_right: bool) -> np.ndarray:
        """Bisect a rejection boundary between two shifts, row-wise.

        ``rejecting_at_right`` states which end is the rejecting one, so the same
        routine serves both tails.
        """
        a, b = left.copy(), right.copy()
        for _ in range(max_iter):
            mid = 0.5 * (a + b)
            rej = _p_at_shift(d, mid, signs, side_, batch) <= level
            if rejecting_at_right:
                b = np.where(rej, mid, b)
                a = np.where(rej, a, mid)
            else:
                a = np.where(rej, mid, a)
                b = np.where(rej, b, mid)
            if np.all(b - a < tol):
                break
        return b if rejecting_at_right else a

    if side == "one":
        # The one-sided test rejects for small shifts and stops rejecting at the
        # bound: p(shift) is non-decreasing in shift.
        lower = bisect(mean - span, mean, "one", alpha, rejecting_at_right=False)
        return {"lower": lower, "upper": np.full_like(lower, np.inf)}

    # Two-sided: the test rejects outside a bounded interval, so the interval has
    # two distinct endpoints. Each is inverted from its own tail -- negating the
    # lower endpoint is not the upper endpoint unless the distribution is
    # exactly symmetric about zero, which small-n sign-flip intervals are not.
    lower = bisect(mean - span, mean, "two", alpha, rejecting_at_right=False)
    upper = bisect(mean, mean + span, "two", alpha, rejecting_at_right=True)
    return {"lower": lower, "upper": upper}


# --------------------------------------------------------------------------
# Decision rules (issue #14: test and interval must use one convention)
# --------------------------------------------------------------------------
def rule_consistent(d: np.ndarray, signs: np.ndarray, delta: float,
                    alpha: float = 0.05) -> dict:
    """The preregistered rule, with test and interval on the same footing.

    Primary test: exact ONE-SIDED sign-flip test at ``alpha``.
    Primary interval: one-sided 95% randomization lower bound, inverted from
    that same one-sided test.

    Under this convention "the interval excludes 0" and "the test rejects at
    ``alpha``" are the same event (checked in the test suite), so the composite
    rule ``point > delta AND interval excludes 0`` has type-I error <= alpha by
    construction, and the two clauses cannot pull in different directions.
    """
    p_one = exact_p_batched(d, signs, "one")
    ci = randomization_ci(d, signs, alpha, side="one")
    point = d.mean(axis=1)
    return {
        "p_one_sided": p_one,
        "ci_lower": ci["lower"],
        "point": point,
        "test_rejects": p_one <= alpha,
        "interval_excludes_zero": ci["lower"] > 0.0,
        "point_over_delta": point > delta,
        "fires": (p_one <= alpha) & (ci["lower"] > 0.0) & (point > delta),
    }


def rule_two_sided_ci(d: np.ndarray, signs: np.ndarray, delta: float,
                      alpha: float = 0.05) -> dict:
    """The inconsistent variant issue #14 identified, kept for comparison.

    One-sided test, but the positive-result rule asks a *two-sided* 95%
    interval to exclude zero.  Its lower limit sits at the ``alpha/2`` tail, so
    the direction that carries the decision is held to a stricter standard than
    the test that motivated it.
    """
    p_one = exact_p_batched(d, signs, "one")
    ci = randomization_ci(d, signs, alpha, side="two")
    point = d.mean(axis=1)
    return {
        "p_one_sided": p_one,
        "ci_lower": ci["lower"],
        "fires": (p_one <= alpha) & (ci["lower"] > 0.0) & (point > delta),
    }


# --------------------------------------------------------------------------
# Winner's curse / selection margin (issue #12)
# --------------------------------------------------------------------------
def winner_margin(scores: np.ndarray) -> np.ndarray:
    """Gap between the best and second-best score in each row of ``scores``.

    ``max(score) - second_largest(score)`` over *all* candidates, not the
    difference between two fixed columns: the winner's identity and the
    runner-up's identity both depend on the row, so a fixed-column difference
    measures something else entirely (wrong gap whenever the argmax is not
    column 0, and even wrongly signed).

    Works for any k >= 2 candidates and is invariant to which column wins.
    """
    scores = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    if scores.shape[1] < 2:
        raise ValueError("need at least two candidates to define a winner/runner-up margin")
    part = np.partition(scores, kth=scores.shape[1] - 2, axis=1)
    return part[:, -1] - part[:, -2]


def top_two_indices(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column indices of the winner and runner-up per row."""
    scores = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    order = np.argsort(-scores, axis=1, kind="stable")
    return order[:, 0], order[:, 1]


# --------------------------------------------------------------------------
# Effect-size units (issue #13)
# --------------------------------------------------------------------------
def sd_ratio(rho: float, q: float = 1.0) -> float:
    """r = sd_delta / sd_baseline from Var(d) = Var(B) + Var(T) - 2*Cov(B,T).

    ``q = sd_treatment / sd_baseline`` is the arm-variance ratio.  The general
    form is::

        r = sqrt(1 + q^2 - 2 q rho)

    which reduces to ``sqrt(2(1 - rho))`` at q = 1.  Carrying q explicitly
    matters because equal arm variances is an *assumption about the outcome*, not
    a mathematical identity: a noise arm can plausibly have a different
    seed-to-seed spread than the deterministic arm it is compared against, and
    assuming q = 1 when q > 1 makes the design look more sensitive than it is.
    """
    return float(np.sqrt(1.0 + q * q - 2.0 * q * rho))


def rho_from_sd_ratio(r: float, q: float = 1.0) -> float:
    """Inverse of :func:`sd_ratio` in rho, for a given arm ratio q."""
    return float((1.0 + q * q - r * r) / (2.0 * q))


def delta_in_delta_units(delta_baseline_units: float, rho: float, q: float = 1.0) -> float:
    """A threshold written in sd_baseline units, expressed in sd_delta units.

    ``delta = k * sd_baseline`` is an effect of ``k / r`` in the units the test
    lives in.  Forgetting to convert is precisely how a design ends up claiming
    more power than its deltas can deliver.
    """
    return float(delta_baseline_units / sd_ratio(rho, q))


def delta_in_baseline_units(effect_delta_units: float, rho: float, q: float = 1.0) -> float:
    """Inverse of :func:`delta_in_delta_units`."""
    return float(effect_delta_units * sd_ratio(rho, q))


def worst_case_sd_ratio(q_max: float | None = None, rho_min: float | None = None) -> float:
    """Largest r the design must tolerate over the predeclared (q, rho) box.

    r increases in q and decreases in rho, so the worst case sits at the corner
    (q = q_max, rho = rho_min): ``sqrt(1 + q_max^2 - 2 q_max rho_min)``.
    """
    q_max = Q_TOLERANCE_MAX if q_max is None else q_max
    rho_min = RHO_WORST_CASE if rho_min is None else rho_min
    return sd_ratio(rho_min, q_max)


# --------------------------------------------------------------------------
# Sampling helpers (issue #15)
# --------------------------------------------------------------------------
def make_rng(seed: int) -> np.random.Generator:
    """The single RNG constructor used by every analysis script."""
    return np.random.default_rng(seed)


def paired_differences(rng: np.random.Generator, n: int, mu: float, sigma: float,
                       trials: int) -> np.ndarray:
    """``trials`` rows of ``n`` paired differences, d_i ~ N(mu, sigma)."""
    return rng.normal(loc=mu, scale=sigma, size=(trials, n))


def chi2_scaled(rng: np.random.Generator, df: int, trials: int) -> np.ndarray:
    """Sample SDs ``s = sqrt(chi2_df / df)`` using the generator directly.

    Uses ``Generator.chisquare`` rather than a SciPy call so the sampling stream
    is explicitly the documented one; equivalent to ``scipy.stats.chi2.rvs``
    with ``random_state=rng``.
    """
    return np.sqrt(rng.chisquare(df, size=trials) / df)
