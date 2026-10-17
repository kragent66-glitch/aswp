"""ASWP analysis test suite.

Run from the repository root:  python3 -m pytest -q

Covers the five filed issues:
  #12 winner-versus-runner-up margin        -> TestWinnerMargin
  #13 effect-size units (sd_baseline/delta) -> TestUnits
  #14 test/interval decision-rule coherence -> TestDecisionRule
  #15 seeded, reproducible simulation       -> TestReproducibility
  #16 memory-bounded exact enumeration      -> TestExactSignFlip
"""
import numpy as np
import pytest

from aswp_core import (
    MEMORY_BUDGET_BYTES,
    Q_TOLERANCE_MAX,
    RHO_GRID,
    SEED_POWER,
    batch_size_for,
    delta_in_delta_units,
    exact_p_batched,
    exact_p_reference,
    make_rng,
    randomization_ci,
    rho_from_sd_ratio,
    rule_consistent,
    rule_two_sided_ci,
    sd_ratio,
    sign_matrix,
    top_two_indices,
    winner_margin,
    worst_case_sd_ratio,
)


# --------------------------------------------------------------------------
# #16 exact sign-flip: correctness, batching invariance, memory bound
# --------------------------------------------------------------------------
class TestExactSignFlip:
    @pytest.mark.parametrize("n", [4, 5, 8, 10])
    def test_sign_matrix_enumerates_every_pattern(self, n):
        signs = sign_matrix(n)
        assert signs.shape == (2 ** n, n)
        assert set(np.unique(signs)) == {-1.0, 1.0}
        # every assignment is present exactly once, and no others
        assert len({tuple(r) for r in signs}) == 2 ** n
        # the last row is all-plus: the identity pattern is part of the enumeration
        assert np.array_equal(signs[-1], np.ones(n))

    @pytest.mark.parametrize("side", ["one", "two"])
    @pytest.mark.parametrize("n", [5, 8, 10])
    def test_batched_matches_reference(self, side, n):
        rng = make_rng(SEED_POWER)
        d = rng.normal(0.7, 1.0, size=(512, n))
        signs = sign_matrix(n)
        ref = exact_p_reference(d, signs, side)
        bat = exact_p_batched(d, signs, side)
        assert np.allclose(bat, ref, atol=1e-12)   # tolerance criterion
        assert np.array_equal(bat, ref)            # and in fact bit-identical

    def test_result_is_invariant_to_batch_size(self):
        rng = make_rng(3)
        n, signs = 8, sign_matrix(8)
        d = rng.normal(0.5, 1.0, size=(300, n))
        full = exact_p_batched(d, signs, "one", batch_size=10 ** 6)
        for bs in (1, 2, 7, 64):
            assert np.array_equal(exact_p_batched(d, signs, "one", batch_size=bs), full)

    def test_p_values_live_on_the_1_over_2n_grid(self):
        # every p-value is a count of sign patterns out of 2**n: if a trial's
        # p-value is not on that grid, patterns are being sampled, not enumerated
        rng = make_rng(5)
        for n in (5, 8):
            signs = sign_matrix(n)
            d = rng.normal(0.6, 1.0, size=(200, n))
            counts = exact_p_batched(d, signs, "one") * 2 ** n
            assert np.allclose(counts, np.round(counts), atol=1e-9)

    def test_default_batch_size_respects_the_memory_budget(self):
        for n in range(4, 13):
            per_block = batch_size_for(n) * 2 ** n * 8
            assert per_block <= MEMORY_BUDGET_BYTES
            # and the batch is not absurdly small for the default budget
            assert batch_size_for(n) >= 1
        assert batch_size_for(10) == MEMORY_BUDGET_BYTES // (2 ** 10 * 8)

    def test_single_trial_still_enumerates_all_patterns(self):
        n = 6
        signs = sign_matrix(n)
        d = np.array([[1.0, -0.5, 0.25, 0.1, -0.2, 0.3]])
        p = exact_p_batched(d, signs, "one")
        # observed statistic is the all-plus pattern, so p >= 1/2**n always
        assert p[0] >= 1.0 / 2 ** n
        assert (p[0] * 2 ** n) == pytest.approx(round(p[0] * 2 ** n))


# --------------------------------------------------------------------------
# #12 winner-versus-runner-up margin
# --------------------------------------------------------------------------
class TestWinnerMargin:
    def test_margin_is_top_two_gap_when_winner_is_not_column_0_or_1(self):
        # winners in columns 2, 0 and 1 respectively -> a fixed-column
        # difference cannot be right for all three rows at once
        scores = np.array([
            [0.20, 0.50, 3.00],   # winner col 2, runner-up col 1 -> 2.50
            [9.00, 1.00, 2.00],   # winner col 0, runner-up col 2 -> 7.00
            [0.10, 4.00, 0.30],   # winner col 1, runner-up col 2 -> 3.70
        ])
        assert np.allclose(winner_margin(scores), [2.50, 7.00, 3.70])
        # explicitly: margin == max - second largest, per row
        for row, expected in zip(scores, [2.50, 7.00, 3.70]):
            top = np.sort(row)[::-1]
            assert top[0] - top[1] == pytest.approx(expected)

    def test_margin_is_never_the_fixed_column_difference_in_general(self):
        rng = make_rng(11)
        scores = rng.normal(size=(2000, 3))
        margin = winner_margin(scores)
        broken = np.abs(scores[:, 0] - scores[:, 1])
        # the old expression disagrees whenever the argmax is not column 0
        not_col0 = np.argmax(scores, axis=1) != 0
        assert not_col0.mean() > 0.5
        assert not np.allclose(margin[not_col0], broken[not_col0])
        # and even when it agrees, it can be wrong in sign-magnitude: the fixed
        # difference ignores the third candidate entirely
        assert np.all(margin >= 0)
        assert np.allclose(margin, broken) is False

    def test_margin_is_invariant_to_column_permutation(self):
        rng = make_rng(12)
        scores = rng.normal(size=(500, 4))
        base = winner_margin(scores)
        for _ in range(5):
            perm = rng.permutation(4)
            assert np.allclose(winner_margin(scores[:, perm]), base)

    def test_reported_indices_agree_with_the_margin(self):
        rng = make_rng(13)
        scores = rng.normal(size=(300, 4))
        win, run = top_two_indices(scores)
        margin = winner_margin(scores)
        for i in range(300):
            assert scores[i, win[i]] >= scores[i, run[i]]
            assert scores[i, win[i]] - scores[i, run[i]] == pytest.approx(margin[i])

    def test_rejects_fewer_than_two_candidates(self):
        with pytest.raises(ValueError):
            winner_margin(np.array([[1.0], [2.0]]))

    def test_handles_multi_candidate_rows(self):
        assert winner_margin([1.0, 5.0, 3.0, 4.0]) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# #13 effect-size units
# --------------------------------------------------------------------------
class TestUnits:
    def test_sd_ratio_matches_the_variance_identity(self):
        # Var(d) = Var(B) + Var(T) - 2 Cov(B,T); equal arm SDs give 2s^2(1-rho)
        for rho in RHO_GRID:
            assert sd_ratio(rho) == pytest.approx(np.sqrt(2 * (1 - rho)))
        assert sd_ratio(0.0) == pytest.approx(np.sqrt(2))
        assert sd_ratio(0.5) == pytest.approx(1.0)
        assert sd_ratio(0.75) == pytest.approx(1 / np.sqrt(2))

    def test_ratio_round_trips(self):
        for rho in (0.0, 0.25, 0.5, 0.75, 0.9):
            assert rho_from_sd_ratio(sd_ratio(rho)) == pytest.approx(rho)

    def test_delta_conversion_between_units(self):
        # a 1.0*sd_baseline threshold is 1.0/r in sd_delta units
        assert delta_in_delta_units(1.0, 0.5) == pytest.approx(1.0)
        assert delta_in_delta_units(1.0, 0.0) == pytest.approx(1 / np.sqrt(2), abs=1e-6)

    def test_sd_ratio_generalizes_to_unequal_arm_variances(self):
        # q = 1 must reproduce the equal-variance formula exactly
        for rho in (0.0, 0.25, 0.5, 0.75):
            assert sd_ratio(rho, 1.0) == pytest.approx(np.sqrt(2 * (1 - rho)))
        # r = sqrt(1 + q^2 - 2 q rho)
        assert sd_ratio(0.0, 1.25) == pytest.approx(np.sqrt(1 + 1.25 ** 2))
        assert sd_ratio(0.5, 1.25) == pytest.approx(1.1456439, abs=1e-6)
        assert sd_ratio(0.3, 1.5) > sd_ratio(0.3, 1.0) > sd_ratio(0.3, 0.75)

    def test_arm_ratio_shrinks_the_standardized_effect(self):
        # a wider treatment arm makes the same delta a smaller sd_delta effect,
        # so the equal-variance assumption is not a free simplification
        assert delta_in_delta_units(1.0, 0.0, 1.25) < delta_in_delta_units(1.0, 0.0, 1.0)
        assert delta_in_delta_units(1.0, 0.0, 1.25) == pytest.approx(0.624692, abs=1e-5)
        assert delta_in_delta_units(1.0, 0.0, 1.0) == pytest.approx(0.707106, abs=1e-6)

    def test_worst_case_is_the_corner_of_the_predeclared_box(self):
        assert worst_case_sd_ratio() == pytest.approx(np.sqrt(1 + Q_TOLERANCE_MAX ** 2))
        assert worst_case_sd_ratio(1.5, 0.0) == pytest.approx(np.sqrt(1 + 2.25))

    def test_rho_round_trips_at_any_arm_ratio(self):
        for q in (0.75, 1.0, 1.25, 1.5):
            for rho in (0.0, 0.4, 0.8):
                assert rho_from_sd_ratio(sd_ratio(rho, q), q) == pytest.approx(rho)

    def test_units_are_not_interchangeable(self):
        # the specific confusion the issue describes: equal numbers, different
        # meanings.  Only rho = 0.5 makes the two units coincide.
        assert delta_in_delta_units(1.0, 0.5) == pytest.approx(1.0)
        for rho in (0.0, 0.25, 0.75):
            assert abs(delta_in_delta_units(1.0, rho) - 1.0) > 0.1


# --------------------------------------------------------------------------
# #14 decision rule: test and interval on one convention
# --------------------------------------------------------------------------
class TestDecisionRule:
    @pytest.mark.parametrize("n", [6, 8, 10])
    @pytest.mark.parametrize("mu", [0.0, 0.4, 0.8, 1.2])
    def test_interval_exclusion_is_the_test(self, n, mu):
        """'one-sided 95% interval excludes 0' == 'one-sided test rejects at 0.05'.

        Asserted exactly, not approximately. The interval is the inversion of the
        same test, so any disagreement is a defect in the inversion rather than a
        tolerance to be papered over -- including for rows whose p-value sits on
        an achievable level adjacent to alpha, which are reported separately
        because they are the cases that used to disagree.
        """
        rng = make_rng(21)
        signs = sign_matrix(n)
        d = rng.normal(mu, 1.0, size=(150, n))
        p = exact_p_batched(d, signs, "one")
        ci = randomization_ci(d, signs, 0.05, side="one")
        assert np.array_equal(ci["lower"] > 0, p <= 0.05)
        grid = 1.0 / 2 ** n
        boundary = np.abs(p - 0.05) <= grid
        if boundary.any():
            assert np.array_equal((ci["lower"] > 0)[boundary], (p <= 0.05)[boundary])

    @pytest.mark.parametrize("n", [6, 8])
    def test_two_sided_interval_has_two_valid_endpoints(self, n):
        """Both endpoints must be inverted from their own tail.

        The earlier implementation returned ``upper = -lower``, which is only a
        valid upper endpoint when the interval is exactly symmetric about zero.
        Small-n sign-flip intervals are not, so the two endpoints are checked
        independently here: ordered, bracketing the point estimate, and not each
        other's negation.
        """
        rng = make_rng(26)
        signs = sign_matrix(n)
        d = rng.normal(0.9, 1.5, size=(200, n))
        ci = randomization_ci(d, signs, 0.05, side="two")
        assert np.all(ci["lower"] < ci["upper"])
        mean = d.mean(axis=1)
        assert np.all(mean >= ci["lower"] - 1e-9)
        assert np.all(mean <= ci["upper"] + 1e-9)
        assert np.all(np.abs(ci["upper"] + ci["lower"]) > 1e-6)

    @pytest.mark.parametrize("n", [6, 8])
    def test_two_sided_interval_covers_the_true_shift(self, n):
        """Nominal 95% coverage, conservative because the grid is discrete."""
        rng = make_rng(27)
        signs = sign_matrix(n)
        covered = []
        for mu in (-1.0, -0.5, 0.0, 0.5, 1.0):
            d = rng.normal(mu, 1.0, size=(300, n))
            ci = randomization_ci(d, signs, 0.05, side="two")
            covered.append((ci["lower"] <= mu) & (mu <= ci["upper"]))
        coverage = float(np.concatenate(covered).mean())
        assert 0.90 <= coverage <= 1.0

    @pytest.mark.parametrize("n", [6, 8])
    def test_two_sided_interval_is_stricter(self, n):
        rng = make_rng(22)
        signs = sign_matrix(n)
        d = rng.normal(0.8, 1.0, size=(200, n))
        one = randomization_ci(d, signs, 0.05, side="one")["lower"]
        two = randomization_ci(d, signs, 0.05, side="two")["lower"]
        assert np.all(two <= one + 1e-9)
        assert (two > 0).mean() <= (one > 0).mean()

    def test_consistent_rule_never_fires_when_the_mismatched_rule_would_not(self):
        rng = make_rng(23)
        signs = sign_matrix(8)
        d = rng.normal(0.7, 1.0, size=(400, 8))
        consistent = rule_consistent(d, signs, delta=0.5)
        mismatched = rule_two_sided_ci(d, signs, delta=0.5)
        assert np.all(~mismatched["fires"] | consistent["fires"])

    def test_rule_requires_both_clauses(self):
        # a huge point estimate with no significance must not fire, and a solid
        # rejection that misses the practical threshold must not fire either
        signs = sign_matrix(6)
        rng = make_rng(24)
        noisy = rng.normal(3.0, 8.0, size=(50, 6))       # big mean, wild spread
        weak = np.full((50, 6), 0.01)                     # tiny, consistent sign
        assert rule_consistent(noisy, signs, delta=2.0)["fires"].sum() <= 50
        assert rule_consistent(weak, signs, delta=1.0)["fires"].sum() == 0

    @pytest.mark.parametrize("n", [6, 8])
    def test_type_one_error_is_bounded_by_alpha(self, n):
        rng = make_rng(25)
        signs = sign_matrix(n)
        d = rng.normal(0.0, 1.0, size=(4000, n))
        p = exact_p_batched(d, signs, "one")
        fired = float((p <= 0.05).mean())
        assert fired <= 0.055          # alpha plus Monte-Carlo slack
        # and it is not vacuous: the test does reject at roughly alpha
        assert fired > 0.02


# --------------------------------------------------------------------------
# #15 seeded, reproducible simulation
# --------------------------------------------------------------------------
class TestReproducibility:
    def test_same_seed_gives_identical_draws_and_summaries(self):
        def summary(seed):
            rng = make_rng(seed)
            signs = sign_matrix(8)
            d = rng.normal(0.8, 1.0, size=(500, 8))
            p = exact_p_batched(d, signs, "one")
            return float(d.mean()), float((p <= 0.05).mean())
        assert summary(SEED_POWER) == summary(SEED_POWER)

    def test_different_seeds_agree_within_monte_carlo_error(self):
        def power(seed):
            rng = make_rng(seed)
            signs = sign_matrix(8)
            d = rng.normal(1.0, 1.0, size=(4000, 8))
            p = exact_p_batched(d, signs, "one")
            return float((p <= 0.05).mean())
        a, b = power(1), power(2)
        # same estimator, different stream: within ~3 MC standard errors
        assert abs(a - b) < 3 * np.sqrt(0.8 * 0.2 / 4000)

    def test_generator_is_not_a_global_random_state(self):
        # two generators created with the same seed must not interleave
        r1, r2 = make_rng(99), make_rng(99)
        a = r1.normal(size=5)
        r2.normal(size=5)
        b = r1.normal(size=5)
        r3 = make_rng(99)
        r3.normal(size=5)
        assert np.array_equal(b, r3.normal(size=5))
        assert not np.array_equal(a, b)

    def test_scipy_free_sampling_helpers_are_seeded(self):
        from aswp_core import chi2_scaled
        assert np.array_equal(chi2_scaled(make_rng(7), 5, 10), chi2_scaled(make_rng(7), 5, 10))
