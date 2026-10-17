"""Arms and their noise schedules.

The protocol's arm list, and the arithmetic that makes the constant-noise arm a *fair*
comparator rather than merely a second run.

Why the constant-noise arm needs variance matching: the annealing schedule
`sigma(t) = sigma0 * 0.5 * (1 + cos(pi t / T))` delivers a *decaying* dose, so its total
perturbation energy is not `sigma0^2`. Integrating over the run,

    mean_t sigma(t)^2 / sigma0^2 = mean_x (0.5 (1 + cos x))^2 = 3/8,   x ~ U[0, pi]

so the variance-matched constant noise is `sigma0 * sqrt(3/8) = 0.6124 * sigma0` -- which is
where that number in the protocol comes from. A constant-noise arm at `sigma0` would be a
*dose-extreme* arm, not a matched control, and comparing against it would confound annealing
with simply applying less noise.

`annealed_schedule_rms` is computed numerically over the run's actual steps, so the finite-T
matching can be checked against the locked asymptotic constant instead of assumed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: sqrt(3/8): the asymptotic variance-matching factor between a cosine-annealed schedule and
#: a constant one, locked in docs/06 and derived above.
CONSTANT_VARIANCE_FACTOR = math.sqrt(3.0 / 8.0)


def annealed_schedule(step: int, total_steps: int, sigma0: float) -> float:
    """sigma0 * 0.5 * (1 + cos(pi t / T)); exactly 0.0 at the final step."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if step < 1 or step > total_steps:
        raise ValueError(f"step must be in 1..{total_steps}, got {step}")
    if sigma0 < 0:
        raise ValueError(f"sigma0 must be non-negative, got {sigma0}")
    if step == total_steps:
        return 0.0
    return sigma0 * 0.5 * (1.0 + math.cos(math.pi * step / total_steps))


def constant_schedule(step: int, total_steps: int, sigma: float) -> float:
    """Constant noise: the same scale on every update, including the last."""
    if step < 1 or step > total_steps:
        raise ValueError(f"step must be in 1..{total_steps}, got {step}")
    return float(sigma)


def annealed_schedule_rms(sigma0: float, total_steps: int) -> float:
    """sqrt(mean_t sigma(t)^2) over the run's steps -- the quantity being matched."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    total = sum(annealed_schedule(t, total_steps, sigma0) ** 2
                for t in range(1, total_steps + 1))
    return math.sqrt(total / total_steps)


def variance_match_report(sigma0: float, total_steps: int) -> dict:
    """The locked matching factor against the one this run actually realizes.

    `sqrt(3/8)` is the *asymptotic* value: it is the continuous average of
    `(0.5(1 + cos x))^2` over one period. A finite schedule sums over the discrete steps
    `t = 1..T`, and because the schedule ends at 0 the discrete average is slightly lower --
    0.6121 rather than 0.6124 at T = 1526, a 0.04% difference. That is far below anything a
    250M-token paired test can resolve, so the preregistered constant stands; the empirical
    value is recorded next to it so the match is documented rather than asserted.
    """
    empirical = annealed_schedule_rms(sigma0, total_steps) / sigma0 if sigma0 else 0.0
    return {
        "locked_factor": CONSTANT_VARIANCE_FACTOR,
        "empirical_factor": empirical,
        "delta_pct": 100.0 * (empirical / CONSTANT_VARIANCE_FACTOR - 1.0) if sigma0 else 0.0,
        "note": ("locked sqrt(3/8) is the T -> inf limit; empirical is sum over steps 1..T, "
                 "which converges to it from below"),
    }


@dataclass(frozen=True)
class ArmSpec:
    """One arm: what it is, and the schedule function its controller uses."""

    name: str
    kind: str            # "baseline" | "annealed" | "constant"
    sigma: float
    description: str

    def sigma_at(self, step: int, total_steps: int) -> float:
        if self.kind == "annealed":
            return annealed_schedule(step, total_steps, self.sigma)
        if self.kind == "constant":
            return constant_schedule(step, total_steps, self.sigma)
        if self.kind == "baseline":
            return 0.0
        raise ValueError(f"unknown arm kind {self.kind!r}")

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "sigma": self.sigma,
                "description": self.description}


def baseline_arm() -> ArmSpec:
    """C0. Still wrapped by the controller, with sigma 0, so every arm runs one code path."""
    return ArmSpec("C0", "baseline", 0.0,
                   "deterministic baseline; perturbation attached but inert (sigma = 0)")


def annealed_arm(sigma0: float, label: str = "C") -> ArmSpec:
    return ArmSpec(f"{label}{{{sigma0:g}}}", "annealed", sigma0,
                   f"annealed noise, sigma0 = {sigma0:g}, cosine to exactly 0 at the endpoint")


def constant_matched_arm(sigma0: float) -> ArmSpec:
    """C4 variance-matched: the comparator a claim about *annealing* rests on."""
    return ArmSpec("C4-matched", "constant", sigma0 * CONSTANT_VARIANCE_FACTOR,
                   f"constant noise at sigma0 * sqrt(3/8) = {sigma0 * CONSTANT_VARIANCE_FACTOR:.6f} "
                   f"(variance-matched to the annealed arm at sigma0 = {sigma0:g})")


def constant_dose_extreme_arm(sigma0: float) -> ArmSpec:
    """C4 second variant: constant noise at sigma0. Robustness row, not the matched control."""
    return ArmSpec("C4-dose", "constant", sigma0,
                   f"constant noise at sigma0 = {sigma0:g} (the dose extreme, kept for "
                   f"robustness; NOT variance-matched)")
