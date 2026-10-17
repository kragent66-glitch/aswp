"""ASWP: annealed stochastic weight perturbation.

The intervention, per docs/01:

    W_eff(t) = W + sigma(t) * RMS(W) * Z,      Z ~ N(0, I)

with `sigma(t) = sigma0 * 0.5 * (1 + cos(pi * t / T))` reaching **exactly zero** at the
final update, the noise constructed in fp32, the RMS scale detached from the graph (so
the optimizer updates only the base weight), and one perturbation per matrix per optimizer
update shared across microbatches of that update.

Three properties carry almost all the risk, and each is enforced here rather than assumed:

1. **Zero-noise equivalence.** With `sigma0 = 0`, or in eval mode, the wrapped module
   returns the *same tensor object* as the unwrapped one, so a zero-noise run is
   bit-identical to the baseline path rather than merely close to it. A run that is
   "nearly" the baseline cannot support a claim about the perturbation.
2. **Train-only.** Evaluation never perturbs, so the reported endpoint is a deterministic
   function of the weights.
3. **RNG isolation.** Every draw comes from a dedicated `torch.Generator`, so enabling the
   perturbation cannot shift the data-loader or dropout streams.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .model import ELIGIBLE_SUFFIXES


def cosine_sigma(step: int, total_steps: int, sigma0: float) -> float:
    """The annealing schedule, evaluated at the update about to be taken.

    Updates are numbered `1..T`. `step = T` returns **exactly** 0.0: the cosine term is
    short-circuited rather than relying on `cos(pi)` rounding to -1.0 in floating point,
    because the final step is the frozen endpoint and the model evaluated there must be
    the deterministic model.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if step < 1 or step > total_steps:
        raise ValueError(f"step must be in 1..{total_steps}, got {step}")
    if sigma0 < 0:
        raise ValueError(f"sigma0 must be non-negative, got {sigma0}")
    if step == total_steps:
        return 0.0
    return sigma0 * 0.5 * (1.0 + math.cos(math.pi * step / total_steps))


def realized_relative_perturbation(weight: torch.Tensor, perturbed: torch.Tensor) -> float:
    """||W_eff - W||_rms / ||W||_rms, measured *after* any dtype cast.

    Under the declared law `dW = sigma * RMS(W) * Z` this quantity is exactly
    `sigma * RMS(Z)`: the weight's own scale cancels, which is what makes RMS-relative
    scaling scale-free. So a run's realized perturbation should sit within a few percent
    of `sigma` for any matrix large enough that RMS(Z) concentrates around 1 -- and a
    measurement that drifts with the matrix's magnitude instead indicates an absolute
    step rather than an RMS-relative one.

    Phase 0 requires the realized perturbation to be measured on the tensor actually used
    in the matmul, not on the fp32 tensor it was constructed from: a bf16 cast quantises
    the noise, and that is a property of the run that should be recorded, not a detail.
    """
    w = weight.detach().float()
    d = (perturbed.detach().float() - w)
    denom = w.pow(2).mean().sqrt()
    if denom == 0:
        return float("inf")
    return float(d.pow(2).mean().sqrt() / denom)


class AnnealedLinear(nn.Module):
    """An `nn.Linear` whose forward uses the perturbed effective weight during training.

    The wrapped weight is the *same* `nn.Parameter` object, so an optimizer built before
    wrapping still sees it, and checkpoint keys are unchanged.
    """

    def __init__(self, base: nn.Linear, generator: torch.Generator, name: str = "") -> None:
        super().__init__()
        if base.bias is not None:
            raise ValueError("ASWP targets bias-free projections (docs/01)")
        self.base = base
        self.name = name
        self.generator = generator
        self.sigma = 0.0
        self._z: torch.Tensor | None = None

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    @property
    def perturbing(self) -> bool:
        """True only when a perturbation will actually be applied on the next forward."""
        return self.training and self.sigma != 0.0

    def start_step(self, sigma: float) -> None:
        """Begin one optimizer update: set the scale and draw that update's noise once.

        Drawing here (rather than lazily inside `forward`) makes "one perturbation per
        matrix per optimizer update, shared across microbatches" a property of the data
        flow rather than of call order, and it keeps the number of RNG draws independent
        of how many microbatches the update is split into.
        """
        self.sigma = float(sigma)
        if self.perturbing:
            w = self.base.weight
            if w.device != self.generator.device:
                raise RuntimeError(
                    f"{self.name}: weight is on {w.device} but the perturbation generator "
                    f"is on {self.generator.device}; move the model before wrapping it"
                )
            self._z = torch.randn(
                w.shape, generator=self.generator, dtype=torch.float32, device=w.device
            )
        else:
            self._z = None

    def clear_step(self) -> None:
        self.sigma = 0.0
        self._z = None

    def effective_weight(self) -> torch.Tensor:
        w = self.base.weight
        if not self.perturbing:
            return w  # the same object: zero-noise runs are bit-identical to baseline
        w32 = w.float()
        rms = w32.pow(2).mean().sqrt().detach()   # detached: no gradient path through scale
        assert self._z is not None
        return (w32 + self.sigma * rms * self._z).to(w.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.effective_weight()
        # Resolve the dtype here rather than leaving it to autocast, so the tensor whose
        # perturbation is measured is the tensor the matmul actually uses.
        if x.dtype != w.dtype:
            x = x.to(w.dtype)
        return F.linear(x, w, self.base.bias)


def eligible_module_names(model: nn.Module, eligible=ELIGIBLE_SUFFIXES) -> list[str]:
    """Dotted names of the projections ASWP may perturb, in sorted (deterministic) order.

    Raises if the model contains an eligible-sounding projection that is *not* a bias-free
    linear, because silently skipping it would make the treatment weaker than declared.
    """
    found = []
    for name, module in model.named_modules():
        if not any(name.endswith(suffix) for suffix in eligible):
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"eligible module {name} is {type(module).__name__}, not nn.Linear")
        if module.bias is not None:
            raise ValueError(f"eligible module {name} has a bias; docs/01 excludes biases")
        found.append(name)
    if not found:
        raise ValueError("no eligible projections found; check the model's module names")
    return sorted(found)


@dataclass
class PerturbationReport:
    """What the controller did, for the run ledger."""

    sigma0: float
    total_steps: int
    modules: list[str]
    steps_perturbed: int
    last_realized_ratio: dict[str, float]


class AnnealedPerturbation:
    """Owns the schedule, the dedicated generator, and the wrapped modules.

    Usage inside a training step:

        ctl.start_step(step)          # draws that update's noise once
        ...forward/backward/optimizer.step() for every microbatch...
        ctl.finish_step()             # optional: records the realized perturbation
    """

    def __init__(self, model: nn.Module, *, sigma0: float, total_steps: int,
                 perturbation_seed: int, eligible=ELIGIBLE_SUFFIXES) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self.model = model
        self.sigma0 = float(sigma0)
        self.total_steps = int(total_steps)
        # The generator lives on the model's device: a CPU generator cannot draw for a
        # CUDA weight, and the restored RNG state must resume on the same device.
        self.device = next(model.parameters()).device
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(perturbation_seed))
        self.modules: dict[str, AnnealedLinear] = {}
        self.steps_perturbed = 0
        self.last_realized_ratio: dict[str, float] = {}

        for name in eligible_module_names(model, eligible):
            parent_path, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, attr, AnnealedLinear(getattr(parent, attr), self.generator, name))
            self.modules[name] = getattr(parent, attr)

    # ------------------------------------------------------------------ lifecycle
    def sigma_at(self, step: int) -> float:
        return cosine_sigma(step, self.total_steps, self.sigma0)

    def start_step(self, step: int) -> float:
        """Set the update's scale and draw its noise. Returns the sigma used."""
        sigma = self.sigma_at(step)
        for module in self.modules.values():
            module.start_step(sigma)
        if sigma != 0.0:
            self.steps_perturbed += 1
        return sigma

    def finish_step(self) -> dict[str, float]:
        """Record the realized relative perturbation per module, then clear the step."""
        ratios = {}
        if any(m.perturbing for m in self.modules.values()):
            for name, module in self.modules.items():
                ratios[name] = realized_relative_perturbation(
                    module.base.weight, module.effective_weight()
                )
        self.last_realized_ratio = ratios
        for module in self.modules.values():
            module.clear_step()
        return ratios

    def disable(self) -> None:
        """Perturbation off (used for the baseline arm and for evaluation)."""
        for module in self.modules.values():
            module.clear_step()

    def report(self) -> PerturbationReport:
        return PerturbationReport(
            sigma0=self.sigma0,
            total_steps=self.total_steps,
            modules=sorted(self.modules),
            steps_perturbed=self.steps_perturbed,
            last_realized_ratio=dict(self.last_realized_ratio),
        )

    # ------------------------------------------------------------------ checks
    def state_dict(self) -> dict:
        return {
            "sigma0": self.sigma0,
            "total_steps": self.total_steps,
            "modules": sorted(self.modules),
            "generator_state": self.generator.get_state().tolist(),
            "steps_perturbed": self.steps_perturbed,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore the perturbation stream so a resumed run continues it, not restarts it."""
        if sorted(state["modules"]) != sorted(self.modules):
            raise ValueError("perturbation modules differ from the checkpoint's")
        self.sigma0 = float(state["sigma0"])
        self.total_steps = int(state["total_steps"])
        self.generator.set_state(
            torch.tensor(state["generator_state"], dtype=torch.uint8).to(self.device)
        )
        self.steps_perturbed = int(state["steps_perturbed"])
