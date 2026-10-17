"""Run configuration: every value a run depends on, explicit and serialized.

The point of this module is that a run directory is self-describing. `docs/06` requires the
tokenizer, budget, endpoint, optimizer and training settings to be frozen *before*
confirmatory training, and a config that is reconstructed from defaults at read time is not
frozen. So `RunConfig` refuses to infer anything silently: the budget is expressed in steps
and tokens together, the realized values are recorded alongside the requested ones, and the
per-role seeds come from the frozen manifest rather than from a fresh draw.

Locked values and where they come from:
  tokenizer          GPT-2 BPE, so token-level NLL is comparable to the reference family
                     (docs/07 section 3)
  context            512, forced by the reference family's untrained position embeddings
                     above row 511 (docs/07 section 4)
  optimizer          AdamW, lr 5e-4 constant, betas (0.9, 0.95), wd 0.1 -- the reference
                     recipe, so the benchmark row is not confounded by the optimizer
  dropout            0.0 in the primary arms; dropout is comparator C5b, not a default
  budget             ~250M tokens; the step count is derived and both are recorded
  grad clipping      1.0, applied identically to every arm
  endpoint           the final step, where the annealed schedule is exactly 0
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .arms import ArmSpec, baseline_arm
from .data import DATASET_REVISION, EOT_MARKER, EOT_TOKEN_ID, TOKENIZER_NAME
from .model import ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "config" / "manifest.json"
#: Roles the protocol requires to be independent RNG streams (docs/01, Reproducibility).
RNG_ROLES = ("global", "init", "data-loader", "dropout", "perturbation")


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 5e-4
    lr_schedule: str = "constant"
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    #: Recorded because the reference recipe does not document clipping; if this ever
    #: changes, it is a protocol deviation and the config says so.
    clip_is_our_choice: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d


@dataclass(frozen=True)
class BudgetConfig:
    """The token budget, and the step count that realizes it at this batch shape."""

    target_tokens: int = 250_000_000
    batch_sequences: int = 80
    grad_accum: int = 4
    ctx: int = 512

    @property
    def tokens_per_step(self) -> int:
        return self.batch_sequences * self.grad_accum * self.ctx

    @property
    def total_steps(self) -> int:
        return max(1, round(self.target_tokens / self.tokens_per_step))

    @property
    def realized_tokens(self) -> int:
        return self.total_steps * self.tokens_per_step

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update({
            "tokens_per_step": self.tokens_per_step,
            "total_steps": self.total_steps,
            "realized_tokens": self.realized_tokens,
            "realized_vs_target_pct": 100.0 * self.realized_tokens / self.target_tokens,
            "bpv_note": (
                "total_steps = round(target_tokens / tokens_per_step); the endpoint is the "
                "final step, so the realized token count is what the run actually sees"
            ),
        })
        return d


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation cadence and streams.

    The endpoint is the pristine stream; the shared stream is evaluated too, because that is
    the row that can be compared against the six public reference checkpoints (docs/07).
    """

    primary_role: str = "confirmatory-primary"
    comparability_role: str = "reference-eval"
    eval_every_steps: int = 250
    interim_eval_tokens: int = 1_000_000
    eval_batch_blocks: int = 16

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RunConfig:
    paired_seed: int
    arm: ArmSpec = field(default_factory=baseline_arm)
    model: ModelConfig = field(default_factory=ModelConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    precision: str = "fp32"           # "fp32" | "bf16"
    #: Logit-chunk size for the training loss. An implementation parameter, not an
    #: experimental one: the loss is a mean over tokens, so slicing the vocabulary
    #: projection changes nothing except the peak memory. Recorded because the run's
    #: numerics (summation order) depend on it.
    loss_chunk_tokens: int = 8192
    data_dir: str = "data"
    out_dir: str = "runs"
    stage: str = "phase0"             # "phase0" | "screen" | "confirm"
    notes: str = ""

    # ------------------------------------------------------------------ derived
    @property
    def tokens_per_step(self) -> int:
        return self.budget.tokens_per_step

    @property
    def total_steps(self) -> int:
        return self.budget.total_steps

    @property
    def run_id(self) -> str:
        """Stable identity: same config, same id. No clock, so a rerun is recognisable."""
        sigma = f"{self.arm.sigma:g}"
        return (f"{self.stage}-{self.arm.name.replace('{','').replace('}','')}-"
                f"seed{self.paired_seed}-d{self.model.d_model}l{self.model.n_layers}-"
                f"s{sigma}")

    def seeds(self, manifest_path: Path = MANIFEST_PATH) -> dict[str, int]:
        """The frozen per-role seeds for this paired seed, from config/manifest.json.

        Read rather than derived here: the manifest is the preregistered artifact, and a
        second derivation in the harness is exactly how a seed mapping drifts.
        """
        manifest = json.loads(manifest_path.read_text())
        table = manifest["per_seed_role_seeds"]
        key = str(self.paired_seed)
        if key not in table:
            raise KeyError(
                f"paired seed {self.paired_seed} is not in the frozen manifest "
                f"(have {sorted(table)})"
            )
        seeds = table[key]
        missing = [r for r in RNG_ROLES if r not in seeds]
        if missing:
            raise KeyError(f"manifest is missing role seeds {missing} for {key}")
        return {r: int(seeds[r]) for r in RNG_ROLES}

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "paired_seed": self.paired_seed,
            "arm": self.arm.to_dict(),
            "model": self.model.to_dict(),
            "budget": self.budget.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "precision": self.precision,
            "loss_chunk_tokens": self.loss_chunk_tokens,
            "data_dir": self.data_dir,
            "out_dir": self.out_dir,
            "notes": self.notes,
            "tokenizer": TOKENIZER_NAME,
            "dataset_revision": DATASET_REVISION,
            "eot_marker": EOT_MARKER,
            "eot_token_id": EOT_TOKEN_ID,
            "seeds": self.seeds(),
        }

    def validate(self) -> list[str]:
        """Reasons this config must not be run. Empty means it is internally coherent."""
        problems = []
        if self.model.max_seq_len < self.budget.ctx:
            problems.append(
                f"model max_seq_len {self.model.max_seq_len} < ctx {self.budget.ctx}"
            )
        if self.optimizer.lr_schedule != "constant":
            problems.append(
                "the locked recipe uses a constant learning rate; a schedule here would be a "
                "protocol change, not a setting"
            )
        if self.model.dropout != 0.0:
            problems.append(
                f"model dropout {self.model.dropout} must be 0 in the primary arms "
                f"(dropout is comparator C5b)"
            )
        if self.arm.kind == "constant" and abs(self.arm.sigma) < 1e-12:
            problems.append("a constant-noise arm at sigma 0 is the baseline; use kind=baseline")
        if self.arm.kind == "annealed" and self.arm.sigma <= 0:
            problems.append("an annealed arm needs a positive sigma0")
        if math.isnan(self.optimizer.max_grad_norm) or self.optimizer.max_grad_norm <= 0:
            problems.append("max_grad_norm must be positive")
        if self.precision not in {"fp32", "bf16"}:
            problems.append(f"unknown precision {self.precision!r}")
        return problems


def git_commit(repo_root: Path = REPO_ROOT) -> str | None:
    """The commit the run's code came from, recorded in the checkpoint and the manifest."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
                             text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None
