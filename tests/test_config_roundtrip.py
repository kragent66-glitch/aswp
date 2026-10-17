"""Config round-trip across the process boundary.

Enough of a hazard to deserve its own file: a run's config is serialized on the host, shipped
to a GPU container, and rebuilt there. If the rebuild does not produce an *identical* config,
the run that executes is not the run that was described -- and the failure mode is a
`TypeError`, or worse, a silently different `run_id` and a run directory nobody looks at.

These tests are torch-free so they run in the fast analysis CI job rather than only where a
GPU-sized dependency set is installed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.arms import annealed_arm, baseline_arm, constant_matched_arm  # noqa: E402
from harness.config import (  # noqa: E402
    BudgetConfig,
    EvalConfig,
    OptimizerConfig,
    RunConfig,
    constructor_kwargs,
)
from harness.model import CANDIDATE_CONFIGS  # noqa: E402

SMOKE = CANDIDATE_CONFIGS["smoke_d64_l2"]


def test_derived_keys_are_not_passed_to_the_constructor():
    """`BudgetConfig.to_dict()` carries derived values; the constructor does not take them.

    Feeding the payload straight back raised `TypeError: unexpected keyword argument
    'tokens_per_step'` -- the runner's first act after receiving a job description.
    """
    payload = BudgetConfig().to_dict()
    for derived in ("tokens_per_step", "total_steps", "realized_tokens",
                    "realized_vs_target_pct", "bpv_note"):
        assert derived in payload, f"{derived} should be recorded in the payload"
        assert derived not in constructor_kwargs(BudgetConfig, payload)

    rebuilt = BudgetConfig(**constructor_kwargs(BudgetConfig, payload))
    assert rebuilt == BudgetConfig(), "the inputs must survive the round trip exactly"
    assert rebuilt.total_steps == payload["total_steps"]


def test_every_config_dataclass_round_trips():
    for cls in (BudgetConfig, EvalConfig, OptimizerConfig):
        payload = cls().to_dict()
        rebuilt = cls(**constructor_kwargs(cls, payload))
        assert rebuilt == cls(), cls.__name__


def test_optimizer_betas_survive_json():
    """A tuple becomes a list in JSON; it must still rebuild into an equal config."""
    payload = json.loads(json.dumps(OptimizerConfig().to_dict()))
    rebuilt = OptimizerConfig(**constructor_kwargs(OptimizerConfig, payload))
    assert tuple(rebuilt.betas) == (0.9, 0.95)
    assert rebuilt.lr == 5e-4 and rebuilt.max_grad_norm == 1.0


def test_a_remote_rebuild_reproduces_the_same_run_identity():
    """The property the runner actually depends on: same config, same run_id, same steps."""
    original = RunConfig(paired_seed=1857649130, arm=annealed_arm(0.03), model=SMOKE,
                         budget=BudgetConfig(target_tokens=128 * 6, batch_sequences=2,
                                             grad_accum=1, ctx=64),
                         evaluation=EvalConfig())
    payload = json.loads(json.dumps(original.to_dict()))

    arm = payload["arm"]
    rebuilt = RunConfig(
        paired_seed=payload["paired_seed"],
        arm=annealed_arm(arm["sigma"]),
        model=SMOKE,
        budget=BudgetConfig(**constructor_kwargs(BudgetConfig, payload["budget"])),
        optimizer=OptimizerConfig(**constructor_kwargs(OptimizerConfig,
                                                       payload["optimizer"])),
        evaluation=EvalConfig(**constructor_kwargs(EvalConfig, payload["evaluation"])),
        precision=payload["precision"],
        stage=payload["stage"],
    )
    assert rebuilt.run_id == original.run_id
    assert rebuilt.total_steps == original.total_steps
    assert rebuilt.tokens_per_step == original.tokens_per_step
    assert rebuilt.seeds() == original.seeds()
    assert rebuilt.arm.kind == original.arm.kind


def test_the_three_confirmatory_arms_are_distinguishable_by_run_id():
    """A run directory must identify the arm, or two arms overwrite each other's ledger."""
    seed = 1857649130
    ids = {
        RunConfig(paired_seed=seed, arm=spec, model=SMOKE,
                  budget=BudgetConfig(target_tokens=128 * 4, batch_sequences=2, grad_accum=1,
                                      ctx=64)).run_id
        for spec in (baseline_arm(), annealed_arm(0.03), constant_matched_arm(0.03))
    }
    assert len(ids) == 3, ids


def test_invalid_configs_are_refused_before_a_gpu_is_allocated():
    """Cheap local validation, so a mis-specified job fails on the host and not in a container."""
    cfg = RunConfig(paired_seed=1857649130, arm=baseline_arm(), model=SMOKE,
                    budget=BudgetConfig(target_tokens=128 * 4, batch_sequences=2, grad_accum=1,
                                        ctx=64))
    assert cfg.validate() == []
    bad = RunConfig(paired_seed=1857649130, arm=annealed_arm(0.03), model=SMOKE,
                    budget=BudgetConfig(target_tokens=128 * 4, batch_sequences=2, grad_accum=1,
                                        ctx=1024))
    assert any("max_seq_len" in p for p in bad.validate())


def test_unknown_paired_seed_fails_loudly():
    cfg = RunConfig(paired_seed=7, arm=baseline_arm(), model=SMOKE,
                    budget=BudgetConfig(target_tokens=128 * 4, batch_sequences=2, grad_accum=1,
                                        ctx=64))
    with pytest.raises(KeyError):
        cfg.seeds()
