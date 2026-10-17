"""Tests for the arms, the config, and the training loop.

The loop is exercised on synthetic data at a size that runs in seconds, so the properties
that matter at 250M tokens are still checkable in CI: the budget arithmetic, the
variance-matching that makes C4 a fair control, one-code-path behaviour across arms, and --
the one that would silently ruin a long run -- that a resumed run continues the identical
trajectory rather than a similar one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _strict import import_torch_or_skip  # noqa: E402

torch = import_torch_or_skip()

from harness.arms import (  # noqa: E402
    CONSTANT_VARIANCE_FACTOR,
    annealed_arm,
    annealed_schedule,
    annealed_schedule_rms,
    baseline_arm,
    constant_dose_extreme_arm,
    constant_matched_arm,
    constant_schedule,
    variance_match_report,
)
from harness.config import BudgetConfig, EvalConfig, RunConfig  # noqa: E402
from harness.loop import Ledger, denormalize_state_dict, normalize_state_dict, train  # noqa: E402
from harness.model import CANDIDATE_CONFIGS, ModelConfig, TinyDecoder  # noqa: E402

SMOKE = CANDIDATE_CONFIGS["smoke_d64_l2"]        # d64, 2 layers, max_seq_len 64
CTX = 64
TINY_BUDGET = dict(target_tokens=128 * 4, batch_sequences=2, grad_accum=1, ctx=CTX)


# ----------------------------------------------------------------- budget arithmetic
def test_the_locked_budget_realizes_the_specified_step_count():
    """80 x accum 4 x ctx 512 = 163,840 tokens/step; 250M tokens is 1526 steps."""
    b = BudgetConfig()
    assert b.tokens_per_step == 80 * 4 * 512 == 163_840
    assert b.total_steps == 1526
    assert b.realized_tokens == 250_019_840
    assert abs(b.realized_tokens - b.target_tokens) / b.target_tokens < 1e-4


def test_the_batch_shape_is_the_one_decided_not_a_default():
    """The step count is a design choice, so it is pinned rather than left to a default.

    ASWP perturbs once per optimizer update, so this number is how many annealing events a
    run contains. The reference recipe's 80 x accum 16 would give 381 updates; the decided
    configuration gives 1526 at the same token cost.
    """
    decided = BudgetConfig()
    reference_recipe = BudgetConfig(grad_accum=16)
    assert reference_recipe.total_steps == 381
    assert decided.total_steps > 4 * reference_recipe.total_steps


def test_budget_arithmetic_is_reported_both_ways():
    d = BudgetConfig().to_dict()
    for key in ("tokens_per_step", "total_steps", "realized_tokens", "realized_vs_target_pct"):
        assert key in d
    assert d["realized_vs_target_pct"] == pytest.approx(100.007, abs=0.01)


# ------------------------------------------------------------------- noise schedules
def test_annealed_schedule_endpoints_and_shape():
    T = 100
    assert annealed_schedule(T, T, 0.03) == 0.0
    assert annealed_schedule(1, T, 0.03) < 0.03
    assert annealed_schedule(T // 2, T, 0.03) == pytest.approx(0.015)
    values = [annealed_schedule(t, T, 0.03) for t in range(1, T + 1)]
    assert all(b <= a + 1e-15 for a, b in zip(values, values[1:]))


def test_constant_schedule_does_not_anneal():
    """The comparator's defining property: it is still noisy at the endpoint."""
    T = 100
    assert constant_schedule(T, T, 0.02) == 0.02
    assert constant_schedule(1, T, 0.02) == constant_schedule(T, T, 0.02)


def test_variance_matching_factor_is_three_eighths_of_sigma_squared():
    """C4 is variance-matched, and the factor is derived rather than chosen.

    mean_t sigma(t)^2 / sigma0^2 -> 3/8 as T grows, so the matched constant scale is
    sigma0 * sqrt(3/8). The locked value is the T -> inf limit; the discrete sum over the
    run's steps converges to it *from below*, which is why the test asserts the direction of
    the finite-T delta rather than pretending it is zero.
    """
    assert CONSTANT_VARIANCE_FACTOR == pytest.approx(0.6123724356957945)
    sigma0 = 0.03

    # convergence from below, monotonically
    factors = [annealed_schedule_rms(sigma0, T) / sigma0 for T in (100, 1526, 5000, 50000)]
    assert factors == sorted(factors), "the discrete factor must approach the limit upward"
    assert all(f < CONSTANT_VARIANCE_FACTOR for f in factors)
    assert factors[-1] == pytest.approx(CONSTANT_VARIANCE_FACTOR, rel=1e-4)

    # at the run's actual length the gap is negligible (0.044%), which is why the
    # preregistered constant is used as-is and the empirical value is only reported
    at_run_length = annealed_schedule_rms(sigma0, 1526) / sigma0
    assert at_run_length == pytest.approx(CONSTANT_VARIANCE_FACTOR, rel=1e-3)
    report = variance_match_report(sigma0, 1526)
    assert report["empirical_factor"] == pytest.approx(at_run_length)
    assert -0.1 < report["delta_pct"] < 0.0

    matched = constant_matched_arm(sigma0)
    assert matched.sigma == pytest.approx(sigma0 * CONSTANT_VARIANCE_FACTOR)
    assert matched.sigma < sigma0, "the matched control must not be the dose extreme"

    # the dose-extreme arm is retained but is explicitly not the matched control
    extreme = constant_dose_extreme_arm(sigma0)
    assert extreme.sigma == sigma0
    assert "NOT variance-matched" in extreme.description


def test_a_constant_arm_at_sigma0_would_not_match_the_annealed_dose():
    """The failure this prevents: comparing annealing against the wrong amount of noise."""
    sigma0 = 0.03
    matched_sq = constant_matched_arm(sigma0).sigma ** 2
    annealed_sq = annealed_schedule_rms(sigma0, 1526) ** 2
    assert matched_sq == pytest.approx(annealed_sq, rel=5e-3)
    # at the dose extreme the noise energy is 1/(3/8) = 2.67x too high
    assert (sigma0 ** 2) / matched_sq == pytest.approx(1 / (3 / 8), rel=1e-3)


def test_arms_share_the_paired_seed_by_construction():
    """Both arms of a pair read the same per-role seeds from the frozen manifest."""
    cfg_a = RunConfig(paired_seed=1857649130, arm=baseline_arm(), model=SMOKE,
                      budget=BudgetConfig(**TINY_BUDGET))
    cfg_b = RunConfig(paired_seed=1857649130, arm=annealed_arm(0.03), model=SMOKE,
                      budget=BudgetConfig(**TINY_BUDGET))
    assert cfg_a.seeds() == cfg_b.seeds()
    assert set(cfg_a.seeds()) == {"global", "init", "data-loader", "dropout", "perturbation"}
    assert cfg_a.run_id != cfg_b.run_id


def test_config_rejects_settings_that_would_be_protocol_changes():
    def cfg(**over):
        base = dict(paired_seed=1857649130, model=SMOKE, budget=BudgetConfig(**TINY_BUDGET))
        base.update(over)
        return RunConfig(**base)

    assert cfg().validate() == []

    from harness.config import OptimizerConfig
    assert any("protocol change" in p
               for p in cfg(optimizer=OptimizerConfig(lr_schedule="cosine")).validate())

    dropout_model = ModelConfig(d_model=64, n_layers=2, n_heads=4, max_seq_len=64, dropout=0.1)
    assert any("C5b" in p for p in cfg(model=dropout_model).validate())

    assert any("baseline" in p for p in cfg(arm=constant_matched_arm(0.0)).validate())


def test_unknown_paired_seed_is_refused():
    cfg = RunConfig(paired_seed=12345, model=SMOKE, budget=BudgetConfig(**TINY_BUDGET))
    with pytest.raises(KeyError, match="not in the frozen manifest"):
        cfg.seeds()


# --------------------------------------------------------------- checkpoint plumbing
def test_state_dict_keys_are_independent_of_wrapping():
    """A checkpoint must not encode whether the perturbation happened to be attached."""
    model = TinyDecoder(SMOKE)
    plain = model.state_dict()
    from harness.perturbation import AnnealedPerturbation

    AnnealedPerturbation(model, sigma0=0.02, total_steps=5, perturbation_seed=1)
    wrapped = model.state_dict()
    assert any(".base." in k for k in wrapped), "wrapping should move the parameters"
    assert set(normalize_state_dict(wrapped)) == set(plain)
    restored = denormalize_state_dict(normalize_state_dict(wrapped), model)
    assert set(restored) == set(wrapped)
    for key, tensor in restored.items():
        assert torch.equal(tensor, wrapped[key]), key


# ---------------------------------------------------------------- the loop, end to end
def synthetic_streams(n_tokens: int = 4096) -> dict:
    rng = np.random.default_rng(0)
    ids = rng.integers(0, CANDIDATE_CONFIGS["smoke_d64_l2"].vocab_size,
                       size=n_tokens).astype(np.uint16)
    meta = {"stream_id": "synthetic", "bytes_per_token": 4.0, "pristine": False}
    return {"confirmatory-primary": (ids, meta), "reference-eval": (ids, meta)}


def tiny_cfg(arm, tmp_path, steps: int = 4, **over) -> RunConfig:
    budget = BudgetConfig(target_tokens=128 * steps, batch_sequences=2, grad_accum=1, ctx=CTX)
    kwargs = dict(paired_seed=1857649130, arm=arm, model=SMOKE, budget=budget,
                  evaluation=EvalConfig(eval_every_steps=0), out_dir=str(tmp_path))
    kwargs.update(over)
    return RunConfig(**kwargs)


def train_ids(n_tokens: int = 4096) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, SMOKE.vocab_size, size=n_tokens).astype(np.uint16)


def losses_from(ledger_path: Path) -> list[tuple[int, float]]:
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    return [(r["step"], r["train_loss"]) for r in rows if "train_loss" in r]


def test_a_tiny_run_produces_a_complete_ledger_and_summary(tmp_path):
    cfg = tiny_cfg(baseline_arm(), tmp_path / "a")
    summary = train(cfg, device="cpu", train_ids=train_ids(), eval_streams=synthetic_streams(),
                    out_root=tmp_path / "a")
    run_dir = tmp_path / "a" / cfg.run_id

    assert (run_dir / "config.json").exists()
    assert (run_dir / "data_manifest.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "checkpoint" / "final.pt").exists()
    assert len(losses_from(run_dir / "metrics.jsonl")) == cfg.total_steps == 4

    assert summary["steps"] == 4
    assert summary["tokens_seen"] == 4 * cfg.tokens_per_step
    assert summary["endpoint_sigma"] == 0.0, "the baseline endpoint is exactly unperturbed"
    assert "confirmatory-primary" in summary["endpoint"]
    assert summary["endpoint"]["confirmatory-primary"]["n_tokens_scored"] > 0
    assert summary["param_counts"]["total"] > 0
    assert summary["git_commit"]


def test_the_baseline_is_reproducible_and_the_arm_changes_the_run(tmp_path):
    """Same config twice must match exactly; a different arm must not.

    Without the second half, a bug that made every arm identical would pass.
    """
    cfg = tiny_cfg(baseline_arm(), tmp_path / "a")
    s1 = train(cfg, device="cpu", train_ids=train_ids(), eval_streams=synthetic_streams(),
               out_root=tmp_path / "a1")
    s2 = train(cfg, device="cpu", train_ids=train_ids(), eval_streams=synthetic_streams(),
               out_root=tmp_path / "a2")
    l1 = [l for _, l in losses_from(tmp_path / "a1" / cfg.run_id / "metrics.jsonl")]
    l2 = [l for _, l in losses_from(tmp_path / "a2" / cfg.run_id / "metrics.jsonl")]
    assert l1 == l2, "the same config must reproduce the same losses bit for bit"
    assert s1["endpoint"]["confirmatory-primary"]["nll_nats_per_token"] == \
        s2["endpoint"]["confirmatory-primary"]["nll_nats_per_token"]

    noisy = tiny_cfg(annealed_arm(0.05), tmp_path / "b")
    train(noisy, device="cpu", train_ids=train_ids(), eval_streams=synthetic_streams(),
          out_root=tmp_path / "b")
    l3 = [l for _, l in losses_from(tmp_path / "b" / noisy.run_id / "metrics.jsonl")]
    assert l3 != l1, "an annealed arm must actually perturb the run"


def test_resume_continues_the_identical_trajectory(tmp_path):
    """Phase 0 item 7. A resumed run must not be merely similar to an uninterrupted one."""
    ids = train_ids()
    cfg = tiny_cfg(annealed_arm(0.03), tmp_path / "a", steps=6)

    uninterrupted = train(cfg, device="cpu", train_ids=ids, eval_streams=synthetic_streams(),
                          out_root=tmp_path / "full")
    full_losses = dict(losses_from(tmp_path / "full" / cfg.run_id / "metrics.jsonl"))

    split_root = tmp_path / "split"
    train(cfg, device="cpu", train_ids=ids, eval_streams=synthetic_streams(),
          out_root=split_root, checkpoint_every=3)
    ckpt = split_root / cfg.run_id / "checkpoint" / "step_3.pt"
    assert ckpt.exists()

    train(cfg, device="cpu", train_ids=ids, eval_streams=synthetic_streams(),
          out_root=split_root, resume_from=ckpt)
    split_losses = dict(losses_from(split_root / cfg.run_id / "metrics.jsonl"))

    for step in (1, 2, 3):
        assert split_losses[step] == pytest.approx(full_losses[step])
    for step in (4, 5, 6):
        assert split_losses[step] == pytest.approx(full_losses[step]), (
            f"step {step} diverged after resume: {split_losses[step]} != {full_losses[step]}"
        )


def test_interrupt_then_resume_matches_an_uninterrupted_run(tmp_path):
    """The preemption case: same config, interrupted at a step boundary, resumed later.

    This is the scenario the first version of this check got wrong -- it resumed with a
    *different* step budget, which is not the same run at all (see the next test).
    """
    ids = train_ids()
    cfg = tiny_cfg(annealed_arm(0.03), tmp_path / "a", steps=4)

    full_dir = tmp_path / "full"
    train(cfg, device="cpu", train_ids=ids, eval_streams=synthetic_streams(),
          out_root=full_dir, log_every=4)
    ref = dict(losses_from(full_dir / cfg.run_id / "metrics.jsonl"))

    split_dir = tmp_path / "split"
    partial = train(cfg, device="cpu", train_ids=ids, eval_streams=synthetic_streams(),
                    out_root=split_dir, stop_after=2, log_every=4)
    assert partial["interrupted"] is True
    assert partial["steps_completed"] == 2
    ckpt = split_dir / cfg.run_id / "checkpoint" / "interrupt.pt"
    assert ckpt.exists()

    train(cfg, device="cpu", train_ids=ids, eval_streams=synthetic_streams(),
          out_root=split_dir, resume_from=ckpt, log_every=4)
    got = dict(losses_from(split_dir / cfg.run_id / "metrics.jsonl"))

    assert set(got) == {1, 2, 3, 4}
    for step in (1, 2, 3, 4):
        assert got[step] == ref[step], f"step {step}: {got[step]} != {ref[step]}"


def test_resuming_with_a_different_step_budget_is_refused(tmp_path):
    """The schedule is defined over the run length, so a changed budget is not the same run."""
    ids = train_ids()
    cfg4 = tiny_cfg(annealed_arm(0.03), tmp_path / "a", steps=4)
    train(cfg4, device="cpu", train_ids=ids, eval_streams=synthetic_streams(),
          out_root=tmp_path / "four", stop_after=2, log_every=9)
    ckpt = tmp_path / "four" / cfg4.run_id / "checkpoint" / "interrupt.pt"

    cfg6 = tiny_cfg(annealed_arm(0.03), tmp_path / "b", steps=6)
    with pytest.raises(ValueError, match="annealing schedule is defined over the run length"):
        train(cfg6, device="cpu", train_ids=ids, eval_streams=synthetic_streams(),
              out_root=tmp_path / "six", resume_from=ckpt, log_every=9)


def test_a_non_finite_loss_is_flagged_rather_than_silently_dropped(tmp_path):
    """Item 9: the run is retained and marked, and the poisoned update is skipped."""
    from harness.model import TinyDecoder as _Tiny

    cfg = tiny_cfg(annealed_arm(0.03), tmp_path / "a", steps=3)
    ids = train_ids(2048)
    original = _Tiny.hidden_states
    calls = {"n": 0}

    def poisoned(self, tokens):                     # noqa: ANN001
        calls["n"] += 1
        out = original(self, tokens)
        return out * float("nan") if calls["n"] == 2 else out

    # the loop computes the loss through hidden_states -> logits_from_hidden, so that is the
    # method to poison; poisoning forward() would no longer intercept the training path
    _Tiny.hidden_states = poisoned
    try:
        summary = train(cfg, device="cpu", train_ids=ids, eval_streams=synthetic_streams(),
                        out_root=tmp_path / "a", log_every=9)
    finally:
        _Tiny.hidden_states = original

    rows = Ledger(tmp_path / "a" / cfg.run_id).read()
    flagged = [r for r in rows if r.get("event") == "non_finite_loss"]
    skipped = [r for r in rows if r.get("optimizer_step_taken") is False]
    assert len(flagged) == 1 and len(skipped) == 1
    assert summary["instability_events"] == 1
    assert summary["steps"] == 3, "the run must complete and be retained, not dropped"
    assert summary["endpoint"]["confirmatory-primary"]["nll_nats_per_token"] > 0


def test_chunked_loss_equals_the_unchunked_loss():
    """The vocabulary-slice must not change the loss or the gradients.

    This is the equivalence that makes the chunking safe to use for real runs: the head is a
    per-position linear map and the loss is a mean over tokens, so slicing changes only the
    summation order. The chunk size is deliberately not a divisor of the token count, so the
    final partial chunk is exercised too.
    """
    from harness.loop import chunked_lm_loss

    tokens = torch.from_numpy(train_ids(4 * CTX).astype(np.int64)).reshape(4, CTX)
    model = TinyDecoder(SMOKE)

    logits = model(tokens)
    full = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
    full.backward()
    full_loss = float(full.detach())
    full_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                  if p.grad is not None}
    assert full_grads, "the unchunked path must produce gradients"

    model.zero_grad(set_to_none=True)
    chunked_loss = chunked_lm_loss(model, tokens, chunk_tokens=7,
                                   scale=1.0 / 2.0)   # a scale, as accumulation would use
    chunked_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                     if p.grad is not None}

    assert chunked_loss == pytest.approx(full_loss, rel=1e-5)
    assert set(chunked_grads) == set(full_grads)
    for name, grad in full_grads.items():
        assert torch.allclose(chunked_grads[name], grad * 0.5, atol=1e-6, rtol=1e-4), name
    # everything above was driven by position-slicing the head only; the body ran once
    assert model.hidden_states(tokens).shape == (4, CTX, SMOKE.d_model)


def test_a_single_token_chunk_still_matches():
    """The extreme case, which would expose any per-chunk state that is not reset."""
    from harness.loop import chunked_lm_loss

    tokens = torch.from_numpy(train_ids(2 * CTX).astype(np.int64)).reshape(2, CTX)
    reference = TinyDecoder(SMOKE)
    chunked_model = TinyDecoder(SMOKE)
    chunked_model.load_state_dict(reference.state_dict())

    logits = reference(tokens)
    expected = float(torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)).detach())

    got = chunked_lm_loss(chunked_model, tokens, chunk_tokens=1)
    assert got == pytest.approx(expected, rel=1e-5)


def test_bf16_forward_and_backward_survive_the_attention_path():
    """The bug this pins: fp32 RoPE tables silently upcast q and k, leaving v in bf16.

    CUDA's scaled_dot_product_attention rejects that mix outright; CPU promotes it silently, so
    this check is worth having on any device, and the GPU cost segment is where it caught it.
    """
    model = TinyDecoder(SMOKE).to(torch.bfloat16)
    tokens = torch.from_numpy(train_ids(2 * CTX).astype(np.int64)).reshape(2, CTX)
    hidden = model.hidden_states(tokens)
    assert hidden.dtype == torch.bfloat16, "the body must not silently upcast to fp32"
    loss = torch.nn.functional.cross_entropy(
        model.logits_from_hidden(hidden[:, :-1, :]).reshape(-1, SMOKE.vocab_size).float(),
        tokens[:, 1:].reshape(-1))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())


def test_resume_refuses_a_different_batch_shape_with_the_same_step_count(tmp_path):
    """The trap `run_id` cannot catch, and the one the OOM decision walked straight toward.

    80 x accum 4 and 40 x accum 8 realize the identical token count and therefore the identical
    step count, so they share a run id -- but the gradient is averaged over a different number
    of microbatches, so they are different experiments. Resuming across them would continue one
    run's state under another run's identity and report it as the requested run.
    """
    from harness.config import trajectory_mismatches

    a = tiny_cfg(baseline_arm(), tmp_path / "a", steps=2)
    # 2 x accum 1 and 1 x accum 2: identical tokens/step and identical step count.
    b = tiny_cfg(baseline_arm(), tmp_path / "b", steps=2,
                 budget=BudgetConfig(target_tokens=128 * 2, batch_sequences=1, grad_accum=2,
                                     ctx=CTX))
    assert a.budget.batch_sequences != b.budget.batch_sequences
    assert a.run_id == b.run_id, "run_id is expected to be blind to the microbatch shape"
    assert a.total_steps == b.total_steps
    assert a.tokens_per_step == b.tokens_per_step

    problems = trajectory_mismatches(a.to_dict(), b)
    assert any("batch_sequences" in p for p in problems), problems
    assert any("grad_accum" in p for p in problems), problems


def test_resume_refuses_a_different_precision(tmp_path):
    """bf16 and fp32 share a run id and differ in every gradient."""
    from harness.config import trajectory_mismatches

    a = tiny_cfg(baseline_arm(), tmp_path / "a", steps=2)
    b = tiny_cfg(baseline_arm(), tmp_path / "b", steps=2)
    b = RunConfig(**{**b.__dict__, "precision": "bf16"})
    assert a.run_id == b.run_id
    assert any("precision" in p for p in trajectory_mismatches(a.to_dict(), b))


def test_resume_refuses_a_checkpoint_from_a_different_arm(tmp_path):
    """End to end: the refusal happens before any state is loaded."""
    import torch

    from harness.loop import read_checkpoint

    ids = train_ids(2048)
    a = tiny_cfg(baseline_arm(), tmp_path / "a", steps=2)
    train(a, device="cpu", train_ids=ids, log_every=10, checkpoint_every=1)
    ckpt_path = sorted(p for p in (tmp_path / "a" / a.run_id / "checkpoint").glob("step_*.pt"))[0]

    b = tiny_cfg(annealed_arm(0.03), tmp_path / "b", steps=2)
    fresh_model = TinyDecoder(b.model)
    before = {k: v.clone() for k, v in fresh_model.state_dict().items()}
    with pytest.raises(ValueError, match="refusing to resume"):
        train(b, device="cpu", train_ids=ids, resume_from=ckpt_path, log_every=10)
    # the refusal fired before loading, so the model it would have resumed into is untouched
    for k, v in fresh_model.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert read_checkpoint(ckpt_path)["step"] == 1


def test_resume_refuses_a_different_token_stream(tmp_path):
    """A same-shaped stream with different contents: the sampler's block count agrees."""
    from harness.loop import resume_mismatches

    cfg = tiny_cfg(baseline_arm(), tmp_path / "a", steps=2)
    ckpt = {"config": cfg.to_dict(),
            "data_manifest_hashes": {"stream_id": "tinystories-train-aaaa",
                                     "token_stream_sha256": "a" * 64}}
    problems = resume_mismatches(ckpt, cfg, {"stream_id": "tinystories-train-bbbb",
                                             "token_stream_sha256": "b" * 64})
    assert any("stream_id" in p for p in problems)
    assert any("token_stream_sha256" in p for p in problems)
    # and an identical stream is not a problem
    assert resume_mismatches(ckpt, cfg, dict(ckpt["data_manifest_hashes"])) == []


def test_a_resumed_run_records_the_fingerprint_it_resumed_as(tmp_path):
    """The ledger must say which trajectory the resumed segment belongs to."""
    from harness.config import trajectory_fingerprint

    ids = train_ids(2048)
    cfg = tiny_cfg(baseline_arm(), tmp_path / "a", steps=2)
    train(cfg, device="cpu", train_ids=ids, log_every=10, checkpoint_every=1,
          stop_after=1)
    ckpt = tmp_path / "a" / cfg.run_id / "checkpoint" / "interrupt.pt"
    train(cfg, device="cpu", train_ids=ids, resume_from=ckpt, log_every=10)

    events = [json.loads(line) for line in
              (tmp_path / "a" / cfg.run_id / "metrics.jsonl").read_text().splitlines()]
    resume_events = [e for e in events if e.get("event") == "resumed"]
    assert len(resume_events) == 1
    assert resume_events[0]["from_step"] == 1
    assert resume_events[0]["trajectory_fingerprint"] == trajectory_fingerprint(cfg.to_dict())


def test_the_ledger_records_what_a_budget_decision_needs(tmp_path):
    """Throughput and memory per step are the numbers that decide n=6 vs n=8."""
    cfg = tiny_cfg(baseline_arm(), tmp_path / "a")
    train(cfg, device="cpu", train_ids=train_ids(), eval_streams=synthetic_streams(),
          out_root=tmp_path / "a")
    rows = Ledger(tmp_path / "a" / cfg.run_id).read()
    first = rows[0]
    for key in ("step", "tokens_seen", "sigma", "train_loss", "grad_norm",
                "step_seconds", "tokens_per_second"):
        assert key in first, key
    assert first["tokens_per_second"] > 0
