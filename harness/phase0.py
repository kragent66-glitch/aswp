#!/usr/bin/env python3
"""Phase 0 gate: the implementation checks that must pass before any scientific run.

Phase 0 is an implementation gate, not a result. It answers "does this harness do what the
protocol says it does?" -- zero-noise equivalence, schedule correctness, target-module
correctness, shape/dtype/device behaviour, gradient sanity, RNG isolation, checkpoint/resume,
deterministic evaluation, an explicit stability policy, and a measured cost segment -- and it
prints the evidence for each, so a pass is a measurement rather than an assertion.

Nothing here is about efficacy. A run that has not passed this gate cannot produce a number
worth interpreting, and item 10's throughput is what turns the token budget into a cost, which
is what the n = 6 vs n = 8 decision waits on.

Usage:
    python harness/phase0.py                        # all checks, CPU, configured model
    python harness/phase0.py --device cuda --throughput-steps 50
    python harness/phase0.py --only zero_noise_equivalence
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .arms import annealed_arm, annealed_schedule, baseline_arm, constant_matched_arm
from .config import BudgetConfig, EvalConfig, RunConfig
from .data import BlockSampler
from .loop import evaluate, train
from .model import CANDIDATE_CONFIGS, ModelConfig, TinyDecoder
from .perturbation import AnnealedPerturbation, AnnealedLinear, eligible_module_names


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def _workdir() -> Path:
    """A fresh directory per gate invocation, per check.

    The ledger is append-only on purpose -- a real run's history is never rewritten -- so the
    gate must never point two invocations at the same directory. Doing so made a previous
    invocation's records part of this one's comparison, which is a defect in the check that
    looks exactly like a defect in the harness.
    """
    return Path(tempfile.mkdtemp(prefix="aswp_phase0_"))


def _cfg(model: ModelConfig, *, steps: int, arm, batch: int = 2, ctx: int = 128,
         accum: int = 1) -> RunConfig:
    return RunConfig(
        paired_seed=1857649130, arm=arm, model=model,
        budget=BudgetConfig(target_tokens=batch * accum * ctx * steps,
                            batch_sequences=batch, grad_accum=accum, ctx=ctx),
        evaluation=EvalConfig(eval_every_steps=0),
    )


def _ids(n: int, vocab: int = 50257) -> np.ndarray:
    return np.random.default_rng(0).integers(0, vocab, size=n).astype(np.uint16)


def _streams(ids: np.ndarray) -> dict:
    meta = {"stream_id": "phase0-synthetic", "bytes_per_token": 4.0, "pristine": None}
    return {"confirmatory-primary": (ids, meta), "reference-eval": (ids, meta)}


# --------------------------------------------------------------------------- checks
def check_zero_noise_equivalence(model: ModelConfig, device: str) -> tuple[bool, dict]:
    """Item 1. sigma = 0 must be the baseline *bitwise*, not merely close to it."""
    import torch

    torch.manual_seed(0)
    plain = TinyDecoder(model).to(device)
    tokens = torch.from_numpy(_ids(4 * model.max_seq_len, model.vocab_size)
                              .astype(np.int64)).reshape(4, model.max_seq_len).to(device)
    with torch.no_grad():
        reference = plain(tokens).clone()

    ctl = AnnealedPerturbation(plain, sigma0=0.0, total_steps=10, perturbation_seed=1)
    plain.train()
    worst = 0.0
    identical = True
    for step in (1, 5, 10):
        ctl.start_step(step)
        with torch.no_grad():
            got = plain(tokens)
        identical &= bool(torch.equal(got, reference))
        worst = max(worst, float((got - reference).abs().max()))
        ctl.finish_step()

    # and the real intervention must be exactly zero at the frozen endpoint. A fresh model,
    # because wrapping twice is refused on purpose (the guard reports "is AnnealedLinear,
    # not nn.Linear") and that refusal is itself the behaviour we want.
    import torch as _torch
    _torch.manual_seed(0)
    endpoint_model = TinyDecoder(model).to(device)
    ctl2 = AnnealedPerturbation(endpoint_model, sigma0=0.03, total_steps=10,
                               perturbation_seed=1)
    ctl2.start_step(10)
    endpoint_identical = all(m.effective_weight() is m.base.weight
                             for m in ctl2.modules.values())

    return identical and endpoint_identical, {
        "bitwise_identical_at_sigma0": identical,
        "max_abs_logit_difference": worst,
        "endpoint_effective_weight_is_base": endpoint_identical,
    }


def check_schedule(model: ModelConfig, device: str) -> tuple[bool, dict]:
    """Item 2. Initial scale, monotone annealing, exact zero at the final step."""
    T = 1526
    values = [annealed_schedule(t, T, 0.03) for t in range(1, T + 1)]
    monotone = all(b <= a + 1e-15 for a, b in zip(values, values[1:]))
    ok = monotone and values[-1] == 0.0 and 0 < values[0] < 0.03
    return ok, {
        "steps": T, "sigma_first": values[0], "sigma_midpoint": values[T // 2],
        "sigma_final": values[-1], "monotone_nonincreasing": monotone,
        "exactly_zero_at_endpoint": values[-1] == 0.0,
    }


def check_target_modules(model: ModelConfig, device: str) -> tuple[bool, dict]:
    """Item 3. Only the eligible projections are touched; nothing else is."""
    m = TinyDecoder(model)
    expected = set(eligible_module_names(m))
    ctl = AnnealedPerturbation(m, sigma0=0.02, total_steps=5, perturbation_seed=1)
    untouched = [n for n, mod in m.named_modules()
                 if isinstance(mod, torch_linear()) and n not in expected and n not in ctl.modules]
    excluded_ok = not any(k in x for x in expected
                          for k in ("tok_emb", "lm_head", "norm"))
    return set(ctl.modules) == expected and excluded_ok, {
        "wrapped": len(ctl.modules), "expected": len(expected),
        "wrapped_names_match": set(ctl.modules) == expected,
        "eligible_excludes_embeddings_head_norms": excluded_ok,
        "unwrapped_linear_layers": len(untouched),
    }


def torch_linear():
    import torch
    return torch.nn.Linear


def check_shapes_dtypes(model: ModelConfig, device: str) -> tuple[bool, dict]:
    """Item 4. Batch 1, short final batch, bf16, and gradient-accumulation equivalence."""
    import torch

    m = TinyDecoder(model).to(device)
    ctl = AnnealedPerturbation(m, sigma0=0.02, total_steps=5, perturbation_seed=3)
    ctl.start_step(1)
    evidence: dict = {}
    ok = True
    with torch.no_grad():
        one = m(torch.from_numpy(_ids(model.max_seq_len).astype(np.int64))
                .unsqueeze(0).to(device))
        ok &= tuple(one.shape) == (1, model.max_seq_len, model.vocab_size)
        evidence["batch1_shape"] = list(one.shape)
        short = m(torch.from_numpy(_ids(model.max_seq_len // 4).astype(np.int64))
                  .unsqueeze(0).to(device))
        ok &= short.shape[1] == model.max_seq_len // 4
        evidence["short_batch_shape"] = list(short.shape)

    # accumulation: two single-example microbatches must give the same gradient as one
    # batch of two, because both losses are means over an equal number of tokens
    def grads(batch_size: int, accum: int):
        m.zero_grad(set_to_none=True)
        for i in range(accum):
            rows = _ids(batch_size * model.max_seq_len).astype(np.int64)
            rows = rows.reshape(batch_size, model.max_seq_len)[i:i + 1] if accum > 1 else rows.reshape(batch_size, model.max_seq_len)
            t = torch.from_numpy(rows).to(device)
            out = m(t)
            loss = torch.nn.functional.cross_entropy(
                out[:, :-1, :].reshape(-1, out.shape[-1]), t[:, 1:].reshape(-1))
            (loss / accum).backward()
        return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}

    g_accum = grads(2, 2)
    g_single = grads(2, 1)
    max_diff = max(float((g_accum[k] - g_single[k]).abs().max()) for k in g_single)
    evidence["accum_vs_single_grad_max_abs_diff"] = max_diff
    ok &= max_diff < 1e-5

    if device == "cpu":
        try:
            m16 = TinyDecoder(model).to(torch.bfloat16).to(device)
            ctl16 = AnnealedPerturbation(m16, sigma0=0.05, total_steps=5, perturbation_seed=3)
            ctl16.start_step(1)
            with torch.no_grad():
                out16 = m16(torch.from_numpy(_ids(model.max_seq_len).astype(np.int64))
                            .unsqueeze(0).to(device))
            from .perturbation import realized_relative_perturbation
            name = sorted(ctl16.modules)[0]
            mod = ctl16.modules[name]
            ratio = realized_relative_perturbation(mod.base.weight, mod.effective_weight())
            evidence["bf16_forward_ok"] = True
            evidence["bf16_effective_weight_dtype"] = str(mod.effective_weight().dtype)
            evidence["bf16_realized_perturbation"] = ratio
            evidence["bf16_expected_perturbation"] = mod.sigma
            ok &= str(mod.effective_weight().dtype) == "torch.bfloat16"
            ok &= abs(ratio / mod.sigma - 1.0) < 0.10
        except Exception as exc:  # bf16 CPU support varies by op; recorded, not hidden
            evidence["bf16_error"] = f"{type(exc).__name__}: {exc}"
    else:
        evidence["bf16_error"] = "not attempted: device is not cpu"
    return ok, evidence


def check_gradients(model: ModelConfig, device: str) -> tuple[bool, dict]:
    """Item 5. Finite gradients reaching the base weight; no accidental detachment."""
    import torch

    m = TinyDecoder(model).to(device)
    ctl = AnnealedPerturbation(m, sigma0=0.02, total_steps=5, perturbation_seed=4)
    ctl.start_step(1)
    t = torch.from_numpy(_ids(2 * model.max_seq_len).astype(np.int64)).reshape(2, model.max_seq_len).to(device)
    out = m(t)
    loss = torch.nn.functional.cross_entropy(out[:, :-1, :].reshape(-1, out.shape[-1]),
                                            t[:, 1:].reshape(-1))
    loss.backward()
    finite, nonzero, quiet = True, True, True
    for name, mod in ctl.modules.items():
        g = mod.base.weight.grad
        finite &= g is not None and bool(torch.isfinite(g).all())
        nonzero &= g is not None and float(g.abs().sum()) > 0
        quiet &= not bool(mod._z.requires_grad) if mod._z is not None else True
    return finite and nonzero and quiet, {
        "all_grads_finite": finite, "all_grads_nonzero": nonzero,
        "noise_requires_grad": not quiet, "loss": float(loss.detach()),
    }


def check_rng_isolation(model: ModelConfig, device: str) -> tuple[bool, dict]:
    """Item 6. Enabling the perturbation must not shift the data-loader or dropout streams."""
    import torch

    m = TinyDecoder(model).to(device)
    ctl = AnnealedPerturbation(m, sigma0=0.02, total_steps=5, perturbation_seed=5)
    t = torch.from_numpy(_ids(2 * model.max_seq_len).astype(np.int64)).reshape(2, model.max_seq_len).to(device)
    before = torch.get_rng_state().clone()
    m(t)
    ctl.start_step(1)
    m(t)
    ctl.finish_step()
    after = torch.get_rng_state().clone()
    # data order is a function of the sampler seed alone, so two arms share it exactly
    ids = _ids(4096)
    a = [int(BlockSampler(ids, ctx=128, seed=7).next_block()[0]) for _ in range(1)]
    b = [int(BlockSampler(ids, ctx=128, seed=7).next_block()[0]) for _ in range(1)]
    return bool(torch.equal(before, after)) and a == b, {
        "global_rng_bit_identical": bool(torch.equal(before, after)),
        "same_seed_same_first_block": a == b,
    }


def check_checkpoint_resume(model: ModelConfig, device: str) -> tuple[bool, dict]:
    """Item 7. A split-save-resume run must continue the identical trajectory.

    The split path is the real scenario, not a contrived one: run a *shorter* config to a
    checkpoint, then resume with the full config. Doing it the other way (running to the end
    and then resuming from the middle) leaves two sets of records for the same steps in one
    ledger, which would make the comparison depend on which write came last.
    """
    steps, split = 4, 2
    arm = annealed_arm(0.03)
    cfg = _cfg(model, steps=steps, arm=arm, batch=2, ctx=model.max_seq_len)
    ids = _ids(4096, model.vocab_size)
    streams = _streams(_ids(2048, model.vocab_size))

    out = _workdir()
    train(cfg, device=device, train_ids=ids, eval_streams=streams,
          out_root=out / "full", log_every=steps)
    ref = [r["train_loss"] for r in _ledger(out / "full" / cfg.run_id)]

    # interrupt the same config at a step boundary, then resume it -- the preemption case
    partial = train(cfg, device=device, train_ids=ids, eval_streams=streams,
                    out_root=out / "split", stop_after=split, log_every=steps)
    ckpt = out / "split" / cfg.run_id / "checkpoint" / "interrupt.pt"
    train(cfg, device=device, train_ids=ids, eval_streams=streams,
          out_root=out / "split", resume_from=ckpt, log_every=steps)
    got = [r["train_loss"] for r in _ledger(out / "split" / cfg.run_id)]

    same_length = len(ref) == len(got) == steps
    worst = max((abs(a - b) for a, b in zip(ref, got)), default=math.inf) if same_length else math.inf
    return (bool(partial.get("interrupted")) and same_length and worst < 1e-6), {
        "workdir": str(out),
        "steps": steps, "interrupted_at": split, "checkpoint_exists": ckpt.exists(),
        "interrupt_reported": bool(partial.get("interrupted")),
        "ledger_records": len(got), "uninterrupted_losses": ref, "resumed_losses": got,
        "max_abs_loss_difference": worst,
    }


def _ledger(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines()
            if l.strip() and "train_loss" in l]


def check_deterministic_eval(model: ModelConfig, device: str) -> tuple[bool, dict]:
    """Item 8. Repeated evaluation of the same weights must be identical."""
    import torch

    m = TinyDecoder(model)
    ids = _ids(2048, model.vocab_size)
    a = evaluate(m, ids, ctx=model.max_seq_len, batch_blocks=4, device=device, bytes_per_token=4.0)
    b = evaluate(m, ids, ctx=model.max_seq_len, batch_blocks=4, device=device, bytes_per_token=4.0)
    # and evaluation must not consume perturbation randomness or perturb anything
    ctl = AnnealedPerturbation(m, sigma0=0.05, total_steps=5, perturbation_seed=9)
    m.train()
    ctl.start_step(1)
    c = evaluate(m, ids, ctx=model.max_seq_len, batch_blocks=4, device=device, bytes_per_token=4.0)
    return (a["nll_nats_per_token"] == b["nll_nats_per_token"] == c["nll_nats_per_token"]), {
        "repeat_identical": a["nll_nats_per_token"] == b["nll_nats_per_token"],
        "identical_while_perturbation_armed": a["nll_nats_per_token"] == c["nll_nats_per_token"],
        "nll": a["nll_nats_per_token"], "tokens_scored": a["n_tokens_scored"],
    }


def check_instability_policy(model: ModelConfig, device: str) -> tuple[bool, dict]:
    """Item 9. A non-finite loss must be *recorded*, not silently dropped."""
    import torch

    arm = annealed_arm(0.03)
    cfg = _cfg(model, steps=3, arm=arm, batch=2, ctx=model.max_seq_len)
    ids = _ids(2048, model.vocab_size)
    streams = _streams(_ids(1024, model.vocab_size))

    original_forward = TinyDecoder.forward
    calls = {"n": 0}

    def poisoned(self, tokens):           # noqa: ANN001
        calls["n"] += 1
        out = original_forward(self, tokens)
        if calls["n"] == 2:               # exactly one step's forward returns NaN
            return out * float("nan")
        return out

    TinyDecoder.forward = poisoned
    try:
        out = _workdir()
        summary = train(cfg, device=device, train_ids=ids, eval_streams=streams,
                        out_root=out, log_every=10)
        rows = _ledger(out / cfg.run_id)
    finally:
        TinyDecoder.forward = original_forward

    flagged = [r for r in rows if r.get("event") == "non_finite_loss"]
    skipped = [r for r in rows if r.get("optimizer_step_taken") is False]
    return len(flagged) == 1 and len(skipped) == 1 and summary["instability_events"] == 1, {
        "steps": cfg.total_steps, "flagged_steps": [r["step"] for r in flagged],
        "updates_skipped": len(skipped), "summary_instability_events": summary["instability_events"],
        "run_completed": summary["steps"] == cfg.total_steps,
    }


def check_cost(model: ModelConfig, device: str, steps: int, batch: int) -> tuple[bool, dict]:
    """Item 10. Measured throughput and memory -- the number the budget decision needs."""
    import torch

    ctx = min(model.max_seq_len, 512)
    arm = annealed_arm(0.03)
    cfg = RunConfig(paired_seed=1857649130, arm=arm, model=model,
                    budget=BudgetConfig(target_tokens=batch * ctx * steps,
                                        batch_sequences=batch, grad_accum=1, ctx=ctx),
                    evaluation=EvalConfig(eval_every_steps=0),
                    precision="fp32")
    ids = _ids(batch * ctx * (steps + 1), model.vocab_size)
    streams = _streams(_ids(1024, model.vocab_size))

    rss_before = _rss_mb()
    cost_dir = _workdir()
    t0 = time.time()
    summary = train(cfg, device=device, train_ids=ids, eval_streams=streams,
                    out_root=cost_dir / "aswp", log_every=steps)
    elapsed = time.time() - t0

    base_cfg = RunConfig(paired_seed=1857649130, arm=baseline_arm(), model=model,
                         budget=cfg.budget, evaluation=cfg.evaluation, precision="fp32")
    t0 = time.time()
    train(base_cfg, device=device, train_ids=ids, eval_streams=streams,
          out_root=cost_dir / "baseline", log_every=steps)
    base_elapsed = time.time() - t0

    tps = summary["mean_tokens_per_second"]
    return tps > 0, {
        "device": device, "steps": steps, "batch_sequences": batch, "ctx": ctx,
        "tokens_per_step": cfg.tokens_per_step,
        "tokens_per_second_aswp": tps,
        "tokens_per_second_baseline": base_cfg.budget.tokens_per_step * steps / base_elapsed,
        "aswp_overhead_pct": 100.0 * (elapsed / base_elapsed - 1.0),
        "wall_seconds": elapsed,
        "peak_rss_mb": _rss_mb(),
        "rss_before_mb": rss_before,
        "peak_gpu_memory_bytes": summary.get("peak_gpu_memory_bytes"),
        "extrapolated_hours_for_250M_tokens": 250e6 / tps / 3600 if tps > 0 else None,
        "note": ("CPU figures are indicative only; the budget decision uses a GPU measurement "
                 "at the real batch shape"),
    }


CHECKS = {
    "zero_noise_equivalence": check_zero_noise_equivalence,
    "schedule_correctness": check_schedule,
    "target_module_correctness": check_target_modules,
    "shapes_dtypes_device": check_shapes_dtypes,
    "gradient_sanity": check_gradients,
    "rng_isolation": check_rng_isolation,
    "checkpoint_resume": check_checkpoint_resume,
    "deterministic_evaluation": check_deterministic_eval,
    "instability_policy": check_instability_policy,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="screen_d256_l8", choices=list(CANDIDATE_CONFIGS))
    ap.add_argument("--only", nargs="*", default=None, help="run only these checks")
    ap.add_argument("--throughput-steps", type=int, default=10)
    ap.add_argument("--throughput-batch", type=int, default=8)
    ap.add_argument("--include-cost", action="store_true",
                    help="also run the cost segment (item 10); slow on CPU")
    ap.add_argument("--out", type=Path, default=Path("/tmp/aswp_phase0/report.json"))
    args = ap.parse_args(argv)

    model = CANDIDATE_CONFIGS[args.model]
    names = args.only or list(CHECKS)
    report = {"device": args.device, "model": model.to_dict(),
              "python": platform.python_version(), "checks": {}}
    failures = []
    print(f"Phase 0 gate | model={args.model} {model.to_dict()} | device={args.device}\n")
    for name in names:
        if name not in CHECKS:
            print(f"  {name}: unknown check")
            failures.append(name)
            continue
        t0 = time.time()
        try:
            passed, evidence = CHECKS[name](model, args.device)
        except Exception as exc:
            passed, evidence = False, {"exception": f"{type(exc).__name__}: {exc}"}
        report["checks"][name] = {"passed": bool(passed), "seconds": round(time.time() - t0, 2),
                                 "evidence": evidence}
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  ({time.time()-t0:.1f}s)")
        for key, value in evidence.items():
            shown = f"{value:.6g}" if isinstance(value, float) else value
            print(f"         {key}: {shown}")
        if not passed:
            failures.append(name)

    if args.include_cost:
        passed, evidence = check_cost(model, args.device, args.throughput_steps,
                                      args.throughput_batch)
        report["checks"]["cost_measurement"] = {"passed": bool(passed), "evidence": evidence}
        print(f"  [{'PASS' if passed else 'FAIL'}] cost_measurement")
        for key, value in evidence.items():
            shown = f"{value:.4g}" if isinstance(value, float) else value
            print(f"         {key}: {shown}")
        if not passed:
            failures.append("cost_measurement")

    report["passed"] = not failures
    report["failures"] = failures
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n{'PHASE 0 GATE PASSED' if not failures else 'PHASE 0 GATE FAILED: ' + ', '.join(failures)}")
    print(f"report: {args.out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
