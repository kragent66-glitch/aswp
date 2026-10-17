"""The training loop: one code path per arm, one scoring path per model.

Three properties this module is built around, each of which was a deliberate choice:

1. **One code path for every arm.** The baseline runs with the perturbation attached but
   inert (sigma = 0), because the zero-noise equivalence test proves that path is
   bit-identical to the unwrapped model. Two code paths would mean the contrast between
   arms could come from the *code*, which is not a difference any claim can be about.
2. **One scoring path.** Evaluation calls `bench/reference_eval.block_nll` -- the same
   function that produced the six reference rows -- through a thin adapter, so our NLL and
   theirs cannot drift apart in how the mean is aggregated.
3. **Resumable exactly.** A checkpoint carries the model, optimizer, step, the sampler's
   epoch permutation and the perturbation generator state, so a split-and-resume run
   continues the identical trajectory instead of a similar one.

The ledger is append-only JSONL plus a summary; nothing here overwrites a previous run's
output, and the run id is a function of the config with no clock in it.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from .arms import variance_match_report
from .config import REPO_ROOT, RunConfig, git_commit
from .data import BlockSampler, load_tokens
from .model import TinyDecoder
from .perturbation import AnnealedLinear, AnnealedPerturbation

#: bench/ holds the evaluator that produced the reference table; imported from there on
#: purpose so there is exactly one scoring implementation in the repository.
sys.path.insert(0, str(REPO_ROOT / "bench"))
from reference_eval import block_nll, find_frozen_stream  # noqa: E402

FROZEN_DIR = REPO_ROOT / "bench" / "frozen"


class _LogitsAdapter:
    """Presents TinyDecoder through the interface `block_nll` expects (`.logits`)."""

    def __init__(self, model: TinyDecoder) -> None:
        self.model = model

    def __call__(self, input_ids=None, **kwargs):
        return SimpleNamespace(logits=self.model(input_ids))


def normalize_state_dict(state: dict) -> dict:
    """Strip the wrapper's `.base.` from checkpoint keys.

    Wrapping a projection in `AnnealedLinear` moves its weight to `...q_proj.base.weight`.
    Storing that would make every checkpoint's key names depend on whether the perturbation
    was attached, and would make the weights unreadable by anything that knows the model.
    """
    return {k.replace(".base.", "."): v for k, v in state.items()}


def denormalize_state_dict(state: dict, model: TinyDecoder) -> dict:
    """The inverse of `normalize_state_dict`, driven by the model's actual module types."""
    wrapped = {name for name, mod in model.named_modules() if isinstance(mod, AnnealedLinear)}
    out = {}
    for key, value in state.items():
        parent, _, leaf = key.rpartition(".")
        out[f"{parent}.base.{leaf}" if parent in wrapped else key] = value
    return out


class Ledger:
    """Append-only per-step metrics. One JSON object per line, no overwriting."""

    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


def evaluate(model: TinyDecoder, ids, *, ctx: int, batch_blocks: int, device: str,
             bytes_per_token: float) -> dict:
    """Deterministic validation NLL through the shared scoring path."""
    import torch

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            nats, scored = block_nll(_LogitsAdapter(model), ids, ctx, batch_blocks, device)
    finally:
        if was_training:
            model.train()
    if scored == 0:
        raise RuntimeError("evaluation scored no tokens")
    nll = nats / scored
    return {
        "nll_nats_per_token": nll,
        "perplexity": math.exp(nll),
        "bits_per_byte": nll / (math.log(2) * bytes_per_token),
        "n_tokens_scored": scored,
    }


def save_checkpoint(path: Path, *, model, optimizer, controller, sampler, cfg: RunConfig,
                    step: int, data_manifest: dict, extra: dict | None = None) -> Path:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": int(step),
        "config": cfg.to_dict(),
        "model": normalize_state_dict(model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "sampler": asdict(sampler.state_dict()),
        "perturbation": controller.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "git_commit": git_commit(),
        "data_manifest_hashes": {
            k: data_manifest.get(k) for k in
            ("stream_id", "raw_sha256", "token_stream_sha256", "n_tokens", "dataset_revision")
        },
        "extra": extra or {},
    }, path)
    return path


def load_checkpoint(path: Path, *, model, optimizer, controller, sampler) -> dict:
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(denormalize_state_dict(ckpt["model"], model))
    optimizer.load_state_dict(ckpt["optimizer"])
    sampler.load_state_dict(ckpt["sampler"])
    controller.load_state_dict(ckpt["perturbation"])
    if "torch_rng_state" in ckpt:
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
    return ckpt


def build(cfg: RunConfig, *, device: str, seeds: dict, train_ids=None, data_dir: Path | None = None):
    """Model, optimizer, sampler and controller, seeded from the frozen per-role seeds."""
    import torch

    if cfg.model.max_seq_len < cfg.budget.ctx:
        raise ValueError("model context is shorter than the training context")
    torch.manual_seed(seeds["init"])
    model = TinyDecoder(cfg.model)
    if cfg.precision == "bf16":
        model = model.to(torch.bfloat16)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )
    if train_ids is None:
        manifest_path = Path(cfg.data_dir) / "train_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        ids = load_tokens(Path(cfg.data_dir), manifest)
    else:
        manifest = {"stream_id": "synthetic", "n_tokens": int(len(train_ids)),
                    "raw_sha256": None, "token_stream_sha256": None,
                    "dataset_revision": None, "bytes_per_token": 4.0}
        ids = train_ids
    sampler = BlockSampler(ids, ctx=cfg.budget.ctx, seed=seeds["data-loader"])
    controller = AnnealedPerturbation(
        model, sigma0=cfg.arm.sigma, total_steps=cfg.total_steps,
        perturbation_seed=seeds["perturbation"],
    )
    return model, optimizer, sampler, controller, manifest


def train(cfg: RunConfig, *, device: str = "cpu", train_ids=None, eval_streams=None,
          resume_from: Path | None = None, out_root: Path | None = None,
          log_every: int = 1, checkpoint_every: int | None = None,
          stop_after: int | None = None, on_step=None) -> dict:
    """Run one arm to its fixed endpoint and return the summary.

    `train_ids` and `eval_streams` exist so the loop can be exercised on tiny synthetic data
    (the test suite does exactly that); when they are None the frozen streams are used.

    `stop_after` interrupts the run cleanly at a step boundary, writing `interrupt.pt`. That
    is the real resume scenario rather than a contrived one: a container gets preempted
    mid-run and the run continues later *with the same config*. Resuming with a different
    `total_steps` is refused, because the annealing schedule is defined over the run length
    -- a run planned for 2 steps cannot be extended to 4 and still be the same trajectory.
    """
    import torch

    problems = cfg.validate()
    if problems:
        raise ValueError("invalid run config: " + "; ".join(problems))

    seeds = cfg.seeds()
    run_dir = (out_root or Path(cfg.out_dir)) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(run_dir)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True) + "\n")

    model, optimizer, sampler, controller, data_manifest = build(
        cfg, device=device, seeds=seeds, train_ids=train_ids
    )
    (run_dir / "data_manifest.json").write_text(
        json.dumps({k: data_manifest.get(k) for k in
                    ("stream_id", "n_tokens", "raw_sha256", "token_stream_sha256",
                     "dataset_revision", "bytes_per_token")}, indent=2, sort_keys=True) + "\n")

    # Loaded once: the interim evals score a fixed prefix of the primary stream, so they are
    # comparable across steps, and the endpoint scores the whole thing.
    streams = eval_streams if eval_streams is not None else _load_eval_streams(cfg)

    start_step = 0
    if resume_from is not None:
        ckpt = load_checkpoint(resume_from, model=model, optimizer=optimizer,
                              controller=controller, sampler=sampler)
        ckpt_steps = ckpt.get("config", {}).get("budget", {}).get("total_steps")
        if ckpt_steps is not None and int(ckpt_steps) != cfg.total_steps:
            raise ValueError(
                f"cannot resume a run planned for {ckpt_steps} steps with a config of "
                f"{cfg.total_steps} steps: the annealing schedule is defined over the run "
                f"length, so the trajectory would not be the same run"
            )
        start_step = int(ckpt["step"])
        ledger.write({"event": "resumed", "from_step": start_step, "checkpoint": str(resume_from)})

    log_path = run_dir / "train.log"

    def log(msg: str) -> None:
        line = f"[{cfg.run_id}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    log(f"arm={cfg.arm.name} kind={cfg.arm.kind} sigma={cfg.arm.sigma:g} "
        f"steps={cfg.total_steps} tokens/step={cfg.tokens_per_step} "
        f"realized_tokens={cfg.budget.realized_tokens:,} device={device} precision={cfg.precision}")
    log(f"seeds={seeds}")

    t0 = time.time()
    tokens_seen = start_step * cfg.tokens_per_step
    instability_events = 0
    for step in range(start_step + 1, cfg.total_steps + 1):
        step_t0 = time.time()
        sigma = controller.start_step(step)
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        non_finite = False
        for _ in range(cfg.budget.grad_accum):
            batch = torch.from_numpy(sampler.next_batch(cfg.budget.batch_sequences)).to(device)
            logits = model(batch)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                batch[:, 1:].reshape(-1),
            )
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                # A NaN/Inf update would poison every weight, so this step's update is
                # skipped -- but the event is recorded and counted, never silently dropped:
                # the protocol's rule is that an unstable run is retained and flagged, not
                # quietly excluded from the paired comparison.
                non_finite = True
                break
            (loss / cfg.budget.grad_accum).backward()
            micro_losses.append(loss_value)
        if non_finite:
            instability_events += 1
            optimizer.zero_grad(set_to_none=True)
            grad_norm = float("nan")
            optimizer_step_taken = False
        else:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.optimizer.max_grad_norm))
            optimizer.step()
            optimizer_step_taken = True
        ratios = controller.finish_step()
        tokens_seen += cfg.tokens_per_step
        step_time = time.time() - step_t0

        record = {
            "step": step,
            "tokens_seen": tokens_seen,
            "sigma": sigma,
            "train_loss": (sum(micro_losses) / len(micro_losses)) if micro_losses else None,
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "optimizer_step_taken": optimizer_step_taken,
            "step_seconds": step_time,
            "tokens_per_second": cfg.tokens_per_step / step_time,
            "epoch": sampler.epoch,
            "realized_perturbation": ratios.get(sorted(ratios)[0]) if ratios else 0.0,
        }
        if non_finite:
            record["event"] = "non_finite_loss"
        if device.startswith("cuda"):
            record["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
        ledger.write(record)
        if on_step is not None:
            on_step(record)
        if step % log_every == 0 or step == cfg.total_steps:
            log(f"step {step}/{cfg.total_steps} loss={record['train_loss']} "
                f"sigma={sigma:.6f} grad_norm={grad_norm:.3f} "
                f"{record['tokens_per_second']/1000:.1f}k tok/s"
                + (" [NON-FINITE: update skipped]" if non_finite else ""))
        if cfg.evaluation.eval_every_steps and step % cfg.evaluation.eval_every_steps == 0:
            pref_ids, pref_meta = streams[cfg.evaluation.primary_role]
            res = evaluate(model, pref_ids[: cfg.evaluation.interim_eval_tokens],
                           ctx=cfg.budget.ctx, batch_blocks=cfg.evaluation.eval_batch_blocks,
                           device=device, bytes_per_token=pref_meta["bytes_per_token"])
            ledger.write({"step": step, "event": "interim_eval",
                          "n_tokens_scored": res["n_tokens_scored"],
                          "nll_nats_per_token": res["nll_nats_per_token"],
                          "bits_per_byte": res["bits_per_byte"]})
        if checkpoint_every and step % checkpoint_every == 0 and step < cfg.total_steps:
            save_checkpoint(run_dir / "checkpoint" / f"step_{step}.pt", model=model,
                            optimizer=optimizer, controller=controller, sampler=sampler,
                            cfg=cfg, step=step, data_manifest=data_manifest)
        if stop_after is not None and step >= stop_after:
            save_checkpoint(run_dir / "checkpoint" / "interrupt.pt", model=model,
                            optimizer=optimizer, controller=controller, sampler=sampler,
                            cfg=cfg, step=step, data_manifest=data_manifest)
            log(f"interrupted at step {step} (stop_after={stop_after}); "
                f"resume with the same config from checkpoint/interrupt.pt")
            summary = {
                "run_id": cfg.run_id, "stage": cfg.stage, "arm": cfg.arm.to_dict(),
                "interrupted": True, "steps_completed": step,
                "steps": cfg.total_steps, "tokens_seen": tokens_seen,
                "instability_events": instability_events,
                "config": cfg.to_dict(),
                "git_commit": git_commit(), "device": device, "precision": cfg.precision,
            }
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
            return summary

    # ------------------------------------------------------------------ endpoint eval
    endpoint = {}
    for role, (ids, meta) in streams.items():
        res = evaluate(model, ids, ctx=cfg.budget.ctx, batch_blocks=cfg.evaluation.eval_batch_blocks,
                       device=device, bytes_per_token=meta["bytes_per_token"])
        endpoint[role] = {**res, "stream_id": meta.get("stream_id"),
                          "n_tokens_in_stream": int(len(ids)),
                          "pristine": meta.get("pristine")}
        log(f"endpoint[{role}] NLL={res['nll_nats_per_token']:.4f} "
            f"ppl={res['perplexity']:.3f} bpb={res['bits_per_byte']:.4f} "
            f"scored={res['n_tokens_scored']:,}")

    summary = {
        "run_id": cfg.run_id,
        "stage": cfg.stage,
        "arm": cfg.arm.to_dict(),
        "paired_seed": cfg.paired_seed,
        "seeds": seeds,
        "steps": cfg.total_steps,
        "tokens_seen": tokens_seen,
        "endpoint_step": cfg.total_steps,
        "endpoint_sigma": controller.sigma_at(cfg.total_steps),
        "endpoint": endpoint,
        "wall_seconds": round(time.time() - t0, 1),
        "mean_tokens_per_second": tokens_seen / max(1e-9, time.time() - t0),
        "param_counts": model.param_counts() if hasattr(model, "param_counts") else None,
        "perturbation": controller.report().__dict__,
        "variance_match": variance_match_report(cfg.arm.sigma, cfg.total_steps),
        "instability_events": instability_events,
        "config": cfg.to_dict(),
        "data_manifest": {k: data_manifest.get(k) for k in
                          ("stream_id", "raw_sha256", "token_stream_sha256", "n_tokens")},
        "git_commit": git_commit(),
        "device": device,
        "precision": cfg.precision,
    }
    if device.startswith("cuda"):
        summary["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
    save_checkpoint(run_dir / "checkpoint" / "final.pt", model=model, optimizer=optimizer,
                    controller=controller, sampler=sampler, cfg=cfg, step=cfg.total_steps,
                    data_manifest=data_manifest)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    log(f"done: {json.dumps(endpoint, sort_keys=True)}")
    return summary


def _load_eval_streams(cfg: RunConfig) -> dict:
    """The endpoint streams: the pristine primary plus the reference-comparability row."""
    import numpy as np

    out = {}
    for role in (cfg.evaluation.primary_role, cfg.evaluation.comparability_role):
        npy, meta = find_frozen_stream(FROZEN_DIR, role)
        out[role] = (np.load(npy), meta)
    return out
