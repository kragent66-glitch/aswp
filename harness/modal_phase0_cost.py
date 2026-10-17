#!/usr/bin/env python3
"""Phase 0, GPU half: the item-10 cost segment, and the bf16-at-scale check.

This produces the one number Phase 2 is waiting on. The token budget is already fixed at
250,019,840 tokens, so the cost of a run is set entirely by throughput, and throughput on
this box cannot be measured -- the VPS has 4 CPU cores, where a single step at the real batch
shape (80 x 512, accumulation 4 = 163,840 tokens) is minutes rather than tens of milliseconds.

What it measures, and why each piece exists:

* **Per-step throughput at the real batch shape and the real model** (d256/L8), not a scaled
  proxy: cost does not scale linearly with batch size, so a small-batch figure would be a
  plausible-looking number that is simply wrong.
* **The ASWP overhead against the baseline arm**, because the intervention adds a perturbation
  materialisation per matrix per update, and the decision rule needs the *arm's* cost.
* **bf16 at scale, with the realized perturbation measured after the cast.** Phase 0's precision
  policy says noise is built in fp32 and the realized perturbation is measured after any bf16
  cast; on CPU that was a small-shape check, here it is the real model.
* **Peak GPU memory**, so a container that would OOM at step 800 is discovered at step 30.
* **The endpoint-evaluation cost on the full pristine stream**, because it is part of a run's
  cost and is paid 22-28 times across the design.

The segment feeds **real tokens** from the frozen validation stream (mounted, 9.4 MB), not
random ids, so the data path exercised is the one runs use. It runs the *real* config with
`stop_after=N`: the annealing schedule depends on total steps, so a short schedule would be a
different run's timings.

The decision rule is implemented here rather than applied by hand, so the arithmetic that
picks n is the arithmetic that was preregistered:

    usable budget   $30 credit x 0.75 reserve rule = $22.50
    design A        screen 4 + n=6 x 3 arms = 22 runs
    design B        screen 4 + n=8 x 3 arms = 28 runs
    n = 8           if cost/run <= 22.50 / 28 = 0.8036
    n = 6           if 0.8036 < cost/run <= 0.94   (22 runs, leaves a retry margin)
    n = 6, reserve  if 0.94 < cost/run <= 1.0227   (22 runs, reserve partly consumed)
    neither         if cost/run > 1.0227 -- the design of record does not fit

Usage
-----
    MODAL_PROFILE=<workspace> modal run harness/modal_phase0_cost.py
    MODAL_PROFILE=<workspace> modal run harness/modal_phase0_cost.py --steps 60
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
HARNESS_FILES = ("__init__.py", "model.py", "perturbation.py", "data.py", "arms.py",
                 "config.py", "loop.py")


def host_repo() -> Path | None:
    """The repository on the *host* that builds the image, or None inside the container.

    Modal copies this entrypoint module to /root/ before importing it, so
    `Path(__file__).parents[1]` is `/` in the container, not the repo -- which is how a
    module-level `(REPO / "bench" / "frozen").iterdir()` becomes `/bench/frozen` and
    crash-loops every container before it starts. Inside the container the files are already
    mounted at absolute paths, so the host path is only needed when it exists.
    """
    if (REPO / "harness").is_dir() and (REPO / "bench").is_dir():
        return REPO
    return None


HOST_REPO = host_repo()

#: The hourly rate Modal bills per GPU, used only to turn throughput into a cost.
GPU_RATES = {"T4": 0.59, "L4": 0.80, "A10": 1.10, "A100": 2.10, "H100": 3.90}
USABLE_BUDGET = 30.0 * 0.75
SCREEN_RUNS = 4
ARMS = 3


def cost_per_run(tokens_per_second: float, rate_usd_per_hour: float,
                 total_tokens: int = 250_019_840) -> float:
    """US$ to train one arm-run at the measured throughput."""
    return (total_tokens / tokens_per_second / 3600.0) * rate_usd_per_hour


def decide_n(cost: float) -> dict:
    """The preregistered rule, in code so it cannot be applied loosely after the fact."""
    if cost <= USABLE_BUDGET / (SCREEN_RUNS + 8 * ARMS):
        return {"n": 8, "runs": SCREEN_RUNS + 8 * ARMS,
                "design_cost": cost * (SCREEN_RUNS + 8 * ARMS),
                "band": "fits with the reserve intact"}
    if cost <= 0.94:
        return {"n": 6, "runs": SCREEN_RUNS + 6 * ARMS,
                "design_cost": cost * (SCREEN_RUNS + 6 * ARMS),
                "band": "fits with a retry margin"}
    if cost <= USABLE_BUDGET / (SCREEN_RUNS + 6 * ARMS):
        return {"n": 6, "runs": SCREEN_RUNS + 6 * ARMS,
                "design_cost": cost * (SCREEN_RUNS + 6 * ARMS),
                "band": "fits only by consuming the reserve"}
    return {"n": None, "runs": SCREEN_RUNS + 6 * ARMS,
            "design_cost": cost * (SCREEN_RUNS + 6 * ARMS),
            "band": "does not fit -- protocol owner decides"}


app = modal.App("aswp-phase0-cost")

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "numpy==2.4.6", "torch", "transformers==5.20.1"
)
if HOST_REPO is not None:
    for name in HARNESS_FILES:
        image = image.add_local_file(str(HOST_REPO / "harness" / name), f"/root/harness/{name}")
    image = image.add_local_file(str(HOST_REPO / "config" / "manifest.json"),
                                "/root/config/manifest.json")
    image = image.add_local_file(str(HOST_REPO / "bench" / "reference_eval.py"),
                                "/root/bench/reference_eval.py")
    for frozen in sorted((HOST_REPO / "bench" / "frozen").iterdir()):
        if frozen.suffix in {".npy", ".json"}:
            image = image.add_local_file(str(frozen), f"/root/bench/frozen/{frozen.name}")


@app.function(image=image, gpu="T4", timeout=3600, max_containers=6)
def cost_segment(steps: int, precision: str, arm: str, label: str) -> dict:
    """Train `steps` steps of the real config and report throughput, memory and the cast check."""
    import time

    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from harness.arms import annealed_arm, baseline_arm
    from harness.config import BudgetConfig, EvalConfig, RunConfig, git_commit
    from harness.model import CANDIDATE_CONFIGS, TinyDecoder
    from harness.perturbation import AnnealedPerturbation, realized_relative_perturbation

    frozen = Path("/root/bench/frozen")
    meta_path = next(p for p in sorted(frozen.glob("*.json"))
                     if json.loads(p.read_text())["role"] == "reference-eval")
    meta = json.loads(meta_path.read_text())
    ids = np.load(frozen / f"{meta['stream_id']}.npy")
    print(f"[{label}] token source: {meta['stream_id']} ({len(ids):,} tokens, real data)",
          flush=True)

    model_cfg = CANDIDATE_CONFIGS["screen_d256_l8"]
    spec = baseline_arm() if arm == "baseline" else annealed_arm(0.03)
    cfg = RunConfig(paired_seed=1857649130, arm=spec, model=model_cfg,
                    budget=BudgetConfig(), evaluation=EvalConfig(eval_every_steps=0),
                    precision=precision, stage="phase0")
    print(f"[{label}] config: {cfg.total_steps} steps total, segment stops at {steps}; "
          f"{cfg.tokens_per_step:,} tokens/step; precision={precision}; arm={spec.name}", flush=True)

    # small evaluation slices: the endpoint-eval cost is measured separately, so it must not
    # be charged to the per-step throughput
    small = {role: (ids[:2048], meta) for role in ("confirmatory-primary", "reference-eval")}
    torch.cuda.reset_peak_memory_stats()
    from harness.loop import train

    t0 = time.time()
    summary = train(cfg, device="cuda", train_ids=ids, eval_streams=small,
                    out_root=Path(f"/root/runs/{label}"), log_every=steps,
                    stop_after=steps)
    elapsed = time.time() - t0

    tokens = steps * cfg.tokens_per_step
    tps = tokens / elapsed
    peak = torch.cuda.max_memory_allocated()
    device_name = torch.cuda.get_device_name(0)

    # the precision policy: realized perturbation measured after any bf16 cast
    torch.manual_seed(0)
    probe = TinyDecoder(model_cfg)
    if precision == "bf16":
        probe = probe.to(torch.bfloat16)
    probe = probe.to("cuda")
    ctl = AnnealedPerturbation(probe, sigma0=0.03, total_steps=cfg.total_steps,
                              perturbation_seed=7)
    ctl.start_step(1)
    name = sorted(ctl.modules)[0]
    module = ctl.modules[name]
    realized = realized_relative_perturbation(module.base.weight, module.effective_weight())
    print(f"[{label}] realized perturbation on {name}: {realized:.6f} vs sigma "
          f"{module.sigma:.6f} (ratio {realized/module.sigma:.4f}), dtype "
          f"{module.effective_weight().dtype}", flush=True)

    return {
        "label": label, "steps": steps, "precision": precision, "arm": arm,
        "device": device_name, "tokens": tokens, "wall_seconds": round(elapsed, 2),
        "tokens_per_second": tps,
        "seconds_per_step": elapsed / steps,
        "peak_gpu_memory_bytes": peak,
        "realized_perturbation_ratio": realized / module.sigma,
        "perturbation_dtype": str(module.effective_weight().dtype),
        "instability_events": summary.get("instability_events"),
        "steps_completed": summary.get("steps_completed"),
        "git_commit": git_commit(),
        "gpu_rate_usd_per_hour": GPU_RATES.get(gpu_key(device_name, label) or "", None),
    }


@app.function(image=image, gpu="T4", timeout=1800)
def endpoint_eval_cost(batch_blocks: int = 16) -> dict:
    """Forward-only cost of scoring the full endpoint streams -- paid by every run."""
    import time

    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from harness.config import EvalConfig, RunConfig
    from harness.loop import evaluate
    from harness.model import CANDIDATE_CONFIGS, TinyDecoder

    frozen = Path("/root/bench/frozen")
    out = {}
    torch.manual_seed(0)
    model = TinyDecoder(CANDIDATE_CONFIGS["screen_d256_l8"]).to("cuda")
    for meta_path in sorted(frozen.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        ids = np.load(frozen / f"{meta['stream_id']}.npy")
        t0 = time.time()
        res = evaluate(model, ids, ctx=512, batch_blocks=batch_blocks, device="cuda",
                       bytes_per_token=meta["bytes_per_token"])
        out[meta["role"]] = {
            "stream_id": meta["stream_id"], "n_tokens_in_stream": int(len(ids)),
            "n_tokens_scored": res["n_tokens_scored"], "seconds": round(time.time() - t0, 2),
            "nll_nats_per_token": res["nll_nats_per_token"],
            "tokens_per_second": res["n_tokens_scored"] / (time.time() - t0),
        }
        print(f"[eval] {meta['role']}: {res['n_tokens_scored']:,} tokens in "
              f"{out[meta['role']]['seconds']}s = "
              f"{out[meta['role']]['tokens_per_second']:,.0f} tok/s", flush=True)
    out["total_seconds"] = sum(v["seconds"] for k, v in out.items() if isinstance(v, dict))
    return out


def gpu_key(device_name: str, label: str = "") -> str | None:
    """Map a torch device name onto the Modal rate table.

    `torch.cuda.get_device_name()` returns things like "NVIDIA L4", "NVIDIA A10G" and
    "Tesla T4", so taking the first token gives "NVIDIA" or "Tesla" and silently drops every
    segment out of the decision table -- which is exactly what happened, leaving the table
    empty after a run that had produced all the numbers it needed.
    """
    aliases = {"T4": "T4", "A10": "A10", "A10G": "A10", "L4": "L4",
               "A100": "A100", "H100": "H100"}
    for token in device_name.replace("-", " ").replace(",", " ").split():
        if token in aliases:
            return aliases[token]
    if label:
        head = label.split("-")[0]
        if head in GPU_RATES:
            return head
    return None


def derive_decisions(segments: list[dict], endpoint_seconds: float) -> list[dict]:
    """The rate table, cost per run and n decision for each measured segment."""
    decisions = []
    for r in segments:
        rate = GPU_RATES.get(gpu_key(r.get("device", ""), r.get("label", "")) or "")
        if rate is None:
            continue
        c = cost_per_run(r["tokens_per_second"], rate)
        eval_hours = endpoint_seconds / 3600.0
        with_eval = c + eval_hours * rate
        decisions.append({
            "label": r.get("label"), "device": r.get("device"), "gpu_key":
            gpu_key(r.get("device", ""), r.get("label", "")),
            "precision": r["precision"], "arm": r["arm"],
            "tokens_per_second": r["tokens_per_second"],
            "cost_per_run_usd": c,
            "cost_per_run_incl_endpoint_eval_usd": with_eval,
            "decision_training_only": decide_n(c),
            "decision_with_eval": decide_n(with_eval),
        })
    return decisions


@app.local_entrypoint()
def main(steps: int = 30, gpu: str = "T4", include_l4: bool = True, profiles: str = "") -> None:
    """`profiles` selects segments explicitly as "GPU:precision[:arm]" comma-separated.

    The default set measures the decided configuration; the selector exists because the
    questions change as answers arrive (e.g. "is bf16 on a tensor-core GPU cheaper per token
    than fp32 on L4?"), and editing the entrypoint for each one invites a typo in the thing
    being measured.
    """
    if profiles:
        specs = []
        for spec in [p for p in profiles.split(",") if p.strip()]:
            parts = spec.strip().split(":")
            seg_gpu, precision = parts[0], parts[1]
            arm = parts[2] if len(parts) > 2 else "aswp"
            specs.append({"steps": steps, "precision": precision, "arm": arm,
                          "label": f"{seg_gpu}-{precision}-{arm}", "gpu": seg_gpu})
    else:
        specs = [
            {"steps": steps, "precision": "fp32", "arm": "aswp", "label": f"{gpu}-fp32-aswp",
             "gpu": gpu},
            {"steps": steps, "precision": "fp32", "arm": "baseline",
             "label": f"{gpu}-fp32-baseline", "gpu": gpu},
            {"steps": steps, "precision": "bf16", "arm": "aswp", "label": f"{gpu}-bf16-aswp",
             "gpu": gpu},
        ]
        if include_l4 and gpu != "L4":
            specs.append({"steps": steps, "precision": "fp32", "arm": "aswp",
                          "label": "L4-fp32-aswp", "gpu": "L4"})
    print(f"launching {len(specs)} segments in parallel: {[s['label'] for s in specs]}\n",
          flush=True)

    handles = [(s["label"], cost_segment.with_options(gpu=s.pop("gpu")).spawn(**s))
               for s in specs]
    results, failures = [], []
    for label, handle in handles:
        # per-segment isolation: last run, the first failure propagated out of h.get() and the
        # local entrypoint stopped the whole app, cancelling segments that were still running
        try:
            results.append(handle.get())
        except Exception as exc:                      # noqa: BLE001 - reported, not swallowed
            failures.append(f"{label}: {type(exc).__name__}: {str(exc)[:300]}")
            print(f"  FAILED {label}: {type(exc).__name__}", flush=True)
    if not results:
        raise SystemExit("every segment failed:\n" + "\n".join(failures))
    print()
    for r in results:
        print(f"  {r['label']:22s} {r['device']:12s} {r['tokens']:>10,} tok in "
              f"{r['wall_seconds']:7.2f}s = {r['tokens_per_second']:>10,.0f} tok/s  "
              f"step {r['seconds_per_step']:.3f}s  peak {r['peak_gpu_memory_bytes']/2**30:.2f} GiB  "
              f"perturb ratio {r['realized_perturbation_ratio']:.4f}")

    print("\n=== endpoint evaluation cost (forward-only, full streams) ===")
    try:
        ev = endpoint_eval_cost.with_options(gpu=gpu).remote(batch_blocks=16)
    except Exception as exc:                          # noqa: BLE001
        ev = {"total_seconds": 0.0, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        print(f"  endpoint eval segment failed: {ev['error']}")
    for role in ("confirmatory-primary", "reference-eval"):
        if role in ev:
            e = ev[role]
            print(f"  {role:22s} {e['n_tokens_scored']:>10,} tokens in {e['seconds']:6.2f}s "
                  f"(NLL {e['nll_nats_per_token']:.4f})")
    print(f"  total {ev['total_seconds']:.2f}s per run")

    print("\n=== the decision this was run for ===")
    decisions = derive_decisions(results, ev["total_seconds"])
    for d in decisions:
        print(f"  {d['label']:22s} {d['gpu_key']:5s} {d['precision']:4s} {d['arm']:8s} "
              f"${d['cost_per_run_usd']:.4f}/run -> n={d['decision_training_only']['n']} "
              f"({d['decision_training_only']['band']}); incl. endpoint eval "
              f"${d['cost_per_run_incl_endpoint_eval_usd']:.4f} -> "
              f"n={d['decision_with_eval']['n']}")
    baseline = next((r for r in results if r["arm"] == "baseline"), None)
    aswp = next((r for r in results if r["arm"] == "aswp" and r["device"] == baseline["device"]
                 and r["precision"] == baseline["precision"]), None) if baseline else None
    if baseline and aswp:
        overhead = 100.0 * (aswp["seconds_per_step"] / baseline["seconds_per_step"] - 1.0)
        print(f"\n  ASWP overhead vs baseline: {overhead:+.1f}% "
              f"({aswp['seconds_per_step']:.3f}s vs {baseline['seconds_per_step']:.3f}s per step)")

    out = (HOST_REPO / "bench" / "results" / "phase0_cost.json") if HOST_REPO else Path("/root/phase0_cost.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Merged by label, not overwritten: comparing precision and GPU options means running the
    # segment more than once, and a file that only holds the latest run cannot be compared to
    # anything. Newest measurement wins for a given label.
    previous: dict = {}
    if out.exists():
        try:
            previous = json.loads(out.read_text())
        except json.JSONDecodeError:
            previous = {}
    by_label = {s["label"]: s for s in previous.get("segments", [])}
    for r in results:
        by_label[r["label"]] = r
    by_decision = {(d["device"], d["precision"], d["arm"]): d
                   for d in previous.get("decisions", [])}
    for d in decisions:
        by_decision[(d["device"], d["precision"], d["arm"])] = d
    out.write_text(json.dumps({
        "segments": [by_label[k] for k in sorted(by_label)],
        "decisions": [by_decision[k] for k in sorted(by_decision)],
        "endpoint_eval": ev,
        "failures": failures,
        "usable_budget": USABLE_BUDGET,
        "rule": "n=8 if cost<=0.8036/run; n=6 if <=0.94; "
                "n=6 consuming reserve if <=1.0227; else owner decides",
    }, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")
