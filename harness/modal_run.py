#!/usr/bin/env python3
"""Run arms to completion on Modal GPUs, resuming across preemption.

The cost segment measures 30-step slices; this is the thing that actually trains an arm to
its frozen endpoint. It exists as a separate runner because the requirements differ in ways
that matter:

* **The training stream is 939 MB and lives on a Modal volume, not in the image.** Uploaded
  once and verified by hash after upload, so a container cannot silently train on a truncated
  or stale stream -- the same integrity check the reference evaluator performs on the
  validation streams.
* **A preempted run resumes rather than restarts.** Checkpoints are written every
  `checkpoint_every` steps and the *host* retries with `resume_from` pointing at the newest
  one. Resume equivalence is exact by construction (the loop refuses a checkpoint whose step
  budget differs, and the sampler and perturbation generator states are restored), so a retry
  continues the identical trajectory rather than a similar one.
* **Every run's output outlives the container.** Ledger, config, data manifest and summary go
  to the volume, and the summaries are merged locally into a JSON the analysis reads. A run
  that dies at step 1400 must not have to be repeated to be readable.

Cost accounting is explicit: tokens seen, wall seconds, and the implied dollars at the
profile's rate, because the design's whole budget rule is built on cost per run.

Usage
-----
    MODAL_PROFILE=<workspace> modal run harness/modal_run.py --upload-data
    MODAL_PROFILE=<workspace> modal run harness/modal_run.py --stage screen --steps 30
    MODAL_PROFILE=<workspace> modal run harness/modal_run.py --stage screen
    MODAL_PROFILE=<workspace> modal run harness/modal_run.py --stage confirm --arms C0,ASWP,C4 --gpu A10G
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
HARNESS_FILES = ("__init__.py", "model.py", "perturbation.py", "data.py", "arms.py",
                 "config.py", "loop.py")
VOLUME_NAME = "aswp-runs"
DATA_DIR = "/data"
RUNS_DIR = "/runs"
FROZEN_DIR = "/bench/frozen"

#: Hourly rates, used only to report what a run cost. Same table as the cost segment.
GPU_RATES = {"T4": 0.59, "L4": 0.80, "A10": 1.10, "A100": 2.10, "H100": 3.90}


def host_repo() -> Path | None:
    """The repository on the host that builds the image, or None inside the container.

    Modal copies this entrypoint to /root/ before importing it, so `__file__` is not the
    repo there. Getting this wrong crash-loops every container before it starts.
    """
    if (REPO / "harness").is_dir() and (REPO / "bench").is_dir():
        return REPO
    return None


HOST_REPO = host_repo()

app = modal.App("aswp-runs")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

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
            image = image.add_local_file(str(frozen), f"{FROZEN_DIR}/{frozen.name}")


# --------------------------------------------------------------------------- data upload
@app.function(image=image, volumes={DATA_DIR: volume}, timeout=3600)
def verify_data() -> dict:
    """Read the stream back from the volume and check it against its manifest.

    Upload goes through `modal volume put` (host side, no 939 MB Python serialization); this is
    the half that matters -- a container that trains on a truncated or stale stream would look
    like a perfectly normal run with a wrong result.
    """
    import hashlib

    from numpy import fromfile

    manifest_path = Path(DATA_DIR) / "train_manifest.json"
    if not manifest_path.exists():
        return {"ok": False, "error": "train_manifest.json is not on the volume"}
    manifest = json.loads(manifest_path.read_text())
    bin_path = Path(DATA_DIR) / manifest["bin_file"]
    if not bin_path.exists():
        return {"ok": False, "error": f"{manifest['bin_file']} is not on the volume"}
    tokens = fromfile(bin_path, dtype="uint16")
    digest = hashlib.sha256(tokens.tobytes()).hexdigest()
    ok = (tokens.size == manifest["n_tokens"] and digest == manifest["token_stream_sha256"])
    return {"ok": bool(ok), "stream_id": manifest["stream_id"], "n_tokens": int(tokens.size),
            "expected_tokens": manifest["n_tokens"], "sha256": digest,
            "expected_sha256": manifest["token_stream_sha256"],
            "bytes": bin_path.stat().st_size}


# --------------------------------------------------------------------------------- runs
def checkpoint_step(path: Path) -> int:
    """Sort key for checkpoint files: numbered steps, then `interrupt`, never `final`.

    `final.pt` means the run finished, so it is not a resume point; and a plain
    `int(stem.split("_")[-1])` raises on the literal names the loop also writes
    (`interrupt.pt`), which would have crashed every retry.
    """
    stem = path.stem
    if stem == "final":
        return -1
    if stem == "interrupt":
        return 0
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return 0


@app.function(image=image, gpu="A10G", timeout=10800, volumes={DATA_DIR: volume,
                                                               RUNS_DIR: volume})
def run_arm(config: dict, label: str, checkpoint_every: int = 250,
            stop_after: int = 0, resume: bool = False) -> dict:
    """Train one arm from `config` to its endpoint, resuming if asked and a checkpoint exists."""
    import hashlib
    import time

    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from harness.arms import ArmSpec
    from harness.config import (BudgetConfig, EvalConfig, ModelConfig, OptimizerConfig,
                                RunConfig, constructor_kwargs)
    from harness.loop import train
    from harness.model import CANDIDATE_CONFIGS

    # The stream on the volume is verified against its manifest before a single step runs:
    # training on a truncated stream would look like a normal run with a wrong result.
    manifest = json.loads((Path(DATA_DIR) / "train_manifest.json").read_text())
    tokens = np.fromfile(Path(DATA_DIR) / manifest["bin_file"], dtype=np.uint16)
    if tokens.size != manifest["n_tokens"]:
        return {"label": label, "error": f"stream has {tokens.size:,} tokens, manifest says "
                                         f"{manifest['n_tokens']:,}"}
    digest = hashlib.sha256(tokens.tobytes()).hexdigest()
    if digest != manifest["token_stream_sha256"]:
        return {"label": label, "error": f"stream hash {digest[:16]} != manifest "
                                         f"{manifest['token_stream_sha256'][:16]}"}
    print(f"[{label}] stream verified: {manifest['stream_id']} "
          f"({tokens.size:,} tokens, {digest[:12]})", flush=True)

    frozen = Path(FROZEN_DIR)
    streams = {}
    for meta_path in sorted(frozen.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        streams[meta["role"]] = (np.load(frozen / f"{meta['stream_id']}.npy"), meta)

    budget = BudgetConfig(**constructor_kwargs(BudgetConfig, config["budget"]))
    model_cfg = CANDIDATE_CONFIGS[config["model_key"]]
    arm = ArmSpec(name=config["arm"]["name"], kind=config["arm"]["kind"],
                  sigma=config["arm"]["sigma"], description=config["arm"]["description"])
    cfg = RunConfig(
        paired_seed=config["paired_seed"], arm=arm, model=model_cfg, budget=budget,
        optimizer=OptimizerConfig(**constructor_kwargs(OptimizerConfig,
                                                       config.get("optimizer", {}))),
        evaluation=EvalConfig(**constructor_kwargs(EvalConfig, config.get("evaluation", {}))),
        precision=config.get("precision", "bf16"), stage=config["stage"],
        loss_chunk_tokens=config.get("loss_chunk_tokens", 8192),
    )
    print(f"[{label}] {cfg.total_steps} steps | {cfg.tokens_per_step:,} tokens/step | "
          f"{cfg.precision} | {cfg.arm.name}", flush=True)

    run_dir = Path(RUNS_DIR) / cfg.run_id
    ckpt_dir = run_dir / "checkpoint"
    resume_from = None
    if resume and ckpt_dir.exists():
        candidates = [p for p in ckpt_dir.glob("*.pt")]
        if candidates:
            resume_from = max(candidates, key=checkpoint_step)
            print(f"[{label}] resuming from {resume_from.name}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    summary = train(cfg, device="cuda", train_ids=tokens,
                    eval_streams={k: v for k, v in streams.items()
                                  if k in (cfg.evaluation.primary_role,
                                           cfg.evaluation.comparability_role)},
                    out_root=Path(RUNS_DIR), resume_from=resume_from,
                    checkpoint_every=checkpoint_every, log_every=100,
                    stop_after=(stop_after or None))
    wall = time.time() - t0
    volume.commit()                       # make the ledger and summary durable

    rate = next((v for k, v in GPU_RATES.items() if k in torch.cuda.get_device_name(0)), None)
    hours = wall / 3600.0
    out = {
        "label": label, "run_id": cfg.run_id, "arm": cfg.arm.to_dict(),
        "paired_seed": cfg.paired_seed, "precision": cfg.precision,
        "device": torch.cuda.get_device_name(0), "steps_completed": summary.get("steps_completed",
                                                                               cfg.total_steps),
        "interrupted": bool(summary.get("interrupted")),
        "tokens_seen": summary.get("tokens_seen"),
        "wall_seconds": round(wall, 1), "hours": round(hours, 4),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "instability_events": summary.get("instability_events"),
        "endpoint": summary.get("endpoint"),
        "endpoint_sigma": summary.get("endpoint_sigma"),
        "perturbation": summary.get("perturbation"),
        "data_stream_id": manifest["stream_id"],
        "gpu_rate_usd_per_hour": rate,
        "cost_usd": round(hours * rate, 4) if rate else None,
    }
    (run_dir / "runner_record.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    volume.commit()
    ep = (summary.get("endpoint") or {}).get(cfg.evaluation.primary_role) or {}
    print(f"[{label}] {'INTERRUPTED at ' + str(out['steps_completed']) if out['interrupted'] else 'done'}"
          f" | {out['tokens_seen']:,} tokens in {wall/60:.1f} min "
          f"(${out['cost_usd']}) | peak {out['peak_gpu_memory_bytes']/2**30:.2f} GiB"
          + (f" | pristine NLL {ep.get('nll_nats_per_token'):.4f}" if ep else ""), flush=True)
    return out


@app.local_entrypoint()
def main(stage: str = "screen", arms: str = "", seeds: str = "", gpu: str = "A10G",
         steps: int = 0, checkpoint_every: int = 250, upload_data: bool = False,
         sigma: float = 0.03, retries: int = 1, out: str = "") -> None:
    if upload_data:
        if HOST_REPO is None:
            raise SystemExit("--upload-data must run on the host")
        import subprocess

        for name in ("train_manifest.json", "train_tokens.bin"):
            path = HOST_REPO / "data" / name
            if not path.exists():
                raise SystemExit(f"missing {path}; build it with harness/data.py first")
            # `modal volume put` rather than passing bytes through a function: the stream is
            # 939 MB and serializing it as an argument would load it entirely into memory on
            # both sides for no benefit.
            subprocess.run(["modal", "volume", "put", VOLUME_NAME, str(path), name], check=True)
        report = verify_data.remote()
        print(f"  volume check: {'OK' if report.get('ok') else 'FAILED'} "
              f"{report.get('stream_id')} {report.get('n_tokens', 0):,} tokens "
              f"sha256={str(report.get('sha256'))[:16]}")
        if not report.get("ok"):
            raise SystemExit(f"uploaded stream does not match its manifest: {report}")
        if not arms:
            return

    from harness.arms import (  # noqa: E402
        annealed_arm, baseline_arm, constant_matched_arm,
    )
    from harness.config import BudgetConfig, EvalConfig  # noqa: E402

    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    if not seed_list:
        manifest = json.loads((HOST_REPO / "config" / "manifest.json").read_text())
        seed_list = [int(s) for s in list(manifest["per_seed_role_seeds"])[: (8 if stage ==
                      "confirm" else 4)]]
    arm_names = [a.strip() for a in arms.split(",") if a.strip()]
    if not arm_names:
        arm_names = ["C0", "ASWP", "C4"] if stage == "confirm" else ["C0", "ASWP"]

    def spec_for(name: str, s0: float):
        if name == "C0":
            return baseline_arm()
        if name in ("ASWP", "C"):
            return annealed_arm(s0)
        if name == "C4":
            return constant_matched_arm(s0)
        raise SystemExit(f"unknown arm {name!r}")

    jobs = []
    for seed in seed_list:
        for name in arm_names:
            spec = spec_for(name, sigma)
            config = {
                "stage": stage, "paired_seed": seed, "arm": spec.to_dict(),
                "model_key": "screen_d256_l8",
                "budget": BudgetConfig().to_dict(),
                "evaluation": EvalConfig().to_dict(),
                "precision": "bf16",
            }
            jobs.append((config, f"{stage}-{name}-seed{seed}"))
    print(f"launching {len(jobs)} run(s) on {gpu}: {[j[1] for j in jobs]}\n", flush=True)

    records, failures = [], []
    for config, label in jobs:
        attempt = 0
        while True:
            attempt += 1
            try:
                res = run_arm.with_options(gpu=gpu).remote(
                    config, label, checkpoint_every=checkpoint_every, stop_after=steps,
                    resume=(attempt > 1))
                break
            except Exception as exc:                       # noqa: BLE001 - reported
                if attempt > retries:
                    failures.append(f"{label}: {type(exc).__name__}: {str(exc)[:200]}")
                    res = {"label": label, "error": str(exc)[:200]}
                    break
                print(f"  {label}: attempt {attempt} failed ({type(exc).__name__}); "
                      f"retrying from the newest checkpoint", flush=True)
        records.append(res)
        if res.get("error"):
            print(f"  FAILED {label}: {res['error']}")
        elif res.get("interrupted"):
            print(f"  INTERRUPTED {label} at step {res['steps_completed']} "
                  f"(rerun without --steps to finish)")

    ok = [r for r in records if not r.get("error")]
    total_cost = sum(r.get("cost_usd") or 0.0 for r in ok)
    print(f"\n{len(ok)}/{len(records)} runs completed; measured GPU spend ${total_cost:.4f}")
    for r in sorted(ok, key=lambda r: (r["run_id"])):
        ep = (r.get("endpoint") or {}).get("confirmatory-primary") or {}
        print(f"  {r['label']:28s} {r.get('tokens_seen', 0):>12,} tok {r['wall_seconds']/60:6.1f} min "
              f"${r.get('cost_usd') or 0:.4f} " +
              (f"NLL(pristine) {ep.get('nll_nats_per_token'):.4f} "
               f"bpb {ep.get('bits_per_byte'):.4f}" if ep else "no endpoint"))
    for f in failures:
        print(f"  FAILED {f}")

    target = Path(out) if out else (HOST_REPO / "bench" / "results" /
                                    f"{stage}_runs.json" if HOST_REPO else None)
    if target:
        merged: dict = {}
        if target.exists():
            try:
                merged = json.loads(target.read_text())
            except json.JSONDecodeError:
                merged = {}
        merged.setdefault("runs", {})
        for r in ok:
            merged["runs"][r["run_id"]] = r
        merged["total_cost_usd"] = round(sum(v.get("cost_usd") or 0.0
                                             for v in merged["runs"].values()), 4)
        merged["failures"] = failures
        target.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
        print(f"wrote {target}")
