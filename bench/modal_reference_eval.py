#!/usr/bin/env python3
"""Run bench/reference_eval.py on Modal GPUs, on our frozen validation stream.

Why not locally: the VPS that hosts this repository has 4 cores. A smoke run of the
1M checkpoint took 48 s per 100k tokens, which puts the full 4.72M-token stream across
all six reference checkpoints at roughly 8-15 hours and would monopolise the box. T4s
on Modal do the same arithmetic in minutes for cents of the workspace's free compute.

Design notes, both learned the hard way:

* **One container per checkpoint, run in parallel.** A single container fetching all
  six checkpoints serially spent 20 minutes without finishing, because ~1 GB of
  unauthenticated Hub downloads dominate the runtime while the GPU sits idle. Per-repo
  containers download concurrently and let failures be per-model rather than total.
* **The evaluator is invoked, never reimplemented.** The container calls
  `bench/reference_eval.py` verbatim, so the reference column and our own model's
  numbers come from one code path, and it streams the child's output line by line --
  with `capture_output=True` a slow run is indistinguishable from a hung one.

The container also asserts the sha256 of the mounted token stream against the committed
metadata, which makes every GPU run a second-machine check on the frozen stream.

Usage
-----
    MODAL_PROFILE=<workspace> modal run bench/modal_reference_eval.py
    MODAL_PROFILE=<workspace> modal run bench/modal_reference_eval.py --repos gpt2
    MODAL_PROFILE=<workspace> modal run bench/modal_reference_eval.py --limit-tokens 200000
    MODAL_PROFILE=<workspace> modal run bench/modal_reference_eval.py --gpu L4
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

CONTAINER_BENCH = "/root/bench"


def _resolve_bench_dir() -> Path:
    """Locate reference_eval.py on the host (repo/bench) and in the container.

    Modal copies this module to /root/ for the entrypoint, so a bare sibling import
    fails there; the image places the evaluator at /root/bench/.
    """
    here = Path(__file__).resolve().parent
    for cand in (here, Path(CONTAINER_BENCH), here.parent / CONTAINER_BENCH.lstrip("/")):
        if (cand / "reference_eval.py").exists():
            return cand
    raise SystemExit(f"cannot locate reference_eval.py; looked near {here}")


BENCH_DIR = _resolve_bench_dir()
sys.path.insert(0, str(BENCH_DIR))
from reference_eval import REFERENCE_REPOS, merge_results  # noqa: E402

FROZEN_DIR = BENCH_DIR / "frozen"
RESULTS_PATH = BENCH_DIR / "results" / "reference_results.json"


def _local_stream_files() -> tuple[Path | None, Path | None]:
    """The frozen stream pair as it exists on the *host* building the image.

    Inside the container these files are already baked in, so a missing host path is
    not an error -- it means we are the container side of the same module.
    """
    npy = sorted(FROZEN_DIR.glob("tinystories-valid-*.npy"))
    meta = sorted(FROZEN_DIR.glob("tinystories-valid-*.json"))
    return (npy[0] if len(npy) == 1 else None, meta[0] if len(meta) == 1 else None)


_LOCAL_NPY, _LOCAL_META = _local_stream_files()

app = modal.App("aswp-reference-eval")

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "numpy==2.4.6", "torch", "transformers==5.20.1"
)
image = image.add_local_file(str(BENCH_DIR / "reference_eval.py"), f"{CONTAINER_BENCH}/reference_eval.py")
if _LOCAL_NPY is not None and _LOCAL_NPY.exists():
    image = image.add_local_file(str(_LOCAL_NPY), f"{CONTAINER_BENCH}/frozen/{_LOCAL_NPY.name}")
if _LOCAL_META is not None and _LOCAL_META.exists():
    image = image.add_local_file(str(_LOCAL_META), f"{CONTAINER_BENCH}/frozen/{_LOCAL_META.name}")


@app.function(image=image, gpu="T4", timeout=3600, max_containers=8)
def run_one(repo: str, ctx_tokens: int, batch_blocks: int,
            limit_tokens: int | None, device: str = "cuda") -> dict:
    import hashlib
    import json
    import os
    import subprocess

    frozen = Path(f"{CONTAINER_BENCH}/frozen")
    npys = sorted(frozen.glob("tinystories-valid-*.npy"))
    metas = sorted(frozen.glob("tinystories-valid-*.json"))
    if len(npys) != 1 or len(metas) != 1:
        found = [p.name for p in frozen.iterdir()] if frozen.exists() else f"{frozen} missing"
        return {"repo": repo, "error": f"expected one frozen stream pair, found {found}"}

    # Second-machine integrity check. The recorded hash is of the token array's raw
    # bytes, not of the .npy file (which carries a header), so hash the loaded array.
    import numpy as np

    meta = json.loads(metas[0].read_text())
    digest = hashlib.sha256(np.load(npys[0]).tobytes()).hexdigest()
    if digest != meta["token_stream_sha256"]:
        return {"repo": repo,
                "error": f"mounted stream array {digest} != frozen {meta['token_stream_sha256']}"}
    print(f"[{repo}] frozen stream verified: {meta['stream_id']} ({digest[:12]})", flush=True)

    out = "/root/results.json"
    cmd = [
        "python", f"{CONTAINER_BENCH}/reference_eval.py",
        "--repos", repo,
        "--ctx", str(ctx_tokens),
        "--batch-blocks", str(batch_blocks),
        "--frozen-dir", str(frozen),
        "--out", out,
        "--device", device,
    ]
    if limit_tokens:
        cmd += ["--limit-tokens", str(limit_tokens)]

    # stream the child's output line by line (see module docstring)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail: list[str] = []
    for line in proc.stdout:  # type: ignore[union-attr]
        print(line, end="", flush=True)
        tail.append(line)
        if len(tail) > 200:
            tail.pop(0)
    proc.wait()
    if not os.path.exists(out):
        return {"repo": repo, "exit_code": proc.returncode, "error": "".join(tail[-40:])}
    runs = json.loads(Path(out).read_text())["runs"]
    return {"repo": repo, "exit_code": proc.returncode, "runs": runs}


@app.local_entrypoint()
def main(repos: str = "", ctx_tokens: int = 512, batch_blocks: int = 16,
         limit_tokens: int = 0, gpu: str = "T4") -> None:
    repo_list = [r for r in repos.split(",") if r] or REFERENCE_REPOS
    print(f"repos={repo_list}\nctx={ctx_tokens} batch_blocks={batch_blocks} "
          f"limit_tokens={limit_tokens or 'all'} gpu={gpu} "
          f"(one container per repo, in parallel)", flush=True)

    args = [(r, ctx_tokens, batch_blocks, limit_tokens or None) for r in repo_list]
    records: list[dict] = []
    failures: list[str] = []
    for res in run_one.with_options(gpu=gpu).starmap(args):
        if res.get("runs"):
            records.extend(res["runs"])
        else:
            failures.append(f"{res.get('repo')}: {res.get('error', 'no runs')}")

    if not records:
        raise SystemExit("no results from any container:\n" + "\n".join(failures))

    merge_results(RESULTS_PATH, records)
    for r in sorted(records, key=lambda r: r.get("params") or 0):
        if "nll_nats_per_token" in r:
            print(f"  {r['repo'].split('/')[-1]:26s} params={r['params']/1e6:6.2f}M "
                  f"NLL={r['nll_nats_per_token']:.4f} ppl={r['perplexity']:7.3f} "
                  f"bpb={r['bits_per_byte']:.4f} scored={r['n_tokens_scored']:,} "
                  f"{r['wall_seconds']}s on {r['env']['device']}")
        else:
            print(f"  {r['repo']}: ERROR {r.get('error')}")
    for f in failures:
        print(f"  FAILED {f}")
    print(f"\nmerged into {RESULTS_PATH}")
