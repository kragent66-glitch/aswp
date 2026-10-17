#!/usr/bin/env python3
"""Recompute the cost table from the stored measurements, without touching a GPU.

The numbers in `bench/results/phase0_cost.json` are measurements; the cost per run and the n
decision are arithmetic on them. The arithmetic lives in one place (the same functions the
runner uses), so recomputing after a fix cannot disagree with the run that produced the
measurements -- as it did for the first pass, where the rate lookup silently dropped every
segment and printed an empty decision table.

Usage:
    python bench/recompute_cost_decisions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.modal_phase0_cost import (  # noqa: E402
    USABLE_BUDGET,
    derive_decisions,
    gpu_key,
)

RESULTS = REPO / "bench" / "results" / "phase0_cost.json"

payload = json.loads(RESULTS.read_text())
segments = payload.get("segments", [])
endpoint_seconds = float(payload.get("endpoint_eval", {}).get("total_seconds") or 0.0)
decisions = derive_decisions(segments, endpoint_seconds)
payload["decisions"] = decisions

print(f"{'segment':24s} {'gpu':5s} {'prec':5s} {'tok/s':>10s} {'$ / run':>9s} "
      f"{'incl eval':>10s}  n   band")
for d in sorted(decisions, key=lambda d: d["cost_per_run_incl_endpoint_eval_usd"]):
    print(f"{d['label']:24s} {d['gpu_key'] or '?':5s} {d['precision']:5s} "
          f"{d['tokens_per_second']:10,.0f} {d['cost_per_run_usd']:9.4f} "
          f"{d['cost_per_run_incl_endpoint_eval_usd']:10.4f}  "
          f"{str(d['decision_with_eval']['n']):>3s}  {d['decision_with_eval']['band']}")

# ASWP's overhead, measured on the one GPU where both arms were run in the same precision
baseline = next((s for s in segments if s["arm"] == "baseline"), None)
if baseline:
    peer = next((s for s in segments if s["arm"] == "aswp"
                 and gpu_key(s["device"], s["label"]) == gpu_key(baseline["device"],
                                                                baseline["label"])
                 and s["precision"] == baseline["precision"]), None)
    if peer:
        overhead = 100.0 * (peer["seconds_per_step"] / baseline["seconds_per_step"] - 1.0)
        print(f"\nASWP overhead vs baseline ({baseline['precision']} on "
              f"{gpu_key(baseline['device'], baseline['label'])}): {overhead:+.1f}% "
              f"({peer['seconds_per_step']:.3f}s vs {baseline['seconds_per_step']:.3f}s/step)")
        payload["aswp_overhead_pct"] = overhead

payload["usable_budget"] = USABLE_BUDGET
RESULTS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(f"\nwrote {RESULTS}")
best = min(decisions, key=lambda d: d["cost_per_run_incl_endpoint_eval_usd"]) if decisions else None
if best:
    print(f"cheapest per token: {best['label']} at "
          f"${best['cost_per_run_incl_endpoint_eval_usd']:.4f}/run -> "
          f"n={best['decision_with_eval']['n']} ({best['decision_with_eval']['band']})")
