#!/usr/bin/env python3
"""Verify every frozen validation stream against its committed metadata.

The `.npy` arrays are gitignored (they regenerate deterministically), so a fresh clone has
the sidecars and no streams. This script closes that gap: for each sidecar it checks the
local array if present, and otherwise re-derives the stream from the pinned revision and
compares every recorded value -- so "the numbers were measured on this stream" is checkable
by anyone, not just by the machine that produced it.

    python bench/verify_stream.py                 # check what is present, re-derive the rest
    python bench/verify_stream.py --array-only    # offline: compare arrays to sidecars only
    python bench/verify_stream.py --role confirmatory-primary

Exit codes: 0 all match, 1 mismatch, 2 usage/IO problem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))


def verify_sidecar(meta_path: Path, frozen_dir: Path) -> int:
    meta = json.loads(meta_path.read_text())
    npy = frozen_dir / f"{meta['stream_id']}.npy"
    role = meta.get("role", "?")
    flag = "pristine" if meta.get("pristine") else "shared"
    print(f"\n[{role}] {meta['stream_id']}  ({flag}, {meta['file']})")

    if npy.exists():
        ids = np.load(npy)
        digest = hashlib.sha256(ids.tobytes()).hexdigest()
        ok = digest == meta["token_stream_sha256"]
        print(f"  local array: {ids.size:,} tokens dtype={ids.dtype} "
              f"sha256(array bytes)={digest[:16]}...")
        print(f"  recorded   : {meta['token_stream_sha256'][:16]}...  "
              f"{'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            return 1
        if ids.size != meta["n_tokens"]:
            print(f"  MISMATCH: {ids.size:,} tokens but the record says {meta['n_tokens']:,}")
            return 1
    else:
        print("  no local array (it is gitignored); re-deriving from the pinned revision")
        return 3  # caller decides whether to re-derive
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--frozen-dir", type=Path, default=BENCH_DIR / "frozen")
    ap.add_argument("--role", default=None, help="verify only this role")
    ap.add_argument("--array-only", action="store_true",
                    help="do not re-derive missing streams (offline use)")
    args = ap.parse_args(argv)

    metas = sorted(args.frozen_dir.glob("*.json"))
    if args.role:
        metas = [m for m in metas if json.loads(m.read_text()).get("role") == args.role]
    if not metas:
        print(f"no sidecars in {args.frozen_dir}"
              + (f" for role {args.role}" if args.role else ""), file=sys.stderr)
        return 2

    print(f"verifying {len(metas)} frozen stream(s) in {args.frozen_dir}")
    missing, worst = [], 0
    for meta_path in metas:
        rc = verify_sidecar(meta_path, args.frozen_dir)
        worst = max(worst, 1 if rc == 1 else 0)
        if rc == 3:
            missing.append(meta_path)

    if missing and not args.array_only:
        from freeze_stream import main as freeze_main

        roles = [json.loads(m.read_text())["role"] for m in missing]
        print(f"\nre-deriving {len(roles)} stream(s): {', '.join(roles)}")
        for role in roles:
            worst = max(worst, freeze_main(["--stream", role]))
        print("\nre-checking after re-derivation")
        for meta_path in missing:
            worst = max(worst, 1 if verify_sidecar(meta_path, args.frozen_dir) == 1 else 0)
    elif missing:
        print(f"\n{len(missing)} stream(s) not derived locally (--array-only): "
              f"{[m.name for m in missing]}")

    print("\n" + ("ALL STREAMS MATCH" if worst == 0 else "MISMATCHES FOUND"))
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
