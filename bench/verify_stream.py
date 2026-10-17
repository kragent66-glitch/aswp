#!/usr/bin/env python3
"""Verify the frozen validation stream against its committed metadata.

The `.npy` array is gitignored (it is regenerated deterministically), so a fresh
clone has the sidecar JSON and no stream. This script closes that gap: it re-derives
the stream from the pinned dataset revision and checks every recorded value, so the
claim "the numbers were measured on this stream" is checkable by anyone.

    python bench/verify_stream.py            # re-derive and compare
    python bench/verify_stream.py --array-only  # just check the local array's hash

Exit codes: 0 match, 1 mismatch, 2 usage/IO problem.
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--frozen-dir", type=Path, default=BENCH_DIR / "frozen")
    ap.add_argument("--array-only", action="store_true",
                    help="check only the local array against its metadata (no download)")
    args = ap.parse_args(argv)

    metas = sorted(args.frozen_dir.glob("tinystories-valid-*.json"))
    if len(metas) != 1:
        print(f"expected exactly one metadata sidecar, found {[m.name for m in metas]}",
              file=sys.stderr)
        return 2
    meta = json.loads(metas[0].read_text())
    npy = args.frozen_dir / f"{meta['stream_id']}.npy"

    if npy.exists():
        ids = np.load(npy)
        dig = hashlib.sha256(ids.tobytes()).hexdigest()
        ok = dig == meta["token_stream_sha256"]
        print(f"local array: {ids.size:,} tokens, dtype={ids.dtype}, "
              f"sha256(array bytes)={dig}")
        print(f"  recorded : {meta['token_stream_sha256']}  {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            return 1
    elif args.array_only:
        print(f"no local array at {npy}", file=sys.stderr)
        return 2
    else:
        print("no local array; re-deriving from the pinned revision")

    if args.array_only:
        return 0

    from freeze_stream import main as freeze_main

    rc = freeze_main([
        "--out-dir", str(args.frozen_dir),
    ])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
