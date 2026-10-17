#!/usr/bin/env python3
"""Freeze the TinyStories validation token stream.

Every number this repository reports as a validation NLL -- ours, and the
reference column measured from public checkpoints -- is read off the token
stream this script writes. That stream must therefore be immutable and
hash-addressed: `docs/06-preregistration-checklist.md` item 5 requires the
validation split and the post-tokenization token-stream hash to be frozen
before any run, and a stream that differs between two runs makes every
comparison in the report meaningless.

What it does
------------
1. Downloads `TinyStories-valid.txt` at a pinned dataset revision (or reuses a
   local copy) and verifies the sha256 the Hub reports for that blob.
2. Tokenizes it with GPT-2 BPE -- the same token unit the public
   `roneneldan/TinyStories-*` checkpoints were trained in (verified: gpt2 and
   EleutherAI/gpt-neo-125M have identical token->id maps, and every reference
   checkpoint ships an identically-keyed 50257-entry vocab).
3. Writes `bench/frozen/<stream_id>.npy` (uint16 token ids) plus a sidecar JSON
   carrying the hashes, counts and provenance. The stream id is derived from the
   content, so a changed stream cannot silently overwrite a frozen one.

Nothing else in the repository may tokenize the validation text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

DATASET_REPO = "roneneldan/TinyStories"
# Pinned revision of the dataset repository (the sha the Hub served on
# 2026-10-17). Recorded in docs/07-reference-benchmark.md.
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
VALID_FILENAME = "TinyStories-valid.txt"
TOKENIZER_NAME = "gpt2"
TOKENIZER_ALIAS = "EleutherAI/gpt-neo-125M"  # identical token->id map, see module docstring

BENCH_DIR = Path(__file__).resolve().parent
FROZEN_DIR = BENCH_DIR / "frozen"
CACHE_DIR = BENCH_DIR / ".cache"

RAW_URL = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}"
BLOBS_API = "https://huggingface.co/api/datasets/{repo}?blobs=true"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hub_blob_sha256(repo: str, revision: str, filename: str) -> str | None:
    """The LFS sha256 the Hub publishes for one file, if it reports one."""
    with urllib.request.urlopen(BLOBS_API.format(repo=repo), timeout=60) as fh:
        meta = json.load(fh)
    if meta.get("sha") != revision:
        print(
            f"  warning: dataset HEAD is {meta.get('sha')}, pinned revision is {revision}",
            file=sys.stderr,
        )
    for sib in meta.get("siblings", []):
        if sib.get("rfilename") == filename:
            lfs = sib.get("lfs") or {}
            return lfs.get("sha256") or lfs.get("oid")
    return None


def fetch_raw(repo: str, revision: str, filename: str, dest: Path) -> Path:
    """Download one file at an exact revision, caching it by revision."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  reusing {dest} ({dest.stat().st_size:,} bytes)")
        return dest
    url = RAW_URL.format(repo=repo, rev=revision, name=filename)
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as resp, open(tmp, "wb") as out:
        total = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
            if total % (16 << 20) < (1 << 20):
                print(f"    {total/1e6:8.1f} MB", flush=True)
    tmp.replace(dest)
    print(f"  wrote {dest} ({dest.stat().st_size:,} bytes)")
    return dest


def build_tokenizer():
    """GPT-2 BPE, fast (rust) implementation.

    Imported lazily so that `--help` works without transformers installed.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    return tok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--local-file", type=Path, default=None,
                    help="use an already-downloaded TinyStories-valid.txt instead of fetching it")
    ap.add_argument("--out-dir", type=Path, default=FROZEN_DIR)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the Hub sha256 comparison (offline use)")
    args = ap.parse_args(argv)

    import numpy as np

    args.out_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] raw validation text @ {DATASET_REPO}@{DATASET_REVISION[:12]}")
    if args.local_file:
        raw_path = args.local_file
    else:
        raw_path = CACHE_DIR / f"{DATASET_REVISION[:12]}_{VALID_FILENAME}"
        raw_path = fetch_raw(DATASET_REPO, DATASET_REVISION, VALID_FILENAME, raw_path)

    raw = raw_path.read_bytes()
    raw_sha = sha256_bytes(raw)
    print(f"  bytes={len(raw):,}  sha256={raw_sha}")

    if not args.no_verify:
        print("[2/4] verifying against the Hub's reported blob sha256")
        expected = hub_blob_sha256(DATASET_REPO, DATASET_REVISION, VALID_FILENAME)
        if expected is None:
            print("  Hub reports no sha256 for this file (not an LFS blob) -- "
                  "recording the locally computed hash only", file=sys.stderr)
        else:
            expected = expected.removeprefix("sha256:")
            if expected != raw_sha:
                print(f"  FATAL: sha256 mismatch\n    hub:    {expected}\n    local:  {raw_sha}",
                      file=sys.stderr)
                return 2
            print(f"  ok: matches the Hub's {expected}")
    else:
        print("[2/4] sha256 verification skipped (--no-verify)")

    print("[3/4] tokenizing with GPT-2 BPE")
    tok = build_tokenizer()
    text = raw.decode("utf-8", errors="replace")
    # truncation and padding are explicitly off: a silently truncated validation
    # stream would score fewer tokens than the frozen metadata claims.
    enc = tok(text, add_special_tokens=False, truncation=False, padding=False)
    ids = np.asarray(enc["input_ids"], dtype=np.uint16)
    print(f"  tokens={int(ids.size):,}  bytes/token={len(raw)/ids.size:.4f}  "
          f"distinct ids={int(np.unique(ids).size)}")

    stream_sha = sha256_bytes(ids.tobytes())
    stream_id = f"tinystories-valid-{stream_sha[:12]}"
    meta = {
        "stream_id": stream_id,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "file": VALID_FILENAME,
        "raw_bytes": len(raw),
        "raw_sha256": raw_sha,
        "tokenizer": TOKENIZER_NAME,
        "tokenizer_equivalent_to": TOKENIZER_ALIAS,
        "tokenizer_vocab_size": int(tok.vocab_size),
        "n_tokens": int(ids.size),
        "bytes_per_token": len(raw) / int(ids.size),
        "distinct_token_ids": int(np.unique(ids).size),
        "dtype": "uint16",
        "token_stream_sha256": stream_sha,
        "token_stream_sha256_scope": (
            "sha256 of the token array's raw bytes (ids.tobytes()), NOT of the .npy "
            "file: a .npy carries a header, so file and array hashes can never match. "
            "Verify with bench/verify_stream.py or the test suite."
        ),
        "note": ("Validation stream for every NLL this repository reports. "
                 "Scoring convention lives in bench/reference_eval.py."),
    }

    npy_path = args.out_dir / f"{stream_id}.npy"
    meta_path = args.out_dir / f"{stream_id}.json"
    if npy_path.exists():
        existing = json.loads(meta_path.read_text())
        if existing != meta:
            print(f"  FATAL: {npy_path} exists with different metadata; refusing to overwrite",
                  file=sys.stderr)
            return 2
        print(f"[4/4] already frozen and identical: {npy_path}")
        return 0

    np.save(npy_path, ids)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"[4/4] froze {npy_path}\n      meta {meta_path}\n      stream_id={stream_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
