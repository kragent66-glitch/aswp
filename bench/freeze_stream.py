#!/usr/bin/env python3
"""Freeze the validation token streams the repository reports numbers on.

Every validation NLL this repository publishes -- ours, and the reference column measured
from public checkpoints -- is read off a token stream this script writes. Those streams
must therefore be immutable and hash-addressed: `docs/06-preregistration-checklist.md`
item 5 requires the validation split and the post-tokenization token-stream hash to be
frozen before any run, and a stream that differs between two runs makes every comparison
meaningful only against itself.

Two streams, one implementation, deliberately:

* `reference-eval` -- `TinyStories-valid.txt`. The file the dataset card nominates for
  validation loss, and the stream the reference table (docs/07) is measured on. Kept so
  our model has a row that is directly comparable to the six public checkpoints. Its one
  weakness: the reference authors may have used it for model selection.
* `confirmatory-primary` -- `TinyStoriesV2-GPT4-valid.txt`. **Pristine**: nothing in the
  comparison trained on it or selected against it. This is the primary endpoint for the
  confirmatory test, precisely because it removes that weakness.

Method: download the file at a pinned dataset revision, verify the sha256 the Hub reports
for that blob, tokenize with GPT-2 BPE (the same token unit the reference family uses --
verified in docs/07 section 3), and write `<slug>-<hash12>.npy` (uint16 ids) plus a
sidecar JSON carrying role, hashes, counts and provenance. The stream id is derived from
the content, so a changed stream cannot silently overwrite a frozen one.

Usage:
    python bench/freeze_stream.py                     # both streams
    python bench/freeze_stream.py --stream reference-eval
    python bench/freeze_stream.py --stream all --local-file-dir /path/to/downloads
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DATASET_REPO = "roneneldan/TinyStories"
# Pinned revision of the dataset repository (the sha the Hub served on 2026-10-17).
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TOKENIZER_NAME = "gpt2"
TOKENIZER_ALIAS = "EleutherAI/gpt-neo-125M"  # identical token->id map, see module docstring

BENCH_DIR = Path(__file__).resolve().parent
FROZEN_DIR = BENCH_DIR / "frozen"
CACHE_DIR = BENCH_DIR / ".cache"

RAW_URL = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}"
BLOBS_API = "https://huggingface.co/api/datasets/{repo}?blobs=true"


@dataclass(frozen=True)
class StreamSpec:
    """One validation stream: which file, under which role, and why it exists."""

    role: str
    filename: str
    slug: str
    pristine: bool
    rationale: str


STREAMS: dict[str, StreamSpec] = {
    "reference-eval": StreamSpec(
        role="reference-eval",
        filename="TinyStories-valid.txt",
        slug="tinystories-valid",
        pristine=False,
        rationale=(
            "The file the dataset card nominates for validation loss, and therefore the "
            "stream the public reference checkpoints are measured on (docs/07). Shared with "
            "our own model so it has a directly comparable row. Caveat recorded in docs/07 "
            "section 6: the reference authors may have used it for model selection."
        ),
    ),
    "confirmatory-primary": StreamSpec(
        role="confirmatory-primary",
        filename="TinyStoriesV2-GPT4-valid.txt",
        slug="tinystories-v2-gpt4-valid",
        pristine=True,
        rationale=(
            "Pristine endpoint: no model in the comparison trained on it or selected against "
            "it, which is what the reference-eval stream cannot promise. Primary endpoint for "
            "the confirmatory test (docs/06 item 5)."
        ),
    ),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hub_blob_sha256(repo: str, revision: str, filename: str) -> str | None:
    """The LFS sha256 the Hub publishes for one file, if it reports one."""
    with urllib.request.urlopen(BLOBS_API.format(repo=repo), timeout=60) as fh:
        meta = json.load(fh)
    if meta.get("sha") != revision:
        print(f"  warning: dataset HEAD is {meta.get('sha')}, pinned revision is {revision}",
              file=sys.stderr)
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
    """GPT-2 BPE, fast (rust) implementation. Imported lazily so --help needs no torch."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)


def freeze_one(spec: StreamSpec, *, local_file: Path | None, out_dir: Path,
               verify: bool) -> int:
    import numpy as np

    print(f"\n=== {spec.role} :: {spec.filename} "
          f"({'pristine' if spec.pristine else 'shared with the reference table'}) ===")
    if local_file is not None:
        raw_path = local_file
        if not raw_path.exists():
            print(f"FATAL: --local-file {raw_path} does not exist", file=sys.stderr)
            return 2
    else:
        raw_path = fetch_raw(
            DATASET_REPO, DATASET_REVISION, spec.filename,
            CACHE_DIR / f"{DATASET_REVISION[:12]}_{spec.filename}",
        )

    raw = raw_path.read_bytes()
    raw_sha = sha256_bytes(raw)
    print(f"  bytes={len(raw):,}  sha256={raw_sha}")

    if verify:
        expected = hub_blob_sha256(DATASET_REPO, DATASET_REVISION, spec.filename)
        if expected is None:
            print("  Hub reports no sha256 for this file (not an LFS blob) -- recording the "
                  "locally computed hash only", file=sys.stderr)
        else:
            expected = expected.removeprefix("sha256:")
            if expected != raw_sha:
                print(f"  FATAL: sha256 mismatch\n    hub:    {expected}\n    local:  {raw_sha}",
                      file=sys.stderr)
                return 2
            print(f"  ok: matches the Hub's {expected}")

    tok = build_tokenizer()
    text = raw.decode("utf-8", errors="replace")
    # truncation and padding are explicitly off: a silently truncated validation stream
    # would score fewer tokens than the frozen metadata claims.
    ids = np.asarray(
        tok(text, add_special_tokens=False, truncation=False, padding=False)["input_ids"],
        dtype=np.uint16,
    )
    print(f"  tokens={int(ids.size):,}  bytes/token={len(raw)/ids.size:.4f}  "
          f"distinct ids={int(np.unique(ids).size)}")

    stream_sha = sha256_bytes(ids.tobytes())
    stream_id = f"{spec.slug}-{stream_sha[:12]}"
    meta = {
        "stream_id": stream_id,
        "role": spec.role,
        "pristine": spec.pristine,
        "rationale": spec.rationale,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "file": spec.filename,
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
            "sha256 of the token array's raw bytes (ids.tobytes()), NOT of the .npy file: "
            "a .npy carries a header, so file and array hashes can never match. Verify "
            "with bench/verify_stream.py or the test suite."
        ),
        "note": ("Validation stream for every NLL this repository reports. Scoring "
                 "convention lives in bench/reference_eval.py."),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{stream_id}.npy"
    meta_path = out_dir / f"{stream_id}.json"
    if npy_path.exists():
        existing = json.loads(meta_path.read_text())
        if existing != meta:
            print(f"  FATAL: {npy_path} exists with different metadata; refusing to overwrite.\n"
                  f"    (The stream is content-addressed, so a differing sidecar means the "
                  f"schema or the tokenization changed. Remove the stale pair deliberately.)",
                  file=sys.stderr)
            return 2
        print(f"  already frozen and identical: {npy_path}")
        return 0

    np.save(npy_path, ids)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"  froze {npy_path}\n  meta  {meta_path}\n  stream_id={stream_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--stream", default="all", choices=[*STREAMS, "all"],
                    help="which validation stream to freeze (default: all)")
    ap.add_argument("--local-file", type=Path, default=None,
                    help="use an already-downloaded file (only valid with a single --stream)")
    ap.add_argument("--local-file-dir", type=Path, default=None,
                    help="directory holding already-downloaded copies by filename")
    ap.add_argument("--out-dir", type=Path, default=FROZEN_DIR)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the Hub sha256 comparison (offline use)")
    args = ap.parse_args(argv)

    roles = list(STREAMS) if args.stream == "all" else [args.stream]
    if args.local_file is not None and len(roles) != 1:
        print("FATAL: --local-file requires a single --stream", file=sys.stderr)
        return 2

    rc = 0
    for role in roles:
        spec = STREAMS[role]
        local = args.local_file
        if local is None and args.local_file_dir is not None:
            cand = args.local_file_dir / spec.filename
            local = cand if cand.exists() else None
        rc |= freeze_one(spec, local_file=local, out_dir=args.out_dir,
                         verify=not args.no_verify)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
