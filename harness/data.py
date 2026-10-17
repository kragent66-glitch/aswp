"""Data pipeline: one frozen token stream, one deterministic order.

Two jobs, both of which exist to make arms comparable rather than to be convenient:

1. **Build the training stream canonically.** The TinyStories corpus separates stories with
   the literal marker `<|endoftext|>`, which GPT-2 BPE encodes as a single token (50256).
   Because that marker is a token boundary, the file can be tokenized in chunks that begin
   and end exactly at a marker and the concatenation is *identical* to tokenizing the whole
   file at once -- which is what makes it possible to build a 467M-token stream on a machine
   that cannot hold it in memory as a Python list. `tests/test_harness_data.py` asserts that
   equivalence on a fixture rather than asserting it in a comment.

2. **Sample blocks deterministically.** The stream is cut into non-overlapping blocks of
   `ctx` tokens; an epoch is a permutation of their offsets drawn from a dedicated
   data-loader generator. Baseline and treatment runs share the seed, so the paired
   difference cannot come from one arm having seen different data. The sampler is
   resumable, so a split-and-resume run continues the same order rather than restarting it.

This module is deliberately numpy-only: the sampler is the part that must be reproducible
and it needs no deep-learning framework, so the fast CI job can test it. torch appears
nowhere here, and `transformers` is imported lazily inside the builder.
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASET_REPO = "roneneldan/TinyStories"
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TRAIN_FILENAME = "TinyStories-train.txt"
TOKENIZER_NAME = "gpt2"
EOT_MARKER = "<|endoftext|>"
EOT_TOKEN_ID = 50256
STREAM_DTYPE = np.uint16
#: Piece size for the builder. Kept small deliberately: a 64 MB piece yields ~16M token ids
#: and the intermediate list of Python ints alone is >400 MB, which is enough to have this
#: process OOM-killed inside the worker's memory cgroup. 8 MB keeps peak resident memory in
#: the low hundreds of MB while costing nothing -- piece boundaries are marker-aligned, so
#: the chunk size cannot change the resulting stream (asserted in the test suite).
DEFAULT_CHUNK_BYTES = 8 << 20


# --------------------------------------------------------------------------- building
def tokenizer_provenance(name: str = TOKENIZER_NAME) -> dict:
    """Hashes of the tokenizer artifacts actually on disk.

    Naming the tokenizer is not the same as pinning it: the vocabulary is an input to every
    token id in every stream, so the files themselves have to be identifiable. Recorded for
    the same reason the dataset revision is -- so that "GPT-2 BPE" is a verifiable claim
    rather than a description.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    files: dict[str, str] = {}
    candidates = []
    for attr in ("vocab_file", "merges_file"):
        path = getattr(tok, attr, None)
        if path:
            candidates.append(Path(path))
    if candidates:
        directory = candidates[0].parent
        for extra in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                      "added_tokens.json", "vocab.json", "merges.txt"):
            candidate = directory / extra
            if candidate.exists():
                candidates.append(candidate)
    for path in sorted({p for p in candidates if p.exists()}):
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "name": name,
        "vocab_size": int(tok.vocab_size),
        "revision": (candidates[0].parent.name if candidates else None),
        "files": files,
    }


def _encode(tokenizer, text: str) -> np.ndarray:
    """Tokenize one piece straight into a numpy array.

    `return_tensors="np"` avoids materialising millions of Python ints before the array
    exists, which is the difference between fitting in the cgroup and not.
    """
    enc = tokenizer(text, add_special_tokens=False, truncation=False, padding=False,
                    return_tensors="np")
    return np.asarray(enc["input_ids"], dtype=STREAM_DTYPE).reshape(-1)


def tokenize_chunked(path: Path, tokenizer, chunk_bytes: int = DEFAULT_CHUNK_BYTES,
                     marker: str = EOT_MARKER,
                     max_carry_bytes: int = 512 << 20) -> np.ndarray:
    """Tokenize a marker-separated corpus in memory-bounded pieces.

    Every piece ends exactly after a marker occurrence and the next begins immediately
    after it, so no subword is ever split across pieces and the result equals a
    whole-file tokenization. Returned as one array; `build_train_stream` uses the same
    loop to write incrementally instead.
    """
    pieces = []
    for piece in iter_pieces(path, chunk_bytes=chunk_bytes, marker=marker,
                             max_carry_bytes=max_carry_bytes):
        pieces.append(_encode(tokenizer, piece))
    if not pieces:
        return np.zeros(0, dtype=STREAM_DTYPE)
    return np.concatenate(pieces)


def iter_pieces(path: Path, chunk_bytes: int = DEFAULT_CHUNK_BYTES, marker: str = EOT_MARKER,
                max_carry_bytes: int = 512 << 20):
    """Yield text pieces that each end exactly after a marker occurrence.

    A marker split across two reads is reassembled because the unprocessed remainder is
    carried forward, so `rfind` always sees it whole.
    """
    marker_b = marker.encode()
    carry = b""
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk_bytes)
            if not buf:
                break
            data = carry + buf
            cut = data.rfind(marker_b)
            if cut == -1:
                carry = data
                if len(carry) > max_carry_bytes:
                    raise RuntimeError(
                        f"no {marker!r} found in {len(carry)} bytes; refusing to split a "
                        f"document that large (would break canonical tokenization)"
                    )
                continue
            end = cut + len(marker_b)
            yield data[:end].decode("utf-8", errors="replace")
            carry = data[end:]
    if carry:
        # Flush on *any* remainder, whitespace included. An earlier version used
        # `carry.strip()`, which silently dropped a trailing newline -- one token missing
        # from every stream, and the chunked result no longer equal to a whole-file
        # tokenization. Whitespace is tokenized data.
        yield carry.decode("utf-8", errors="replace")


def build_train_stream(out_dir: Path, *, local_file: Path | None = None,
                       chunk_bytes: int = DEFAULT_CHUNK_BYTES, verify: bool = True,
                       delete_raw_after: bool = False) -> dict:
    """Download/verify the training split, tokenize it, and write stream + manifest.

    The raw file is streamed in pieces and the tokens are appended to a `.bin`, so peak
    memory stays at one piece rather than the whole 467M-token stream. The manifest records
    the raw sha256, the token count, the token-stream sha256, the marker invariant, and the
    chunking rule, because those are the values `docs/06` item 5 asks to freeze.
    """
    import urllib.request

    from transformers import AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    if local_file is None:
        raw_path = out_dir / f"{DATASET_REVISION[:12]}_{TRAIN_FILENAME}"
        if not raw_path.exists():
            url = (f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/"
                   f"{DATASET_REVISION}/{TRAIN_FILENAME}")
            print(f"downloading {url}", flush=True)
            tmp = raw_path.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=3600) as resp, open(tmp, "wb") as out:
                total = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    total += len(chunk)
                    if total % (256 << 20) < (1 << 20):
                        print(f"  {total/1e6:8.1f} MB", flush=True)
            tmp.replace(raw_path)
        print(f"raw: {raw_path} ({raw_path.stat().st_size:,} bytes)", flush=True)
    else:
        raw_path = local_file

    raw_sha = hashlib.sha256()
    with open(raw_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            raw_sha.update(chunk)
    raw_sha_hex = raw_sha.hexdigest()
    print(f"raw sha256 {raw_sha_hex}", flush=True)

    if verify:
        expected = hub_train_sha256()
        if expected and expected.removeprefix("sha256:") != raw_sha_hex:
            raise SystemExit(
                f"sha256 mismatch for {TRAIN_FILENAME}\n  hub:   {expected}\n  local: {raw_sha_hex}"
            )
        print(f"  ok: matches the Hub's {expected}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    bin_path = out_dir / "train_tokens.bin"
    stream_sha = hashlib.sha256()
    n_tokens = 0
    eot_seen = 0
    counts = np.zeros(tokenizer.vocab_size, dtype=np.int64)
    marker_bytes = EOT_MARKER.encode()
    with open(bin_path, "wb") as out:
        for i, piece in enumerate(iter_pieces(raw_path, chunk_bytes=chunk_bytes)):
            ids = _encode(tokenizer, piece)
            counts += np.bincount(ids.astype(np.int64), minlength=tokenizer.vocab_size)
            eot_seen += int((ids == EOT_TOKEN_ID).sum())
            out.write(ids.tobytes())
            stream_sha.update(ids.tobytes())
            n_tokens += int(ids.size)
            # Measured, not guessed: without this the tokenizer's per-piece objects are only
            # reclaimed by the cycle collector, and RSS grows ~120 MB per 8 MiB piece -- which
            # is what OOM-killed this build inside the worker's memory cgroup partway through
            # the corpus. Explicitly dropping the piece and collecting keeps RSS flat at
            # ~1.4 GB for the whole file.
            del ids, piece
            gc.collect()
            if (i + 1) % 8 == 0:
                print(f"  {i+1} pieces, {n_tokens:,} tokens", flush=True)

    raw_bytes = raw_path.stat().st_size
    with open(raw_path, "rb") as fh:
        raw_marker_count = 0
        overlap = b""
        while True:
            block = fh.read(1 << 24)
            if not block:
                break
            raw_marker_count += (overlap + block).count(marker_bytes) - overlap.count(marker_bytes)
            overlap = block[-len(marker_bytes):]
    manifest = {
        "role": "train",
        "stream_id": f"tinystories-train-{stream_sha.hexdigest()[:12]}",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "file": TRAIN_FILENAME,
        "raw_bytes": raw_bytes,
        "raw_sha256": raw_sha_hex,
        "tokenizer": TOKENIZER_NAME,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_provenance": tokenizer_provenance(),
        "n_tokens": n_tokens,
        "bytes_per_token": raw_bytes / n_tokens if n_tokens else None,
        "distinct_token_ids": int((counts > 0).sum()),
        "dtype": np.dtype(STREAM_DTYPE).name,
        "bin_file": bin_path.name,
        "chunk_bytes": chunk_bytes,
        "raw_file_retained": not delete_raw_after,
        "token_stream_sha256": stream_sha.hexdigest(),
        "eot_marker": EOT_MARKER,
        "eot_token_id": EOT_TOKEN_ID,
        "eot_tokens": eot_seen,
        "eot_markers_in_raw": raw_marker_count,
        "chunking": (
            "pieces end exactly after an <|endoftext|> occurrence and the next begins "
            "immediately after it, so no subword spans a piece boundary; the concatenation "
            "is identical to tokenizing the whole file at once"
        ),
        "note": ("Training stream for every run. Cut into non-overlapping 512-token blocks "
                 "by BlockSampler; the validation endpoints are bench/frozen/*."),
    }
    # The marker invariant: every marker occurrence in the raw text becomes exactly one
    # 50256 token, so the two counts must agree. A mismatch means a piece boundary split a
    # marker (or the tokenizer stopped treating it as special) and the stream is not
    # canonical.
    manifest["marker_invariant_holds"] = raw_marker_count == eot_seen
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if delete_raw_after and local_file is None:
        # the raw sha256 is in the manifest, so the file is re-obtainable; on a 97%-full
        # disk the 1.9 GB is worth more than the convenience of keeping it
        raw_path.unlink(missing_ok=True)
        print(f"  removed the cached raw file ({raw_bytes:,} bytes) after hashing it")
    print(f"\nwrote {bin_path} ({bin_path.stat().st_size:,} bytes = {n_tokens:,} tokens)")
    print(f"  bytes/token {manifest['bytes_per_token']:.4f}, distinct ids "
          f"{manifest['distinct_token_ids']:,}")
    print(f"  eot tokens {eot_seen:,} vs markers in raw {raw_marker_count:,}: "
          f"invariant {'holds' if manifest['marker_invariant_holds'] else 'VIOLATED'}")
    print(f"  stream id {manifest['stream_id']}")
    return manifest


def hub_train_sha256() -> str | None:
    import urllib.request

    url = f"https://huggingface.co/api/datasets/{DATASET_REPO}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as fh:
        meta = json.load(fh)
    for sib in meta.get("siblings", []):
        if sib.get("rfilename") == TRAIN_FILENAME:
            lfs = sib.get("lfs") or {}
            return lfs.get("sha256") or lfs.get("oid")
    return None


# --------------------------------------------------------------------------- loading
def load_tokens(out_dir: Path, manifest: dict | None = None) -> np.memmap:
    """Memory-map the training stream, so the loop never holds 900 MB in RAM."""
    manifest = manifest or json.loads((out_dir / "train_manifest.json").read_text())
    shape = (manifest["n_tokens"],)
    return np.memmap(out_dir / manifest["bin_file"], dtype=STREAM_DTYPE, mode="r", shape=shape)


# --------------------------------------------------------------------------- sampling
@dataclass
class SamplerState:
    """Enough to resume the exact order, not merely the exact RNG.

    The current epoch's permutation is stored explicitly rather than reconstructed. A
    resume that only restored the generator state could not rebuild the epoch's order --
    the draw that produced it already happened -- and a split-and-resume run would then
    silently see different data from an uninterrupted one.
    """

    epoch: int
    position: int
    generator_state: dict
    n_blocks: int
    order: list


class BlockSampler:
    """Non-overlapping `ctx`-token blocks in a per-epoch permutation.

    Determinism is the point: with `seed` fixed, the order is fixed, so two runs of
    different arms see identical tokens in an identical order and the paired difference
    cannot be attributed to data order. `ctx` defaults to 512 because the reference family
    cannot be evaluated beyond that (docs/07 section 4) and because the protocol fixes it.
    """

    def __init__(self, ids, *, ctx: int = 512, seed: int = 0, drop_last: bool = True) -> None:
        if ctx < 2:
            raise ValueError("ctx must be at least 2")
        n_tokens = int(len(ids))
        n_blocks = n_tokens // ctx if drop_last else -(-n_tokens // ctx)
        if n_blocks == 0:
            raise ValueError(f"{n_tokens} tokens cannot fill one block of {ctx}")
        self.ids = ids
        self.ctx = ctx
        self.seed = int(seed)
        self.drop_last = drop_last
        self.n_blocks = n_blocks
        self.epoch = 0
        self.position = 0
        self.generator = np.random.default_rng(self.seed)
        self._order = self._permutation()

    def _permutation(self) -> np.ndarray:
        order = self.generator.permutation(self.n_blocks)
        return order.astype(np.int64)

    def __len__(self) -> int:
        return self.n_blocks - self.position

    def next_block(self) -> np.ndarray:
        """The next block of ids as int64 (the dtype a framework's embedding wants)."""
        if self.position >= self.n_blocks:
            self.epoch += 1
            self.position = 0
            self._order = self._permutation()
        start = int(self._order[self.position]) * self.ctx
        self.position += 1
        return np.asarray(self.ids[start:start + self.ctx], dtype=np.int64)

    def next_batch(self, batch_size: int) -> np.ndarray:
        """A (batch_size, ctx) batch, crossing an epoch boundary if it lands there."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return np.stack([self.next_block() for _ in range(batch_size)])

    def state_dict(self) -> SamplerState:
        return SamplerState(epoch=self.epoch, position=self.position,
                            generator_state=self.generator.bit_generator.state,
                            n_blocks=self.n_blocks,
                            order=self._order.tolist())

    def load_state_dict(self, state: SamplerState | dict) -> None:
        """Resume the exact order: same epoch, same position, same permutation.

        Everything is validated *before* anything is assigned. The earlier version assigned the
        epoch, position and generator and only then checked the permutation, so a bad state left
        the sampler half-updated while the caller saw an exception -- and a same-length but
        malformed order loaded happily, repeating some blocks and omitting others, which is a
        resume that appears to work while silently changing which data the run sees.
        """
        if isinstance(state, dict):
            state = SamplerState(**state)
        if int(state.n_blocks) != self.n_blocks:
            raise ValueError(
                f"sampler state was taken over {state.n_blocks} blocks but this stream has "
                f"{self.n_blocks}; the stream or ctx changed, so the order cannot be resumed"
            )
        order = np.asarray(state.order, dtype=np.int64)
        if order.size != self.n_blocks:
            raise ValueError("stored permutation does not cover every block")
        if self.n_blocks:
            if int(order.min()) < 0 or int(order.max()) >= self.n_blocks:
                raise ValueError(
                    f"stored permutation contains an out-of-range block index "
                    f"[{int(order.min())}, {int(order.max())}] for {self.n_blocks} blocks"
                )
            if np.unique(order).size != self.n_blocks:
                raise ValueError("stored permutation repeats or omits blocks")
        # `position == n_blocks` is legitimate: the epoch is exhausted and the next draw rolls
        # over. Anything outside that range is not a state this sampler can produce.
        position = int(state.position)
        if not 0 <= position <= self.n_blocks:
            raise ValueError(f"stored position {position} is outside 0..{self.n_blocks}")

        self.epoch = int(state.epoch)
        self.position = position
        self.generator.bit_generator.state = state.generator_state
        self._order = order


def main(argv: list[str] | None = None) -> int:
    """Build the training stream and its manifest. Long-running; writes into `data/`."""
    import argparse

    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    ap.add_argument("--local-file", type=Path, default=None,
                    help="use an already-downloaded TinyStories-train.txt")
    ap.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the Hub sha256 comparison (offline use)")
    ap.add_argument("--delete-raw-after", action="store_true",
                    help="remove the cached raw text once it has been hashed and tokenized")
    args = ap.parse_args(argv)

    manifest = build_train_stream(args.out_dir, local_file=args.local_file,
                                  chunk_bytes=args.chunk_bytes, verify=not args.no_verify,
                                  delete_raw_after=args.delete_raw_after)
    if not manifest["marker_invariant_holds"]:
        print("\nFATAL: the marker invariant does not hold; the stream is not canonical",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
