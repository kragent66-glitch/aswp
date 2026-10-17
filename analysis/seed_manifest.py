#!/usr/bin/env python3
"""ASWP config manifest: the frozen paired seed list and per-role RNG seeds.

Issue #10 asks for the eight paired seeds (plus the per-role random seeds the
protocol requires) to be generated once, frozen in the config manifest, and never
regenerated. Freezing them by drawing numbers and pasting them into a file would
leave one question unanswerable: *were these drawn before or after seeing
something?* So the seeds are **derived**, not drawn:

    seed = int.from_bytes(sha256("ASWP/v1/<role>/<n>")[:8], "big") % 2**31

Anyone can recompute the list from this rule and check it against
``config/manifest.json``; the repository's test suite does exactly that on every
run, so the frozen values cannot drift or be quietly re-chosen. The derivation is
deliberately not seeded by anything external (no beacon, no clock): the point is
that the list is fully determined by a published string, so it is identical for
anyone who applies the rule.

Usage:
    python analysis/seed_manifest.py            # verify the committed manifest
    python analysis/seed_manifest.py --write    # regenerate + write it
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

MANIFEST_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "manifest.json"

SCHEMA = "aswp-config-manifest/1"
NAMESPACE = "ASWP"
VERSION = "v1"
SEED_MODULUS = 2 ** 31
PAIRED_SEED_COUNT = 8

#: One RNG role per independent stream the protocol requires to be recorded
#: separately (docs/01-research-protocol.md, Reproducibility).
ROLES = ("global", "init", "data-loader", "dropout", "perturbation")


def derive_seed(*parts: str) -> int:
    """Deterministic 31-bit seed from the documented derivation."""
    digest = hashlib.sha256("/".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % SEED_MODULUS


def paired_seeds(count: int = PAIRED_SEED_COUNT) -> list[int]:
    """The frozen paired seed list, index-ordered."""
    seeds = [derive_seed(NAMESPACE, VERSION, "paired", str(i)) for i in range(1, count + 1)]
    if len(set(seeds)) != count:
        raise RuntimeError("derivation produced a duplicate paired seed; change the namespace/version")
    return seeds


def role_seeds(paired: int) -> dict[str, int]:
    """Per-role RNG seeds for one paired seed."""
    return {role: derive_seed(NAMESPACE, VERSION, role, str(paired)) for role in ROLES}


def build_manifest() -> dict:
    seeds = paired_seeds()
    return {
        "schema": SCHEMA,
        "namespace": NAMESPACE,
        "version": VERSION,
        "frozen": True,
        "issue": 10,
        "derivation": {
            "method": "sha256(parts joined by '/'), first 8 bytes big-endian, mod 2**31",
            "parts": [NAMESPACE, VERSION, "<role>", "<n>"],
            "role_parts_for_paired_seeds": ["paired", "1..8"],
            "property": (
                "Fully determined by this published rule: no clock, no beacon, no "
                "draw that could be re-rolled. Recompute with analysis/seed_manifest.py."
            ),
        },
        "paired_seed_count": PAIRED_SEED_COUNT,
        "paired_seeds": seeds,
        "roles": list(ROLES),
        "per_seed_role_seeds": {str(s): role_seeds(s) for s in seeds},
        "notes": (
            "One paired seed per seed-pair: baseline and treatment runs of a pair share the "
            "paired seed and differ only in the perturbation arm, so pairing is enforced by "
            "construction. Each role stream is independent; the perturbation stream must be "
            "drawn from a dedicated torch.Generator. Never regenerate this list: adding seeds "
            "later extends it by index rather than replacing it."
        ),
    }


def write_manifest(path: pathlib.Path = MANIFEST_PATH) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_manifest(), indent=2) + "\n")
    return path


def load_manifest(path: pathlib.Path = MANIFEST_PATH) -> dict:
    return json.loads(path.read_text())


def verify(path: pathlib.Path = MANIFEST_PATH) -> tuple[bool, list[str]]:
    """Check the committed manifest against the derivation. Returns (ok, problems)."""
    problems: list[str] = []
    if not path.exists():
        return False, [f"{path} does not exist"]
    committed = load_manifest(path)
    expected = build_manifest()
    if committed != expected:
        for key in sorted(set(expected) | set(committed)):
            if committed.get(key) != expected.get(key):
                problems.append(f"{key}: committed {committed.get(key)!r} != derived {expected.get(key)!r}")
    return (not problems), problems


def main(argv: list[str]) -> int:
    if "--write" in argv:
        path = write_manifest()
        print(f"wrote {path}")
        return 0
    ok, problems = verify()
    if ok:
        m = load_manifest()
        print(f"{MANIFEST_PATH}: verified against the derivation")
        print(f"  paired seeds ({m['paired_seed_count']}): {m['paired_seeds']}")
        print(f"  roles: {', '.join(m['roles'])}")
        return 0
    print(f"{MANIFEST_PATH}: MISMATCH")
    for p in problems:
        print(f"  {p}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
