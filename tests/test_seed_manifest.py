"""Tests for the frozen seed manifest (issue #10).

The point of these tests is that the committed seed list is *checked against its
published derivation* on every run, so it cannot be silently re-chosen, drifted,
or hand-edited. A test that only counted the seeds would not do that.
"""
import json

import pytest

from seed_manifest import (
    MANIFEST_PATH,
    PAIRED_SEED_COUNT,
    ROLES,
    SEED_MODULUS,
    build_manifest,
    derive_seed,
    load_manifest,
    paired_seeds,
    role_seeds,
    verify,
    write_manifest,
)


class TestFrozenSeeds:
    def test_manifest_exists_and_is_marked_frozen(self):
        manifest = load_manifest()
        assert manifest["frozen"] is True
        assert manifest["schema"] == "aswp-config-manifest/1"
        assert manifest["issue"] == 10

    def test_there_are_exactly_the_required_paired_seeds(self):
        seeds = load_manifest()["paired_seeds"]
        assert len(seeds) == PAIRED_SEED_COUNT
        assert len(set(seeds)) == PAIRED_SEED_COUNT          # no duplicates
        assert all(0 <= s < SEED_MODULUS for s in seeds)     # in range

    def test_committed_manifest_equals_its_derivation(self):
        ok, problems = verify()
        assert ok, f"seed manifest does not match its derivation: {problems}"

    def test_derivation_is_deterministic(self):
        assert build_manifest() == build_manifest()
        assert paired_seeds() == paired_seeds()
        assert derive_seed("ASWP", "v1", "paired", "1") == derive_seed("ASWP", "v1", "paired", "1")

    def test_derivation_is_not_a_trivial_sequence(self):
        # consecutive derivations must not be arithmetic neighbours (which would
        # signal a counter rather than a hash being used)
        seeds = paired_seeds()
        steps = {b - a for a, b in zip(seeds, seeds[1:])}
        assert len(steps) == len(seeds) - 1

    def test_every_role_gets_an_independent_seed(self):
        manifest = load_manifest()
        for paired_seed, mapping in manifest["per_seed_role_seeds"].items():
            assert set(mapping) == set(ROLES)
            assert len(set(mapping.values())) == len(ROLES)
            assert str(role_seeds(int(paired_seed))) == str(mapping)

    def test_roles_match_the_protocol_requirement(self):
        assert set(ROLES) == {"global", "init", "data-loader", "dropout", "perturbation"}

    def test_rewriting_is_byte_idempotent(self, tmp_path):
        first = write_manifest(tmp_path / "m1.json").read_bytes()
        second = write_manifest(tmp_path / "m2.json").read_bytes()
        assert first == second

    def test_a_tampered_seed_is_detected(self, tmp_path):
        # the failure mode this guards: someone edits a seed in the manifest and
        # the suite still passes because nothing recomputes it
        tampered = tmp_path / "manifest.json"
        data = json.loads(MANIFEST_PATH.read_text())
        data["paired_seeds"][0] = data["paired_seeds"][0] + 1
        tampered.write_text(json.dumps(data))
        ok, problems = verify(tampered)
        assert not ok
        assert any("paired_seeds" in p for p in problems)

    def test_a_missing_manifest_is_reported(self, tmp_path):
        ok, problems = verify(tmp_path / "absent.json")
        assert not ok and problems


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
