"""Shared strictness for test modules that need optional heavy dependencies.

A skipped test still exits 0, so a CI job that is *supposed* to exercise a test can
report green while never running it. Modules that require torch or the frozen data call
`import_torch_or_skip()` / `require()`: locally (and in the fast analysis job) the
absence of a dependency produces a clean skip, while any job that sets
`ASWP_BENCH_STRICT=1` gets a failure instead. See .github/workflows/tests.yml.
"""
from __future__ import annotations

import os

import pytest

STRICT = os.environ.get("ASWP_BENCH_STRICT") == "1"


def require(condition: bool, why: str) -> None:
    if condition:
        return
    if STRICT:
        pytest.fail(f"ASWP_BENCH_STRICT=1 but this test cannot run: {why}")
    pytest.skip(why)


def import_torch_or_skip():
    if STRICT:
        import torch  # a missing dependency must fail the job, not skip the test
        return torch
    return pytest.importorskip("torch")
