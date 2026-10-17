"""Phase 0 precursor tests for the harness core.

These pin the properties that make the experiment interpretable at all, on CPU and in
seconds, so that a GPU is never spent discovering them. They cover docs/06 Phase 0 items
1-3 (zero-noise equivalence, schedule correctness, target-module correctness) and item 6
(RNG isolation), plus the determinism and resume properties the gate needs.

Each test is written to fail under the specific defect it names. Where a property could
pass by accident (a "perturbation" that silently does nothing, a schedule that is zero
everywhere), the test includes the positive control that rules that out.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _strict import import_torch_or_skip  # noqa: E402

torch = import_torch_or_skip()  # skips this module where torch is absent
import torch.nn.functional as F  # noqa: E402

from harness.model import (  # noqa: E402
    CANDIDATE_CONFIGS,
    ELIGIBLE_SUFFIXES,
    ModelConfig,
    TinyDecoder,
)
from harness.perturbation import (  # noqa: E402
    AnnealedPerturbation,
    AnnealedLinear,
    cosine_sigma,
    eligible_module_names,
    realized_relative_perturbation,
)

SMOKE = CANDIDATE_CONFIGS["smoke_d64_l2"]           # d64, 2 layers: seconds per test
SIGMA0 = 0.03


def make_model(cfg: ModelConfig = SMOKE, seed: int = 0) -> TinyDecoder:
    torch.manual_seed(seed)
    return TinyDecoder(cfg)


def make_controller(model, sigma0=SIGMA0, total_steps=10, seed=1234):
    return AnnealedPerturbation(model, sigma0=sigma0, total_steps=total_steps,
                                perturbation_seed=seed)


# ------------------------------------------------------------------ the schedule
def test_schedule_is_exactly_zero_at_the_final_step():
    """The endpoint is the deterministic model, so sigma(T) must be 0.0, not ~0."""
    for T in (1, 2, 7, 100):
        assert cosine_sigma(T, T, SIGMA0) == 0.0, T


def test_schedule_starts_at_sigma0_and_halves_at_midpoint():
    T = 100
    assert cosine_sigma(0 + 1, T, SIGMA0) == pytest.approx(SIGMA0 * 0.5 * (1 + math.cos(math.pi / T)))
    assert cosine_sigma(T // 2, T, SIGMA0) == pytest.approx(SIGMA0 / 2)
    # the first update is close to sigma0 but strictly below it
    assert 0 < cosine_sigma(1, T, SIGMA0) < SIGMA0


def test_schedule_is_monotone_non_increasing():
    T = 50
    values = [cosine_sigma(t, T, SIGMA0) for t in range(1, T + 1)]
    assert all(b <= a + 1e-15 for a, b in zip(values, values[1:])), values
    assert values[-1] == 0.0
    assert values[0] > 0.9 * SIGMA0


def test_schedule_rejects_out_of_range_steps():
    for bad in (0, -1, 11):
        with pytest.raises(ValueError):
            cosine_sigma(bad, 10, SIGMA0)
    with pytest.raises(ValueError):
        cosine_sigma(1, 0, SIGMA0)
    with pytest.raises(ValueError):
        cosine_sigma(1, 10, -0.1)


def test_zero_sigma0_gives_zero_everywhere():
    """The baseline arm runs through the same code path with sigma0 = 0."""
    assert all(cosine_sigma(t, 10, 0.0) == 0.0 for t in range(1, 11))


# ------------------------------------------------------- target-module correctness
def test_only_the_declared_projections_are_eligible():
    model = make_model()
    names = eligible_module_names(model)
    n = model.cfg.n_layers
    assert len(names) == 7 * n, names          # 4 attention + 3 MLP per layer
    for suffix in ELIGIBLE_SUFFIXES:
        assert sum(1 for x in names if x.endswith(suffix)) == n
    for excluded in ("tok_emb", "lm_head", "norm1", "norm2", "norm_f"):
        assert not any(excluded in x for x in names), excluded


def test_wrapping_replaces_exactly_the_eligible_modules():
    model = make_model()
    before = dict(model.named_modules())
    ctl = make_controller(model)
    after = dict(model.named_modules())
    became_annealed = {
        k for k, m in before.items()
        if isinstance(m, torch.nn.Linear) and isinstance(after.get(k), AnnealedLinear)
    }
    assert became_annealed == set(ctl.modules)
    assert ctl.modules.keys() == set(names_with_suffixes(before))
    assert isinstance(model.blocks[0].attn.q_proj, AnnealedLinear)
    assert not isinstance(model.tok_emb, AnnealedLinear)
    assert not isinstance(model.lm_head, AnnealedLinear)
    # wrapping must not change the parameter objects, or an already-built optimizer and
    # every checkpoint key would silently point somewhere else
    for name, module in ctl.modules.items():
        assert module.base.weight is before[name].weight


def names_with_suffixes(modules: dict) -> list[str]:
    return [k for k, m in modules.items()
            if isinstance(m, torch.nn.Linear)
            and any(k.endswith(s) for s in ELIGIBLE_SUFFIXES)]


def test_eligible_discovery_rejects_a_biased_linear():
    class WithBias(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.q_proj = torch.nn.Linear(4, 4, bias=True)

    with pytest.raises(ValueError, match="bias"):
        eligible_module_names(WithBias())


# ------------------------------------------------------------------- the weights
def test_zero_noise_is_bit_identical_to_the_unperturbed_model():
    """Phase 0 item 1. 'Close to' is not good enough: it must be the same tensor."""
    model = make_model(seed=7)
    tokens = torch.randint(0, SMOKE.vocab_size, (2, 16))
    with torch.no_grad():
        reference = model(tokens).clone()

    ctl = make_controller(model, sigma0=0.0)
    for step in (1, 5, 10):
        ctl.start_step(step)
        assert all(m.effective_weight() is m.base.weight for m in ctl.modules.values())
        with torch.no_grad():
            assert torch.equal(model(tokens), reference), step
        ctl.finish_step()


def test_eval_mode_never_perturbs_even_with_sigma_above_zero():
    model = make_model(seed=3)
    tokens = torch.randint(0, SMOKE.vocab_size, (2, 16))
    with torch.no_grad():
        reference = model(tokens).clone()

    ctl = make_controller(model, sigma0=SIGMA0)
    model.eval()
    ctl.start_step(1)
    assert not any(m.perturbing for m in ctl.modules.values())
    with torch.no_grad():
        assert torch.equal(model(tokens), reference)
    # positive control: the same scale in train mode does change the output
    model.train()
    ctl.start_step(1)
    with torch.no_grad():
        assert not torch.equal(model(tokens), reference)


def test_final_step_leaves_the_stored_weight_untouched():
    """The optimizer must only ever see W, and at the endpoint W_eff must be W."""
    model = make_model()
    ctl = make_controller(model, total_steps=10)
    w0 = {n: m.base.weight.detach().clone() for n, m in ctl.modules.items()}
    for step in range(1, 11):
        ctl.start_step(step)
        for m in ctl.modules.values():
            if step < 10:
                assert m.effective_weight() is not m.base.weight
                assert m.effective_weight().data_ptr() != m.base.weight.data_ptr()
            else:
                assert m.effective_weight() is m.base.weight
        ctl.finish_step()
    for n, m in ctl.modules.items():
        assert torch.equal(m.base.weight.detach(), w0[n]), "the stored weight never moved"


def test_one_noise_draw_per_update_shared_across_microbatches():
    """docs/01: one perturbation per matrix per optimizer update."""
    model = make_model()
    ctl = make_controller(model, total_steps=4, seed=99)
    tokens = torch.randint(0, SMOKE.vocab_size, (1, 8))

    ctl.start_step(1)
    first = {n: m.effective_weight().clone() for n, m in ctl.modules.items()}
    for _ in range(3):                       # three microbatches inside one update
        for m in ctl.modules.values():
            assert m.perturbing             # no RNG draw, but still perturbed
    with torch.no_grad():
        model(tokens)
    second = {n: m.effective_weight().clone() for n, m in ctl.modules.items()}
    for n in first:
        assert torch.equal(first[n], second[n]), n

    ctl.start_step(2)
    third = {n: m.effective_weight().clone() for n, m in ctl.modules.items()}
    assert any(not torch.equal(first[n], third[n]) for n in first), "step 2 must redraw"


def test_realized_perturbation_follows_the_rms_relative_law_exactly():
    """W_eff = W + sigma * RMS(W) * Z, so the realized ratio is sigma * RMS(Z)/RMS(W).

    Asserted exactly rather than approximately: a loose tolerance would accept a mutant
    that used a fixed absolute scale, or that used sigma0 instead of the schedule's value
    for this step. Halving the scale used must halve the realized perturbation, which is
    what distinguishes the RMS-relative law from any constant-offset variant.
    """
    model = make_model()
    ctl = make_controller(model, sigma0=SIGMA0, total_steps=10, seed=5)
    name = sorted(ctl.modules)[0]
    module = ctl.modules[name]
    ctl.start_step(1)

    ratio = realized_relative_perturbation(module.base.weight, module.effective_weight())
    z = module._z
    # Algebra of the declared law: dW = sigma * RMS(W) * Z, so dividing by RMS(W) cancels
    # the weight's own scale and leaves exactly sigma * RMS(Z). That cancellation is the
    # reason RMS-relative scaling makes the relative perturbation scale-free, and it is
    # what a mutant using an absolute step, or the wrong sigma, cannot reproduce.
    predicted = module.sigma * float(z.pow(2).mean().sqrt())
    assert ratio == pytest.approx(predicted, rel=1e-6)
    assert float(z.pow(2).mean().sqrt()) == pytest.approx(1.0, rel=0.05)  # RMS(Z) ~ 1
    assert module.sigma != SIGMA0, "step 1 of a cos schedule is already below sigma0"
    assert ratio < SIGMA0

    half = AnnealedLinear(module.base, module.generator, name)
    half.sigma = module.sigma / 2
    half._z = z
    assert realized_relative_perturbation(module.base.weight, half.effective_weight()) == \
        pytest.approx(ratio / 2, rel=1e-6)


def test_perturbation_is_measured_after_a_dtype_cast():
    """Phase 0 item 4: with a bf16 weight, the noise is quantised; record the realized value."""
    model = make_model()
    ctl = make_controller(model, sigma0=0.5, total_steps=10, seed=11)  # large sigma, visible
    module = ctl.modules[sorted(ctl.modules)[0]]
    ctl.start_step(1)
    fp32_ratio = realized_relative_perturbation(module.base.weight, module.effective_weight())
    module.base.weight.data = module.base.weight.data.to(torch.bfloat16)
    cast_ratio = realized_relative_perturbation(module.base.weight, module.effective_weight())
    assert ctl.modules[sorted(ctl.modules)[0]].effective_weight().dtype == torch.bfloat16
    assert cast_ratio == pytest.approx(fp32_ratio, rel=0.05)


# ------------------------------------------------------------------ rng isolation
def test_perturbation_does_not_consume_the_global_rng():
    """Phase 0 item 6: the data-loader and dropout streams must be untouched."""
    model = make_model()
    ctl = make_controller(model, sigma0=SIGMA0)
    tokens = torch.randint(0, SMOKE.vocab_size, (1, 8))
    before = torch.get_rng_state().clone()
    for step in (1, 2):
        ctl.start_step(step)
        model(tokens)
        ctl.finish_step()
    assert torch.equal(torch.get_rng_state(), before)


def test_same_seed_reproduces_and_different_seed_does_not():
    a = make_model(seed=42)
    ctl_a = make_controller(a, seed=2024)
    ctl_a.start_step(1)
    wa = {n: m.effective_weight().clone() for n, m in ctl_a.modules.items()}

    b = make_model(seed=42)                  # identical init too
    ctl_b = make_controller(b, seed=2024)
    ctl_b.start_step(1)
    for n, m in ctl_b.modules.items():
        assert torch.equal(wa[n], m.effective_weight()), n

    c = make_model(seed=42)
    ctl_c = make_controller(c, seed=2025)
    ctl_c.start_step(1)
    assert any(not torch.equal(wa[n], m.effective_weight()) for n, m in ctl_c.modules.items())


def test_resume_continues_the_perturbation_stream():
    """Phase 0 item 7 precursor: a resumed run must draw the noise it would have drawn."""
    a = make_model(seed=1)
    ctl_a = make_controller(a, total_steps=4, seed=17)
    for step in (1, 2):
        ctl_a.start_step(step)
        ctl_a.finish_step()
    ctl_a.start_step(3)
    expected = {n: m.effective_weight().clone() for n, m in ctl_a.modules.items()}
    ctl_a.finish_step()

    b = make_model(seed=1)
    ctl_b = make_controller(b, total_steps=4, seed=17)
    for step in (1, 2):
        ctl_b.start_step(step)
        ctl_b.finish_step()
    state = ctl_b.state_dict()

    c = make_model(seed=1)
    ctl_c = make_controller(c, total_steps=4, seed=999)   # deliberately wrong seed
    ctl_c.load_state_dict(state)
    ctl_c.start_step(3)
    for n, m in ctl_c.modules.items():
        assert torch.equal(expected[n], m.effective_weight()), n


def test_load_state_rejects_a_different_module_set():
    a = make_model()
    ctl_a = make_controller(a, total_steps=4)
    state = ctl_a.state_dict()
    smaller = make_model(ModelConfig(d_model=64, n_layers=1, n_heads=4, max_seq_len=64))
    ctl_b = make_controller(smaller, total_steps=4)
    with pytest.raises(ValueError, match="modules differ"):
        ctl_b.load_state_dict(state)


# ------------------------------------------------------------- gradients, accounting
def test_gradients_reach_the_base_weight_and_are_finite():
    """No accidental detachment: the perturbation must not sever the learning signal."""
    model = make_model()
    ctl = make_controller(model, sigma0=SIGMA0, total_steps=10)
    tokens = torch.randint(0, SMOKE.vocab_size, (2, 16))
    targets = torch.randint(0, SMOKE.vocab_size, (2, 16))

    ctl.start_step(1)
    logits = model(tokens)
    loss = F.cross_entropy(logits.view(-1, SMOKE.vocab_size), targets.view(-1))
    loss.backward()
    for name, module in ctl.modules.items():
        grad = module.base.weight.grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, f"{name} received no gradient"
    assert not any(m._z.requires_grad for m in ctl.modules.values())


def test_candidate_scale_matches_the_reference_family_column():
    """The non-embedding count is the column the reference family is named by (docs/07).

    The comparison only means something at matched scale, so the candidate configs are
    checked against the *measured* reference non-embedding counts rather than against a
    number of my own choosing.
    """
    measured_reference = {                    # docs/07 section 5
        "TinyStories-8M": 6_840_000,
        "TinyStories-28M": 26_260_000,
    }
    pairs = [("screen_d256_l8", "TinyStories-8M"), ("verify_d512_l8", "TinyStories-28M")]
    for ours, theirs in pairs:
        counts = make_model(CANDIDATE_CONFIGS[ours]).param_counts()
        ratio = counts["non_embedding"] / measured_reference[theirs]
        assert 0.9 <= ratio <= 1.1, f"{ours} is {ratio:.3f}x {theirs} on non-embedding params"

    # and the accounting itself must be coherent for every candidate
    for name, cfg in CANDIDATE_CONFIGS.items():
        counts = make_model(cfg).param_counts()
        assert counts["total"] == counts["non_embedding"] + counts["embedding"], name
        assert counts["embedding"] == cfg.vocab_size * cfg.d_model, name

    # the embedding dominates at this scale, which is why the non-embedding column is the
    # one to match on: for the screen config it is two thirds of the total
    counts = make_model(CANDIDATE_CONFIGS["screen_d256_l8"]).param_counts()
    assert counts["embedding"] > 2 * counts["non_embedding"]


def test_model_config_rejects_incoherent_shapes():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=100, n_heads=8)
    with pytest.raises(ValueError, match="even"):
        ModelConfig(d_model=12, n_heads=4)      # head dim 3
    with pytest.raises(ValueError, match="context"):
        ModelConfig(max_seq_len=4096)


def test_tied_embeddings_share_one_tensor():
    tied = make_model(ModelConfig(d_model=64, n_layers=1, n_heads=4, max_seq_len=64,
                                  tie_embeddings=True))
    assert tied.lm_head.weight is tied.tok_emb.weight
    untied = make_model(ModelConfig(d_model=64, n_layers=1, n_heads=4, max_seq_len=64,
                                    tie_embeddings=False))
    assert untied.lm_head.weight is not untied.tok_emb.weight


def test_forward_rejects_sequences_beyond_the_frozen_context():
    model = make_model()
    with pytest.raises(ValueError, match="exceeds"):
        model(torch.zeros(1, SMOKE.max_seq_len + 1, dtype=torch.long))
