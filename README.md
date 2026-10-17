# ASWP — Annealed Stochastic Weight Perturbation

> **An open experiment.** Does adding zero-mean Gaussian noise to Transformer weight matrices during training — annealed to exactly zero by the final optimization step — improve final deterministic validation NLL over an otherwise identical deterministic baseline?

**Status:** 🔬 **Pre-results.** This repository contains the experimental protocol, engineering specification, and statistical analysis plan. **No claim of improvement is made** until the experiments complete and the preregistered analysis is run.

## The question

Training a Transformer with noise on the effective forward weights, while the optimizer updates the clean base parameters, is a cheap way to probe the loss landscape. The noise changes where gradients are evaluated but never touches the weights you keep. If the noise is annealed to zero by the end of training, the final model is exactly the deterministic model — only the *trajectory* was different.

The experiment asks five questions:

1. Does annealed stochastic weight perturbation improve final deterministic validation NLL (TinyStories, decoder-only Transformer)?
2. Is any effect larger than baseline seed-to-seed variability, under a predeclared threshold?
3. Does annealing beat a matched constant-noise control — or is any improvement just generic stochastic regularization?
4. Do schedule shape, layer targeting, scaling, and model scale change the outcome?
5. Does a confirmed effect transfer from TinyStories to a fixed FineWeb-Edu subset?

This is deliberately a **controlled, falsifiable experiment**: a credible negative result is treated as a useful result.

## Experimental design (summary)

| | |
|---|---|
| **Method** | `W_eff(t) = W + σ(t) · RMS(W) · Z`, Z ~ 𝒩(0, I), sampled fp32, training-only, detached RMS scale |
| **Schedule** | σ(t) = σ₀ · 0.5 · (1 + cos(π·t/T)), reaching **exactly zero** at the final step |
| **Targets** | Attention q/k/v/o + MLP up/gate/down projections (no embeddings, head, biases, norms) |
| **Data** | TinyStories (screen + confirmation) → fixed FineWeb-Edu subset (transfer) |
| **Models** | ~10M screen → 30–50M verification |
| **Endpoint** | Final **deterministic** validation NLL at a fixed step, frozen validation stream |
| **Analysis** | Paired seeds, exact sign-flip test, predeclared practical threshold δ |

**Arms:** C0 deterministic baseline · C1–C3 annealed screen (σ₀ = 0.01 / 0.03 / 0.05) · C4 constant noise · C5 DropConnect comparator · C6 RWP comparator · C7 LR-coupled noise.

## Repository layout

```
docs/
├── 01-research-protocol.md                  # methodology, arms, phases, decision criteria
├── 02-implementation-operations-manual.md   # reference module, unit tests, logging, checkpoints
├── 03-statistical-analysis-plan.md          # SAP + preregistration (config fields to be locked)
├── 04-design-review.md                      # red-team review of the protocol (historical record)
├── 05-blocker-resolution.md                 # resolutions + regenerated values for issues #1–#16
└── 06-preregistration-checklist.md          # what is frozen, what is open, what blocks what
config/
└── manifest.json                            # frozen paired seed list and per-role RNG seeds (#10)
analysis/                                    # statistical analysis: exact sign-flip test, power,
                                             # selection/margin, δ criteria — with tests
tests/                                       # pytest suite for the analysis core
```

The statistical analysis is executable and tested today (see `analysis/` and `tests/`);
the training harness (`AnnealedLinear`, training loop, Phase 0 gate) is **not yet written**
and is the next milestone — see [06-preregistration-checklist.md](docs/06-preregistration-checklist.md).

## Design review & open issues

The protocol went through an adversarial design review before publication; the findings are tracked as [GitHub issues](https://github.com/kragent66-glitch/aswp/issues), and each is resolved or open in [docs/05-blocker-resolution.md](docs/05-blocker-resolution.md):

- ✅ **n=5 cannot reach α=0.05 two-sided** — resolved: one-sided test locked, minimum raised to n=6, target n=8
- ✅ **C4 constant-noise arm not variance-matched** — resolved: σ_c = 0.6124·σ₀ (both arms run)
- ✅ **Winner's curse in screen → confirm** — resolved: pre-registered selection-stability check
- ⛔ **The SAP config fields are not locked yet** (dataset/tokenizer/validation hashes, model commit, training configuration, selected σ₀) — these cannot be filled before the harness exists and the Phase 0/1 runs produce them. No confirmatory run may start until they are.
- ⛔ **The training harness is not written** — nothing runs on a GPU yet

How to run what exists:

```bash
python -m pip install numpy scipy pytest
python -m pytest                      # 64 tests
python analysis/p1_exact_power.py     # resolution + power, units conversion
python analysis/p23_match_curse.py    # variance matching, winner's curse, margin rule
python analysis/p4_delta_criteria.py  # decision-rule fire rates, type-I error, δ under units
```

## Open science

Pre-results protocol release, immutable configuration manifests, seeds, raw logs, analysis scripts, and a failed-run ledger are all part of the plan. Code and documentation are released under this repository's license (all rights reserved); checkpoints only where dataset/model licensing permits.

## License

All rights reserved — see [LICENSE](LICENSE).
