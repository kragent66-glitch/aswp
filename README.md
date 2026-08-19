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
├── 03-statistical-analysis-plan.md          # SAP + preregistration (to be locked)
└── 04-design-review.md                      # red-team review of the protocol (open issues)
```

Code (training harness, `AnnealedLinear`, Phase 0 unit-test suite) lands here once the implementation phase starts — the protocol is versioned and pushed **before** any confirmatory run, per the open-science plan.

## Design review & open issues

The protocol went through an adversarial design review before publication; the findings are tracked as [GitHub issues](https://github.com/kragent66-glitch/aswp/issues). Known blockers that must be resolved before the SAP is locked:

- n=5 paired seeds cannot reach α=0.05 with a two-sided exact test (min p = 0.0625) — hypothesis direction or sample size must be locked
- C4 constant-noise arm is not variance-matched (matched level ≈ 0.61·σ₀) — the annealing-specific claim needs the matched comparison
- Screen → confirmation selection bias (winner's curse) needs a pre-registered stability check

## Open science

Pre-results protocol release, immutable configuration manifests, seeds, raw logs, analysis scripts, and a failed-run ledger are all part of the plan. Code and documentation are released under this repository's license (all rights reserved); checkpoints only where dataset/model licensing permits.

## License

All rights reserved — see [LICENSE](LICENSE).
