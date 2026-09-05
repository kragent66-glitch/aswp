# ASWP — Annealed Stochastic Weight Perturbation for Transformer Training

**An open experiment.** Pre-results experimental protocol — methodology, design, and decision framework.

**Author:** Utkarsh Bhangale · **Version:** 1.0 · **Date:** August 2026 · **Repository:** [github.com/kragent66-glitch/aswp](https://github.com/kragent66-glitch/aswp)

# Document Status
This is a pre-results technical white paper and preregistered-style experimental protocol. It proposes a method and a falsifiable evaluation plan. It does not claim that the method improves generalization, convergence, robustness, stability, or loss-landscape flatness before experiments are completed.

# Executive Summary
This project tests whether zero-mean, training-only Gaussian perturbations of selected Transformer weight matrices, gradually annealed to zero by the final optimization step, improve deterministic validation negative log-likelihood relative to an otherwise identical deterministic baseline. The method changes the effective weights used in the training forward pass, while the optimizer continues to own and update the unperturbed base parameters.
The central scientific question is not whether noise can act as regularization in general. It is whether an annealed-to-zero weight-noise mechanism produces a reproducible benefit beyond deterministic training and beyond a matched constant-noise control. The study is deliberately structured to make a credible negative result useful.

# Research Questions
Does annealed stochastic weight perturbation improve final deterministic validation NLL in a decoder-only Transformer trained on TinyStories?
Is a detected effect larger than baseline seed-to-seed variability and practically meaningful under a predeclared threshold?
Does annealing outperform a matched constant-noise control, or is any improvement merely generic stochastic regularization?
Do schedule shape, layer targeting, scaling convention, and model scale materially change the outcome?
Does any confirmed effect transfer from TinyStories to a fixed FineWeb-Edu subset?

# Related Work Positioning
The proposal is related to sharpness-aware minimization, SGD-induced stochasticity, Dropout and DropConnect, Entropy-SGD, and especially random weight perturbation (RWP). Existing RWP methods already study random parameter perturbations as a computationally cheaper alternative or complement to sharpness-aware approaches. Accordingly, this project must not claim novelty for weight perturbation itself.
The proposed contribution is narrower: a transparent, controlled, decoder-only Transformer language-model study of isotropic Gaussian weight perturbations with layer-relative scaling and a schedule that reaches exactly zero at terminal training, compared against deterministic, constant-noise, and regularization controls.

# Method Definition
For each eligible weight matrix W at training step t, define the effective training weight as W_eff(t) = W + epsilon(t). Let Z be sampled from a standard normal distribution with the same shape as W. The primary method uses epsilon(t) = sigma(t) times RMS(W) times Z, where RMS(W) = sqrt(mean(W squared)).
The perturbation is sampled without gradient tracking, while W_eff remains connected to W through ordinary addition. Thus gradients flow to W but are evaluated at the perturbed point. Evaluation, checkpoint loading, and inference always use W_eff = W with no sampled perturbation.

# Primary Design Choices
Distribution: independent Gaussian noise.
Scaling: RMS-relative scaling in the primary experiment; standard-deviation-relative scaling is a later ablation.
Target modules: attention and MLP projection linear layers; exclude embeddings, output head, biases, LayerNorm/RMSNorm parameters, and tied weights in the first study.
Schedule: cosine annealing from sigma0 at step zero to exactly zero at final step.
Sampling cadence: one perturbation per eligible matrix per optimizer update, shared across all gradient-accumulation microbatches.
Precision: construct noise and perturbation in fp32; explicitly measure realized perturbation after any bf16/fp16 cast.

# Schedules
Let p = t / max(T - 1, 1). The primary schedule is sigma(t) = sigma0 times 0.5 times (1 + cos(pi p)). At t = T - 1, sigma(t) must be forcibly set to zero exactly. Secondary schedules are linear and learning-rate-coupled schedules. The learning-rate-coupled ablation is essential because optimizer stochasticity often changes as learning rate decays.

# Theoretical Motivation
Training with sampled perturbations can be interpreted as minimizing a Gaussian-smoothed objective: the expected loss over local perturbations of the current parameter vector. Under a small-noise second-order approximation, the difference from the point loss contains a curvature-related contribution. This motivates—but does not prove—the possibility that the method may favor solutions with lower local sensitivity.
The stochastic gradient is evaluated at W + epsilon rather than W. A first-order expansion implies added gradient variability proportional to perturbation variance and local curvature. This may aid escape or regularization early in optimization but can hinder final convergence; annealing is intended to reduce the added variability to zero by the end.

# Experimental Conditions

## C0: Deterministic baseline
sigma0 = 0, executed through the same replacement/code path whenever possible.

## C1–C3: Annealed screen
Cosine RMS-relative sigma0 values of 0.01, 0.03, and 0.05, one exploratory seed each.

## C4: Constant noise — two arms (locked, basis: docs/05-blocker-resolution.md §P2)
- **C4a: variance-matched, σ_const = 0.6124·σ0.** For f(p) = 0.5(1+cos πp), ∫₀¹ f(p)² dp = 3/8, so σ_const = σ0·√(3/8) = 0.6124·σ0 delivers the same integrated noise variance as the annealed schedule. This is the arm that separates "the schedule shape matters" from "any noise at this dose is equally good" — the annealing-specific claim requires C4a.
- **C4b: σ0 held constant.** The original specification, delivering 8/3 = 2.67× the annealed noise dose; retained as the "too much noise" extreme that documents why annealing exists.

## C5: DropConnect comparator
A tuned multiplicative weight-noise comparator with comparable module coverage.

## C5b: Vanilla dropout comparator
Activation dropout — the standard, cheapest, best-understood regularizer — tuned with the SAME fixed budget and the same screen rule as the treatment (symmetric effort; the screen rule thresholds in Phase 1 apply unchanged). Naive per-weight variance matching is explicitly NOT used as the tuning target: sigma0 in [0.01, 0.05] corresponds to per-weight drop rates of roughly 0.01–0.5%, while standard dropout operates at 5–20% — matched-strength dropout is not a meaningful comparator regime (calibration in docs/05-blocker-resolution.md §P2). The paper must state this dose asymmetry and report the tuned dropout level. This arm answers the practical question: does ASWP beat just cranking dropout at equal cost?

## C6: RWP comparator
Optional reproduction of a documented RWP formulation where engineering budget permits.

## C7: LR-coupled noise
Noise proportional to the learning-rate schedule, testing whether independently chosen annealing adds value.

# Phased Execution

## Phase 0: Correctness and Cost
Use a 1M–3M model and fixed tiny shard.
Verify sigma=0 output, loss, and gradient equivalence against the untouched baseline.
Verify nonzero realized perturbation and deterministic eval output.
Measure throughput, peak memory, and perturbation-generation overhead.
Test checkpoint/resume and gradient accumulation semantics.

## Phase 1: Exploratory Screen
Use a 10M model and fixed token budget.
Run C0 and the three annealed sigma conditions.
Select one candidate using a prewritten fixed rule: final deterministic validation NLL at a fixed endpoint, subject to stability and throughput requirements.
Do not select based on best checkpoint, visual curve preference, or unlogged trial reruns.

## Phase 2: Confirmation
Run baseline and the single selected annealed configuration on at least 6 paired seed pairs (minimum; 8–10 are targeted — one-sided α=0.05 requires n≥6, and n=8 gives 80% power at a 1.0·SD effect; see docs/05-blocker-resolution.md §P1 and SAP §Sample Size).
Pair model initialization, data order, dropout, optimizer settings, and all non-perturbation RNG streams.
Run the selected constant-noise controls (C4a variance-matched at 0.6124·σ0 and C4b at σ0) on the same seeds before claiming that annealing itself matters.

## Phase 3: Ablations
Schedule shape, target modules, scaling convention, DropConnect, RWP comparator, weight-decay/dropout controls, and initialization sensitivity.
Ablations remain exploratory unless separately preregistered and sufficiently replicated.

## Phase 4: Transfer
Verify a pre-selected configuration at 30M–50M parameters on TinyStories.
Only then evaluate the same locked configuration on a fixed FineWeb-Edu subset.
Do not perform a new wide sweep in the transfer phase to rescue an inconclusive first-stage result.

# Statistical Plan
The confirmatory estimand is d_i = NLL_baseline,i minus NLL_annealed,i at an exactly fixed terminal training endpoint, evaluated deterministically on exactly the same validation token stream. Positive d_i favors the treatment.
Report all seed pairs, mean difference, median difference, standard deviation, and a 95% bootstrap confidence interval.
Use an exact paired sign-flip/randomization test as the primary hypothesis test; report paired t-test only as a sensitivity analysis.
Define a practical improvement threshold delta from a baseline-only calibration pilot before confirmatory treatment results are examined.
Treat sigma sweeps as exploratory. Only one pre-selected configuration advances to the confirmatory analysis.
Never interpret a single seed, best checkpoint, or minimum validation loss across a sweep as confirmation.

# Metrics

## Primary
Final deterministic validation NLL at a pre-specified step/token budget.
Perplexity = exp(validation NLL), reported descriptively.

## Secondary
Validation NLL trajectory, token-weighted area under the curve, and time to fixed NLL thresholds.
Training loss, clean evaluation loss, and optionally the noisy-forward training loss.
Gradient norm, clipped-gradient rate, update norm, parameter norm, activation statistics, NaN/Inf events.
Perturbation norm, perturbation-to-weight ratio, layerwise ratios, throughput, peak memory, and cost per million tokens.
Post-training local sensitivity, Hessian proxies, and interpolation analyses, all exploratory only.

# Stability Rules
Stop and retain a run after persistent non-finite loss or gradient events according to a predeclared recovery policy.
Define a rolling clipping-rate threshold and sustained validation degradation threshold before launch.
Apply identical rules to every arm. Infrastructure failures are rerunnable; numerical instability is a scientific outcome.

# Dataset and Model
TinyStories is used as a controlled low-cost causal-language-modeling environment, not as a broad capability benchmark. Freeze the exact dataset revision, preprocessing code, tokenizer, split membership, packed-sequence behavior, and post-tokenization hash before the first run.
Pilot model: approximately 1M–3M parameters.
Screen model: approximately 10M parameters.
Verification model: approximately 30M–50M parameters.
Architecture: fixed decoder-only Transformer with fixed sequence length, vocabulary, optimizer, learning-rate schedule, batch configuration, precision, clipping, and token budget.

# Reproducibility
Record global, initialization, data-loader, dropout, and perturbation RNG seeds separately.
Use a dedicated torch.Generator for perturbations.
Record source commit, environment lock file, CUDA/PyTorch versions, GPU, driver, precision mode, deterministic settings, and hardware details.
Store immutable configuration manifests, append-only raw metric logs, checkpoint metadata, and a ledger of all failed/aborted runs.
Save deterministic base parameters only; never permanently write sampled perturbations into checkpoints.

# Known Limitations
A TinyStories gain may be synthetic-corpus-specific and may not transfer.
Flatness measures are parameterization-dependent, so they are supporting evidence rather than proof.
The experiment may reproduce known RWP behavior rather than establish a novel mechanism.
Small seed counts create wide uncertainty; lack of significance is not proof of no effect.
Noise overhead or mixed-precision quantization can erase practical value even if NLL improves.

# Decision Criteria
Confirmed favorable signal: paired effect exceeds the predeclared practical threshold with compatible uncertainty, no unacceptable stability regression, and the result survives the constant-noise comparison.
Regularization-only finding: treatment and constant-noise control perform similarly and exceed baseline.
Negative result: treatment is inferior, unstable, or excludes a meaningful benefit at tested settings.
Inconclusive result: uncertainty includes both meaningful benefit and harm/no-benefit.

# Open Science Plan
Release code, immutable configurations, seeds, raw logs, data-preprocessing recipe and hashes, analysis scripts, environment specification, and failed-run ledger. Release checkpoints only where dataset and model licensing permits. Tag a pre-results protocol release before confirmed experimentation begins.

# References to Include
Foret et al., Sharpness-Aware Minimization for Efficiently Improving Generalization. Li et al., Efficient Generalization Improvement Guided by Random Weight Perturbation. Li et al., Revisiting Random Weight Perturbation for Efficiently Improving Generalization. Eldan and Li, TinyStories: How Small Can Language Models Be and Still Speak Coherent English. PyTorch Reproducibility Documentation.