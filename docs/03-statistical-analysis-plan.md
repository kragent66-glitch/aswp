# ASWP — Statistical Analysis Plan and Preregistration

**Lock this document before confirmatory training starts.**

**Author:** Utkarsh Bhangale · **Version:** 1.0 · **Date:** August 2026 · **Repository:** [github.com/kragent66-glitch/aswp](https://github.com/kragent66-glitch/aswp)

# Purpose and Status
Complete every bracketed field before confirmatory data collection. Commit the completed document alongside the code and configuration manifest. Modifications afterward are protocol deviations and must be reported rather than silently incorporated.

# Confirmatory Question
For the selected annealed RMS-relative Gaussian perturbation configuration, does deterministic final validation NLL improve compared with a deterministic baseline across fixed paired training seeds under otherwise identical conditions?

# Locked Configuration Fields
Dataset revision and SHA-256: <value>.
Tokenizer identity/version/hash: <value>.
Validation split and token-stream hash: <value>.
Model source commit and architecture: <value>.
Total optimizer steps and tokens: <value>.
Optimizer, learning-rate schedule, warmup, decay, clipping, dropout, precision: <values>.
Eligible module list: <values>.
Noise distribution: Gaussian; scaling: RMS-relative; schedule: cosine.
Selected sigma0: <value>.
Selection rule and exploratory parent run ID: <value>.
Paired seed list: 8 integers (absolute minimum 6), generated once and frozen in the config manifest (one-sided α=0.05 needs n≥6 to be achievable; n=8 gives 80% power at a 1.0·SD effect — see # Sample Size).

# Endpoint and Estimand
Primary endpoint: deterministic validation NLL at checkpoint step <value>, evaluated on the frozen validation token stream. Let d_i equal baseline NLL minus annealed NLL for paired seed i. A positive d_i favors annealed perturbation. Best checkpoint and minimum-over-time NLL are not confirmatory endpoints.

# Hypotheses
H0: paired differences are symmetric around zero. H1: annealed treatment improves NLL, such that the paired difference tends to be positive.
**Primary test: ONE-SIDED (locked 2026-09-05, basis: docs/05-blocker-resolution.md §P1).** Justification: the decision ladder has no use for "significantly worse than baseline" — a worse-than-baseline result is reported as an estimate regardless, so only the positive direction carries a decision. A two-sided test is retained only as a reviewer fallback; it is achievable from n=6 (min two-sided p = 0.0313).

# Sample Size
Minimum: **6 paired seeds** — below this a two-sided α=0.05 test is mathematically impossible (n=5: min achievable two-sided p = 2/32 = 0.0625; simulated power at α=0.05 is exactly 0.000).
Target: **8 paired seeds** (preferred 8–10). The exact one-sided sign-flip test at α=0.05 has 80% power at a 1.0·SD effect and 98% at 1.5·SD at n=8 (40,000-trial simulation, seed 42 — docs/05-blocker-resolution.md §P1 power table). At n=10 power at 1.0·SD rises to 89.7%. A result with 6 pairs should be described conservatively (only 64.7% power at 1.0·SD).

# Primary Analysis
Collect final deterministic validation NLL for every completed paired run.
Compute and publish each d_i, then its mean, median, standard deviation, and a **95% randomization confidence interval** (invert the exact sign-flip test: the set of δ such that the test of d−δ fails to reject — exact under the same exchangeability assumption as the primary test).
Compute an exact paired sign-flip/randomization p-value by enumerating the 2^n signs of paired differences.
Report a 95% percentile bootstrap CI (10000 resamples) only as descriptive sensitivity — at n≤8 the percentile bootstrap is quantized to a coarse grid (126 distinct resampled means at n=5) with unreliable tail coverage.
Report a paired t-test and a Wilcoxon signed-rank test only as sensitivity analyses; do not rely on untestable normality assumptions at small n.
Plot individual pairs; do not show only aggregated bars.

# Practical Significance
**Delta is locked: δ = 1.0 × SD_baseline** (SD of baseline NLL across seeds, measured in Phase 0/1 baseline-only calibration before any treatment result is examined). It is the smallest practically important mean paired improvement.
**"Compatible uncertainty" is locked:** confirmed favorable signal = point estimate > δ AND 95% randomization CI excludes 0 (primary; fires with ~91% probability at a true 1.5·SD effect at n=8, ~47% at 1.0·SD — docs/05-blocker-resolution.md §P4). Sensitivity: strong criterion — 95% CI lower bound > 0.5·SD_baseline. The review's original strong-k=1 rule is explicitly rejected as unconfirmable at n≤8 (≤23% fire rate at 1.5·SD).

# Multiplicity and Selection
All sigma, schedule, scaling, initialization, layer-target, and regularization sweeps are exploratory.
Only one configuration selected by the written screen rule may enter this confirmatory analysis.
If multiple configurations are confirmed, control multiplicity or label all results exploratory.
The constant-noise comparison is required before making an annealing-specific claim.
**Screen selection rule (numeric, locked):** chosen σ0 must complete with ≤ 1 instability event and throughput ≥ 80% of baseline; otherwise the next-best σ0 is considered, in order.
**Selection-stability check (locked):** the chosen σ0 AND the runner-up σ0 each run on 2 fresh seeds in Phase 2. Any ranking reversal downgrades the confirmatory claim to exploratory. (The check has 50% flip probability at a true tie, ≤ 31% at 0.5·SD separation, 7% at 1.5·SD — docs/05-blocker-resolution.md §P3. Cost: 2 extra 10M runs.) A margin rule m = 0.5·SD_baseline is kept as a cheap pre-filter only, never as the sole arbiter.
The raw screen table is published with the confirmatory write-up (selection-conditional estimate disclosure; tied candidates overstate the winner by ~0.85·screen-SD).

# Failures and Missingness
Infrastructure failures may be rerun with the same intended seed/configuration and must remain in the run ledger.
Instability is an experimental outcome; retain it and report frequency and symptoms.
Do not replace an incomplete pair with a different seed without documenting the deviation.
Do not exclude completed unfavorable runs based on curve shape, loss spikes, or post-hoc quality criteria.
**Instability trigger (locked):** if ≥ 30% of treatment pairs fail stability criteria, the primary conclusion is "instability-dominated negative" regardless of the surviving pairs — survivors are a biased subsample (the most stable noise realizations). Unstable pairs' d_i are reported both as missing and as the worst observed value.

# Decision Language
Confirmed favorable signal: point estimate > δ (= 1.0·SD_baseline) AND 95% randomization CI excludes 0, with no unacceptable stability/cost regression.
Annealing-specific evidence: the above plus favorable comparison to matched constant noise.
Regularization-only evidence: annealed and constant noise are comparable and favorable to baseline.
Negative at tested setting: observed uncertainty excludes a practically relevant favorable effect, or instability dominates.
Inconclusive: uncertainty spans meaningful benefit and no meaningful benefit/harm.

# Reporting Template
Configuration: <id>
Dataset/tokenizer/configuration hashes: <values>
Seeds: <list>
Endpoint: <step and token count>
Baseline NLL: <mean ± SD>
Annealed NLL: <mean ± SD>
Paired differences: <all values>
Mean paired difference and 95% CI: <value>
Exact sign-flip p-value: <value>
Practical threshold delta: <value>
Instability and throughput: <summary>
Locked conclusion language: <text>
Protocol deviations: <none or list>

# Signatures
Protocol owner: <name>. Date/time committed: <timestamp>. Repository commit hash: <hash>. Archive DOI or release tag: <identifier>.