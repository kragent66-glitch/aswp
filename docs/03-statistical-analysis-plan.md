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
Paired seed list: <at least five integers>.

# Endpoint and Estimand
Primary endpoint: deterministic validation NLL at checkpoint step <value>, evaluated on the frozen validation token stream. Let d_i equal baseline NLL minus annealed NLL for paired seed i. A positive d_i favors annealed perturbation. Best checkpoint and minimum-over-time NLL are not confirmatory endpoints.

# Hypotheses
H0: paired differences are symmetric around zero. H1: annealed treatment improves NLL, such that the paired difference tends to be positive. The primary test is two-sided unless a one-sided direction and justification are explicitly locked before training.

# Sample Size
Minimum: 5 paired seeds. Preferred: 8 to 10 paired seeds. Five pairs are suitable for a compute-limited confirmation estimate but produce coarse exact-test resolution and wide uncertainty. A result with five pairs should be described conservatively.

# Primary Analysis
Collect final deterministic validation NLL for every completed paired run.
Compute and publish each d_i, then its mean, median, standard deviation, and a 95 percent percentile bootstrap confidence interval using <10000> resamples.
Compute an exact paired sign-flip/randomization p-value by enumerating the 2^n signs of paired differences.
Report a paired t-test only as sensitivity analysis; do not rely on untestable normality assumptions at small n.
Plot individual pairs; do not show only aggregated bars.

# Practical Significance
Before confirmation, set delta = <value> NLL as the smallest practically important mean paired improvement. Delta must be selected from baseline-only calibration variability and/or a documented application requirement, not from the observed treatment result. Report whether the estimated effect and its confidence interval exceed delta.

# Multiplicity and Selection
All sigma, schedule, scaling, initialization, layer-target, and regularization sweeps are exploratory.
Only one configuration selected by the written screen rule may enter this confirmatory analysis.
If multiple configurations are confirmed, control multiplicity or label all results exploratory.
The constant-noise comparison is required before making an annealing-specific claim.

# Failures and Missingness
Infrastructure failures may be rerun with the same intended seed/configuration and must remain in the run ledger.
Instability is an experimental outcome; retain it and report frequency and symptoms.
Do not replace an incomplete pair with a different seed without documenting the deviation.
Do not exclude completed unfavorable runs based on curve shape, loss spikes, or post-hoc quality criteria.

# Decision Language
Confirmed favorable signal: effect exceeds delta with compatible uncertainty and no unacceptable stability/cost regression.
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