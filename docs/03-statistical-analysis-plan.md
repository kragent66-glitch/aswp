# ASWP — Statistical Analysis Plan and Preregistration

**Lock this document before confirmatory training starts.**

**Author:** Utkarsh Bhangale · **Version:** 1.0 · **Date:** August 2026 · **Repository:** [github.com/kragent66-glitch/aswp](https://github.com/kragent66-glitch/aswp)

# Purpose and Status
Complete every bracketed field before confirmatory data collection. Commit the completed document alongside the code and configuration manifest. Modifications afterward are protocol deviations and must be reported rather than silently incorporated.

No confirmatory outcome feeds back into a design choice: δ and its units, the SD_delta/SD_baseline ratio convention including the predeclared arm-ratio tolerance q ≤ 1.25, the interval sidedness, the selection margin and the analysis seeds are all fixed here, before any treatment result exists (issues #13, #14). The realized values of q, ρ and SD_delta/SD_baseline are *reported* after the runs, never used to pick the threshold.

# Confirmatory Question
For the selected annealed RMS-relative Gaussian perturbation configuration, does deterministic final validation NLL improve compared with a deterministic baseline across fixed paired training seeds under otherwise identical conditions?

# Locked Configuration Fields
Dataset revision and SHA-256: TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` (CDLA-Sharing-1.0). The training raw file is `c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f` (1,924,281,556 bytes), tokenized once into 469,429,735 tokens — stream `tinystories-train-41272badd4c7`, token-stream SHA-256 `41272badd4c705b52920c77a3985acc65ae62aa9ffabab26ed4c26e5cccd5b7c`, 4.0992 bytes/token. The manifest records that the marker invariant holds exactly (2,119,718 `<|endoftext|>` markers in the raw bytes = 2,119,718 eot token ids) and that pieces end exactly after an eot occurrence, so the concatenation is identical to tokenizing the whole file at once.
Tokenizer identity/version/hash: `gpt2` at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, 50,257 entries. Artifact SHA-256: `vocab.json 19613966…`, `merges.txt 1ce16647…`, `tokenizer.json 8414cab9…`, `tokenizer_config.json 5e04eb60…` (full digests in `data/train_manifest.json` and both `bench/frozen/*.json`, all written by one shared helper, `harness/data.py::tokenizer_provenance()`).
Validation split and token-stream hash: **primary/confirmatory** `tinystories-v2-gpt4-valid-fd5d9b500ac0` — raw SHA-256 `6874bae9a4c1…c0585d`, token-stream SHA-256 `fd5d9b500ac0…5a9aee`, 5,532,654 tokens, 11,517 distinct ids; **comparability** `tinystories-valid-aa85f8db4e23` — raw SHA-256 `94e431816c4c…8cb38b4`, token-stream SHA-256 `aa85f8db4e23…c64154`, 4,724,503 tokens, the stream the six published reference checkpoints were scored on. Both are held out of training entirely.
Model source commit and architecture: the confirmatory commit is the one carrying the pre-results protocol release tag `v0.1.0-preregistration` (later confirmatory tags supersede it); every run records `git_commit()` into its checkpoint, ledger and manifest. Architecture (deviating from the reference family's GPT-Neo on the same data): d_model 256, 8 layers, 8 heads, d_ff 4×d_model, RoPE base 10000, tied embeddings, dropout 0.0, max_seq_len 512, vocab 50,257 → **6.43M non-embedding parameters**, scale-matched to the reference "8M" row (that family counts non-embedding parameters).
Total optimizer steps and tokens: **1526 steps** × 163,840 tokens/step (80 sequences × accum 4 × ctx 512) = **250,019,840 tokens** (100.007% of the 250M target). The step count is the number of annealing events, which is why this batch shape was chosen over the reference recipe's 80 × accum 16 (381 steps).
Optimizer, learning-rate schedule, warmup, decay, clipping, dropout, precision: AdamW; lr 5e-4 **constant** (no warmup, no decay); betas (0.9, 0.95); weight decay 0.1; `max_grad_norm` **1.0** (predeclared, applied identically to every arm — a divergent update could silently drop a pair); dropout 0.0; training precision **bf16** on a tensor-core GPU. Noise and perturbation are constructed in fp32 regardless, and the realized perturbation is measured *after* any cast (Phase 0 verifies this at scale).
Eligible module list: the **56 bias-free projection matrices** — attention q/k/v/out projections and MLP in/out projections — across the 8 blocks. Excluded by construction: token embeddings, the tied output head, every norm, and every bias.
Noise distribution: Gaussian; scaling: RMS-relative; schedule: cosine.
Selected sigma0: **OPEN until the screen.** Chosen from the predeclared grid **{0.01, 0.03, 0.05}** (the plan's stated 0.01–0.05 treatment range), one σ per screen arm. It cannot be fixed in advance without peeking at the confirmatory data, which is precisely why the screen precedes the confirmatory set.
Selection rule and exploratory parent run ID: the screen selects the σ with the best deterministic endpoint NLL on the primary stream; the winner is then confirmed by the pre-registered selection-stability check (**chosen + runner-up, 2 fresh seeds each** — docs/05 §P3; any ranking reversal downgrades the claim to exploratory). Chosen σ, runner-up, and the screen run IDs are written to `bench/results/screen_runs.json` **before** any confirmatory run starts.
Paired seed list (**frozen**, issue #10): `1857649130, 1434172854, 1387619681, 536290471, 1202662852, 1106087492, 1881574181, 1859844390` — 8 integers, the absolute minimum being 6 (one-sided α=0.05 needs n≥6 to be achievable; n=8 gives 80% power at a 1.0·SD_delta effect — see # Sample Size). Stored with the derivation rule and the per-role seeds in `config/manifest.json`; `analysis/seed_manifest.py` recomputes and the test suite verifies the file against that rule on every run, so the list cannot be silently re-chosen.
Per-role RNG seeds: one independent stream per role — `global`, `init`, `data-loader`, `dropout`, `perturbation` — each derived per paired seed from the same rule (`sha256("ASWP/v1/<role>/<n>")`). Perturbations must be drawn from a dedicated `torch.Generator` (protocol §Reproducibility); baseline and treatment runs of a pair share the paired seed and differ only in the perturbation arm, so pairing is enforced by construction.
Open config fields and their blockers are tracked in `docs/06-preregistration-checklist.md`; no confirmatory run may start until that document's open list is empty and committed.
Analysis reproducibility (issue #15): all analysis randomness is seeded in `analysis/aswp_core.py` — **SEED_POWER = 42** (power/resolution), **SEED_MATCH_CURSE = 7** (selection and margin simulations), **SEED_DELTA = 20260905** (decision-rule simulation); 40,000 trials per cell, 2,000,000 screen draws, and a **256 MiB** memory budget for the exact 2ⁿ enumeration (issue #16). Each script prints its seed, trial counts, assumptions and budget, and `tests/test_aswp_core.py` asserts same-seed determinism.

# Endpoint and Estimand
Two standard deviations are distinguished throughout and must not be conflated (issue #13; development: docs/05-blocker-resolution.md §P5):
**SD_baseline** — seed-to-seed SD of the baseline arm's NLL. The practical threshold δ is written in this unit.
**SD_delta** — seed-to-seed SD of the paired differences d_i. The sign-flip test and every power statement are in this unit.
With the arm ratio q = SD_treatment/SD_baseline and paired correlation ρ, Var(d) = Var(B) + Var(T) − 2·Cov(B,T) gives **SD_delta = SD_baseline·√(1 + q² − 2qρ)** at q = 1 this is the familiar √(2(1−ρ)). Equal arm variances is an assumption about the outcome rather than an identity — a noise arm may have a wider seed-to-seed spread than the deterministic arm — so the design **predeclares the tolerance box q ≤ 1.25, ρ ≥ 0** and must clear its worst corner (q = 1.25, ρ = 0: SD_delta = 1.6008·SD_baseline). The realized q and ρ are reported after the runs; if q exceeds the tolerance the power statements are restated as a protocol deviation rather than reinterpreted.

Primary endpoint: deterministic validation NLL at the **fixed final checkpoint, step 1526** (the schedule reaches exactly σ = 0 there, so the endpoint is the unperturbed model), evaluated on the frozen validation token stream. Let d_i equal baseline NLL minus annealed NLL for paired seed i. A positive d_i favors annealed perturbation. Best checkpoint and minimum-over-time NLL are not confirmatory endpoints.

# Hypotheses
H0: paired differences are symmetric around zero. H1: annealed treatment improves NLL, such that the paired difference tends to be positive.
**Primary test: ONE-SIDED (locked 2026-09-05, basis: docs/05-blocker-resolution.md §P1).** Justification: the decision ladder has no use for "significantly worse than baseline" — a worse-than-baseline result is reported as an estimate regardless, so only the positive direction carries a decision. A two-sided test is retained only as a reviewer fallback; it is achievable from n=6 (min two-sided p = 0.0313).

# Sample Size
Minimum: **6 paired seeds** — below this a two-sided α=0.05 test is mathematically impossible (n=5: min achievable two-sided p = 2/32 = 0.0625; simulated power at α=0.05 is exactly 0.000).
Target: **8 paired seeds** (preferred 8–10). The exact one-sided sign-flip test at α=0.05 has 80% power at a **1.0·SD_delta** effect and 98% at 1.5·SD_delta at n=8 (40,000-trial simulation, seed 42 — docs/05-blocker-resolution.md §P1). At n=10 power at 1.0·SD_delta rises to 89.7%; at n=6 it is 64.7%, so a 6-pair result is described conservatively. These figures are in **SD_delta** units and are *not* the power of the δ threshold: δ = 1.0·SD_baseline equals a 1.0·SD_delta effect only when ρ = 0.5, and at the predeclared worst corner of the tolerance box (q = 1.25, ρ = 0) it is a **0.625·SD_delta** effect, where n=8 power at δ is **0.458** (docs/05-blocker-resolution.md §P5). The design therefore cannot claim 80% power at its own practical threshold without either more seeds or an explicit argument that q ≈ 1 and ρ ≈ 0.5; that is stated here rather than left implicit in a table.

# Primary Analysis
Collect final deterministic validation NLL for every completed paired run.
Compute and publish each d_i, then its mean, median, standard deviation, and a **one-sided 95% randomization confidence bound** (lower limit, inverted from the same one-sided sign-flip test: the set of δ such that the one-sided test of d−δ fails to reject). The interval must be **one-sided**, matching the one-sided α=0.05 test; a two-sided 95% interval would put its lower limit at the 2.5% tail and hold the decision direction to α/2, which the test did not ask for (issue #14; docs/05-blocker-resolution.md §P6). Under the one-sided convention 'the bound excludes 0' and 'the test rejects at α' are the same event, so the rule carries no extra, unstated stringency.
Compute an exact paired sign-flip/randomization p-value by enumerating the 2^n signs of paired differences.
Report a 95% percentile bootstrap CI (10000 resamples) only as descriptive sensitivity — at n≤8 the percentile bootstrap is quantized to a coarse grid (126 distinct resampled means at n=5) with unreliable tail coverage.
Report a paired t-test and a Wilcoxon signed-rank test only as sensitivity analyses; do not rely on untestable normality assumptions at small n.
Plot individual pairs; do not show only aggregated bars.

# Practical Significance
**Delta is locked: δ = 1.0 × SD_baseline** (SD of baseline NLL across seeds, measured in Phase 0/1 baseline-only calibration before any treatment result is examined). It is the smallest practically important mean paired improvement.
**"Compatible uncertainty" is locked:** confirmed favorable signal = point estimate > δ AND the **one-sided 95% randomization bound** > 0 (primary). The bound is inverted from the same one-sided test, so this is the test of H1 at α=0.05 plus a practical threshold — one convention, not two (issue #14; §P6 of the blocker document). Recompute fire rates, exact sign-flip test, seed 20260905, n=8: 0.494 at μ=1.0, **0.921 at μ=1.5**, 0.997 at μ=2.0 (all SD_delta), with type-I error 0.0022 at μ=0. The figure previously quoted (~91% at 1.5·SD) corresponds to ρ = 0.5, i.e. it assumed δ's sd_baseline unit and the test's sd_delta unit were the same number; across the predeclared (q, ρ) box the same rule at the same fixed 1.5·SD_delta effect spans **0.797 (q=1.25, ρ=0.75) to 0.979 (q=1.25, ρ=0.00)** at n=8, and its power at the δ threshold spans **0.408 to 0.505** — docs/05-blocker-resolution.md §P4, §P5.
Sensitivity: strong criterion — **one-sided** 95% bound > 0.5·SD_delta (0.801 at n=8, μ=1.5·SD_delta). The review's original strong-k=1 rule is explicitly rejected as unconfirmable at n≤8 (0.344 at μ=1.5·SD_delta).

# Multiplicity and Selection
All sigma, schedule, scaling, initialization, layer-target, and regularization sweeps are exploratory.
Only one configuration selected by the written screen rule may enter this confirmatory analysis.
If multiple configurations are confirmed, control multiplicity or label all results exploratory.
The constant-noise comparison is required before making an annealing-specific claim.
**Screen selection rule (numeric, locked):** chosen σ0 must complete with ≤ 1 instability event and throughput ≥ 80% of baseline; otherwise the next-best σ0 is considered, in order.
**Selection-stability check (locked):** the chosen σ0 AND the runner-up σ0 each run on 2 fresh seeds in Phase 2. Any ranking reversal downgrades the confirmatory claim to exploratory. (The check has 50% flip probability at a true tie, ≤ 31% at 0.5·SD separation, 7% at 1.5·SD — docs/05-blocker-resolution.md §P3. Cost: 2 extra 10M runs.) A margin rule m = 0.5·SD_baseline is kept as a cheap pre-filter only, never as the sole arbiter — and it is weaker than it first appeared: with the margin computed as the **top-two gap over all candidates** (issue #12), a genuine 1.0·σ gap still fails it 32% of the time, and under the global null a tied screen clears m = 1.0·σ 34% of the time; the null margin has median 0.691·σ and is below 1·σ in 66% of draws (docs/05-blocker-resolution.md §P3).
The raw screen table is published with the confirmatory write-up (selection-conditional estimate disclosure; tied candidates overstate the winner by ~0.85·screen-SD).

# Failures and Missingness
Infrastructure failures may be rerun with the same intended seed/configuration and must remain in the run ledger.
Instability is an experimental outcome; retain it and report frequency and symptoms.
Do not replace an incomplete pair with a different seed without documenting the deviation.
Do not exclude completed unfavorable runs based on curve shape, loss spikes, or post-hoc quality criteria.
**Instability trigger (locked):** if ≥ 30% of treatment pairs fail stability criteria, the primary conclusion is "instability-dominated negative" regardless of the surviving pairs — survivors are a biased subsample (the most stable noise realizations). Unstable pairs' d_i are reported both as missing and as the worst observed value.

# Decision Language
Confirmed favorable signal: point estimate > δ (= 1.0·SD_baseline) AND the one-sided 95% randomization bound excludes 0, with no unacceptable stability/cost regression. δ is reported together with its SD_delta equivalent (δ/r) so the threshold and the evidence are in the same units.
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