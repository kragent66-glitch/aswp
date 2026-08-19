# ASWP — Implementation and Operations Manual

Engineering specification, tests, logging, and run operations for the experiment.

**Author:** Utkarsh Bhangale · **Version:** 1.0 · **Date:** August 2026 · **Repository:** [github.com/kragent66-glitch/aswp](https://github.com/kragent66-glitch/aswp)

# Purpose
This manual defines the engineering constraints necessary for a fair ASWP experiment. The implementation must alter only the intended training-forward weights, preserve base optimizer parameters, isolate stochastic sources, disable noise at evaluation, and make all realized intervention magnitudes observable.

# Reference Layer

```python
class AnnealedLinear(nn.Module):
    def __init__(self, base_linear, generator=None):
        super().__init__()
        self.weight = base_linear.weight
        self.bias = base_linear.bias
        self.sigma = 0.0
        self.enabled = True
        self.generator = generator

    def forward(self, x):
        if not self.training or not self.enabled or self.sigma == 0.0:
            return F.linear(x, self.weight, self.bias)
        w32 = self.weight.float()
        scale = w32.square().mean().sqrt().detach().clamp_min(1e-12)
        z = torch.randn(w32.shape, device=w32.device, dtype=torch.float32,
                        generator=self.generator)
        w_eff = w32 + self.sigma * scale * z
        return F.linear(x, w_eff.to(x.dtype), self.bias)
```

This reference preserves the original Parameter object, uses no in-place mutation, samples fp32 noise, and detaches the RMS scale so the initial method does not include derivatives of the scaling statistic. Production code should integrate autocast and fused-kernel constraints carefully.

# Non-Negotiable Semantics
Never overwrite model.weight in place and restore it later.
Do not add perturbation to optimizer state or checkpoint state.
Sample noise once per optimizer update, not separately per microbatch under gradient accumulation.
Keep perturbation fixed through recomputation if activation/gradient checkpointing replays a forward pass.
Disable perturbation unconditionally in eval mode and in explicit deterministic evaluation mode.
Use a separate RNG stream for perturbations so changing sigma does not shift data-order, dropout, or initialization RNG consumption.

# Module Selection
Start with q, k, v, o attention projections and MLP up/gate/down projections.
Exclude normalization parameters, embeddings, biases, positional parameters, and output heads in Phase 1.
If parameters are tied, do not replace one reference without inspecting all aliases.
Log module name, shape, parameter count, layer index, and whether it was selected or excluded.

# Schedule Contract
def cosine_sigma(sigma0, step, total_steps):
    if total_steps <= 1 or step >= total_steps - 1:
        return 0.0
    p = step / (total_steps - 1)
    return sigma0 * 0.5 * (1.0 + math.cos(math.pi * p))
Update every eligible module with this scalar once per optimizer update. Log requested sigma, realized perturbation RMS, and perturbation-to-weight RMS ratio. A requested sigma is not sufficient proof that the actual intervention has the configured scale after precision casting.

# Unit-Test Suite
Zero-noise equivalence: sigma=0 gives matching logits, loss, and gradients versus original Linear modules.
Evaluation determinism: repeated eval-mode calls yield identical outputs with enabled=true and nonzero configured sigma.
Noise existence: nonzero training sigma produces nonzero realized perturbation.
Endpoint correctness: first step equals sigma0 and last step equals exact zero.
Gradient connectivity: base weight has finite gradients after backward.
No mutation: base weight is unchanged before versus after forward.
Optimizer identity: optimizer references original parameter identity after module replacement.
Checkpoint resume: restoring all states reproduces next-step behavior within configured determinism tolerance.
Mixed precision: selected sigmas survive cast and do not quantize to zero materially.
Gradient accumulation: perturbation is shared across microbatches belonging to a single update.
Distributed behavior: test and document whether perturbations are synchronized or rank-local.

# Logging
step,tokens_seen,split,train_loss,clean_train_loss,val_nll,val_ppl,learning_rate,sigma,grad_norm,param_norm,update_norm,clip_applied,perturb_norm,perturb_weight_ratio,throughput_tok_s,peak_mem_mb,seed,config_hash,git_commit
Store layer diagnostics in a separate long-format file: step, layer_name, layer_index, weight_rms, perturb_rms, ratio, gradient_rms, update_rms. Keep raw data; do not only write aggregates.

# Checkpoint Contract
model_state_dict containing base deterministic parameters.
optimizer and scheduler states.
global step, tokens seen, configuration hash, git commit, and environment manifest.
all RNG states, including the dedicated perturbation generator.
no sampled perturbation tensors unless required transiently for an interrupted in-progress gradient-accumulation step.

# Run Ledger
run_id, configuration hash, hypothesis category, seed, parent run, start/end, hardware, software, cost rate, status, failure reason, artifacts.
Valid statuses: queued, running, completed, unstable, infrastructure_failed, invalidated_by_bug.
Do not overwrite runs. Retries get a new ID and retain linkage to the original.

# Common Bugs
Gradient checkpointing can resample random noise on recomputation unless RNG behavior is controlled.
Replacing a module after optimizer creation can cause optimizer state to refer to the wrong parameters.
Calling model.train() after evaluation may re-enable perturbation but requires sigma to remain correct.
bf16 casting can make small perturbations ineffective; measure effective weights after casting.
Rank-local noise changes the distributed gradient estimator; this must be deliberate and documented.
RNG stream coupling can invalidate paired comparisons by changing unrelated dropout masks or data ordering.

# Phase 0 Acceptance Gate
Do not initiate the 10M screening matrix until every unit test passes; sigma=0 equivalence is verified; evaluation is deterministic; realized noise statistics match configuration; throughput and memory overhead are measured; and checkpoint/resume is validated.