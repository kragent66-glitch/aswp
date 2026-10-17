"""Host architecture for the ASWP experiment.

A modern decoder-only Transformer -- pre-norm RMSNorm, rotary position embeddings, SwiGLU
MLP, no biases -- rather than the GPT-Neo family the public reference checkpoints use.
Two consequences, both deliberate and both recorded in the docs:

* The reference comparison is valid at the **token** level because the tokenizer is
  identical (GPT-2 BPE, verified in docs/07) and the context length matches (512, forced
  by the reference family's untrained position embeddings above row 511). It is **not** a
  recipe-matched comparison, so a row in that table mixes architecture with recipe.
* Inside the experiment the architecture is shared by every arm, so the paired
  baseline-vs-ASWP comparison isolates the perturbation and nothing else.

ROPES and no biases make the eligible-module list exactly the protocol's: attention
q/k/v/o and the three MLP projections. The token embedding (tied to the output head) and
the norms are never perturbed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

#: The projections ASWP is allowed to perturb (docs/01, "Eligible modules").
ELIGIBLE_SUFFIXES = (
    "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


@dataclass(frozen=True)
class ModelConfig:
    """A fixed architecture description. Frozen before any run; see docs/06 item 6."""

    vocab_size: int = 50257          # GPT-2 BPE, locked (docs/07 section 3)
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int | None = None          # default: 8/3 * d_model rounded to a multiple of 64
    max_seq_len: int = 512           # matches the reference protocol and their checkpoints
    rope_base: float = 10_000.0
    tie_embeddings: bool = True
    dropout: float = 0.0             # 0.0 in the primary arms: dropout is comparator C5b

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
            )
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("head dimension must be even for rotary embeddings")
        if self.max_seq_len > 2048:
            raise ValueError("sequence length beyond the frozen context length")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_dim(self) -> int:
        if self.d_ff is not None:
            return self.d_ff
        return 64 * math.ceil((8 * self.d_model / 3) / 64)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["d_ff"] = self.ffn_dim
        d["head_dim"] = self.head_dim
        return d


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32.to(dtype)) * self.weight


def rope_cache(seq_len: int, head_dim: int, base: float, device, dtype=torch.float32):
    """cos/sin tables of shape (seq_len, head_dim/2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, Dh); cos/sin: (T, Dh/2). Rotates adjacent even/odd pairs."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, Dh = self.cfg.n_heads, self.cfg.head_dim
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.cfg.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.o_proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    """SwiGLU: gate/up/down, the three projections the protocol lists as eligible."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class TinyDecoder(nn.Module):
    """Decoder-only LM returning logits; losses are computed by the caller.

    Returning logits rather than a loss keeps one scoring convention across the
    repository (bench/reference_eval.py), so our rows and the reference rows cannot drift
    apart in how the mean is aggregated.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight  # re-tie after init

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")
        cos, sin = rope_cache(T, self.cfg.head_dim, self.cfg.rope_base, tokens.device)
        x = self.tok_emb(tokens)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.lm_head(self.norm_f(x))

    # ---------------------------------------------------------------- accounting
    def param_counts(self) -> dict:
        """Total and non-embedding parameter counts.

        The non-embedding column is the one to scale-match on: the public reference
        checkpoints are named by it, and at vocab 50 257 the embedding dominates every
        model at this size (docs/07 section 5).
        """
        total = sum(p.numel() for p in self.parameters())
        emb = self.tok_emb.weight.numel()
        return {
            "total": total,
            "non_embedding": total - emb,
            "embedding": emb,
            "tied_embeddings": self.cfg.tie_embeddings,
        }


#: Architectures under consideration for the two experiment stages. The screen stage is
#: parameter-matched to the reference `8M` row (6.84M non-embedding, see docs/07), the
#: verification stage to `28M` (26.26M). Which one is run is a protocol-owner decision;
#: the harness is config-driven either way.
CANDIDATE_CONFIGS = {
    "screen_d256_l8": ModelConfig(d_model=256, n_layers=8, n_heads=8),
    "screen_d320_l8": ModelConfig(d_model=320, n_layers=8, n_heads=8),
    "verify_d512_l8": ModelConfig(d_model=512, n_layers=8, n_heads=8),
    "smoke_d64_l2": ModelConfig(d_model=64, n_layers=2, n_heads=4, max_seq_len=64),
}
