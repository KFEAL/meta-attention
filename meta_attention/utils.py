# Copyright 2024-2025 Alan Ferrari
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility helpers — routing statistics, mask builders, and schedulers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Routing statistics
# ---------------------------------------------------------------------------


@dataclass
class RoutingStats:
    """Per-forward-pass routing diagnostics (all values are detached).

    Attributes
    ----------
    weights:
        Mean routing weight per expert, shape ``(K,)``.
    entropy:
        Mean routing entropy over all tokens in the batch (nats).
    expected_cost:
        ``mean_token( Σ_i alpha_i * cost_i )`` — average normalised FLOP.
    n_collapsed:
        Fraction of tokens where a single expert has weight > ``collapse_threshold``.
    """

    weights: torch.Tensor
    entropy: float
    expected_cost: float
    n_collapsed: float

    def __repr__(self) -> str:
        w = " | ".join(f"E{i+1}={v:.3f}" for i, v in enumerate(self.weights.tolist()))
        return (
            f"RoutingStats({w}  entropy={self.entropy:.4f}"
            f"  cost={self.expected_cost:.4f}  collapsed={self.n_collapsed:.2%})"
        )


def compute_routing_stats(
    alpha: torch.Tensor,
    costs: torch.Tensor,
    collapse_threshold: float = 0.9,
) -> RoutingStats:
    """Compute ``RoutingStats`` from a routing-weight tensor.

    Parameters
    ----------
    alpha : (B, T, K)
        Softmax routing weights.
    costs : (K,)
        Normalised expert costs.
    collapse_threshold:
        A token is considered "collapsed" when its max weight exceeds this.
    """
    with torch.no_grad():
        mean_weights = alpha.mean(dim=(0, 1))
        eps = 1e-8
        H = -(alpha * (alpha + eps).log()).sum(dim=-1)
        entropy = H.mean().item()
        expected_cost = (alpha * costs).sum(dim=-1).mean().item()
        collapsed = (alpha.max(dim=-1).values > collapse_threshold).float().mean().item()
    return RoutingStats(
        weights=mean_weights.detach().cpu(),
        entropy=entropy,
        expected_cost=expected_cost,
        n_collapsed=collapsed,
    )


# ---------------------------------------------------------------------------
# Mask builders
# ---------------------------------------------------------------------------


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Boolean lower-triangular causal mask, shape ``(1, 1, T, T)``."""
    return torch.tril(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    ).view(1, 1, seq_len, seq_len)


def additive_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Additive causal mask (0 / -inf), shape ``(1, 1, T, T)``.

    Pass directly to ``attn_scores + mask`` before softmax.
    """
    mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=dtype)
    mask.masked_fill_(
        ~torch.tril(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        ).view(1, 1, seq_len, seq_len),
        float("-inf"),
    )
    return mask


def hf_attention_mask_to_additive(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a HuggingFace ``(B, T)`` boolean mask to additive form ``(B, 1, 1, T)``.

    HF convention: 1 = real token, 0 = padding.
    Output: 0 for real tokens, -inf for padding.
    """
    additive = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2)
    return additive.masked_fill(additive.bool(), float("-inf")).to(dtype)


# ---------------------------------------------------------------------------
# Gumbel-Softmax helpers (hard routing)
# ---------------------------------------------------------------------------


def gumbel_softmax_top_k(
    logits: torch.Tensor,
    k: int,
    tau: float = 1.0,
    hard: bool = True,
) -> torch.Tensor:
    """Top-k Gumbel-softmax routing weights.

    Parameters
    ----------
    logits : (B, T, K)
    k:
        Number of experts to activate per token.
    tau:
        Temperature for Gumbel noise.
    hard:
        If ``True``, returns a k-hot sparse tensor via straight-through
        estimator (inference mode).  If ``False``, returns differentiable
        soft weights with Gumbel noise (training mode).

    Returns
    -------
    weights : (B, T, K)  — non-negative, sums to 1 along the K dimension.
    """
    if hard:
        indices = logits.topk(k, dim=-1).indices
        weights = torch.zeros_like(logits).scatter_(-1, indices, 1.0 / k)
        return weights

    gumbel = -torch.empty_like(logits).exponential_().log()
    noisy = (logits + gumbel) / max(tau, 1e-6)
    soft = F.softmax(noisy, dim=-1)
    if k < logits.shape[-1]:
        _, topk_idx = soft.topk(k, dim=-1)
        mask = torch.zeros_like(soft).scatter_(-1, topk_idx, 1.0)
        soft = soft * mask
        soft = soft / (soft.sum(dim=-1, keepdim=True) + 1e-8)
    return soft


# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------


def cosine_ramp(
    step: int,
    warmup_steps: int,
    max_steps: int,
    start: float = 0.0,
    end: float = 1.0,
) -> float:
    """Cosine ramp from *start* to *end* over ``[warmup_steps, max_steps]``."""
    if step < warmup_steps:
        return start
    if step >= max_steps:
        return end
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return start + (end - start) * (1 - math.cos(math.pi * progress)) / 2


def linear_ramp(
    step: int,
    warmup_steps: int,
    max_steps: int,
    start: float = 0.0,
    end: float = 1.0,
) -> float:
    """Linear ramp from *start* to *end* over ``[warmup_steps, max_steps]``."""
    if step < warmup_steps:
        return start
    if step >= max_steps:
        return end
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return start + (end - start) * progress
