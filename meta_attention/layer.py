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

"""MetaAttentionLayer — the core drop-in replacement for any attention sub-layer."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .config import MetaAttnConfig
from .controller import BayesianMetaController
from .experts.base import AttentionExpert
from .experts.full import FullAttention
from .experts.linear import LinearAttention
from .experts.local import LocalAttention
from .utils import RoutingStats, compute_routing_stats, dirichlet_kl, gumbel_softmax_top_k


class MetaAttentionLayer(nn.Module):
    """Core Meta-Attention layer.

    A drop-in replacement for any standard attention sub-layer.  Accepts
    ``(B, T, D)`` input and returns ``(B, T, D)`` output alongside an
    ELBO auxiliary loss and routing diagnostics.

    Routing modes
    -------------
    **Soft routing** (``cfg.hard_routing=False``, default)::

        output = Σ_i α_i(x) · E_i(x)

    During training ``α_t`` is sampled from Dir(β̂_t) via the
    reparameterisation trick (when ``cfg.sample_posterior=True``).
    At evaluation the posterior mean is used.  Fully differentiable.

    **Uncertainty-gated hard routing** (``cfg.hard_routing=True``)

    Training: Gumbel-softmax top-k with straight-through gradient.
    Inference: tokens with U_t < η route to argmax expert only; tokens
    with U_t ≥ η fall back to soft (posterior-mean) merge.

    Custom experts
    --------------
    Pass a ``List[AttentionExpert]`` to ``experts``.  Each expert must
    have its ``.cost`` property set consistently with ``cfg.expert_costs``.

    Example
    -------
    >>> cfg = MetaAttnConfig.small()
    >>> layer = MetaAttentionLayer(cfg)
    >>> x = torch.randn(2, 64, 128)
    >>> out, aux_loss, stats = layer(x)
    >>> out.shape
    torch.Size([2, 64, 128])
    """

    def __init__(
        self,
        cfg: MetaAttnConfig,
        experts: Optional[List[AttentionExpert]] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        if experts is None:
            experts = [
                FullAttention(cfg),
                LinearAttention(cfg),
                LocalAttention(cfg),
            ]
        self.experts = nn.ModuleList(experts)

        if len(cfg.expert_costs) != len(self.experts):
            raise ValueError(
                f"cfg.expert_costs length ({len(cfg.expert_costs)}) must equal "
                f"number of experts ({len(self.experts)})"
            )
        self.register_buffer("costs", torch.tensor(cfg.expert_costs, dtype=torch.float32))

        self.controller = BayesianMetaController(cfg)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, RoutingStats]:
        """Run Meta-Attention.

        Parameters
        ----------
        x : (B, T, D)
            Pre-normalised token features (LayerNorm applied in the block).
        mask:
            Optional additive attention mask ``(B, 1, T, T)`` or
            ``(1, 1, T, T)``.  0 = attend, -inf = mask out.

        Returns
        -------
        output : (B, T, D)
        aux_loss : scalar tensor
            ELBO KL term ``elbo_weight · mean_t KL[Dir(β̂_t) ‖ Dir(β)]``.
            Zero when ``cfg.elbo_weight == 0``.
        stats : RoutingStats
            Detached routing diagnostics including per-token uncertainty.
        """
        alpha, beta_hat, uncertainty = self.controller(x)

        if self.cfg.hard_routing:
            output = self._hard_route(x, alpha, uncertainty, mask)
        else:
            output = self._soft_route(x, alpha, mask)

        aux_loss = self._aux_loss(beta_hat)
        stats = compute_routing_stats(alpha, self.costs, uncertainty=uncertainty.detach())
        return output, aux_loss, stats

    # ------------------------------------------------------------------
    # Routing implementations
    # ------------------------------------------------------------------

    def _soft_route(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        expert_outs = torch.stack([e(x, mask) for e in self.experts], dim=-1)  # (B,T,D,K)
        return (expert_outs * alpha.unsqueeze(2)).sum(dim=-1)

    def _hard_route(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
        uncertainty: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.training:
            return self._uncertainty_gated_route(x, alpha, uncertainty, mask)

        # Training: Gumbel-softmax top-k with straight-through gradient
        logits = self.controller.net(
            torch.cat(
                [
                    x,
                    x.detach().norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5),
                    torch.linspace(0, 1, x.shape[1], device=x.device)
                    .view(1, -1, 1)
                    .expand(x.shape[0], -1, -1),
                ],
                dim=-1,
            )
        )
        weights = gumbel_softmax_top_k(
            logits,
            k=self.cfg.hard_top_k,
            tau=self.cfg.gumbel_temp,
            hard=False,
        )
        expert_outs = torch.stack([e(x, mask) for e in self.experts], dim=-1)
        return (expert_outs * weights.unsqueeze(2)).sum(dim=-1)

    def _uncertainty_gated_route(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
        uncertainty: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Inference-time routing: argmax when U_t < η, soft merge otherwise."""
        eta = self.cfg.uncertainty_threshold
        use_hard = uncertainty < eta   # (B, T)
        use_soft = ~use_hard           # (B, T)

        B, T, D = x.shape
        out = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)

        # Hard tokens: winner-takes-all
        if use_hard.any():
            expert_idx = alpha.argmax(dim=-1)  # (B, T)
            for k_idx, expert in enumerate(self.experts):
                token_mask = (use_hard & (expert_idx == k_idx)).unsqueeze(-1).to(x.dtype)
                if token_mask.any():
                    out = out + expert(x, mask) * token_mask

        # Soft tokens: posterior-mean weighted merge
        if use_soft.any():
            expert_outs = torch.stack([e(x, mask) for e in self.experts], dim=-1)
            soft_out = (expert_outs * alpha.unsqueeze(2)).sum(dim=-1)
            out = out + soft_out * use_soft.unsqueeze(-1).to(x.dtype)

        return out

    # ------------------------------------------------------------------
    # ELBO auxiliary loss
    # ------------------------------------------------------------------

    def _aux_loss(self, beta_hat: torch.Tensor) -> torch.Tensor:
        loss = beta_hat.new_zeros(())

        if self.cfg.elbo_weight != 0.0:
            # KL[Dir(β̂) ‖ Dir(β)] averaged over all tokens in the batch
            kl = dirichlet_kl(beta_hat, self.controller.prior_beta)  # (B, T)
            loss = loss + self.cfg.elbo_weight * kl.mean()

        # Legacy ad-hoc terms kept for ablation (active only when elbo_weight == 0)
        if self.cfg.elbo_weight == 0.0:
            alpha = beta_hat / beta_hat.sum(dim=-1, keepdim=True)
            if self.cfg.cost_lambda != 0.0:
                expected_cost = (alpha * self.costs).sum(dim=-1).mean()
                loss = loss + self.cfg.cost_lambda * expected_cost
            if self.cfg.entropy_coeff != 0.0:
                eps = 1e-8
                entropy = -(alpha * (alpha + eps).log()).sum(dim=-1).mean()
                loss = loss - self.cfg.entropy_coeff * entropy

        return loss

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def set_temperature(self, tau: float) -> None:
        """Update routing temperature in-place (kept for API compatibility)."""
        self.cfg.temperature = tau

    def expert_names(self) -> List[str]:
        return [type(e).__name__ for e in self.experts]

    def extra_repr(self) -> str:
        return (
            f"experts=[{', '.join(self.expert_names())}], "
            f"hard={self.cfg.hard_routing}, costs={self.costs.tolist()}"
        )
