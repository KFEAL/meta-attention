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
from .controller import MetaController
from .experts.base import AttentionExpert
from .experts.full import FullAttention
from .experts.linear import LinearAttention
from .experts.local import LocalAttention
from .utils import RoutingStats, compute_routing_stats, gumbel_softmax_top_k


class MetaAttentionLayer(nn.Module):
    """Core Meta-Attention layer.

    A drop-in replacement for any standard attention sub-layer.  Accepts
    ``(B, T, D)`` input and returns ``(B, T, D)`` output alongside an
    auxiliary-loss scalar and routing diagnostics.

    Routing modes
    -------------
    **Soft routing** (``cfg.hard_routing=False``, default)::

        output = Σ_i α_i(x) · E_i(x)

    All experts run on every forward pass.  Fully differentiable.

    **Hard routing** (``cfg.hard_routing=True``)

    Training: Gumbel-softmax top-k with straight-through gradient.
    Inference: argmax — only the winning expert runs, delivering real FLOP savings.

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

        self.controller = MetaController(cfg)

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
        mask:
            Optional additive attention mask ``(B, 1, T, T)`` or
            ``(1, 1, T, T)``.  0 = attend, -inf = mask out.

        Returns
        -------
        output : (B, T, D)
        aux_loss : scalar tensor
            ``cost_lambda * E[cost] - entropy_coeff * H[routing]``.
            Zero when both coefficients are 0.
        stats : RoutingStats
            Detached routing diagnostics.
        """
        alpha = self.controller(x)

        if self.cfg.hard_routing:
            output = self._hard_route(x, alpha, mask)
        else:
            output = self._soft_route(x, alpha, mask)

        aux_loss = self._aux_loss(alpha)
        stats = compute_routing_stats(alpha, self.costs)
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
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
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
            hard=not self.training,
        )

        if not self.training:
            expert_idx = weights.argmax(dim=-1)
            B, T, D = x.shape
            out = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
            for k_idx in range(len(self.experts)):
                token_mask = (expert_idx == k_idx).unsqueeze(-1).float()
                if token_mask.any():
                    out = out + self.experts[k_idx](x, mask) * token_mask
            return out
        else:
            expert_outs = torch.stack([e(x, mask) for e in self.experts], dim=-1)
            return (expert_outs * weights.unsqueeze(2)).sum(dim=-1)

    # ------------------------------------------------------------------
    # Auxiliary loss
    # ------------------------------------------------------------------

    def _aux_loss(self, alpha: torch.Tensor) -> torch.Tensor:
        loss = alpha.new_zeros(())

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
        """Update controller routing temperature in-place (useful for annealing)."""
        self.controller.temperature = tau
        self.cfg.temperature = tau

    def expert_names(self) -> List[str]:
        return [type(e).__name__ for e in self.experts]

    def extra_repr(self) -> str:
        return (
            f"experts=[{', '.join(self.expert_names())}], "
            f"hard={self.cfg.hard_routing}, costs={self.costs.tolist()}"
        )
