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

"""Bayesian Meta-Controller: amortised variational posterior over routing weights."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MetaAttnConfig
from .utils import dirichlet_entropy


class BayesianMetaController(nn.Module):
    """Bayesian Meta-Controller for per-token attention routing.

    Treats per-token mechanism selection as posterior inference under a
    compute-aware Dirichlet prior p(α) = Dir(β), where the floored prior:

        β_i = ε + β₀ · (1 − c_i)

    ensures all concentration parameters are strictly positive.  The
    amortised variational posterior is:

        q_φ(α | x_t) = Dir(β̂_t),  β̂_t = β + δ_φ(x_t)

    where δ_φ(x_t) = softplus(W₂ · GELU(W₁ · [x_t; ‖x_t‖/√D; pos])) ≥ 0.

    Routing weights are the posterior mean at evaluation time:

        α_t = β̂_t / Σ_i β̂_{t,i}

    During training (when ``cfg.sample_posterior=True``), α_t is drawn via
    the Dirichlet reparameterisation trick for gradient flow.

    Returns
    -------
    alpha : (B, T, K)
        Per-token routing weights (sum to 1 over K).
    beta_hat : (B, T, K)
        Posterior concentration parameters (all > 0).
    uncertainty : (B, T)
        Per-token Dirichlet entropy H[q_φ(α | x_t)].
    """

    def __init__(self, cfg: MetaAttnConfig) -> None:
        super().__init__()
        costs = torch.tensor(cfg.expert_costs, dtype=torch.float32)
        prior_beta = cfg.epsilon + cfg.beta_0 * (1.0 - costs)
        self.register_buffer("prior_beta", prior_beta)  # (K,)

        self.sample_posterior = cfg.sample_posterior

        in_dim = cfg.d_model + 2  # x, salience, position
        h = cfg.controller_hidden
        n = len(cfg.expert_costs)

        # Two-layer MLP; softplus applied to output so δ_φ > 0
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, n),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-token routing weights and uncertainty.

        Parameters
        ----------
        x : (B, T, D)
            Pre-normalised token features (LayerNorm applied upstream).

        Returns
        -------
        alpha : (B, T, K)
        beta_hat : (B, T, K)
        uncertainty : (B, T)
        """
        B, T, D = x.shape
        salience = x.detach().norm(dim=-1, keepdim=True) / (D ** 0.5)
        pos = torch.linspace(0, 1, T, device=x.device).view(1, T, 1).expand(B, -1, -1)
        feat = torch.cat([x, salience, pos], dim=-1)

        delta = F.softplus(self.net(feat))          # (B, T, K), all > 0
        beta_hat = self.prior_beta + delta           # (B, T, K)

        if self.training and self.sample_posterior:
            # Dirichlet reparameterisation: sample for gradient-aware exploration
            alpha = torch.distributions.Dirichlet(beta_hat).rsample()
        else:
            # Posterior mean at evaluation (and training when sampling disabled)
            alpha = beta_hat / beta_hat.sum(dim=-1, keepdim=True)

        uncertainty = dirichlet_entropy(beta_hat)    # (B, T)
        return alpha, beta_hat, uncertainty

    def extra_repr(self) -> str:
        b = self.prior_beta.tolist()
        return f"prior_beta={[round(v, 4) for v in b]}, sample={self.sample_posterior}"


# ---------------------------------------------------------------------------
# Alias kept for backward compatibility
# ---------------------------------------------------------------------------

MetaController = BayesianMetaController
