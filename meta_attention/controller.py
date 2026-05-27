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

"""Meta-Controller: lightweight MLP that produces per-token routing weights."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MetaAttnConfig


class MetaController(nn.Module):
    """Lightweight MLP that maps token features to per-token routing weights.

    Input features per token (all derived *before* any attention):

    * The token embedding ``x``  — ``D`` dims.
    * Token salience ``‖x‖ / √D`` — 1 dim (detached).
    * Normalised position in ``[0, 1]`` — 1 dim.

    This design avoids the chicken-and-egg problem: routing is decided
    entirely from input-side information, with no dependence on attention
    outputs.

    Architecture::

        Linear(D+2 → H) → GELU → Dropout → Linear(H → H/2) → GELU → Linear(H/2 → K)
        → Softmax(τ)

    Parameters
    ----------
    cfg:
        Uses ``d_model``, ``controller_hidden``, ``n_experts``,
        ``temperature``, and ``dropout``.
    """

    def __init__(self, cfg: MetaAttnConfig) -> None:
        super().__init__()
        self.temperature = cfg.temperature
        in_dim = cfg.d_model + 2
        h = cfg.controller_hidden
        n = cfg.n_experts

        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Linear(h // 2, n),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-token routing weights.

        Parameters
        ----------
        x : (B, T, D)

        Returns
        -------
        alpha : (B, T, K)  — softmax weights summing to 1 over K.
        """
        B, T, D = x.shape
        norms = x.detach().norm(dim=-1, keepdim=True) / (D ** 0.5)
        pos = torch.linspace(0, 1, T, device=x.device).view(1, T, 1).expand(B, -1, -1)
        feat = torch.cat([x, norms, pos], dim=-1)
        logits = self.net(feat)
        return F.softmax(logits / max(self.temperature, 1e-6), dim=-1)

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}"
