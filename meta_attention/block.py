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

"""MetaTransformerBlock — Pre-LN transformer block with Meta-Attention."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import MetaAttnConfig
from .layer import MetaAttentionLayer
from .utils import RoutingStats


class MetaTransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block using Meta-Attention.

    Architecture::

        x → LayerNorm → MetaAttentionLayer ──┐
                                              + → x'
                        x ──────────────────┘
        x' → LayerNorm → FFN (4×) → GELU → Dropout ──┐
                                                       + → output
        x' ────────────────────────────────────────────┘

    Parameters
    ----------
    cfg:
        Model configuration.
    ffn_multiplier:
        Width of the FFN hidden layer as a multiple of ``d_model``.
    """

    def __init__(self, cfg: MetaAttnConfig, ffn_multiplier: int = 4) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(cfg.d_model)
        self.attn = MetaAttentionLayer(cfg)
        self.ln_ffn = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, ffn_multiplier * cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(ffn_multiplier * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, RoutingStats]:
        """
        Returns
        -------
        x : (B, T, D)
        aux_loss : scalar tensor
        stats : RoutingStats
        """
        attn_out, aux_loss, stats = self.attn(self.ln_attn(x), mask)
        x = x + attn_out
        x = x + self.ffn(self.ln_ffn(x))
        return x, aux_loss, stats
