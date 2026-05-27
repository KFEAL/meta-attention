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

"""Full O(T²) softmax attention expert (E1)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AttentionExpert
from .registry import register_expert
from ..config import MetaAttnConfig

_HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


@register_expert("full")
class FullAttention(AttentionExpert):
    """Standard multi-head scaled dot-product attention.

    Complexity: O(T² · D).
    Normalised cost: 1.0.

    When ``cfg.use_flash=True`` and PyTorch >= 2.0, uses
    ``F.scaled_dot_product_attention``, which dispatches to FlashAttention
    on CUDA with supported dtypes.
    """

    _cost: float = 1.0

    def __init__(self, cfg: MetaAttnConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.scale = cfg.d_head ** -0.5
        self.use_flash = cfg.use_flash and _HAS_SDPA

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_flash:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores + mask
            attn = self.attn_drop(F.softmax(scores, dim=-1))
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

    def extra_repr(self) -> str:
        return (
            f"n_heads={self.n_heads}, d_head={self.d_head}, "
            f"flash={self.use_flash}, cost={self.cost}"
        )
