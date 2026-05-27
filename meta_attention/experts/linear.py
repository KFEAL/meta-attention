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

"""Linear (kernel) attention expert — Performer-style ELU+1 feature map (E2)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AttentionExpert
from .registry import register_expert
from ..config import MetaAttnConfig


@register_expert("linear")
class LinearAttention(AttentionExpert):
    """Performer-style linear attention using the ELU+1 feature map.

    Approximates softmax attention in O(T · D) by replacing the T×T
    attention matrix with its low-rank decomposition::

        Attn(Q, K, V) ≈ φ(Q) · (φ(K)ᵀV)

    where φ(x) = ELU(x) + 1 (positive, non-degenerate).

    Complexity: O(T · d_head · D).
    Normalised cost: 0.15.

    Parameters
    ----------
    cfg:
        Model config.
    causal:
        If ``True``, use the causal cumulative-sum variant.
        Required for autoregressive language modelling.
    """

    _cost: float = 0.15

    def __init__(self, cfg: MetaAttnConfig, causal: bool = False) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.causal = causal

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        """ELU+1 feature map — ensures positivity for a valid kernel."""
        return F.elu(x) + 1.0

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

        q = self._phi(q)
        k = self._phi(k)

        if self.causal:
            out = self._causal_attn(q, k, v)
        else:
            out = self._noncausal_attn(q, k, v)

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

    @staticmethod
    def _noncausal_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        kv = torch.einsum("bhnd,bhnm->bhdm", k, v)
        z = torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + 1e-6
        return torch.einsum("bhnd,bhdm,bhn->bhnm", q, kv, z.reciprocal())

    @staticmethod
    def _causal_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        B, H, T, d = q.shape
        dv = v.shape[-1]
        out = torch.zeros(B, H, T, dv, device=q.device, dtype=q.dtype)
        kv_sum = torch.zeros(B, H, d, dv, device=q.device, dtype=q.dtype)
        k_sum = torch.zeros(B, H, d, device=q.device, dtype=q.dtype)
        for t in range(T):
            kv_sum = kv_sum + torch.einsum("bhd,bhe->bhde", k[:, :, t], v[:, :, t])
            k_sum = k_sum + k[:, :, t]
            z = torch.einsum("bhd,bhd->bh", q[:, :, t], k_sum) + 1e-6
            num = torch.einsum("bhd,bhde->bhe", q[:, :, t], kv_sum)
            out[:, :, t] = num / z.unsqueeze(-1)
        return out

    def extra_repr(self) -> str:
        return f"n_heads={self.n_heads}, causal={self.causal}, cost={self.cost}"
