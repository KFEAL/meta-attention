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

"""Sliding-window local attention expert (E3)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AttentionExpert
from .registry import register_expert
from ..config import MetaAttnConfig

_HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


@register_expert("local")
class LocalAttention(AttentionExpert):
    """Sliding-window multi-head attention.

    Each token ``i`` attends only to positions in ``[i - w, i + w]``.

    Complexity: O(T · w · D).
    Normalised cost: 0.30.

    For production use at large ``T``, consider replacing the inner loop
    with a chunked CUDA kernel (e.g. xFormers ``local_attention``).

    Parameters
    ----------
    cfg:
        Model config.  Uses ``window_size``, ``n_heads``, ``d_head``,
        ``dropout``, and ``use_flash``.
    """

    _cost: float = 0.30

    def __init__(self, cfg: MetaAttnConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.w = cfg.window_size
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
        w = min(self.w, T - 1)

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = self._windowed_attn(q, k, v, w, B, T)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

    def _windowed_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: int,
        B: int,
        T: int,
    ) -> torch.Tensor:
        out = torch.zeros(B, self.n_heads, T, self.d_head, device=q.device, dtype=q.dtype)
        drop_p = self.attn_drop.p if self.training else 0.0

        for i in range(T):
            lo = max(0, i - w)
            hi = min(T, i + w + 1)
            q_i = q[:, :, i : i + 1, :]
            k_w = k[:, :, lo:hi, :]
            v_w = v[:, :, lo:hi, :]

            if self.use_flash:
                out[:, :, i : i + 1, :] = F.scaled_dot_product_attention(
                    q_i, k_w, v_w, dropout_p=drop_p
                )
            else:
                scores = (q_i @ k_w.transpose(-2, -1)) * self.scale
                attn = F.softmax(scores, dim=-1)
                if self.training and drop_p > 0:
                    attn = F.dropout(attn, p=drop_p)
                out[:, :, i : i + 1, :] = attn @ v_w
        return out

    def extra_repr(self) -> str:
        return f"n_heads={self.n_heads}, window={self.w}, cost={self.cost}"
