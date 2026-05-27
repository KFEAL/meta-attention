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

"""PyTorch native SDPA backend (default).

Uses ``torch.nn.functional.scaled_dot_product_attention``, which
automatically dispatches to FlashAttention on CUDA with fp16/bf16 when
cuDNN/Flash-Attention is available, and falls back to a memory-efficient
math kernel otherwise.

Available in PyTorch >= 2.0.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import AttentionBackend


class TorchSDPABackend(AttentionBackend):
    """Wrapper around ``F.scaled_dot_product_attention`` (PyTorch >= 2.0)."""

    @property
    def name(self) -> str:
        return "torch_sdpa"

    @property
    def is_available(self) -> bool:
        return hasattr(F, "scaled_dot_product_attention")

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        if not self.is_available:
            raise RuntimeError(
                "TorchSDPABackend requires PyTorch >= 2.0.  "
                "Upgrade with: pip install --upgrade torch"
            )
        kwargs = dict(
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
        if scale is not None:
            kwargs["scale"] = scale
        return F.scaled_dot_product_attention(q, k, v, **kwargs)
