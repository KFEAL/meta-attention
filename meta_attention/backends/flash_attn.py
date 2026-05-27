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

"""FlashAttention-2 backend.

Requires: ``pip install flash-attn>=2.0.0``

FlashAttention-2 provides the fastest known GPU kernel for full softmax
attention.  It only supports CUDA and requires fp16 or bf16 inputs.

Input convention for flash_attn: ``(B, T, H, d_head)``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .base import AttentionBackend


class FlashAttnBackend(AttentionBackend):
    """FlashAttention-2 backend via the ``flash_attn`` package.

    Install with ``pip install flash-attn`` (or ``meta-attention[flash]``).

    Constraints
    -----------
    * CUDA only.
    * Inputs must be fp16 or bf16.
    * ``attn_mask`` is not supported; use ``is_causal=True`` for causal masking.
    """

    @property
    def name(self) -> str:
        return "flash_attn"

    @property
    def is_available(self) -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False

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
        """
        Parameters
        ----------
        q, k, v : (B, H, T, d_head)

        Notes
        -----
        ``attn_mask`` is silently ignored — FlashAttention-2 only supports
        causal masking.  Convert custom masks to the ``torch_sdpa`` backend.

        Returns
        -------
        out : (B, H, T, d_head)
        """
        try:
            from flash_attn import flash_attn_func
        except ImportError:
            raise ImportError(
                "flash-attn is not installed.  Install with: "
                "pip install flash-attn  or  pip install 'meta-attention[flash]'"
            )

        if attn_mask is not None:
            import warnings
            warnings.warn(
                "FlashAttnBackend does not support arbitrary attn_mask; "
                "the mask will be ignored.  Use is_causal=True for causal masking "
                "or switch to the 'torch_sdpa' backend.",
                stacklevel=3,
            )

        B, H, T, d = q.shape
        softmax_scale = scale or (d ** -0.5)

        # flash_attn expects (B, T, H, d_head)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        out = flash_attn_func(
            q, k, v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=is_causal,
        )

        return out.transpose(1, 2)
