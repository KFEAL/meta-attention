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

"""xFormers memory-efficient attention backend.

Requires: ``pip install xformers>=0.0.22``

xFormers provides ``xformers.ops.memory_efficient_attention``, which is
particularly efficient for variable-length sequences and avoids materialising
the full T×T attention matrix.

Input convention for xFormers: ``(B, T, H, d_head)`` (different from the
``(B, H, T, d_head)`` convention used elsewhere in this library).
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .base import AttentionBackend


class XFormersBackend(AttentionBackend):
    """xFormers ``memory_efficient_attention`` backend.

    Install with ``pip install xformers`` (or ``meta-attention[xformers]``).
    """

    @property
    def name(self) -> str:
        return "xformers"

    @property
    def is_available(self) -> bool:
        try:
            import xformers.ops  # noqa: F401
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
        q, k, v : (B, H, T, d_head)  — standard library convention.

        Returns
        -------
        out : (B, H, T, d_head)
        """
        try:
            import xformers.ops as xops
        except ImportError:
            raise ImportError(
                "xFormers is not installed.  Install with: "
                "pip install xformers  or  pip install 'meta-attention[xformers]'"
            )

        B, H, T, d = q.shape
        scale = scale or (d ** -0.5)

        # xFormers expects (B, T, H, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_bias = None
        if is_causal:
            attn_bias = xops.LowerTriangularMask()
        elif attn_mask is not None:
            # xFormers accepts (B, 1, T, T) float additive bias
            attn_bias = attn_mask.expand(B, H, T, T) if attn_mask.shape[1] == 1 else attn_mask

        out = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=attn_bias,
            p=dropout_p,
            scale=scale,
        )

        # Return to (B, H, T, d_head)
        return out.transpose(1, 2)
