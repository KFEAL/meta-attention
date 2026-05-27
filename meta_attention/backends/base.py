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

"""Abstract base class for attention compute backends.

A backend is a thin, stateless adapter that wraps a specific attention
kernel (PyTorch SDPA, xFormers, FlashAttention-2, …) behind a uniform
``sdpa``-style interface.  Experts call the active backend rather than
hard-coding kernel calls, making it easy to swap implementations at
construction time via ``cfg.backend``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch


class AttentionBackend(ABC):
    """Uniform interface for attention kernels.

    All backends implement :meth:`scaled_dot_product_attention`, mirroring
    ``torch.nn.functional.scaled_dot_product_attention``.

    Parameters for every concrete backend constructor:
    - ``dropout_p`` — applied during training only.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``"torch_sdpa"``."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """``True`` if the underlying library is installed and importable."""
        ...

    @abstractmethod
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
        """Compute scaled dot-product attention.

        Parameters
        ----------
        q, k, v : (B, H, T, d_head)
        attn_mask:
            Additive mask ``(B, 1, T, T)`` or ``None``.
        dropout_p:
            Attention dropout probability (applied during training).
        is_causal:
            If ``True``, apply a causal mask automatically.
            Mutually exclusive with ``attn_mask``.
        scale:
            Override the default ``1 / sqrt(d_head)`` scaling.

        Returns
        -------
        out : (B, H, T, d_head)
        """
        ...
