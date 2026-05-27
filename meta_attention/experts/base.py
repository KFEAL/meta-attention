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

"""Abstract base class for Meta-Attention experts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class AttentionExpert(nn.Module, ABC):
    """Interface every Meta-Attention expert must implement.

    Each expert maps ``(B, T, D) → (B, T, D)`` and exposes a normalised
    compute-cost scalar used by the auxiliary loss.

    Subclassing
    -----------
    Override :meth:`forward` and set the class-level ``_cost`` attribute::

        class MyExpert(AttentionExpert):
            _cost = 0.5

            def forward(self, x, mask=None):
                ...
                return out

    The cost should be normalised relative to ``FullAttention`` (cost=1.0).
    Register custom experts with :func:`meta_attention.experts.registry.register_expert`
    to make them discoverable by name.
    """

    #: Normalised compute cost in [0, 1].  1.0 == full O(T²) softmax attention.
    _cost: float = 1.0

    def __init__(self, cfg: Optional[object] = None, **kwargs: object) -> None:
        """Accept ``MetaAttnConfig`` for registry-built experts; ignore if unused."""
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run attention.

        Parameters
        ----------
        x : (B, T, D)
        mask:
            Optional additive attention mask ``(B, 1, T, T)``.
            0 = attend, -inf = mask out.  May be ignored by sub-quadratic experts.

        Returns
        -------
        out : (B, T, D)
        """
        ...

    @property
    def cost(self) -> float:
        """Normalised compute cost.  1.0 == full softmax attention."""
        return self._cost

    def extra_repr(self) -> str:
        return f"cost={self.cost}"
