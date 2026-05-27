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

"""Generic attention wrapper compatible with ``nn.MultiheadAttention``.

Provides two helpers:

* :class:`MetaAttentionWrapper` — wraps ``MetaAttentionLayer`` behind the
  ``nn.MultiheadAttention`` call signature so it slots into any
  ``nn.TransformerEncoderLayer`` without source changes.
* :func:`patch_module` — recursively replaces every instance of a target
  class inside an ``nn.Module`` with a ``MetaAttentionWrapper``.
* :func:`collect_aux_losses` — sums auxiliary losses after a forward pass.
"""

from __future__ import annotations

import logging
from typing import Optional, Type

import torch
import torch.nn as nn

from ..config import MetaAttnConfig
from ..layer import MetaAttentionLayer

log = logging.getLogger(__name__)


class MetaAttentionWrapper(nn.Module):
    """Drop-in replacement for ``nn.MultiheadAttention`` using Meta-Attention.

    Implements the ``forward(query, key, value, ...)`` signature of MHA and
    returns ``(output, None)`` so it is transparent to PyTorch internals.

    Auxiliary loss from each forward pass is stored in ``self.last_aux_loss``
    and can be collected with :func:`collect_aux_losses`.

    Parameters
    ----------
    cfg:
        Meta-Attention configuration.
    batch_first:
        Match the surrounding ``TransformerEncoderLayer`` setting.
    pass_mask_kwarg:
        Name of the keyword argument used for masks by the host architecture.
        ``None`` to ignore all mask inputs.
    """

    def __init__(
        self,
        cfg: MetaAttnConfig,
        batch_first: bool = True,
        pass_mask_kwarg: Optional[str] = "attn_mask",
    ) -> None:
        super().__init__()
        self.layer = MetaAttentionLayer(cfg)
        self.pass_mask_kwarg = pass_mask_kwarg
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)
        self.last_stats = None

        # Attributes read by nn.TransformerEncoderLayer internals
        self.batch_first = batch_first
        self.embed_dim = cfg.d_model

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Returns ``(output, None)`` matching ``nn.MultiheadAttention``."""
        x = query
        if not self.batch_first:
            x = x.transpose(0, 1)

        mask = kwargs.get(self.pass_mask_kwarg) if self.pass_mask_kwarg else None
        out, aux_loss, stats = self.layer(x, mask=mask)
        self.last_aux_loss = aux_loss
        self.last_stats = stats

        if not self.batch_first:
            out = out.transpose(0, 1)

        return out, None


def patch_module(
    model: nn.Module,
    target_class: Type[nn.Module],
    cfg: MetaAttnConfig,
    pass_mask_kwarg: Optional[str] = "attn_mask",
    verbose: bool = True,
) -> nn.Module:
    """Replace every ``target_class`` instance in *model* with a :class:`MetaAttentionWrapper`.

    Operates **in-place** and returns *model* for chaining.

    Parameters
    ----------
    model:
        The model to patch.
    target_class:
        Attention class to replace (e.g. ``nn.MultiheadAttention``).
    cfg:
        Config for the replacement layers.
    pass_mask_kwarg:
        Mask keyword argument name used by the host architecture.
    verbose:
        Log each replacement.

    Example
    -------
    >>> encoder = nn.TransformerEncoder(
    ...     nn.TransformerEncoderLayer(d_model=256, nhead=8), num_layers=4
    ... )
    >>> patch_module(encoder, nn.MultiheadAttention, MetaAttnConfig(d_model=256, n_heads=8))
    """
    n_replaced = 0
    for name, module in list(model.named_children()):
        if isinstance(module, target_class):
            batch_first = getattr(module, "batch_first", True)
            setattr(
                model,
                name,
                MetaAttentionWrapper(cfg, batch_first=batch_first, pass_mask_kwarg=pass_mask_kwarg),
            )
            n_replaced += 1
            if verbose:
                log.info("Replaced %s: %s → MetaAttentionWrapper", name, type(module).__name__)
        else:
            patch_module(module, target_class, cfg, pass_mask_kwarg, verbose=False)

    if verbose and n_replaced:
        log.info("Total replacements at top level: %d", n_replaced)
    return model


def collect_aux_losses(model: nn.Module) -> torch.Tensor:
    """Sum ``last_aux_loss`` from all :class:`MetaAttentionWrapper` instances.

    Call after each forward pass and add to the task loss::

        out, _ = model(x)
        loss = task_loss(out) + collect_aux_losses(model)
    """
    total = torch.tensor(0.0)
    for m in model.modules():
        if isinstance(m, MetaAttentionWrapper):
            total = total + m.last_aux_loss
    return total
