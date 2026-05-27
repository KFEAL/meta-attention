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

"""HuggingFace Transformers integration.

Supports GPT-2, LLaMA, Mistral, Phi, and Falcon.  Falls back to a generic
``nn.MultiheadAttention`` scan for unknown architectures.

Quick start
-----------
>>> from transformers import AutoModelForCausalLM
>>> from meta_attention import MetaAttnConfig
>>> from meta_attention.integrations.hf import patch_hf_model, collect_hf_aux_losses

>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> cfg = MetaAttnConfig.gpt2()
>>> patch_hf_model(model, cfg)
>>> outputs = model(**inputs)
>>> loss = outputs.loss + collect_hf_aux_losses(model)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import MetaAttnConfig
from ..layer import MetaAttentionLayer
from ..utils import RoutingStats, hf_attention_mask_to_additive

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HuggingFace-compatible wrapper
# ---------------------------------------------------------------------------


class MetaHFAttention(nn.Module):
    """Wraps :class:`MetaAttentionLayer` behind the HuggingFace attention interface.

    HuggingFace attention modules return tuples; the exact shape varies by
    architecture.  This wrapper handles the impedance mismatch so that the
    surrounding ``forward()`` method in GPT-2, LLaMA, etc. requires no edits.

    Auxiliary loss is stored in ``self.last_aux_loss`` after each forward and
    can be collected with :func:`collect_hf_aux_losses`.

    Parameters
    ----------
    cfg:
        Meta-Attention configuration.
    hf_model_type:
        One of ``"gpt2"``, ``"llama"``, ``"mistral"``, ``"phi"``, ``"generic"``.
        Controls output tuple format.
    """

    def __init__(self, cfg: MetaAttnConfig, hf_model_type: str = "generic") -> None:
        super().__init__()
        self.meta_layer = MetaAttentionLayer(cfg)
        self.hf_model_type = hf_model_type
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)
        self.last_stats: Optional[RoutingStats] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        # GPT-2 extras
        layer_past: Optional[tuple] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        # LLaMA / Mistral extras
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple:
        mask = self._build_mask(hidden_states, attention_mask)
        out, aux_loss, stats = self.meta_layer(hidden_states, mask=mask)
        self.last_aux_loss = aux_loss
        self.last_stats = stats

        if self.hf_model_type == "gpt2":
            # (attn_output, present)  where present = (k, v) or None
            return (out, None)
        elif self.hf_model_type in ("llama", "mistral", "phi"):
            # (attn_output, attn_weights, past_key_value)
            return (out, None, past_key_value)
        else:
            return (out,)

    def _build_mask(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            return attention_mask
        if attention_mask.dim() == 2:
            return hf_attention_mask_to_additive(attention_mask, dtype=x.dtype)
        return None


# ---------------------------------------------------------------------------
# Model surgery helpers
# ---------------------------------------------------------------------------


def patch_hf_model(
    model: nn.Module,
    cfg: MetaAttnConfig,
    verbose: bool = True,
) -> nn.Module:
    """Replace attention layers in a HuggingFace model with Meta-Attention.

    Supports GPT-2, LLaMA, Mistral, Phi, Falcon.  Falls back to a generic
    ``nn.MultiheadAttention`` scan for unknown architectures.

    .. important::

       Ensure ``cfg.d_model`` and ``cfg.n_heads`` match the model's
       ``hidden_size`` and ``num_attention_heads``.

    Parameters
    ----------
    model:
        HuggingFace model (e.g. from ``AutoModelForCausalLM.from_pretrained``).
    cfg:
        Meta-Attention config matching the model's dimensions.
    verbose:
        Print a summary of replaced layers.

    Returns
    -------
    model  (modified in-place, returned for chaining)
    """
    arch = _detect_arch(model)
    if verbose:
        log.info("Detected HF architecture: %s", arch)

    dispatch = {
        "gpt2": _patch_gpt2,
        "llama": lambda m, c, v: _patch_llama_style(m, c, "llama", v),
        "mistral": lambda m, c, v: _patch_llama_style(m, c, "mistral", v),
        "phi": lambda m, c, v: _patch_llama_style(m, c, "phi", v),
    }
    patch_fn = dispatch.get(arch, _patch_generic)
    patch_fn(model, cfg, verbose)
    return model


def collect_hf_aux_losses(model: nn.Module) -> torch.Tensor:
    """Sum ``last_aux_loss`` from all :class:`MetaHFAttention` layers.

    Call after each forward pass::

        outputs = model(**batch)
        loss = outputs.loss + collect_hf_aux_losses(model)
    """
    total = torch.tensor(0.0)
    for m in model.modules():
        if isinstance(m, MetaHFAttention):
            total = total + m.last_aux_loss
    return total


def get_hf_routing_stats(model: nn.Module):
    """Return ``RoutingStats`` from all :class:`MetaHFAttention` layers."""
    return [m.last_stats for m in model.modules() if isinstance(m, MetaHFAttention)]


# ---------------------------------------------------------------------------
# Architecture-specific patches
# ---------------------------------------------------------------------------


def _patch_gpt2(model: nn.Module, cfg: MetaAttnConfig, verbose: bool) -> None:
    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    except ImportError:
        log.warning("transformers not installed; skipping GPT-2 patch")
        return

    replaced = 0
    for name, module in model.named_modules():
        if isinstance(module, GPT2Attention):
            parent, attr = _parent_and_attr(model, name)
            setattr(parent, attr, MetaHFAttention(cfg, hf_model_type="gpt2"))
            replaced += 1
            if verbose:
                log.info("  Replaced GPT2Attention at %s", name)
    if verbose:
        log.info("GPT-2: replaced %d attention layers", replaced)


def _patch_llama_style(
    model: nn.Module,
    cfg: MetaAttnConfig,
    arch: str,
    verbose: bool,
) -> None:
    _TARGET_NAMES = {
        "llama": ("LlamaAttention", "LlamaSdpaAttention", "LlamaFlashAttention2"),
        "mistral": ("MistralAttention", "MistralSdpaAttention"),
        "phi": ("PhiAttention",),
    }
    targets = _TARGET_NAMES.get(arch, ())

    try:
        import transformers  # noqa: F401
    except ImportError:
        log.warning("transformers not installed; skipping %s patch", arch)
        return

    replaced = 0
    for name, module in model.named_modules():
        if type(module).__name__ in targets:
            parent, attr = _parent_and_attr(model, name)
            setattr(parent, attr, MetaHFAttention(cfg, hf_model_type=arch))
            replaced += 1
            if verbose:
                log.info("  Replaced %s at %s", type(module).__name__, name)
    if verbose:
        log.info("%s: replaced %d attention layers", arch, replaced)


def _patch_generic(model: nn.Module, cfg: MetaAttnConfig, verbose: bool) -> None:
    replaced = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            parent, attr = _parent_and_attr(model, name)
            setattr(parent, attr, MetaHFAttention(cfg, hf_model_type="generic"))
            replaced += 1
            if verbose:
                log.info("  Replaced MultiheadAttention at %s", name)
    if verbose:
        log.info("Generic: replaced %d attention layers", replaced)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_arch(model: nn.Module) -> str:
    cls = type(model).__name__.lower()
    for arch in ("gpt2", "llama", "mistral", "phi", "falcon"):
        if arch in cls:
            return arch
    return "generic"


def _parent_and_attr(root: nn.Module, dotted_name: str):
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]
