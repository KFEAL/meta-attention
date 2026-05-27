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

"""Configuration dataclasses for Meta-Attention."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class MetaAttnConfig:
    """Configuration for a single Meta-Attention layer.

    Parameters
    ----------
    d_model:
        Total model dimensionality.
    n_heads:
        Number of attention heads.  Must evenly divide ``d_model``.
    window_size:
        Half-window radius used by ``LocalAttention`` — attends to
        ``[i - window_size, i + window_size]``.
    dropout:
        Dropout probability applied inside experts and the controller.
    temperature:
        Softmax temperature for routing logits.  Lower → sharper routing.
    controller_hidden:
        Hidden width of the Meta-Controller MLP.
    expert_costs:
        Normalised compute cost per expert ``[full, linear, local]``.
        Must be the same length as the number of experts used.
    cost_lambda:
        Weight of the expected-compute-cost auxiliary loss term.
        Set to 0.0 (default) to disable.
    entropy_coeff:
        Weight of the routing-entropy bonus (encourages load balance).
        Set to 0.0 (default) to disable.
    use_flash:
        If ``True``, ``FullAttention`` uses ``F.scaled_dot_product_attention``
        (Flash-style) when PyTorch >= 2.0.  Falls back automatically if
        unavailable.
    hard_routing:
        If ``True``, use Gumbel-softmax (training) / argmax (inference)
        for hard sparse routing.  If ``False`` (default), use soft
        weighted merge — all experts run every forward pass.
    hard_top_k:
        Number of experts to activate per token under hard routing.
    gumbel_temp:
        Temperature for Gumbel-softmax during training.
    backend:
        Compute backend for built-in experts.  One of
        ``"torch_sdpa"`` (default), ``"xformers"``, ``"flash_attn"``.
        Third-party backends must be installed separately.
    """

    d_model: int = 512
    n_heads: int = 8
    window_size: int = 64
    dropout: float = 0.1
    temperature: float = 1.0
    controller_hidden: int = 128
    expert_costs: List[float] = field(default_factory=lambda: [1.0, 0.15, 0.30])
    cost_lambda: float = 0.0
    entropy_coeff: float = 0.0
    use_flash: bool = True
    hard_routing: bool = False
    hard_top_k: int = 1
    gumbel_temp: float = 1.0
    backend: str = "torch_sdpa"

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def d_head(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        return self.d_model // self.n_heads

    @property
    def n_experts(self) -> int:
        return len(self.expert_costs)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MetaAttnConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "MetaAttnConfig":
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @classmethod
    def small(cls) -> "MetaAttnConfig":
        """128-d, 4-head — unit tests and ablations."""
        return cls(d_model=128, n_heads=4, window_size=32, controller_hidden=64)

    @classmethod
    def base(cls) -> "MetaAttnConfig":
        """512-d, 8-head — matches paper Phase-2 training config."""
        return cls(d_model=512, n_heads=8)

    @classmethod
    def large(cls) -> "MetaAttnConfig":
        """1024-d, 16-head — GPT-2 Large / LLaMA-7B compatible."""
        return cls(d_model=1024, n_heads=16, window_size=128, controller_hidden=256)

    @classmethod
    def llama_7b(cls) -> "MetaAttnConfig":
        """Drop-in config for LLaMA-7B (hidden=4096, heads=32)."""
        return cls(d_model=4096, n_heads=32, window_size=256, controller_hidden=512)

    @classmethod
    def gpt2(cls) -> "MetaAttnConfig":
        """Drop-in config for GPT-2 base (hidden=768, heads=12)."""
        return cls(d_model=768, n_heads=12, window_size=64, controller_hidden=192)

    @classmethod
    def gpt2_medium(cls) -> "MetaAttnConfig":
        """Drop-in config for GPT-2 Medium (hidden=1024, heads=16)."""
        return cls(d_model=1024, n_heads=16, window_size=128, controller_hidden=256)


@dataclass
class MetaLMConfig:
    """Configuration for the standalone Meta-Attention language model."""

    attn: MetaAttnConfig = field(default_factory=MetaAttnConfig)
    vocab_size: int = 50257
    n_layers: int = 6
    max_seq_len: int = 1024
    ffn_multiplier: int = 4
    tie_weights: bool = True
