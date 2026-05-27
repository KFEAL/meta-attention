# Copyright 2024-2025 Alan Ferrari
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Registering and using a custom attention expert."""

from typing import Optional

import torch
import torch.nn as nn

from meta_attention import MetaAttnConfig, MetaAttentionLayer, register_expert
from meta_attention.experts import AttentionExpert


@register_expert("identity")
class IdentityAttention(AttentionExpert):
    """Trivial expert that returns the input unchanged (cost ≈ 0)."""

    _cost: float = 0.01

    def __init__(self, cfg: MetaAttnConfig) -> None:
        super().__init__()
        # A minimal learned transform so the expert has parameters.
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.proj(x)


if __name__ == "__main__":
    from meta_attention import build_expert, list_experts

    print("Registered experts:", list_experts())  # ['full', 'identity', 'linear', 'local']

    cfg = MetaAttnConfig(
        d_model=256,
        n_heads=8,
        # Three experts: full, linear, and our custom identity expert.
        expert_costs=[1.0, 0.15, 0.01],
    )

    experts = [
        build_expert("full", cfg),
        build_expert("linear", cfg),
        build_expert("identity", cfg),
    ]
    layer = MetaAttentionLayer(cfg, experts=experts)

    x = torch.randn(1, 32, 256)
    out, aux_loss, stats = layer(x)
    print(f"out: {out.shape}  aux_loss: {aux_loss:.4f}")
    print(f"stats: {stats}")
