# Copyright 2024-2025 Alan Ferrari
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch
from meta_attention import MetaAttnConfig, MetaAttentionLayer, RoutingStats
from meta_attention.utils import additive_causal_mask

CFG = MetaAttnConfig.small()
B, T, D = 2, 32, CFG.d_model


def test_output_shape():
    layer = MetaAttentionLayer(CFG)
    x = torch.randn(B, T, D)
    out, aux_loss, stats = layer(x)
    assert out.shape == (B, T, D)
    assert aux_loss.shape == ()
    assert isinstance(stats, RoutingStats)


def test_soft_routing_gradients():
    layer = MetaAttentionLayer(CFG)
    x = torch.randn(B, T, D, requires_grad=True)
    out, aux_loss, _ = layer(x)
    (out.sum() + aux_loss).backward()
    assert x.grad is not None


def test_hard_routing_forward():
    cfg = MetaAttnConfig.small()
    cfg.hard_routing = True
    layer = MetaAttentionLayer(cfg)
    x = torch.randn(B, T, D)
    out, _, _ = layer(x)
    assert out.shape == (B, T, D)


def test_causal_mask_applied():
    layer = MetaAttentionLayer(CFG)
    mask = additive_causal_mask(T, torch.device("cpu"))
    x = torch.randn(B, T, D)
    out, _, _ = layer(x, mask=mask)
    assert out.shape == (B, T, D)


def test_cost_lambda_aux_loss():
    cfg = MetaAttnConfig.small()
    cfg.cost_lambda = 0.1
    layer = MetaAttentionLayer(cfg)
    x = torch.randn(B, T, D)
    _, aux_loss, _ = layer(x)
    assert aux_loss.item() > 0


def test_temperature_update():
    layer = MetaAttentionLayer(CFG)
    layer.set_temperature(0.5)
    assert layer.controller.temperature == 0.5
    assert layer.cfg.temperature == 0.5


def test_wrong_expert_count_raises():
    cfg = MetaAttnConfig.small()
    cfg.expert_costs = [1.0, 0.5]   # expects 2, but 3 default experts
    with pytest.raises(ValueError, match="must equal"):
        MetaAttentionLayer(cfg)
