# Copyright 2024-2025 Alan Ferrari
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch
import pytest

from meta_attention.utils import (
    RoutingStats,
    additive_causal_mask,
    causal_mask,
    compute_routing_stats,
    cosine_ramp,
    gumbel_softmax_top_k,
    hf_attention_mask_to_additive,
    linear_ramp,
)


def test_causal_mask_shape():
    mask = causal_mask(8, torch.device("cpu"))
    assert mask.shape == (1, 1, 8, 8)
    # lower triangle is True
    assert mask[0, 0, 0, 0].item() is True
    assert mask[0, 0, 0, 1].item() is False


def test_additive_causal_mask():
    mask = additive_causal_mask(4, torch.device("cpu"))
    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 0, 0].item() == 0.0
    assert mask[0, 0, 0, 1].item() == float("-inf")


def test_hf_mask_conversion():
    attn_mask = torch.tensor([[1, 1, 0, 0]])  # (B=1, T=4)
    out = hf_attention_mask_to_additive(attn_mask, dtype=torch.float32)
    assert out.shape == (1, 1, 1, 4)
    assert out[0, 0, 0, 0].item() == 0.0
    assert out[0, 0, 0, 2].item() == float("-inf")


def test_routing_stats():
    alpha = torch.softmax(torch.randn(2, 16, 3), dim=-1)
    costs = torch.tensor([1.0, 0.15, 0.30])
    stats = compute_routing_stats(alpha, costs)
    assert stats.weights.shape == (3,)
    assert 0.0 <= stats.n_collapsed <= 1.0


def test_gumbel_soft():
    logits = torch.randn(2, 16, 3)
    w = gumbel_softmax_top_k(logits, k=2, hard=False)
    assert w.shape == (2, 16, 3)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, 16), atol=1e-5)


def test_gumbel_hard():
    logits = torch.randn(2, 16, 3)
    w = gumbel_softmax_top_k(logits, k=1, hard=True)
    assert w.shape == (2, 16, 3)
    # each token activates exactly 1 expert
    assert (w > 0).sum(dim=-1).eq(1).all()


def test_cosine_ramp():
    assert cosine_ramp(0, 0, 100) == pytest.approx(0.0)
    assert cosine_ramp(100, 0, 100) == pytest.approx(1.0)
    val = cosine_ramp(50, 0, 100)
    assert 0.0 < val < 1.0


def test_linear_ramp():
    assert linear_ramp(0, 0, 100) == pytest.approx(0.0)
    assert linear_ramp(100, 0, 100) == pytest.approx(1.0)
    assert linear_ramp(50, 0, 100) == pytest.approx(0.5)
