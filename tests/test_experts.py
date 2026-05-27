# Copyright 2024-2025 Alan Ferrari
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch
from meta_attention import MetaAttnConfig
from meta_attention.experts import (
    AttentionExpert,
    FullAttention,
    LinearAttention,
    LocalAttention,
    build_expert,
    list_experts,
    register_expert,
    unregister_expert,
)

CFG = MetaAttnConfig.small()
B, T, D = 2, 32, CFG.d_model


@pytest.mark.parametrize("ExpertCls", [FullAttention, LinearAttention, LocalAttention])
def test_expert_output_shape(ExpertCls):
    expert = ExpertCls(CFG)
    x = torch.randn(B, T, D)
    out = expert(x)
    assert out.shape == (B, T, D)


def test_linear_causal():
    expert = LinearAttention(CFG, causal=True)
    x = torch.randn(B, T, D)
    out = expert(x)
    assert out.shape == (B, T, D)


def test_expert_cost_range():
    for ExpertCls in (FullAttention, LinearAttention, LocalAttention):
        e = ExpertCls(CFG)
        assert 0.0 < e.cost <= 1.0


def test_registry_builtin():
    assert set(list_experts()) >= {"full", "linear", "local"}


def test_register_and_build():
    @register_expert("_test_expert_tmp")
    class _TmpExpert(AttentionExpert):
        _cost = 0.5
        def forward(self, x, mask=None):
            return x

    assert "_test_expert_tmp" in list_experts()
    e = build_expert("_test_expert_tmp", CFG)
    x = torch.randn(B, T, D)
    assert e(x).shape == (B, T, D)
    unregister_expert("_test_expert_tmp")
    assert "_test_expert_tmp" not in list_experts()


def test_build_unknown_raises():
    with pytest.raises(KeyError, match="Unknown expert"):
        build_expert("does_not_exist", CFG)


def test_duplicate_register_raises():
    @register_expert("_dup_tmp")
    class _A(AttentionExpert):
        _cost = 0.1
        def forward(self, x, mask=None): return x

    with pytest.raises(KeyError, match="already registered"):
        @register_expert("_dup_tmp")
        class _B(AttentionExpert):
            _cost = 0.2
            def forward(self, x, mask=None): return x

    unregister_expert("_dup_tmp")
