# Copyright 2024-2025 Alan Ferrari
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn as nn
import pytest

from meta_attention import MetaAttnConfig
from meta_attention.integrations.generic import (
    MetaAttentionWrapper,
    collect_aux_losses,
    patch_module,
)

D, H = 128, 4
CFG = MetaAttnConfig(d_model=D, n_heads=H)


def test_wrapper_output_shape():
    wrapper = MetaAttentionWrapper(CFG)
    x = torch.randn(2, 32, D)
    out, weights = wrapper(x)
    assert out.shape == (2, 32, D)
    assert weights is None


def test_wrapper_seq_first():
    wrapper = MetaAttentionWrapper(CFG, batch_first=False)
    x = torch.randn(32, 2, D)   # (T, B, D)
    out, _ = wrapper(x)
    assert out.shape == (32, 2, D)


def test_patch_module_replaces_mha():
    encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=D, nhead=H, batch_first=True),
        num_layers=2,
    )
    patch_module(encoder, nn.MultiheadAttention, CFG, verbose=False)

    n_wrappers = sum(
        1 for m in encoder.modules() if isinstance(m, MetaAttentionWrapper)
    )
    assert n_wrappers == 2


def test_collect_aux_losses():
    encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=D, nhead=H, batch_first=True),
        num_layers=2,
    )
    patch_module(encoder, nn.MultiheadAttention, CFG, verbose=False)

    x = torch.randn(2, 32, D)
    encoder(x)
    aux = collect_aux_losses(encoder)
    assert aux.shape == ()


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("transformers"),
    reason="transformers not installed",
)
def test_hf_patch_gpt2():
    from transformers import AutoModelForCausalLM
    from meta_attention.integrations.hf import (
        MetaHFAttention,
        collect_hf_aux_losses,
        patch_hf_model,
    )

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    cfg = MetaAttnConfig.gpt2()
    patch_hf_model(model, cfg, verbose=False)

    n_replaced = sum(1 for m in model.modules() if isinstance(m, MetaHFAttention))
    assert n_replaced == 12   # GPT-2 base has 12 layers

    inputs = torch.randint(0, 50257, (1, 8))
    with torch.no_grad():
        out = model(inputs)
    assert out.logits.shape == (1, 8, 50257)
    aux = collect_hf_aux_losses(model)
    assert aux.shape == ()
