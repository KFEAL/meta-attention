# Copyright 2024-2025 Alan Ferrari
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch
from meta_attention import MetaAttnConfig, MetaLanguageModel, MetaLMConfig

LM_CFG = MetaLMConfig(
    attn=MetaAttnConfig.small(),
    vocab_size=500,
    n_layers=2,
    max_seq_len=64,
)


def test_logits_shape():
    model = MetaLanguageModel(LM_CFG)
    idx = torch.randint(0, LM_CFG.vocab_size, (2, 32))
    logits, aux_loss, stats = model(idx)
    assert logits.shape == (2, 32, LM_CFG.vocab_size)
    assert len(stats) == LM_CFG.n_layers


def test_compute_loss():
    model = MetaLanguageModel(LM_CFG)
    idx = torch.randint(0, LM_CFG.vocab_size, (2, 32))
    loss, metrics = model.compute_loss(idx)
    assert loss.item() > 0
    assert "lm_loss" in metrics
    assert "ppl" in metrics


def test_generate():
    model = MetaLanguageModel(LM_CFG)
    prompt = torch.randint(0, LM_CFG.vocab_size, (1, 8))
    out = model.generate(prompt, max_new_tokens=4)
    assert out.shape == (1, 12)


def test_param_count():
    model = MetaLanguageModel(LM_CFG)
    counts = model.param_count()
    assert counts["total"] > 0
    assert counts["total"] == counts["embedding"] + counts["attention"] + counts["ffn"]


def test_backward():
    model = MetaLanguageModel(LM_CFG)
    idx = torch.randint(0, LM_CFG.vocab_size, (2, 16))
    loss, _ = model.compute_loss(idx)
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            break
