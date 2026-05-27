# Copyright 2024-2025 Alan Ferrari
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest
from meta_attention import MetaAttnConfig, MetaLMConfig


def test_defaults():
    cfg = MetaAttnConfig()
    assert cfg.d_model == 512
    assert cfg.n_heads == 8
    assert cfg.d_head == 64
    assert cfg.n_experts == 3


def test_d_head_invalid():
    with pytest.raises(ValueError, match="divisible"):
        _ = MetaAttnConfig(d_model=100, n_heads=8).d_head


def test_presets():
    for preset in ("small", "base", "large", "llama_7b", "gpt2", "gpt2_medium"):
        cfg = getattr(MetaAttnConfig, preset)()
        assert cfg.d_model % cfg.n_heads == 0


def test_serialisation_roundtrip():
    cfg = MetaAttnConfig.small()
    assert MetaAttnConfig.from_json(cfg.to_json()) == cfg
    assert MetaAttnConfig.from_dict(cfg.to_dict()) == cfg


def test_lm_config():
    lm_cfg = MetaLMConfig(attn=MetaAttnConfig.small(), vocab_size=1000, n_layers=2)
    assert lm_cfg.attn.d_model == 128
