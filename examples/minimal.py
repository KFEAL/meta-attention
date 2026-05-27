# Copyright 2024-2025 Alan Ferrari
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Minimal usage: MetaAttentionLayer as a standalone module."""

import torch
from meta_attention import MetaAttnConfig, MetaAttentionLayer

cfg = MetaAttnConfig.small()          # 128-d, 4-head
layer = MetaAttentionLayer(cfg)

x = torch.randn(2, 64, 128)           # (batch, seq_len, d_model)
out, aux_loss, stats = layer(x)

print(f"out shape : {out.shape}")     # torch.Size([2, 64, 128])
print(f"aux_loss  : {aux_loss:.4f}")
print(f"stats     : {stats}")
