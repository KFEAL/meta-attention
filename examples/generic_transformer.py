# Copyright 2024-2025 Alan Ferrari
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Replace nn.MultiheadAttention in a vanilla PyTorch TransformerEncoder."""

import torch
import torch.nn as nn

from meta_attention import MetaAttnConfig
from meta_attention.integrations.generic import collect_aux_losses, patch_module

D, H, LAYERS = 256, 8, 4

encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=D, nhead=H, batch_first=True),
    num_layers=LAYERS,
)

cfg = MetaAttnConfig(d_model=D, n_heads=H)
patch_module(encoder, nn.MultiheadAttention, cfg, verbose=True)

x = torch.randn(2, 32, D)
out = encoder(x)
aux = collect_aux_losses(encoder)

print(f"out: {out.shape}   aux_loss: {aux:.4f}")
