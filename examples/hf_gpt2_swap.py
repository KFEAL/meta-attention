# Copyright 2024-2025 Alan Ferrari
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Swap GPT-2 attention layers with Meta-Attention (requires `transformers`)."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from meta_attention import MetaAttnConfig
from meta_attention.integrations.hf import collect_hf_aux_losses, patch_hf_model

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# GPT-2 base: hidden=768, heads=12
cfg = MetaAttnConfig.gpt2()
patch_hf_model(model, cfg, verbose=True)

# Forward pass — the patched model is fully functional.
inputs = tokenizer("Hello, Meta-Attention!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])

print(f"LM loss   : {outputs.loss:.4f}")
print(f"Aux loss  : {collect_hf_aux_losses(model):.4f}")
