# Meta-Attention

[![CI](https://github.com/alanferrari/meta-attention/actions/workflows/ci.yml/badge.svg)](https://github.com/alanferrari/meta-attention/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/meta-attention)](https://pypi.org/project/meta-attention/)

**Per-token adaptive attention routing for efficient transformer inference.**

Meta-Attention replaces the standard attention sub-layer in any transformer block with a lightweight routing system that assigns each token to the most appropriate attention algorithm — full softmax, linear (kernel), or sliding-window local attention — based on per-token salience and position signals.

> "Rather than asking *how* to approximate attention globally and cheaply, Meta-Attention asks *when* exact attention is necessary."

The library is a direct accompaniment to the paper:

> **Meta-Attention: Adaptive Attention Routing for Efficient Transformer Inference**  
> Alan Ferrari · K-Lab, Zürich  
> [`paper/meta-attention.pdf`](paper/meta-attention.pdf)

---

## Installation

```bash
# PyPI (after release)
pip install meta-attention

# From source
git clone https://github.com/alanferrari/meta-attention.git
cd meta-attention
pip install -e .

# Optional extras
pip install "meta-attention[hf]"        # HuggingFace Transformers
pip install "meta-attention[xformers]"  # xFormers backend
pip install "meta-attention[flash]"    # FlashAttention-2 (CUDA, fp16/bf16)
pip install "meta-attention[all]"       # hf + xformers

# Development
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0

---

## Quick Start

### 1. Standalone attention layer

```python
import torch
from meta_attention import MetaAttnConfig, MetaAttentionLayer

cfg = MetaAttnConfig(d_model=512, n_heads=8)
layer = MetaAttentionLayer(cfg)

x = torch.randn(2, 64, 512)          # (batch, seq_len, d_model)
out, aux_loss, stats = layer(x)

print(out.shape)    # torch.Size([2, 64, 512])
print(stats)        # RoutingStats(E1=0.34 | E2=0.38 | E3=0.28  entropy=1.09  cost=0.47)
```

### 2. Drop-in replacement for `nn.MultiheadAttention`

```python
import torch.nn as nn
from meta_attention import MetaAttnConfig
from meta_attention.integrations.generic import patch_module, collect_aux_losses

encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True),
    num_layers=4,
)
patch_module(encoder, nn.MultiheadAttention, MetaAttnConfig(d_model=256, n_heads=8))

x = torch.randn(2, 32, 256)
out = encoder(x)
aux = collect_aux_losses(encoder)     # add to your task loss
```

### 3. HuggingFace model patching

```python
from transformers import AutoModelForCausalLM
from meta_attention import MetaAttnConfig
from meta_attention.integrations.hf import patch_hf_model, collect_hf_aux_losses

model = AutoModelForCausalLM.from_pretrained("gpt2")
patch_hf_model(model, MetaAttnConfig.gpt2())   # auto-detects architecture

outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss + collect_hf_aux_losses(model)
```

Supported: GPT-2, LLaMA, Mistral, Phi, Falcon (generic fallback for others).

### 4. Custom attention expert

```python
from typing import Optional
import torch, torch.nn as nn
from meta_attention import MetaAttnConfig, MetaAttentionLayer, register_expert
from meta_attention.experts import AttentionExpert

@register_expert("my_expert")
class MyExpert(AttentionExpert):
    _cost = 0.5   # normalised cost relative to full attention (1.0)

    def __init__(self, cfg: MetaAttnConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return self.proj(x)

# Use it alongside the built-ins
from meta_attention import build_expert
cfg = MetaAttnConfig(d_model=256, n_heads=8, expert_costs=[1.0, 0.15, 0.5])
layer = MetaAttentionLayer(cfg, experts=[
    build_expert("full",      cfg),
    build_expert("linear",    cfg),
    build_expert("my_expert", cfg),
])
```

---

## Architecture at a glance

```
Input x (B, T, D)
       │
       ├──────────────────────────┐
       │                          │
       ▼                          ▼
 Meta-Controller MLP        3 × Attention Experts
 [x, ‖x‖/√D, pos] → α      E1: Full softmax  (cost 1.00)
                             E2: Linear ELU+1  (cost 0.15)
                             E3: Local window  (cost 0.30)
       │                          │
       └──────────────────────────┘
                    │
             Weighted merge
         Output = Σ αᵢ(x) · Eᵢ(x)
                    │
              (B, T, D) output
```

The routing is fully differentiable (soft mode) and requires no attention outputs as input, avoiding circular dependencies. See [`docs/architecture.md`](docs/architecture.md) for the full technical description.

---

## Documentation

| Document | Description |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | How Meta-Attention works; relation to the paper |
| [`docs/configuration.md`](docs/configuration.md) | All `MetaAttnConfig` fields and presets |
| [`docs/experts.md`](docs/experts.md) | Built-in experts and custom expert guide |
| [`docs/backends.md`](docs/backends.md) | Compute backends (torch SDPA, xFormers, FlashAttention-2) |
| [`docs/integrations.md`](docs/integrations.md) | PyTorch generic and HuggingFace integration |
| [`docs/training.md`](docs/training.md) | Training guide: auxiliary losses, routing collapse, scheduling |

---

## Paper

The companion paper describes the full motivation, architecture, training objective, relationship to Mixture of Depths and Attention Residuals, and an experimental roadmap.

```bibtex
@article{ferrari2025metaattention,
  title   = {Meta-Attention: Adaptive Attention Routing for Efficient Transformer Inference},
  author  = {Ferrari, Alan},
  journal = {arXiv preprint},
  year    = {2025},
  note    = { 2025 preprint. K-Lab, Z{\"u}rich, Switzerland.}
}
```

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Please read [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Publishing / maintainers

This folder is the public GitHub export. See [`PUBLISHING.md`](PUBLISHING.md) for push, release, and PyPI steps.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).  
Copyright 2024–2025 Knoweldge Lab AG, Zurich, Switzerland.
