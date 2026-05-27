# Integrations

Meta-Attention provides two integration paths:

1. **Generic** — drop-in replacement for `nn.MultiheadAttention` in any PyTorch model.
2. **HuggingFace** — surgical replacement of attention layers in GPT-2, LLaMA, Mistral, Phi, and Falcon.

---

## Generic Integration

Suitable for any model built with `nn.TransformerEncoder`, `nn.TransformerDecoder`, or any custom module that contains `nn.MultiheadAttention` layers.

```python
from meta_attention.integrations.generic import (
    patch_module,
    collect_aux_losses,
    MetaAttentionWrapper,
)
```

### patch_module

Recursively scans an `nn.Module` and replaces every instance of a target class with a `MetaAttentionWrapper`. Operates **in-place** and returns the model for chaining.

```python
import torch.nn as nn
from meta_attention import MetaAttnConfig
from meta_attention.integrations.generic import patch_module, collect_aux_losses

encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True),
    num_layers=4,
)

cfg = MetaAttnConfig(d_model=256, n_heads=8)
patch_module(encoder, nn.MultiheadAttention, cfg, verbose=True)

x = torch.randn(2, 32, 256)
out = encoder(x)

# Collect and add auxiliary loss
aux = collect_aux_losses(encoder)
loss = my_task_loss(out) + aux
```

**Signature:**

```python
def patch_module(
    model: nn.Module,
    target_class: Type[nn.Module],           # class to replace, e.g. nn.MultiheadAttention
    cfg: MetaAttnConfig,
    pass_mask_kwarg: Optional[str] = "attn_mask",  # mask kwarg name in the host architecture
    verbose: bool = True,
) -> nn.Module
```

**Important:** `cfg.d_model` and `cfg.n_heads` must match the model's embedding dimension and number of heads.

### collect_aux_losses

Sums `last_aux_loss` from every `MetaAttentionWrapper` in the model after a forward pass:

```python
out = encoder(x)
aux = collect_aux_losses(encoder)
loss = task_loss + aux
loss.backward()
```

This must be called after each forward pass — `last_aux_loss` is overwritten on every call.

### MetaAttentionWrapper

The wrapper that `patch_module` installs. It mirrors the `nn.MultiheadAttention` call signature — `forward(query, key, value, ...)` — and returns `(output, None)`, so it is transparent to PyTorch internals.

```python
wrapper = MetaAttentionWrapper(
    cfg,
    batch_first=True,                  # match the surrounding TransformerEncoderLayer
    pass_mask_kwarg="attn_mask",       # which kwarg contains the attention mask
)
out, _ = wrapper(x)
```

After each forward:
- `wrapper.last_aux_loss` — auxiliary loss scalar.
- `wrapper.last_stats` — `RoutingStats` object.

---

## HuggingFace Integration

Patches attention layers in models loaded via the `transformers` library.

```bash
pip install "meta-attention[hf]"
# or: pip install transformers>=4.35.0
```

```python
from meta_attention.integrations.hf import (
    patch_hf_model,
    collect_hf_aux_losses,
    get_hf_routing_stats,
    MetaHFAttention,
)
```

### Supported architectures

| Architecture | Patched class(es) | `hf_model_type` |
|---|---|---|
| GPT-2 | `GPT2Attention` | `"gpt2"` |
| LLaMA | `LlamaAttention`, `LlamaSdpaAttention`, `LlamaFlashAttention2` | `"llama"` |
| Mistral | `MistralAttention`, `MistralSdpaAttention` | `"mistral"` |
| Phi | `PhiAttention` | `"phi"` |
| Falcon, others | `nn.MultiheadAttention` (generic scan) | `"generic"` |

The architecture is auto-detected from the model class name.

### patch_hf_model

```python
from transformers import AutoModelForCausalLM
from meta_attention import MetaAttnConfig
from meta_attention.integrations.hf import patch_hf_model, collect_hf_aux_losses

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Use the matching preset
cfg = MetaAttnConfig.gpt2()   # d_model=768, n_heads=12

patch_hf_model(model, cfg, verbose=True)

# The model is still fully functional
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])

# Add auxiliary loss during training
loss = outputs.loss + collect_hf_aux_losses(model)
loss.backward()
```

**Signature:**

```python
def patch_hf_model(
    model: nn.Module,
    cfg: MetaAttnConfig,
    verbose: bool = True,
) -> nn.Module
```

> **Important:** Always ensure `cfg.d_model` matches the model's `hidden_size` and `cfg.n_heads` matches `num_attention_heads`. Use the preset for known models:
>
> | Model | Preset |
> |---|---|
> | GPT-2 base | `MetaAttnConfig.gpt2()` (768-d, 12-head) |
> | GPT-2 Medium | `MetaAttnConfig.gpt2_medium()` (1024-d, 16-head) |
> | LLaMA-7B | `MetaAttnConfig.llama_7b()` (4096-d, 32-head) |

### collect_hf_aux_losses

```python
outputs = model(**batch)
loss = outputs.loss + collect_hf_aux_losses(model)
```

Sums `last_aux_loss` from all `MetaHFAttention` layers. Returns `torch.tensor(0.0)` if `cost_lambda == entropy_coeff == 0.0`.

### get_hf_routing_stats

```python
stats_per_layer = get_hf_routing_stats(model)
# returns List[Optional[RoutingStats]], one entry per MetaHFAttention layer
for i, s in enumerate(stats_per_layer):
    print(f"Layer {i}: {s}")
```

### MetaHFAttention

The wrapper installed by `patch_hf_model`. It handles the impedance mismatch between Meta-Attention's `(output, aux_loss, stats)` return and the HuggingFace attention tuple format:

- GPT-2: returns `(out, None)` → `(attn_output, present)`
- LLaMA/Mistral/Phi: returns `(out, None, past_key_value)` → `(attn_output, attn_weights, past_key_value)`
- Generic: returns `(out,)`

You generally do not need to instantiate `MetaHFAttention` directly — use `patch_hf_model` instead.

---

## Full Examples

- [`examples/generic_transformer.py`](../examples/generic_transformer.py) — patch a PyTorch `TransformerEncoder`
- [`examples/hf_gpt2_swap.py`](../examples/hf_gpt2_swap.py) — patch GPT-2 and run a forward pass
