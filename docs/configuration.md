# Configuration

All Meta-Attention behaviour is controlled through two dataclasses: `MetaAttnConfig` (per-layer) and `MetaLMConfig` (standalone language model).

```python
from meta_attention import MetaAttnConfig, MetaLMConfig
```

---

## MetaAttnConfig

Configuration for a single `MetaAttentionLayer`.

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `d_model` | `int` | `512` | Total model dimensionality. |
| `n_heads` | `int` | `8` | Number of attention heads. Must evenly divide `d_model`. |
| `window_size` | `int` | `64` | Half-window radius for `LocalAttention` — token `i` attends to `[i−w, i+w]`. |
| `dropout` | `float` | `0.1` | Dropout probability inside experts and the controller. |
| `temperature` | `float` | `1.0` | Softmax temperature for routing logits. Lower → sharper routing. |
| `controller_hidden` | `int` | `128` | Hidden width of the Meta-Controller MLP. |
| `expert_costs` | `List[float]` | `[1.0, 0.15, 0.30]` | Normalised compute cost per expert `[full, linear, local]`. Must match the number of experts used. |
| `cost_lambda` | `float` | `0.0` | Weight of the expected-compute-cost auxiliary loss. `0.0` disables it. |
| `entropy_coeff` | `float` | `0.0` | Weight of the routing-entropy bonus (encourages balanced routing). `0.0` disables it. |
| `use_flash` | `bool` | `True` | Use `F.scaled_dot_product_attention` (Flash-style) when PyTorch ≥ 2.0. Falls back automatically if unavailable. |
| `hard_routing` | `bool` | `False` | Use Gumbel-softmax (training) / argmax (inference) for sparse routing. See [training guide](training.md#hard-routing). |
| `hard_top_k` | `int` | `1` | Number of experts to activate per token under hard routing. |
| `gumbel_temp` | `float` | `1.0` | Temperature for Gumbel-softmax during training. |
| `backend` | `str` | `"torch_sdpa"` | Compute backend for built-in experts. One of `"torch_sdpa"`, `"xformers"`, `"flash_attn"`. See [backends guide](backends.md). |

### Derived properties

| Property | Type | Description |
|---|---|---|
| `d_head` | `int` | `d_model // n_heads`. Raises `ValueError` if not evenly divisible. |
| `n_experts` | `int` | `len(expert_costs)`. |

### Presets

Pre-configured instances for common model sizes. All presets are classmethods:

```python
cfg = MetaAttnConfig.small()    # 128-d, 4-head
```

| Preset | `d_model` | `n_heads` | `d_head` | `window_size` | `controller_hidden` | Use for |
|---|---|---|---|---|---|---|
| `small()` | 128 | 4 | 32 | 32 | 64 | Unit tests, ablations |
| `base()` | 512 | 8 | 64 | 64 | 128 | Paper Phase 2 training config |
| `large()` | 1024 | 16 | 64 | 128 | 256 | GPT-2 Large / LLaMA-7B |
| `gpt2()` | 768 | 12 | 64 | 64 | 192 | GPT-2 base |
| `gpt2_medium()` | 1024 | 16 | 64 | 128 | 256 | GPT-2 Medium |
| `llama_7b()` | 4096 | 32 | 128 | 256 | 512 | LLaMA-7B |

### Serialisation

Configs can be round-tripped through JSON or plain dicts:

```python
cfg = MetaAttnConfig.gpt2()

# To/from JSON string
json_str = cfg.to_json()
cfg2 = MetaAttnConfig.from_json(json_str)

# To/from dict
d = cfg.to_dict()
cfg3 = MetaAttnConfig.from_dict(d)

assert cfg == cfg2 == cfg3
```

### Common patterns

**Enabling auxiliary losses:**

```python
cfg = MetaAttnConfig(
    d_model=512,
    n_heads=8,
    cost_lambda=0.01,      # penalise expensive experts
    entropy_coeff=0.001,   # encourage routing diversity
)
```

**Enabling hard routing:**

```python
cfg = MetaAttnConfig(
    d_model=512,
    n_heads=8,
    hard_routing=True,
    hard_top_k=1,
    gumbel_temp=1.0,
)
```

**Selecting a backend:**

```python
cfg = MetaAttnConfig(d_model=512, n_heads=8, backend="xformers")
```

See [`docs/backends.md`](backends.md) for backend requirements and constraints.

---

## MetaLMConfig

Configuration for the standalone `MetaLanguageModel`. Wraps a `MetaAttnConfig` with language-model-specific settings.

```python
from meta_attention import MetaLMConfig, MetaAttnConfig, MetaLanguageModel

lm_cfg = MetaLMConfig(
    attn=MetaAttnConfig.small(),
    vocab_size=50257,
    n_layers=6,
)
model = MetaLanguageModel(lm_cfg)
```

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `attn` | `MetaAttnConfig` | `MetaAttnConfig()` | Attention layer configuration. |
| `vocab_size` | `int` | `50257` | Vocabulary size (GPT-2 tokeniser default). |
| `n_layers` | `int` | `6` | Number of `MetaTransformerBlock` layers. |
| `max_seq_len` | `int` | `1024` | Maximum sequence length (positional embedding size). |
| `ffn_multiplier` | `int` | `4` | FFN hidden width as a multiple of `d_model`. |
| `tie_weights` | `bool` | `True` | Tie token embedding and output projection weights. |

### Example: Small model for testing

```python
from meta_attention import MetaAttnConfig, MetaLMConfig, MetaLanguageModel
import torch

lm_cfg = MetaLMConfig(
    attn=MetaAttnConfig.small(),   # 128-d, 4-head
    vocab_size=500,
    n_layers=2,
    max_seq_len=64,
)
model = MetaLanguageModel(lm_cfg)

idx = torch.randint(0, 500, (2, 32))
logits, aux_loss, stats = model(idx)
# logits: (2, 32, 500)
# stats: list of RoutingStats, one per layer
```
