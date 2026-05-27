# Attention Experts

An **attention expert** is any module that maps `(B, T, D) → (B, T, D)` and exposes a normalised compute cost. The three built-in experts cover the spectrum from precise-but-expensive to approximate-but-cheap.

```python
from meta_attention.experts import FullAttention, LinearAttention, LocalAttention
```

---

## Built-in Experts

| Name | Class | Registered as | Complexity | Cost | Description |
|---|---|---|---|---|---|
| Full softmax | `FullAttention` | `"full"` | O(T² · D) | 1.00 | Standard SDPA; reference expert |
| Linear kernel | `LinearAttention` | `"linear"` | O(T · D) | 0.15 | Performer ELU+1 feature map |
| Local window | `LocalAttention` | `"local"` | O(T · w · D) | 0.30 | Sliding-window attention |

Costs are normalised relative to full attention (1.0) and used by the auxiliary loss.

### FullAttention

Standard multi-head scaled dot-product attention with softmax normalisation.

```python
from meta_attention.experts import FullAttention
from meta_attention import MetaAttnConfig

cfg = MetaAttnConfig(d_model=512, n_heads=8)
expert = FullAttention(cfg)
out = expert(x)           # x: (B, T, D)
```

When `cfg.use_flash=True` and PyTorch ≥ 2.0, uses `F.scaled_dot_product_attention`, which automatically dispatches to FlashAttention on CUDA with fp16/bf16. Falls back to a manual matmul implementation otherwise.

### LinearAttention

Performer-style kernel attention using the ELU+1 feature map:

```
Attn(Q, K, V) ≈ φ(Q) · (φ(K)ᵀ V)    where φ(x) = ELU(x) + 1
```

This avoids materialising the T×T attention matrix — complexity is O(T · d_head · D).

```python
from meta_attention.experts import LinearAttention

# Non-causal (default) — for encoder/bidirectional models
expert = LinearAttention(cfg)

# Causal variant — required for autoregressive language modelling
expert = LinearAttention(cfg, causal=True)
```

**Note on the causal variant**: The causal form uses a sequential cumulative-sum loop over positions (O(T · d²) but serial). It is correct for LM use but slower than the non-causal form at long sequences. A parallelised CUDA implementation is a planned Phase 2 improvement.

**Note on `attn_mask`**: The linear attention kernel does not materialise a T×T matrix, so additive attention masks are not applicable and are silently ignored.

### LocalAttention

Sliding-window multi-head attention. Token `i` attends only to positions in `[i − w, i + w]`, where `w = cfg.window_size`.

```python
from meta_attention.experts import LocalAttention

expert = LocalAttention(cfg)   # window_size comes from cfg
```

**Performance note**: The current implementation uses a Python loop over positions, which is correct but not optimised for large sequences. For production use with T > 4k, replace with a chunked CUDA kernel (e.g. xFormers `local_attention`).

---

## The `AttentionExpert` ABC

All experts inherit from `AttentionExpert`:

```python
from meta_attention.experts import AttentionExpert
from typing import Optional
import torch

class AttentionExpert(torch.nn.Module):
    _cost: float = 1.0          # class-level default; override in subclass

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,                          # (B, T, D)
        mask: Optional[torch.Tensor] = None,      # (B, 1, T, T) additive, or None
    ) -> torch.Tensor:                             # (B, T, D)
        ...

    @property
    def cost(self) -> float:
        return self._cost
```

Contracts:
- Input and output shape are both `(B, T, D)`.
- `_cost` must be a float in `(0, 1]`, normalised relative to `FullAttention` (1.0).
- `mask` is an additive mask (0 = attend, −∞ = mask out); sub-quadratic experts may ignore it.

---

## Custom Experts and the Registry

Experts can be registered by name and later instantiated from config strings. This enables third-party packages to provide experts without modifying the library.

### Registering an expert

Use the `@register_expert` decorator on any `AttentionExpert` subclass:

```python
from typing import Optional
import torch, torch.nn as nn
from meta_attention import MetaAttnConfig, register_expert
from meta_attention.experts import AttentionExpert

@register_expert("gated_full")
class GatedFullAttention(AttentionExpert):
    """Full attention with a sigmoid gate (Gated Attention, Qiu et al. 2025)."""
    _cost = 1.05   # slightly above full attention due to gate overhead

    def __init__(self, cfg: MetaAttnConfig):
        super().__init__()
        self.attn = FullAttention(cfg)
        self.gate = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return torch.sigmoid(self.gate(x)) * self.attn(x, mask)
```

Attempting to register the same name twice raises `KeyError`.

### Building and listing experts

```python
from meta_attention import build_expert, list_experts

# All registered experts (built-ins + user-registered)
print(list_experts())   # ['full', 'gated_full', 'linear', 'local']

# Instantiate by name
cfg = MetaAttnConfig(d_model=512, n_heads=8)
expert = build_expert("gated_full", cfg)
```

### Using custom experts in a layer

```python
from meta_attention import MetaAttnConfig, MetaAttentionLayer, build_expert

cfg = MetaAttnConfig(
    d_model=256,
    n_heads=8,
    expert_costs=[1.0, 0.15, 1.05],   # must match the number of experts
)
layer = MetaAttentionLayer(cfg, experts=[
    build_expert("full",        cfg),
    build_expert("linear",      cfg),
    build_expert("gated_full",  cfg),
])
```

### Registry API reference

| Function | Description |
|---|---|
| `register_expert(name)` | Decorator. Registers a class under `name`. Raises `KeyError` if already registered. |
| `build_expert(name, cfg, **kwargs)` | Instantiates the expert registered under `name`. Passes `cfg` as first arg, `**kwargs` as extras. Raises `KeyError` if not found. |
| `list_experts()` | Returns a sorted list of all registered names. |
| `get_expert_class(name)` | Returns the class without instantiating it. Raises `KeyError` if not found. |
| `unregister_expert(name)` | Removes a name from the registry (useful in tests). No-op if not found. |

### Full example: Identity expert

See [`examples/custom_expert.py`](../examples/custom_expert.py) for a complete working example, including a minimal `IdentityAttention` expert and layer construction with `build_expert`.
