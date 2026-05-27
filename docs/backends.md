# Backends

A **backend** is a thin, stateless adapter that wraps a specific attention kernel behind a uniform SDPA-style interface. Built-in experts call the active backend instead of hard-coding kernel calls, making it straightforward to swap implementations at construction time via `cfg.backend`.

```python
from meta_attention import MetaAttnConfig

cfg = MetaAttnConfig(d_model=512, n_heads=8, backend="xformers")
```

---

## Available Backends

| Name | Class | Install | Constraints |
|---|---|---|---|
| `"torch_sdpa"` | `TorchSDPABackend` | Built-in (PyTorch ≥ 2.0) | None |
| `"xformers"` | `XFormersBackend` | `pip install xformers` | Best on CUDA; CPU supported |
| `"flash_attn"` | `FlashAttnBackend` | `pip install flash-attn` | CUDA only, fp16/bf16 only, no custom masks |

---

## torch_sdpa (default)

Wraps `torch.nn.functional.scaled_dot_product_attention`, available since PyTorch 2.0.

On CUDA with fp16/bf16 inputs, PyTorch automatically dispatches to FlashAttention when the cuDNN or SDPA Flash kernel is available. On CPU or with fp32, it falls back to a memory-efficient math implementation.

```python
cfg = MetaAttnConfig(d_model=512, n_heads=8, backend="torch_sdpa")  # default
```

**Compatibility**: Works with all expert types, all mask formats, causal and non-causal. No additional packages required.

---

## xformers

Wraps `xformers.ops.memory_efficient_attention`.

```bash
pip install xformers
# or
pip install "meta-attention[xformers]"
```

```python
cfg = MetaAttnConfig(d_model=512, n_heads=8, backend="xformers")
```

**When to prefer xformers**:
- Variable-length sequences (packing without padding).
- Cases where you want to stay on xFormers-maintained kernels without installing flash-attn.
- Custom attention bias patterns (xFormers has richer bias support than torch_sdpa).

---

## flash_attn

Wraps `flash_attn.flash_attn_func` (FlashAttention-2).

```bash
pip install flash-attn
# or
pip install "meta-attention[flash]"
```

```python
cfg = MetaAttnConfig(d_model=512, n_heads=8, backend="flash_attn")
```

> **Constraints — read carefully:**
> - **CUDA only**: Raises `RuntimeError` on CPU.
> - **fp16 or bf16 only**: fp32 inputs will raise an error inside `flash_attn`.
> - **No custom attention masks**: Arbitrary `attn_mask` tensors are not supported. If you pass a mask, it is silently ignored with a `UserWarning`. Use `is_causal=True` for causal masking instead.

**When to prefer flash_attn**:
- Maximum throughput on CUDA for full-attention workloads with fp16/bf16.
- Long-context training where memory efficiency is the bottleneck.

---

## Backend Interface

All backends implement `AttentionBackend`:

```python
from meta_attention.backends import AttentionBackend

class AttentionBackend:
    @property
    def name(self) -> str: ...           # e.g. "torch_sdpa"

    @property
    def is_available(self) -> bool: ...  # True if library is installed

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,                          # (B, H, T, d_head)
        k: torch.Tensor,                          # (B, H, T, d_head)
        v: torch.Tensor,                          # (B, H, T, d_head)
        attn_mask: Optional[torch.Tensor] = None, # (B, 1, T, T) additive or None
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,            # overrides 1/sqrt(d_head)
    ) -> torch.Tensor:                             # (B, H, T, d_head)
        ...
```

---

## Retrieving a Backend

```python
from meta_attention.backends import get_backend

backend = get_backend("torch_sdpa")
out = backend.scaled_dot_product_attention(q, k, v, is_causal=True)
```

`get_backend` raises:
- `KeyError` — if the name is not recognised.
- `RuntimeError` — if the backend's required library is not installed.

---

## Registering a Custom Backend

Third-party CUDA kernels or custom implementations can be registered:

```python
from meta_attention.backends import register_backend, AttentionBackend

class MyCudaKernel(AttentionBackend):
    @property
    def name(self): return "my_kernel"

    @property
    def is_available(self): return True

    def scaled_dot_product_attention(self, q, k, v, **kwargs):
        ...   # your kernel call

register_backend("my_kernel", MyCudaKernel())
```

After registration, pass `backend="my_kernel"` in `MetaAttnConfig` to use it.
