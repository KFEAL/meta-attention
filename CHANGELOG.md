# Changelog

All notable changes to this project will be documented in this file.

---

## [0.2.0] — 2025-04-10

Initial public release under the Apache License 2.0.

### Added

**Core**
- `MetaAttnConfig` dataclass with all configuration fields, serialisation (`to_json` / `from_json` / `to_dict` / `from_dict`), and six model-size presets (`small`, `base`, `large`, `gpt2`, `gpt2_medium`, `llama_7b`).
- `MetaLMConfig` for the standalone language model.
- `MetaController` — lightweight MLP routing from token features `[x, ‖x‖/√D, pos]` to softmax weights.
- `MetaAttentionLayer` — core drop-in attention replacement; supports soft routing and Gumbel-softmax hard routing.
- `MetaTransformerBlock` — Pre-LN transformer block wrapping `MetaAttentionLayer`.
- `MetaLanguageModel` — standalone GPT-style autoregressive LM with `forward`, `compute_loss`, `generate`, and `param_count`.

**Experts**
- `FullAttention` (E1, cost 1.0) — standard SDPA, FlashAttention-compatible.
- `LinearAttention` (E2, cost 0.15) — Performer ELU+1 feature map; causal variant.
- `LocalAttention` (E3, cost 0.30) — sliding-window attention.
- `AttentionExpert` abstract base class.
- `ExpertRegistry` — `@register_expert` decorator, `build_expert`, `list_experts`, `get_expert_class`, `unregister_expert`.

**Backends**
- `TorchSDPABackend` (`"torch_sdpa"`, default) — `F.scaled_dot_product_attention`.
- `XFormersBackend` (`"xformers"`) — `xformers.ops.memory_efficient_attention`.
- `FlashAttnBackend` (`"flash_attn"`) — `flash_attn.flash_attn_func`.
- `AttentionBackend` ABC; `get_backend`, `register_backend`.

**Integrations**
- `integrations.generic`: `MetaAttentionWrapper`, `patch_module`, `collect_aux_losses`.
- `integrations.hf`: `MetaHFAttention`, `patch_hf_model`, `collect_hf_aux_losses`, `get_hf_routing_stats`. Supports GPT-2, LLaMA, Mistral, Phi, Falcon.

**Utilities**
- `RoutingStats` dataclass with `weights`, `entropy`, `expected_cost`, `n_collapsed`.
- `compute_routing_stats`, `cosine_ramp`, `linear_ramp`, `gumbel_softmax_top_k`.
- Mask builders: `causal_mask`, `additive_causal_mask`, `hf_attention_mask_to_additive`.

**Examples**
- `examples/minimal.py` — standalone layer.
- `examples/hf_gpt2_swap.py` — GPT-2 patching.
- `examples/generic_transformer.py` — PyTorch TransformerEncoder patching.
- `examples/custom_expert.py` — custom expert registration.

**Tests**
- `tests/test_config.py`, `test_experts.py`, `test_layer.py`, `test_model.py`, `test_integrations.py`, `test_utils.py`.

**Documentation**
- `README.md` with quick-start guide and paper citation.
- `docs/architecture.md` — Meta-Attention mechanism and paper reference.
- `docs/configuration.md` — `MetaAttnConfig` and `MetaLMConfig` reference.
- `docs/experts.md` — built-in experts and custom expert guide.
- `docs/backends.md` — compute backend guide.
- `docs/integrations.md` — PyTorch and HuggingFace integration guide.
- `docs/training.md` — auxiliary losses, routing collapse, scheduling, hard routing.
