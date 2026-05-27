# Training Guide

This guide covers training a model with Meta-Attention: auxiliary losses, routing diagnostics, avoiding routing collapse, and the transition from soft to hard routing.

---

## Auxiliary Loss

Meta-Attention adds a compute-aware auxiliary term to the task loss:

```
L = L_task + λ · E[cost] − entropy_coeff · H[α]
```

Both coefficients are set in `MetaAttnConfig`:

```python
cfg = MetaAttnConfig(
    d_model=512,
    n_heads=8,
    cost_lambda=0.01,       # penalise expensive experts
    entropy_coeff=0.001,    # encourage routing diversity
)
```

### cost_lambda

Penalises the expected compute cost per token:

```
λ · mean_token( Σᵢ αᵢ · costᵢ )
```

With `expert_costs=[1.0, 0.15, 0.30]`, pushing routing weight toward `LinearAttention` (E2) minimises this term. Set to `0.0` (default) to train without cost pressure.

**Recommended warm-up:** Start at `0.0` and ramp up after the task loss has stabilised. Introducing cost pressure too early can disrupt learning before routing has had a chance to become meaningful.

### entropy_coeff

Adds a bonus proportional to the Shannon entropy of the routing distribution:

```
entropy_coeff · mean_token( −Σᵢ αᵢ log αᵢ )
```

This discourages premature collapse to a single expert. Set to `0.0` (default) to disable.

### Collecting losses during training

**Standalone model:**

```python
logits, aux_loss, stats = model(idx)
lm_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
loss = lm_loss + aux_loss      # aux_loss already includes cost_lambda scaling
loss.backward()
```

**Patched model (generic):**

```python
from meta_attention.integrations.generic import collect_aux_losses

out = encoder(x)
task_loss = criterion(out, targets)
loss = task_loss + collect_aux_losses(encoder)
loss.backward()
```

**Patched HuggingFace model:**

```python
from meta_attention.integrations.hf import collect_hf_aux_losses

outputs = model(**batch)
loss = outputs.loss + collect_hf_aux_losses(model)
loss.backward()
```

---

## Scheduling `cost_lambda`

Use `cosine_ramp` or `linear_ramp` to warm up the cost penalty:

```python
from meta_attention import cosine_ramp

for step in range(total_steps):
    lam = cosine_ramp(
        step,
        warmup_steps=1000,   # keep λ=0 for first 1000 steps
        max_steps=10000,     # reach target λ at step 10000
        start=0.0,
        end=0.01,            # target cost_lambda
    )
    # Update config for all layers
    for block in model.blocks:
        block.attn.cfg.cost_lambda = lam
    
    # or pass it per-forward-pass (standalone model only):
    logits, aux_loss, stats = model(idx, cost_lambda=lam)
```

---

## Routing Temperature Annealing

Decrease the controller temperature over training to sharpen routing decisions:

```python
from meta_attention import cosine_ramp

# Anneal from τ=2.0 (diffuse) to τ=0.3 (sharp)
for step in range(total_steps):
    tau = cosine_ramp(step, warmup_steps=500, max_steps=5000, start=2.0, end=0.3)
    for block in model.blocks:
        block.attn.set_temperature(tau)
```

`set_temperature` updates both the controller and the config in-place.

---

## Routing Diagnostics with RoutingStats

Every forward pass returns `RoutingStats` (or a list of them for multi-layer models):

```python
logits, aux_loss, stats = model(idx)

for i, s in enumerate(stats):
    print(f"Layer {i}: {s}")
# Layer 0: RoutingStats(E1=0.341 | E2=0.381 | E3=0.278  entropy=1.094  cost=0.474  collapsed=0.00%)
```

| Field | Description | Healthy range |
|---|---|---|
| `weights` | Mean routing weight per expert, shape `(K,)` | No single expert dominates early in training |
| `entropy` | Mean routing entropy over all tokens (nats) | Should start near `log(K)` ≈ 1.1 (uniform) and decrease as routing specialises |
| `expected_cost` | Average normalised FLOP per token | Should decrease as `cost_lambda` pushes toward cheaper experts |
| `n_collapsed` | Fraction of tokens where one expert has weight > 0.9 | Should be low early; rising is normal as routing specialises |

**Routing collapse** is when `n_collapsed → 1.0` and `weights[0] → 1.0` (all tokens routed to E1 forever). This is the primary training failure mode — see the section below.

---

## Routing Collapse

Routing collapse is identified in the paper (§7) as the primary open challenge. Because the model can minimise task loss without committing to non-trivial routing (it can set `α₁ ≈ 1` everywhere and use full attention), the cost regulariser must be strong enough to break this equilibrium.

### Symptoms

```python
# Collapsed routing stats (bad):
# RoutingStats(E1=0.980 | E2=0.012 | E3=0.008  entropy=0.09  cost=0.99  collapsed=97.50%)
```

### Mitigations

1. **Warm up `cost_lambda`** — introduce cost pressure only after the task loss stabilises.

2. **Use `entropy_coeff`** — add an explicit entropy bonus to prevent premature collapse:
   ```python
   cfg.entropy_coeff = 0.005
   ```

3. **Warm up from a diffuse temperature** — start with `τ ≥ 1.5` so the softmax output is near-uniform early in training.

4. **Switch Transformer-style load-balancing loss** — an auxiliary term that penalises load imbalance across experts (from Fedus et al., 2022). Planned for Phase 2.

5. **Monitor entropy, not just loss** — routing collapse can occur without a visible spike in task loss. Track `stats.entropy` in your training loop.

### Predicting the phase transition

From the paper (§2.3), citing Zucchet et al. (2025): routing sparsity is predicted to emerge *suddenly* (a sharp phase transition) rather than gradually. The practical implication:

> The entropy drop (rather than a fixed epoch count) should be treated as the signal that the model has learned a committed routing policy.

A useful heuristic: if `stats.entropy` drops by more than 30% in a single epoch, routing is specialising rapidly — this is the right moment to:
- Reduce `entropy_coeff` to let specialisation proceed.
- Transition from soft to hard routing (see below).

---

## Hard Routing

Hard routing activates only the top-`k` experts per token, delivering real FLOP savings at inference time. This is Phase 3 of the experimental roadmap.

```python
cfg = MetaAttnConfig(
    d_model=512,
    n_heads=8,
    hard_routing=True,
    hard_top_k=1,         # 1 expert per token at inference
    gumbel_temp=1.0,
)
layer = MetaAttentionLayer(cfg)
```

### Training vs. inference behaviour

| Mode | Behaviour |
|---|---|
| `model.train()` | Gumbel-softmax top-k with straight-through gradient. All expert outputs are still computed (soft mix). |
| `model.eval()` | Argmax routing. Only the winning expert runs — real FLOP savings. |

### Recommended transition procedure

1. Train to convergence with soft routing (`cfg.hard_routing=False`).
2. Wait for routing entropy to stabilise (the phase transition mentioned above).
3. Switch to hard routing and fine-tune for a few thousand steps with a small LR.
4. Optionally anneal `gumbel_temp` from `1.0` down to `0.3` during fine-tuning.

---

## Experimental Roadmap

From the paper (§6, Table 3):

| Phase | Target | Status |
|---|---|---|
| **Phase 1** | Forward-pass correctness, routing diversity at init | ✅ Complete |
| **Phase 2** | Train on WikiText-103; soft routing distribution analysis | In progress |
| **Phase 2** | Routing collapse characterisation; load-balancing ablation | In progress |
| **Phase 3** | Hard routing via Gumbel-softmax; FLOP counting | Planned |
| **Phase 3** | Long-sequence benchmark (≥ 8k tokens) | Planned |
| **Phase 3** | SWE-bench agent evaluation | Planned |

**Falsifiability criterion**: Under hard routing, Meta-Attention achieves ≥ 30% FLOP reduction on sequences ≥ 4k tokens without perplexity degradation exceeding 0.5 nats on WikiText-103.
