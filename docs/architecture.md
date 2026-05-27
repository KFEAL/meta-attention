# Architecture

This document describes the Meta-Attention mechanism as presented in the companion paper:

> **Meta-Attention: Adaptive Attention Routing for Efficient Transformer Inference**  
> Alan Ferrari · K-Lab, Zürich · NeurIPS 2025 preprint  
> [`paper/meta-attention.pdf`](../paper/meta-attention.pdf)

---

## Motivation

Standard transformer architectures apply a single attention mechanism uniformly across all tokens and sequence positions, regardless of local context or computational budget. This leads to two failure modes:

- **Sliding-window models** (Longformer, local attention) save quadratic cost globally but sacrifice precise long-range dependencies *everywhere* — even at positions where they are essential (e.g. coreference resolution, cross-section citations).
- **Full attention** squanders compute on tokens whose context is trivially local.

Meta-Attention reframes the problem: rather than asking *how* to approximate attention, it asks **when exact attention is necessary**. This shift — from a fixed algorithmic commitment to a dynamic resource allocation problem — is the central conceptual contribution of the paper.

---

## Components

A Meta-Attention layer has three components that replace the standard attention sub-layer in any transformer block:

```
Input x (B, T, D)
       │
       ├──────────────────────────┐
       ▼                          ▼
 Meta-Controller              Attention Experts
 (routing MLP)                E1, E2, E3, ...
       │                          │
       └──────── α ───────────────┘
                    │
             Weighted merge
         Output = Σ αᵢ(x) · Eᵢ(x)
```

### 1. Attention Experts

Each expert maps `(B, T, D) → (B, T, D)` using a different algorithm. Three experts are defined in the Phase 1 prototype:

| Expert | Registered name | Algorithm | Complexity | Cost |
|---|---|---|---|---|
| `FullAttention` (E1) | `"full"` | Standard multi-head scaled dot-product attention (softmax) | O(T² · D) | 1.00 |
| `LinearAttention` (E2) | `"linear"` | Performer-style kernel approximation, feature map φ(x) = ELU(x) + 1 | O(T · D) | 0.15 |
| `LocalAttention` (E3) | `"local"` | Sliding-window attention over `[i−w, i+w]` | O(T · w · D) | 0.30 |

**Design intent** (from the paper, §3.2):
- **E1** handles tokens requiring precise long-range dependencies.
- **E2** handles long-range but low-precision context (background information).
- **E3** handles locally coherent text, structured sequences, and repetitive regions.

The expert set is extensible — see [`docs/experts.md`](experts.md) for the custom expert API.

### 2. Meta-Controller

The Meta-Controller is a lightweight MLP that produces per-token routing weights *before* any attention is run:

```
M(x) = softmax( W₂ · GELU(W₁ · [x ; ‖x‖/√D ; pos]) / τ )
```

where:
- `x ∈ ℝ^(B×T×D)` — token embeddings (the layer input)
- `‖x‖/√D` — token salience proxy (1 dim per token, detached from the graph)
- `pos ∈ [0, 1]` — normalised position index (1 dim per token)
- `τ` — routing temperature (lower = sharper/more committed routing)

**Why salience and position, not attention entropy?**  
Using attention entropy as a routing signal creates a circular dependency: you would need to run attention before deciding which attention to run. Token embedding norm `‖x‖/√D` correlates empirically with attention entropy ([Clark et al., 2019](https://arxiv.org/abs/1906.04341)) and is available without any attention computation.

The controller MLP architecture:

```
Linear(D+2 → H) → GELU → Dropout → Linear(H → H//2) → GELU → Linear(H//2 → K) → Softmax(τ)
```

where `H = cfg.controller_hidden` (default 128) and `K = number of experts`.

### 3. Routing and Merge

#### Soft routing (default, `cfg.hard_routing=False`)

```
Output = Σᵢ αᵢ(x) · Eᵢ(x)
```

All experts run on every token. The output is a weighted sum. This mode is:
- Fully differentiable through both routing weights and expert outputs.
- Stable to train.
- The primary mode for Phase 1 and Phase 2 of the experimental roadmap.

#### Hard routing (`cfg.hard_routing=True`)

Only the top-`k` experts are executed per token (`cfg.hard_top_k`, default 1):

- **Training**: Gumbel-softmax top-k with straight-through gradient estimator.
- **Inference**: argmax — only the winning expert runs, delivering real FLOP savings.

Hard routing requires load-balancing regularisation to prevent collapse (see [`docs/training.md`](training.md)). It is Phase 3 work in the experimental roadmap.

---

## Training Objective

```
L = L_task + λ · E[cost] − entropy_coeff · H[α]
```

- **L_task**: Standard task loss (cross-entropy for language modelling, etc.).
- **λ · E[cost]**: Compute-cost regulariser — penalises unnecessary use of expensive experts. `λ = cfg.cost_lambda`.
- **−entropy_coeff · H[α]**: Entropy bonus — encourages balanced routing across experts. `entropy_coeff = cfg.entropy_coeff`.

Both coefficients default to `0.0`. The cost regulariser should be warmed up gradually to avoid destabilising the task loss early in training (see [`docs/training.md`](training.md)).

The expected cost term is differentiable because `αᵢ` is a differentiable function of `x` under soft routing:

```python
expected_cost = (alpha * costs).sum(dim=-1).mean()   # alpha: (B, T, K), costs: (K,)
aux_loss = cost_lambda * expected_cost - entropy_coeff * entropy
```

---

## Relationship to Related Work

The paper (§4–5) positions Meta-Attention relative to two concurrent orthogonal works:

| Dimension | Mixture of Depths (MoD) | Meta-Attention | Attention Residuals (AttnRes) |
|---|---|---|---|
| **What varies** | Layer participation (depth axis) | Attention algorithm (mechanism axis) | Residual aggregation source (depth axis) |
| **Routing granularity** | Per-layer, per-token (binary skip) | Per-layer, per-token (soft/hard) | Per-layer (pseudo-query, input-independent) |
| **Long-context cost** | Not addressed — survivors use full attention | Addressed via linear + local branches | Not addressed |
| **FLOP savings** | Exact, predictable (skip = zero cost) | Expected cost via regularisation | Marginal overhead over residuals |
| **Training stability** | Validated at 1B+ scale | Soft routing stable; hard routing: open | Validated at 48B / 1.4T tokens |

**Composability**: The three approaches operate on orthogonal axes and can be combined in a single architecture without interaction:

1. **MoD** decides *which tokens enter* the layer (depth axis).
2. **AttnRes** decides *which depth representations contribute* to the current hidden state (via a layer-level pseudo-query).
3. **Meta-Attention** decides *which attention algorithm* processes the resulting hidden state (mechanism axis).

All three are jointly differentiable under soft routing.

---

## Experimental Roadmap

From the paper (§6, Table 3):

| Phase | Target | Status |
|---|---|---|
| **Phase 1** | Forward-pass correctness; routing diversity at initialisation | ✅ Complete |
| **Phase 2** | Train on WikiText-103; soft routing distribution analysis | In progress |
| **Phase 2** | Routing collapse characterisation; load-balancing ablation | In progress |
| **Phase 3** | Hard routing via Gumbel-softmax; FLOP counting | Planned |
| **Phase 3** | Long-sequence benchmark (≥ 8k tokens) | Planned |
| **Phase 3** | SWE-bench agent evaluation | Planned |

**Falsifiability criterion** (from the paper): Under hard routing, Meta-Attention achieves ≥ 30% FLOP reduction on sequences of length ≥ 4k tokens without perplexity degradation exceeding 0.5 nats on WikiText-103.

---

## NeurIPS 2025 Context

Three concurrent NeurIPS 2025 works sharpen the research agenda (paper §2.3):

- **Gated Attention** (Qiu et al., NeurIPS 2025 Best Paper): Head-specific sigmoid gate after SDPA consistently improves performance and eliminates attention sinks. Natural candidate for an additional E4 expert in Phase 2.
- **Sparse Attention Emergence** (Zucchet et al., NeurIPS 2025 Oral): Routing sparsity is predicted to emerge suddenly (phase transition), not gradually. The practical implication: monitor routing entropy, not epoch count, as the signal to transition from soft to hard routing.
- **Polynomial-Time Learnability of Linear Attention** (Yau et al., NeurIPS 2025 Oral): E2 (linear attention) is provably polynomial-time PAC-learnable — providing theoretical backing for calibrating the cost regulariser in favour of E2 over E3 when both suffice.

---

## References

Full bibliography in [`old/arxiv_package/meta_attention.bib`](../old/arxiv_package/meta_attention.bib). Key citations:

- Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- Choromanski et al., "Rethinking Attention with Performers", ICLR 2021
- Dao et al., "FlashAttention", NeurIPS 2022
- Fedus et al., "Switch Transformers", JMLR 2022
- Raposo et al., "Mixture of Depths", arXiv 2024
- Clark et al., "What Does BERT Look At?", BlackboxNLP 2019
