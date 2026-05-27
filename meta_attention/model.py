# Copyright 2024-2025 Alan Ferrari
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone GPT-style language model built entirely from Meta-Attention blocks."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import MetaTransformerBlock
from .config import MetaAttnConfig, MetaLMConfig
from .utils import RoutingStats, additive_causal_mask


class MetaLanguageModel(nn.Module):
    """GPT-style autoregressive language model using Meta-Attention blocks.

    For using Meta-Attention inside an *existing* model (GPT-2, LLaMA, etc.)
    see ``meta_attention.integrations``.

    Example
    -------
    >>> cfg = MetaLMConfig(attn=MetaAttnConfig.small(), vocab_size=1000, n_layers=2)
    >>> model = MetaLanguageModel(cfg)
    >>> idx = torch.randint(0, 1000, (2, 32))
    >>> logits, aux_loss, stats = model(idx)
    >>> logits.shape
    torch.Size([2, 32, 1000])
    """

    def __init__(self, lm_cfg: MetaLMConfig) -> None:
        super().__init__()
        self.lm_cfg = lm_cfg
        cfg = lm_cfg.attn

        self.tok_emb = nn.Embedding(lm_cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(lm_cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [MetaTransformerBlock(cfg, lm_cfg.ffn_multiplier) for _ in range(lm_cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, lm_cfg.vocab_size, bias=False)

        if lm_cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        idx: torch.LongTensor,
        cost_lambda: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[RoutingStats]]:
        """
        Parameters
        ----------
        idx : (B, T)
        cost_lambda:
            Override ``cfg.cost_lambda`` for this forward pass only.

        Returns
        -------
        logits : (B, T, vocab_size)
        aux_loss : scalar tensor — sum of per-layer auxiliary losses
        all_stats : List[RoutingStats], one per layer
        """
        B, T = idx.shape
        device = idx.device

        if cost_lambda is not None:
            for block in self.blocks:
                block.attn.cfg.cost_lambda = cost_lambda

        mask = additive_causal_mask(T, device, dtype=self._dtype())
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        total_aux = x.new_zeros(())
        all_stats: List[RoutingStats] = []

        for block in self.blocks:
            x, aux_loss, stats = block(x, mask)
            total_aux = total_aux + aux_loss
            all_stats.append(stats)

        logits = self.head(self.ln_f(x))
        return logits, total_aux, all_stats

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        idx: torch.LongTensor,
        cost_lambda: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Cross-entropy LM loss + auxiliary loss.

        Returns
        -------
        loss : scalar tensor
        metrics : dict with keys ``lm_loss``, ``aux_loss``, ``ppl``
        """
        logits, aux_loss, _ = self.forward(idx[:, :-1], cost_lambda=cost_lambda)
        targets = idx[:, 1:]
        lm_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        loss = lm_loss + aux_loss
        metrics = {
            "lm_loss": lm_loss.item(),
            "aux_loss": aux_loss.item(),
            "ppl": lm_loss.exp().item(),
        }
        return loss, metrics

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompt: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.LongTensor:
        """Simple greedy / top-k autoregressive generation."""
        x = prompt
        for _ in range(max_new_tokens):
            x_in = x[:, -self.lm_cfg.max_seq_len :]
            logits, _, _ = self.forward(x_in)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                v, _ = logits.topk(top_k, dim=-1)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_tok], dim=1)
        return x

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def param_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        embedding = sum(p.numel() for p in self.tok_emb.parameters()) + sum(
            p.numel() for p in self.pos_emb.parameters()
        )
        attn = sum(
            p.numel()
            for block in self.blocks
            for p in list(block.attn.parameters()) + list(block.ln_attn.parameters())
        )
        ffn = sum(
            p.numel()
            for block in self.blocks
            for p in list(block.ffn.parameters()) + list(block.ln_ffn.parameters())
        ) + sum(p.numel() for p in self.ln_f.parameters())
        if not self.lm_cfg.tie_weights:
            ffn += sum(p.numel() for p in self.head.parameters())
        return {"total": total, "embedding": embedding, "attention": attn, "ffn": ffn}

    def _dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
