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

"""Meta-Attention: adaptive per-token attention routing for efficient inference.

Public API
----------
**Core**::

    from meta_attention import MetaAttnConfig, MetaAttentionLayer
    cfg = MetaAttnConfig(d_model=512, n_heads=8)
    layer = MetaAttentionLayer(cfg)
    out, aux_loss, stats = layer(x)            # x: (B, T, D)

**Standalone model**::

    from meta_attention import MetaLanguageModel, MetaLMConfig
    lm_cfg = MetaLMConfig(attn=cfg, vocab_size=50257, n_layers=6)
    model = MetaLanguageModel(lm_cfg)
    logits, aux_loss, stats = model(token_ids)

**HuggingFace integration**::

    from meta_attention.integrations.hf import patch_hf_model, collect_hf_aux_losses
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
    patch_hf_model(hf_model, MetaAttnConfig.gpt2())

**Generic replacement** (any ``nn.Module``)::

    from meta_attention.integrations.generic import patch_module
    patch_module(my_model, target_class=nn.MultiheadAttention, cfg=cfg)

**Custom experts**::

    from meta_attention.experts import register_expert, AttentionExpert

    @register_expert("my_expert")
    class MyExpert(AttentionExpert):
        _cost = 0.5
        def forward(self, x, mask=None): ...
"""

from ._version import __version__
from .config import MetaAttnConfig, MetaLMConfig
from .controller import BayesianMetaController, MetaController
from .experts import (
    AttentionExpert,
    FullAttention,
    LinearAttention,
    LocalAttention,
    build_expert,
    list_experts,
    register_expert,
)
from .layer import MetaAttentionLayer
from .block import MetaTransformerBlock
from .model import MetaLanguageModel
from .utils import (
    RoutingStats,
    compute_routing_stats,
    dirichlet_entropy,
    dirichlet_kl,
    cosine_ramp,
    linear_ramp,
)

__all__ = [
    "__version__",
    # Config
    "MetaAttnConfig",
    "MetaLMConfig",
    # Experts
    "AttentionExpert",
    "FullAttention",
    "LinearAttention",
    "LocalAttention",
    "register_expert",
    "build_expert",
    "list_experts",
    # Core
    "BayesianMetaController",
    "MetaController",
    "MetaAttentionLayer",
    "MetaTransformerBlock",
    "MetaLanguageModel",
    # Utils
    "RoutingStats",
    "compute_routing_stats",
    "dirichlet_entropy",
    "dirichlet_kl",
    "cosine_ramp",
    "linear_ramp",
]
