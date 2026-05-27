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

"""Global registry for attention experts.

Third-party packages can register custom experts by name so they can be
instantiated from config strings without importing the concrete class.

Example — registering an expert in your package::

    # my_package/flash_expert.py
    from meta_attention.experts.registry import register_expert
    from meta_attention.experts.base import AttentionExpert

    @register_expert("flash_mla")
    class FlashMLAExpert(AttentionExpert):
        _cost = 0.4
        def forward(self, x, mask=None): ...

Example — building experts by name in a config file::

    from meta_attention.experts.registry import build_expert
    expert = build_expert("flash_mla", cfg)
"""

from __future__ import annotations

from typing import Dict, List, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import MetaAttnConfig
    from .base import AttentionExpert

_REGISTRY: Dict[str, Type["AttentionExpert"]] = {}


def register_expert(name: str):
    """Class decorator that registers an ``AttentionExpert`` under *name*.

    Parameters
    ----------
    name:
        Unique string key (e.g. ``"flash_mla"``, ``"sparse_bigbird"``).

    Raises
    ------
    KeyError
        If *name* is already registered.
    """
    def decorator(cls: Type["AttentionExpert"]) -> Type["AttentionExpert"]:
        if name in _REGISTRY:
            raise KeyError(
                f"Expert '{name}' is already registered as {_REGISTRY[name].__qualname__}. "
                "Use a different name or call unregister_expert() first."
            )
        _REGISTRY[name] = cls
        return cls
    return decorator


def unregister_expert(name: str) -> None:
    """Remove *name* from the registry (useful in tests)."""
    _REGISTRY.pop(name, None)


def build_expert(name: str, cfg: "MetaAttnConfig", **kwargs) -> "AttentionExpert":
    """Instantiate a registered expert by name.

    Parameters
    ----------
    name:
        Registered expert name.
    cfg:
        ``MetaAttnConfig`` passed as the first positional argument.
    **kwargs:
        Extra keyword arguments forwarded to the expert constructor.

    Raises
    ------
    KeyError
        If *name* is not found in the registry.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(
            f"Unknown expert '{name}'. Registered experts: {available}"
        )
    return _REGISTRY[name](cfg, **kwargs)


def list_experts() -> List[str]:
    """Return a sorted list of all registered expert names."""
    return sorted(_REGISTRY)


def get_expert_class(name: str) -> Type["AttentionExpert"]:
    """Return the class registered under *name* without instantiating it."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown expert '{name}'. Registered experts: {available}")
    return _REGISTRY[name]
