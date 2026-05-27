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

"""Pluggable attention compute backends.

Use :func:`get_backend` to retrieve the backend selected by ``cfg.backend``::

    from meta_attention.backends import get_backend
    backend = get_backend("torch_sdpa")
    out = backend.scaled_dot_product_attention(q, k, v)
"""

from __future__ import annotations

from .base import AttentionBackend
from .torch_sdpa import TorchSDPABackend
from .xformers import XFormersBackend
from .flash_attn import FlashAttnBackend

_BACKENDS: dict[str, AttentionBackend] = {
    "torch_sdpa": TorchSDPABackend(),
    "xformers": XFormersBackend(),
    "flash_attn": FlashAttnBackend(),
}


def get_backend(name: str) -> AttentionBackend:
    """Return the backend instance for *name*.

    Parameters
    ----------
    name:
        One of ``"torch_sdpa"``, ``"xformers"``, ``"flash_attn"``.

    Raises
    ------
    KeyError
        If *name* is not recognised.
    RuntimeError
        If the selected backend's required library is not installed.
    """
    if name not in _BACKENDS:
        raise KeyError(
            f"Unknown backend '{name}'.  Available: {list(_BACKENDS)}"
        )
    backend = _BACKENDS[name]
    if not backend.is_available:
        raise RuntimeError(
            f"Backend '{name}' is not available.  "
            "Install the required package (see backend module docstring)."
        )
    return backend


def register_backend(name: str, backend: AttentionBackend) -> None:
    """Register a custom backend under *name*.

    Useful for third-party backends (e.g. custom CUDA kernels) that are
    not bundled with this library.
    """
    _BACKENDS[name] = backend


__all__ = [
    "AttentionBackend",
    "TorchSDPABackend",
    "XFormersBackend",
    "FlashAttnBackend",
    "get_backend",
    "register_backend",
]
