"""Swappable LLM code-generation backends.

Each subfolder (e.g. `claude_proxy/`) is a plugin exposing `LLM_BACKEND` and
`create_backend(config)` in its `backend` module; see `registry.py`.
"""

from .base import CodeGenBackend
from .registry import get_codegen_backend

__all__ = ["CodeGenBackend", "get_codegen_backend"]
