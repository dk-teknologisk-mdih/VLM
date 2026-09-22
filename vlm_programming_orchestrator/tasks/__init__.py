"""Task plugins: what the VLM detects and what plan/prompt is built from it.

Mirrors the `robots/` / `llm/` / `vision/` plugin pattern — see `base.py` for
the `TaskBackend` protocol and `registry.py` for discovery.
"""

from .base import TaskBackend
from .registry import get_task_backend

__all__ = ["TaskBackend", "get_task_backend"]
