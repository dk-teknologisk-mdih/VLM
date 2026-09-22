"""Discovers LLM code-gen plugins (subfolders of this package) by LLM_BACKEND."""

import importlib
import os
import pkgutil

from .base import CodeGenBackend

_PKG_DIR = os.path.dirname(__file__)
_PKG_NAME = __name__.rsplit(".", 1)[0]

_backends: dict[str, str] = {}


def _discover() -> None:
    if _backends:
        return
    for _finder, name, is_pkg in pkgutil.iter_modules([_PKG_DIR]):
        if not is_pkg:
            continue
        try:
            module = importlib.import_module(f"{_PKG_NAME}.{name}.backend")
        except ImportError:
            continue
        llm_backend = getattr(module, "LLM_BACKEND", None)
        if llm_backend:
            _backends[llm_backend] = module.__name__


def get_codegen_backend(name: str, config) -> CodeGenBackend:
    """Instantiate the `CodeGenBackend` registered for `name` (e.g. "claude_proxy")."""
    _discover()
    module_name = _backends.get(name)
    if module_name is None:
        available = ", ".join(sorted(_backends)) or "(none found)"
        raise ValueError(
            f"No LLM backend registered for llm_backend={name!r}. "
            f"Available: {available}"
        )
    module = importlib.import_module(module_name)
    return module.create_backend(config)
