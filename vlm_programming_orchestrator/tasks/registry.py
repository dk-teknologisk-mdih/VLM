"""Discovers task plugins (subfolders of this package) by TASK_TYPE."""

import importlib
import os
import pkgutil

from .base import TaskBackend

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
        task_type = getattr(module, "TASK_TYPE", None)
        if task_type:
            _backends[task_type] = module.__name__


def get_task_backend(task_type: str, config) -> TaskBackend:
    """Instantiate the `TaskBackend` registered for `task_type` (e.g. "stack_blocks")."""
    _discover()
    module_name = _backends.get(task_type)
    if module_name is None:
        available = ", ".join(sorted(_backends)) or "(none found)"
        raise ValueError(
            f"No task backend registered for task_type={task_type!r}. "
            f"Available: {available}"
        )
    module = importlib.import_module(module_name)
    return module.create_backend(config)
