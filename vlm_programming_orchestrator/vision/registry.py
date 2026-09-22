"""Discovers vision/detector plugins (subfolders of this package) by VISION_BACKEND."""

import importlib
import os
import pkgutil

from .base import ObjectDetector

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
        vision_backend = getattr(module, "VISION_BACKEND", None)
        if vision_backend:
            _backends[vision_backend] = module.__name__


def get_detector_backend(name: str, api_key: str, **kwargs) -> ObjectDetector:
    """Instantiate the `ObjectDetector` registered for `name` (e.g. "gemini_realsense")."""
    _discover()
    module_name = _backends.get(name)
    if module_name is None:
        available = ", ".join(sorted(_backends)) or "(none found)"
        raise ValueError(
            f"No vision backend registered for vision_backend={name!r}. "
            f"Available: {available}"
        )
    module = importlib.import_module(module_name)
    return module.create_backend(api_key, **kwargs)
