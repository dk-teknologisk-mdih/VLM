"""Swappable vision/object-detection backends.

Each subfolder (e.g. `gemini_realsense/`) is a plugin exposing
`VISION_BACKEND` and `create_backend(api_key, **kwargs)` in its `backend`
module; see `registry.py`.
"""

from .base import ObjectDetector
from .registry import get_detector_backend

__all__ = ["ObjectDetector", "get_detector_backend"]
