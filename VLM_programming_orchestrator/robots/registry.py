"""Discovers robot vendor plugins (subfolders of this package) by ROBOT_TYPE."""

import importlib
import os
import pkgutil

from .base import RobotBackend

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
        robot_type = getattr(module, "ROBOT_TYPE", None)
        if robot_type:
            _backends[robot_type] = module.__name__


def get_robot_backend(robot_type: str, config) -> RobotBackend:
    """Instantiate the `RobotBackend` registered for `robot_type` (e.g. "ABB")."""
    _discover()
    module_name = _backends.get(robot_type)
    if module_name is None:
        available = ", ".join(sorted(_backends)) or "(none found)"
        raise ValueError(
            f"No robot backend registered for robot_type={robot_type!r}. "
            f"Available: {available}"
        )
    module = importlib.import_module(module_name)
    return module.create_backend(config)
