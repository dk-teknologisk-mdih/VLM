"""Swappable robot backends for the robot-specific pipeline subgraph.

Each subfolder (e.g. `abb/`) is a plugin exposing `ROBOT_TYPE` and
`create_backend(config)` in its `backend` module; see `registry.py`.
"""

from .base import (ErrorDescriber, ModulePreparer, RobotBackend,
                   RobotExecutor, Simulator, SyntaxVerifier)
from .registry import get_robot_backend

__all__ = [
    "RobotBackend",
    "SyntaxVerifier",
    "Simulator",
    "ErrorDescriber",
    "RobotExecutor",
    "ModulePreparer",
    "get_robot_backend",
]
