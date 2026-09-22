"""
Stack Blocks Task Plugin

VLM detection prompts live in `prompts/*.txt` (offline-editable). Generic
plan-schema helpers (depth lookup, pixel->3D, prompt templating) live in
`tasks/common.py` and are shared with other task plugins.
"""

from .backend import TASK_TYPE, StackBlocksTaskBackend, create_backend
from .pipeline import plan_stacking_trajectory

__all__ = [
    "TASK_TYPE",
    "StackBlocksTaskBackend",
    "create_backend",
    "plan_stacking_trajectory",
]
