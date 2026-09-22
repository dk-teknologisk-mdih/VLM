"""
Path Inspection Task Plugin

Detects a configurable target object (default: bike frame) and plans an
ordered 3D waypoint path for a robot to follow while inspecting it. VLM and
LLM prompts live in `prompts/*.txt` (offline-editable), and can be swapped at
runtime via `Config.vlm_prompt_file` / `Config.llm_prompt_file`.
"""

from .backend import TASK_TYPE, PathInspectionTaskBackend, create_backend
from .pipeline import plan_inspection_path

__all__ = [
    "TASK_TYPE",
    "PathInspectionTaskBackend",
    "create_backend",
    "plan_inspection_path",
]
