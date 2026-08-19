"""
VLM Stack Blocks Package

Generic (robot/model-agnostic) trajectory planning + prompt building.
Vision detection lives in `vision/`, LLM code-gen in `llm/`.
"""

from .code_generation import build_stacking_prompt, extract_code_block
from .detection import (add_depth_to_detections, detect_blocks,
                        detect_target_location)
from .pipeline import plan_stacking_trajectory
from .trajectory import (build_trajectory_prompt, convert_plan_to_3d,
                         extract_waypoints_for_visualization)
from .utils import print_summary, save_detections, save_stacking_plan

__all__ = [
    "plan_stacking_trajectory",
    "build_stacking_prompt",
    "extract_code_block",
    "detect_blocks",
    "detect_target_location",
    "add_depth_to_detections",
    "build_trajectory_prompt",
    "convert_plan_to_3d",
    "extract_waypoints_for_visualization",
    "save_detections",
    "save_stacking_plan",
    "print_summary",
]

