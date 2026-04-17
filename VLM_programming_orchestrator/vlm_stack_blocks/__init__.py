"""
VLM Stack Blocks Package

Uses Gemini Robotics-ER to plan a stacking trajectory for blocks.
Uses requests-based API calls through a corporate proxy.
"""

from .code_generation import (build_stacking_prompt,
                              call_claude_for_robot_code,
                              call_claude_to_fix_code, extract_code_block)
from .detection import (add_depth_to_detections, detect_blocks,
                        detect_target_location)
from .detector import GeminiRoboticsDetector
from .pipeline import plan_stacking_trajectory
from .trajectory import (build_trajectory_prompt, convert_plan_to_3d,
                         extract_waypoints_for_visualization)
from .utils import print_summary, save_detections, save_stacking_plan

__all__ = [
    "plan_stacking_trajectory",
    "call_claude_for_robot_code",
    "call_claude_to_fix_code",
    "build_stacking_prompt",
    "extract_code_block",
    "GeminiRoboticsDetector",
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
