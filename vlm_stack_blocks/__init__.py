"""
VLM Stack Blocks Package

Uses Gemini Robotics-ER to plan a stacking trajectory for blocks.
Uses requests-based API calls through a corporate proxy.
"""

from .pipeline import plan_stacking_trajectory
from .code_generation import call_claude_for_robot_code
from .detector import GeminiRoboticsDetector
from .detection import detect_blocks, detect_target_location, add_depth_to_detections
from .trajectory import build_trajectory_prompt, convert_plan_to_3d, extract_waypoints_for_visualization
from .utils import save_detections, save_stacking_plan, print_summary

__all__ = [
    "plan_stacking_trajectory",
    "call_claude_for_robot_code",
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
