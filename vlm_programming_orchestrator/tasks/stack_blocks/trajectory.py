"""Trajectory prompt building (via externalized template) and visualization helpers.

Pixel->3D conversion now lives in `tasks/common.py` (shared across tasks).
"""

import os

from ..common import load_prompt_template

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def build_trajectory_prompt(blocks, target_location, object_to_stack, where_to_stack, stack_order=None):
    """Build the trajectory planning prompt for the VLM.

    Args:
        blocks: List of detected blocks with depth.
        target_location: Target location dict or None.
        object_to_stack: Name of the objects being stacked (string or list).
        where_to_stack: Description of target location.
        stack_order: Optional list specifying stacking order (e.g., ['red', 'blue', 'green']).

    Returns:
        str: Formatted prompt for trajectory planning.
    """
    blocks_info = chr(10).join([
        (
            f"Block {i+1} ({block['label']}): "
            f"y={block['point'][0]}, "
            f"x={block['point'][1]} (normalized 0-1000), "
            f"depth={block['point'][2]} mm"
        )
        for i, block in enumerate(blocks)
    ])

    if target_location:
        target_info = (
            f"Target location ({target_location['label']}): y={target_location['point'][0]}, "
            f"x={target_location['point'][1]} (normalized 0-1000), "
            f"depth={target_location['point'][2]} mm"
        )
    else:
        target_info = f"Target: {where_to_stack} (location not precisely detected)"

    if isinstance(object_to_stack, list):
        object_desc = ", ".join(object_to_stack)
    else:
        object_desc = object_to_stack

    stack_order_info = ""
    if stack_order:
        stack_order_str = " -> ".join(
            [f"{color} block" for color in stack_order])
        stack_order_info = f"\n\nIMPORTANT: Stack the blocks in this specific order (bottom to top): {stack_order_str}"

    return load_prompt_template(
        os.path.join(_PROMPTS_DIR, "vlm_trajectory_prompt.txt"),
        num_blocks=len(blocks),
        object_desc=object_desc,
        stack_order_info=stack_order_info,
        blocks_info=blocks_info,
        target_info=target_info,
        where_to_stack=where_to_stack,
    )
