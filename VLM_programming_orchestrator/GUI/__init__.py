"""
GUI for VLM Block Stacking/Sorting Input

Step-by-step wizard GUI to configure block manipulation tasks.
- Step 1: Choose task (stack or sort)
- Step 2 (Stack): Drag and drop blocks to build stacking order
- Step 2 (Sort): Drag blocks to target areas
- Step 3 (Stack): Choose target area
Includes live camera preview from RealSense camera.
"""

from .draggable_block import DraggableBlock
from .gui import VLMInputGUI, get_user_input
from .stack_builder import StackBuilder

__all__ = ["get_user_input", "VLMInputGUI", "DraggableBlock", "StackBuilder"]
