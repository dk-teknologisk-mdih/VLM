"""
GUI for VLM Block Stacking/Sorting Input

Step-by-step wizard GUI to configure block manipulation tasks.
- Step 1: Choose task (stack or sort)
- Step 2 (Stack): Drag and drop blocks to build stacking order
- Step 2 (Sort): Drag blocks to target areas
- Step 3 (Stack): Choose target area
Includes live camera preview from RealSense camera.
"""

__all__ = ["get_user_input", "VLMInputGUI", "DraggableBlock", "StackBuilder"]


def __getattr__(name):
    if name == "DraggableBlock":
        from .draggable_block import DraggableBlock
        return DraggableBlock
    if name in {"VLMInputGUI", "get_user_input"}:
        from .gui import VLMInputGUI, get_user_input
        return {"VLMInputGUI": VLMInputGUI, "get_user_input": get_user_input}[name]
    if name == "StackBuilder":
        from .stack_builder import StackBuilder
        return StackBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
