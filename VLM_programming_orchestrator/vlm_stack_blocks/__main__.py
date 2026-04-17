"""Allow running as: python -m vlm_stack_blocks"""

import os
from dotenv import load_dotenv

from .pipeline import plan_stacking_trajectory
from .code_generation import call_claude_for_robot_code

load_dotenv()

if __name__ == "__main__":
    # Proxy configuration
    API_KEY = os.environ["API_KEY"]
    BASE_URL = os.environ["BASE_URL"]

    ####################### Configuration-based input #######################
    # Example configuration from GUI or external source
    config = {
        'task': 'stack blocks',
        # Optional: specify stacking order
        'stack_order': ['green', 'light blue', 'yellow', 'dark blue', 'red'],
        # Optional: specify objects to stack
        'objects_to_manipulate': ['green block', 'light blue block', 'yellow block', 'dark blue block', 'red block'],
        'target_position': (434, 127),  # (y, x) normalized coordinates 0-1000
        'target_location': 'clicked position at normalized coordinates (434, 127)'
    }

    # Legacy input mode (used if config is None)
    # object_to_stack = "the middle of the blue and yellow blocks"
    # where_to_stack = "the middle of the white paper"

    ####################### Generate the stacking trajectory plan using VLM #######################
    # Use VLM to plan the stacking trajectory with config
    trajectory_result = plan_stacking_trajectory(
        API_KEY, BASE_URL, config=config)

    # Or use legacy mode without config:
    # trajectory_result = plan_stacking_trajectory(API_KEY, BASE_URL, object_to_stack, where_to_stack)

    ####################### Generate robot control code using LLM #######################
    # # Send the stacking plan to Claude to generate robot control code
    # robot_code = call_claude_for_robot_code(
    #     api_key=API_KEY,
    #     base_url=BASE_URL,
    #     stacking_plan_3d=trajectory_result["stacking_plan_3d"],
    #     robot_type="ABB",
    #     code_language="Rapid"
    # )
