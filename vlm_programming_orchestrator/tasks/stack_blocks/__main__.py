"""Allow running as: python -m vlm_programming_orchestrator.tasks.stack_blocks"""

import os

from dotenv import load_dotenv

from .pipeline import plan_stacking_trajectory

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

    ####################### Generate the stacking trajectory plan using VLM #######################
    trajectory_result = plan_stacking_trajectory(
        API_KEY, BASE_URL, config=config)
