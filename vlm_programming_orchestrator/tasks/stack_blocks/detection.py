"""Block detection and target-location detection, using externalized prompt templates."""

import os

from ..common import load_prompt_template

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def detect_blocks(detector, image, object_to_stack):
    """Detect all blocks to stack in the image.

    Args:
        detector: `ObjectDetector` instance.
        image: PIL Image.
        object_to_stack: Name of the objects to detect (string or list).

    Returns:
        list: Detected blocks with normalized coordinates.
    """
    if isinstance(object_to_stack, list):
        target_description = ", ".join(
            f"the center of the {obj}" for obj in object_to_stack)
    else:
        target_description = f"all {object_to_stack}s"

    prompt = load_prompt_template(
        os.path.join(_PROMPTS_DIR, "vlm_detect_blocks.txt"),
        target_description=target_description,
    )
    return detector.detect_objects(image, prompt)


def detect_target_location(detector, image, where_to_stack):
    """Detect the target stacking location in the image.

    Args:
        detector: `ObjectDetector` instance.
        image: PIL Image.
        where_to_stack: Description of target location.

    Returns:
        dict or None: Target location with normalized coordinates.
    """
    prompt = load_prompt_template(
        os.path.join(_PROMPTS_DIR, "vlm_detect_target.txt"),
        where_to_stack=where_to_stack,
    )

    target_locations = detector.detect_objects(image, prompt)

    if not target_locations:
        print(f"Warning: Could not find {where_to_stack}")
        return None
    return target_locations[0]
