"""Block detection and depth estimation functions."""

import numpy as np
import cv2


def detect_blocks(detector, image, object_to_stack):
    """
    Detect all blocks to stack in the image.

    Args:
        detector: GeminiRoboticsDetector instance
        image: PIL Image
        object_to_stack: Name of the objects to detect (can be string or list)

    Returns:
        list: Detected blocks with normalized coordinates
    """
    # Handle list of objects to detect
    if isinstance(object_to_stack, list):
        object_to_stack = [
            f"the center of the {obj}" for obj in object_to_stack]
        objects_str = ",".join(object_to_stack)
        prompt = f"""Detect each of the following objects in the image: {objects_str}.
Return the location of each object.

The answer should follow the YAML format:
- point: <point>
  label: <label>
- point: <point>
  label: <label>

The points are in [y, x] format normalized to 0-1000.
Use the exact object name as the label (e.g., 'red block', 'blue block')."""
    else:
        prompt = f"""Detect all {object_to_stack}s in the image and return their locations.

The answer should follow the YAML format:
- point: <point>
  label: <label>
- point: <point>
  label: <label>

The points are in [y, x] format normalized to 0-1000."""

    return detector.detect_objects(image, prompt)


def detect_target_location(detector, image, where_to_stack):
    """
    Detect the target stacking location in the image.

    Args:
        detector: GeminiRoboticsDetector instance
        image: PIL Image
        where_to_stack: Description of target location

    Returns:
        dict or None: Target location with normalized coordinates
    """
    prompt = f"""Detect the {where_to_stack} in the image and return its location.

The answer should follow the YAML format:
- point: <point>
  label: <label>

The point is in [y, x] format normalized to 0-1000."""

    target_locations = detector.detect_objects(image, prompt)

    if not target_locations:
        print(f"Warning: Could not find {where_to_stack}")
        return None
    return target_locations[0]


def add_depth_to_detections(detections, depth_image, image_size, hardcoded=False):
    """
    Add depth values to detected objects.

    Args:
        detections: List of detected objects with normalized coordinates
        depth_image: Numpy array of depth values
        image_size: Tuple of (width, height)
        hardcoded: If True, use a fixed depth of 470 mm for all detections

    Returns:
        list: Detections with depth values appended to points
    """
    width, height = image_size

    for detection in detections:
        y_norm, x_norm = detection["point"]
        abs_x = int(x_norm / 1000.0 * width)
        abs_y = int(y_norm / 1000.0 * height)

        if hardcoded:
            depth_value = 570  # Use a fixed depth value for testing
        else:
            depth_value = depth_image[abs_y, abs_x]

            if depth_value == 0:
                print(
                    f"Warning: Depth value at pixel ({abs_x}, {abs_y}) is 0, which may indicate missing depth data.")
                neighbors = depth_image[max(0, abs_y-5):min(depth_image.shape[0], abs_y+5),
                                        max(0, abs_x-5):min(depth_image.shape[1], abs_x+5)]
                non_zero_neighbors = neighbors[neighbors > 0]
                if len(non_zero_neighbors) > 0:
                    depth_value = int(np.mean(non_zero_neighbors))
                    print(
                        f"Estimated depth value from neighbors: {depth_value} mm")
                else:
                    print("No valid neighboring depth values found, setting depth to 0")
                    depth_value = 0

        detection['point'].append(int(depth_value))
        print(
            f"{detection['label']} at pixel ({abs_x}, {abs_y}) with depth {depth_value} mm")

    return detections
