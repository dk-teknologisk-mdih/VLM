"""Shared, task-agnostic helpers reused by task plugins.

None of this file knows about "blocks" or "bike frames" — it operates on the
generic `{point: [y, x], label: ...}` detection schema and the generic
`step -> trajectory -> waypoint` plan schema shared by every task.
"""

import re

import numpy as np


def load_prompt_template(path, **fmt) -> str:
    """Read a prompt template `.txt` file and fill in its `{placeholders}`.

    Keeps prompts as plain, offline-editable text files instead of Python
    f-strings, so they can be tuned or swapped without touching code.
    """
    with open(path, "r", encoding="utf-8") as f:
        template = f.read()
    return template.format(**fmt)


def extract_code_block(text):
    """Strip ```rapid / ``` fences from an LLM response. Returns the raw code."""
    if not text:
        return text

    # Remove em-dashes and en-dashes that may be used as fences instead of backticks
    text = text.replace("—", "-").replace("–", "-")

    m = re.search(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def add_depth_to_detections(detections, depth_image, image_size, hardcoded=False, hardcoded_depth_mm=570):
    """Add a depth value (mm) to each detection's `point` list, in place.

    Args:
        detections: List of `{point: [y_norm, x_norm], label: ...}` dicts.
        depth_image: Numpy array of depth values.
        image_size: Tuple of (width, height).
        hardcoded: If True, use `hardcoded_depth_mm` for all detections instead
            of sampling the depth image (useful when depth data is unreliable).
        hardcoded_depth_mm: Fixed depth value used when `hardcoded` is True.

    Returns:
        list: The same `detections` list, with depth appended to each point.
    """
    width, height = image_size

    for detection in detections:
        y_norm, x_norm = detection["point"]
        abs_x = int(x_norm / 1000.0 * width)
        abs_y = int(y_norm / 1000.0 * height)

        if hardcoded:
            depth_value = hardcoded_depth_mm
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


def convert_plan_to_3d(detector, plan_pixel, image_size, depth_frame, depth_intrinsic):
    """Convert a `step -> trajectory -> waypoint` plan from pixel to 3D camera coordinates.

    Args:
        detector: An `ObjectDetector` instance (for `img_point_to_cam_coord_realsense`).
        plan_pixel: List of steps, each optionally holding a `trajectory` list
            of `{point: [y_norm, x_norm, depth_mm], label: ...}` waypoints.
        image_size: Tuple of (width, height).
        depth_frame: RealSense depth frame.
        depth_intrinsic: Depth camera intrinsics.

    Returns:
        list: The plan with each waypoint's `point` replaced by `[x, y, z]` in
        camera-frame meters.
    """
    width, height = image_size
    plan_3d = []

    for step in plan_pixel:
        step_3d = step.copy()

        if 'trajectory' in step:
            trajectory_3d = []
            for waypoint in step['trajectory']:
                waypoint_3d = waypoint.copy()
                if 'point' in waypoint and len(waypoint['point']) >= 2:
                    y_norm, x_norm = waypoint['point'][0], waypoint['point'][1]
                    abs_x = int(x_norm / 1000.0 * width)
                    abs_y = int(y_norm / 1000.0 * height)

                    point_3d, _ = detector.img_point_to_cam_coord_realsense(
                        [abs_x, abs_y],
                        depth_frame,
                        depth_intrinsic
                    )
                    waypoint_3d['point'] = [float(point_3d[0]), float(
                        point_3d[1]), float(waypoint['point'][2]/1000.0)]
                trajectory_3d.append(waypoint_3d)
            step_3d['trajectory'] = trajectory_3d

        plan_3d.append(step_3d)

    return plan_3d


def extract_waypoints_for_visualization(plan_pixel):
    """Flatten all waypoints from a `step -> trajectory` plan for visualization.

    Args:
        plan_pixel: List of steps with trajectories.

    Returns:
        list: All waypoints, each tagged with its originating step number.
    """
    all_waypoints = []
    for step in plan_pixel:
        if 'trajectory' in step:
            step_number = step.get('step', 0)
            for waypoint in step['trajectory']:
                waypoint_with_step = waypoint.copy()
                waypoint_with_step['step'] = step_number
                all_waypoints.append(waypoint_with_step)
    return all_waypoints
