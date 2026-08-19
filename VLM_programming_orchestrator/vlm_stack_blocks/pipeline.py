"""Main pipeline orchestrator for stacking trajectory planning."""

import numpy as np

from .detection import (add_depth_to_detections, detect_blocks,
                        detect_target_location)
from .detector import GeminiRoboticsDetector
from .trajectory import (build_trajectory_prompt, convert_plan_to_3d,
                         extract_waypoints_for_visualization)
from .utils import print_summary, save_detections, save_stacking_plan


def plan_stacking_trajectory(
    api_key,
    base_url,
    object_to_stack="red block",
    where_to_stack="the middle of the white paper",
    config=None
):
    """
    Plan a trajectory for stacking objects on top of each other.

    Args:
        api_key (str): API key for Gemini proxy
        base_url (str): Base URL for the proxy server
        object_to_stack (str or list): Name of the objects to stack (e.g., "red block" or ['red block', 'blue block'])
        where_to_stack (str): Target location description (e.g., "white paper", "green block")
        config (dict, optional): Configuration dict with keys:
            - task: str (e.g., "stack blocks")
            - stack_order: list (e.g., ['red', 'blue', 'green'])
            - objects_to_manipulate: list (e.g., ['red block', 'blue block', 'green block'])
            - target_position: tuple (y, x) normalized coordinates (0-1000)
            - target_location: str description

    Returns:
        dict: Contains stacking_plan_3d, blocks, target_location, and metadata
    """
    # Extract config values if provided
    stack_order = None
    target_position = None

    if config:
        stack_order = config.get('stack_order')
        target_position = config.get('target_position')
        if config.get('objects_to_manipulate'):
            object_to_stack = config['objects_to_manipulate']
        if config.get('target_position'):
            where_to_stack = config['target_position']

    # Extract camera settings from config
    exposure = config.get('exposure', 450) if config else 450
    contrast = config.get('contrast', 70) if config else 70
    brightness_target = config.get('brightness_target', 20) if config else 20

    # Initialize detector and capture image
    detector = GeminiRoboticsDetector(
        api_key, exposure=exposure, contrast=contrast, brightness_target=brightness_target)
    image, depth_frame, depth_intrinsic = detector.capture_realsense_image(
        "0_raw_realsense_capture.png")
    image_size = image.size
    depth_image = np.asanyarray(depth_frame.get_data())

    # Detect blocks
    blocks = detect_blocks(detector, image, object_to_stack)
    print("\n")

    # Handle target location - either from clicked position or detection
    if target_position:
        # Use clicked position directly (normalized coordinates)
        y_norm, x_norm = target_position
        target_location = {
            'point': [y_norm, x_norm],
            'label': f'target ({where_to_stack})'
        }
        # print(f"\nUsing clicked target position: y={y_norm}, x={x_norm} (normalized)")
    else:
        # Detect target location from image
        target_location = detect_target_location(
            detector, image, where_to_stack)

    # Add depth values to detections
    # Using hardcoded depth for blocks to avoid issues with missing depth data
    add_depth_to_detections(blocks, depth_image, image_size, hardcoded=True)
    if target_location:
        # Using hardcoded depth for target to ensure it's included in the plan even if depth data is missing
        add_depth_to_detections(
            [target_location], depth_image, image_size, hardcoded=True)
    print("\n")

    # Combine and save all detected objects
    all_detected = blocks.copy()
    if target_location:
        all_detected.append(target_location)
    save_detections(detector, all_detected, "1_detected_objects.yaml")
    detector.draw_points_on_image(
        image, all_detected, output_path="1_detected_objects.png")

    # Generate trajectory plan
    trajectory_prompt = build_trajectory_prompt(
        blocks, target_location, object_to_stack, where_to_stack, stack_order)

    prompt_path = detector.ensure_output_dir("2_trajectory_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(trajectory_prompt)
    print(f"Trajectory prompt saved to: {prompt_path}")

    stacking_plan_pixel = detector.detect_objects(
        image, trajectory_prompt, save_path="2_stacking_plan_pixel.yaml")

    # Visualize trajectory
    all_waypoints = extract_waypoints_for_visualization(stacking_plan_pixel)
    all_waypoints.extend(all_detected)
    detector.draw_points_on_image(
        image, all_waypoints, output_path="2_trajectory_with_waypoints.png")

    # Convert to 3D coordinates and save
    stacking_plan_3d = convert_plan_to_3d(
        detector, stacking_plan_pixel, image_size, depth_frame, depth_intrinsic)
    save_stacking_plan(detector, stacking_plan_3d,
                       "3_stacking_trajectory_plan.yaml")

    # Print summary
    print_summary(blocks, target_location, stacking_plan_3d)

    # Return all relevant data for downstream processing
    return {
        "stacking_plan_3d": stacking_plan_3d,
        "stacking_plan_pixel": stacking_plan_pixel,
        "blocks": blocks,
        "target_location": target_location,
        "object_to_stack": object_to_stack,
        "where_to_stack": where_to_stack,
        "stack_order": stack_order,
        "image_size": image_size
    }
