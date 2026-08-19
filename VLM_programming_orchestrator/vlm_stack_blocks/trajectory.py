"""Trajectory planning: prompt building, coordinate conversion, and visualization helpers."""


def build_trajectory_prompt(blocks, target_location, object_to_stack, where_to_stack, stack_order=None):
    """
    Build the trajectory planning prompt for Gemini.

    Args:
        blocks: List of detected blocks with depth
        target_location: Target location dict or None
        object_to_stack: Name of the objects being stacked (string or list)
        where_to_stack: Description of target location
        stack_order: Optional list specifying stacking order (e.g., ['red', 'blue', 'green'])

    Returns:
        str: Formatted prompt for trajectory planning
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

    # Determine object description for prompt
    if isinstance(object_to_stack, list):
        object_desc = ", ".join(object_to_stack)
    else:
        object_desc = object_to_stack

    # Add stack order instructions if provided
    stack_order_info = ""
    if stack_order:
        stack_order_str = " -> ".join(
            [f"{color} block" for color in stack_order])
        stack_order_info = f"\n\nIMPORTANT: Stack the blocks in this specific order (bottom to top): {stack_order_str}"

    return f"""
You are controlling a robot arm to stack {len(blocks)} blocks ({object_desc}) on top of each other.{stack_order_info}

The blocks to be stacked are located at these positions:
{blocks_info}

The target stacking location is at:
{target_info}

Plan a stacking operation where you:
1. Pick up each block one by one
2. Stack them all at the target location: {where_to_stack} on top of each other, with the first block at the bottom and the last block on top
3. For each pick operation, provide waypoints from the gripper starting position to the block 
4. For each place operation, provide waypoints from the picked block to the stacking location
5. After the stacking has been completed, wait for 5 seconds and then destack the blocks back to their original positions

Any time we either pick up or place down a block, include a single approach and retract point above (Z-axis) the target location before and after the target location to avoid collisions
The Z-axis is the depth direction, so for example if the block is at depth 500 mm, the approach point could be at depth 300 mm (200 mm above the block)
Remember to consider the height of the stack as you place each block, each block height is 40 mm.

Return a YAML array with the stacking plan:
- step: 1
  action: pick
  block: Block 2
  trajectory:
    - point: [y, x, z]
      label: approach_point
    - point: [y, x, z]
      label: pick_point
    - point: [y, x, z]
      label: retract_point
- step: 2
  action: place
  block: Block 2
  trajectory:
    - point: [y, x, z]
      label: approach_point
    - point: [y, x, z]
      label: place_point
    - point: [y, x, z]
      label: retract_point

IMPORTANT FORMAT RULES:
- Points are in [y, x, z] format: y and x normalized to 0-1000, z as depth in millimeters
- Do NOT include a description field - only step, action, block, and trajectory
- Label waypoints: approach_point, pick_point/place_point, retract_point"""


def convert_plan_to_3d(detector, stacking_plan_pixel, image_size, depth_frame, depth_intrinsic):
    """
    Convert stacking plan from pixel coordinates to 3D camera coordinates.

    Args:
        detector: GeminiRoboticsDetector instance
        stacking_plan_pixel: List of steps with pixel coordinates
        image_size: Tuple of (width, height)
        depth_frame: RealSense depth frame
        depth_intrinsic: Depth camera intrinsics

    Returns:
        list: Stacking plan with 3D camera coordinates
    """
    width, height = image_size
    stacking_plan_3d = []

    for step in stacking_plan_pixel:
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
                    # print(f"point after {waypoint['label']}: 3D camera coordinates {waypoint_3d['point']}")
                trajectory_3d.append(waypoint_3d)
            step_3d['trajectory'] = trajectory_3d

        stacking_plan_3d.append(step_3d)

    return stacking_plan_3d


def extract_waypoints_for_visualization(stacking_plan_pixel):
    """
    Extract all waypoints from stacking plan for visualization.

    Args:
        stacking_plan_pixel: List of steps with trajectories

    Returns:
        list: All waypoints with step numbers for color grouping
    """
    all_waypoints = []
    for step in stacking_plan_pixel:
        if 'trajectory' in step:
            step_number = step.get('step', 0)
            for waypoint in step['trajectory']:
                waypoint_with_step = waypoint.copy()
                waypoint_with_step['step'] = step_number
                all_waypoints.append(waypoint_with_step)
    return all_waypoints
