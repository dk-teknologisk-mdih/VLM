"""Utility functions for saving detections and stacking plans."""

import yaml


def save_detections(detector, detections, filename):
    """Save detected objects to YAML file."""
    save_path = detector.ensure_output_dir(filename)
    with open(save_path, "w") as f:
        yaml.dump(detections, f, default_flow_style=False, sort_keys=False)
        print(f"\nDetected objects saved to: {save_path}")


def save_stacking_plan(detector, stacking_plan_3d, filename):
    """Save 3D stacking plan to YAML file."""
    save_path = detector.ensure_output_dir(filename)
    with open(save_path, "w") as f:
        yaml.dump(stacking_plan_3d, f,
                  default_flow_style=False, sort_keys=False)
        print(f"Stacking plan with 3D coordinates saved to: {save_path}")


def print_summary(blocks, target_location, stacking_plan_3d):
    """Print summary of the stacking plan."""
    print(f"\n=== Summary ===")
    print(f"Found {len(blocks)} blocks to stack")
    if target_location:
        print(f"Target location: {target_location['label']}")
    print(f"Generated {len(stacking_plan_3d)} steps in stacking plan:")
    # for step in stacking_plan_3d:
    #     print(f"  Step {step.get('step', '?')}: {step.get('action', 'unknown')} - {step.get('description', '')}")
    #     if 'trajectory' in step:
    #         print(f"    Trajectory has {len(step['trajectory'])} waypoints (in 3D camera coordinates)")
