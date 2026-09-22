"""Utility functions for saving detections and stacking plans."""

import yaml


def save_detections(detector, detections, filename):
    """Save detected objects to YAML file."""
    save_path = detector.ensure_output_dir(filename)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(detections, f, default_flow_style=False, sort_keys=False)
        print(f"\nDetected objects saved to: {save_path}")


def save_plan(detector, plan_data, filename):
    """Save the 3D plan to a YAML file."""
    save_path = detector.ensure_output_dir(filename)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(plan_data, f, default_flow_style=False, sort_keys=False)
        print(f"Plan with 3D coordinates saved to: {save_path}")


def print_summary(blocks, target_location, plan_data):
    """Print summary of the stacking plan."""
    print("\n=== Summary ===")
    print(f"Found {len(blocks)} blocks to stack")
    if target_location:
        print(f"Target location: {target_location['label']}")
    print(f"Generated {len(plan_data)} steps in stacking plan:")
