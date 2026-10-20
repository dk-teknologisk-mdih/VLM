"""Offline check of the pixel -> camera -> robot base conversion (no robot/camera needed).

Compares, for every VLM waypoint, what the pipeline currently produces when no
depth frame is available (normalized pixel / 1000 used as meters) with a proper
pinhole deprojection at the fixed depth, and prints the resulting base-frame offset.

    python calibration_tests/test_pixel_to_base_offline.py
    python calibration_tests/test_pixel_to_base_offline.py --yaml VLM_output/1_inspection_path_pixel.yaml --depth-mm 570
"""

import argparse
import json

import numpy as np
import yaml


def deproject(y_norm, x_norm, depth_m, intr):
    u = x_norm / 1000.0 * intr["width"]
    v = y_norm / 1000.0 * intr["height"]
    return np.array([(u - intr["ppx"]) / intr["fx"] * depth_m,
                     (v - intr["ppy"]) / intr["fy"] * depth_m,
                     depth_m])


def to_base(T, p_cam):
    return (T @ np.append(p_cam, 1.0))[:3]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yaml", default="VLM_output/1_inspection_path_pixel.yaml")
    parser.add_argument("--calib", default="vlm_programming_orchestrator/tasks/path_inspection/T_base_cam.json")
    parser.add_argument("--depth-mm", type=float, default=570.0)
    args = parser.parse_args()

    with open(args.calib, encoding="utf-8") as f:
        calib = json.load(f)
    T = np.array(calib["T_base_cam"])
    intr = calib["camera_intrinsics"]
    with open(args.yaml, encoding="utf-8") as f:
        waypoints = yaml.safe_load(f)

    depth_m = args.depth_mm / 1000.0
    print(f"Intrinsics: {intr['width']}x{intr['height']} fx={intr['fx']:.1f} fy={intr['fy']:.1f} "
          f"ppx={intr['ppx']:.1f} ppy={intr['ppy']:.1f}   depth={args.depth_mm:.0f} mm\n")
    header = (f"{'label':<22}{'pixel(u,v)':>14}  {'cam current [m]':>26}  {'cam pinhole [m]':>26}  "
              f"{'base current [m]':>26}  {'base pinhole [m]':>26}  {'offset [mm]':>10}")
    print(header)
    print("-" * len(header))

    offsets = []
    for wp in waypoints:
        y_norm, x_norm = wp["point"][:2]
        u, v = x_norm / 1000.0 * intr["width"], y_norm / 1000.0 * intr["height"]
        cam_current = np.array([x_norm / 1000.0, y_norm / 1000.0, depth_m])
        cam_pinhole = deproject(y_norm, x_norm, depth_m, intr)
        base_current = to_base(T, cam_current)
        base_pinhole = to_base(T, cam_pinhole)
        offset = np.linalg.norm(base_current - base_pinhole) * 1000
        offsets.append(base_current - base_pinhole)
        fmt = lambda a: "[" + ", ".join(f"{x:7.3f}" for x in a) + "]"
        print(f"{wp.get('label', ''):<22}{f'({u:.0f},{v:.0f})':>14}  {fmt(cam_current):>26}  {fmt(cam_pinhole):>26}  "
              f"{fmt(base_current):>26}  {fmt(base_pinhole):>26}  {offset:10.1f}")

    offsets = np.array(offsets) * 1000
    print(f"\nMean base-frame offset (current - pinhole) [mm]: x={offsets[:, 0].mean():.1f} "
          f"y={offsets[:, 1].mean():.1f} z={offsets[:, 2].mean():.1f}")
    print("If the robot's error matches this offset, the pixel->camera conversion is the cause.")


if __name__ == "__main__":
    main()
