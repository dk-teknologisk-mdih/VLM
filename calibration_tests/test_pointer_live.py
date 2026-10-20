"""Live pointer test: compare the predicted base-frame point with where the robot tip really is.

1. Define the pointer tip as the active TCP on the pendant.
2. Jog the tip onto a visible feature (or hold it in front of the camera).
3. Click the tip in the image window, then press SPACE.

For each sample the script prints:
  - predicted base point at the fixed depth and at the measured depth
  - actual TCP position (real-time interface)
  - the actual TCP projected into the image (green circle) -> if it is not on the
    tip, the calibration or TCP definition is wrong, independent of any depth.
  - the depth the camera would need (TCP z in camera frame) vs. the fixed depth.

Keys: left click = select pixel, SPACE = measure, Q/ESC = quit.

    python calibration_tests/test_pointer_live.py --robot-ip 192.168.1.10
"""

import argparse
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CameraCalibration"))
from UR_camera_robot_calibration import read_tcp_pose  # noqa: E402


def deproject(u, v, depth_m, intr):
    return np.array([(u - intr["ppx"]) / intr["fx"] * depth_m,
                     (v - intr["ppy"]) / intr["fy"] * depth_m,
                     depth_m])


def project(p_cam, intr):
    return (intr["fx"] * p_cam[0] / p_cam[2] + intr["ppx"],
            intr["fy"] * p_cam[1] / p_cam[2] + intr["ppy"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--calib", default="vlm_programming_orchestrator/tasks/path_inspection/T_base_cam.json")
    parser.add_argument("--depth-mm", type=float, default=570.0)
    parser.add_argument("--log", default="calibration_tests/pointer_results.json")
    args = parser.parse_args()

    with open(args.calib, encoding="utf-8") as f:
        calib = json.load(f)
    T_base_cam = np.array(calib["T_base_cam"])
    T_cam_base = np.linalg.inv(T_base_cam)
    intr = calib["camera_intrinsics"]

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, intr["width"], intr["height"], rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, intr["width"], intr["height"], rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    for _ in range(30):
        pipeline.wait_for_frames()

    live_intr = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(f"Live color intrinsics fx={live_intr.fx:.1f} ppx={live_intr.ppx:.1f} ppy={live_intr.ppy:.1f} "
          f"(calib file fx={intr['fx']:.1f} ppx={intr['ppx']:.1f} ppy={intr['ppy']:.1f})")

    clicked = {"uv": None}
    cv2.namedWindow("pointer test")
    cv2.setMouseCallback("pointer test",
                         lambda e, x, y, *_: clicked.update(uv=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)

    results = []
    last_tcp_px = None
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color = np.asanyarray(frames.get_color_frame().get_data()).copy()
            depth_frame = frames.get_depth_frame()

            if clicked["uv"]:
                cv2.drawMarker(color, clicked["uv"], (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            if last_tcp_px:
                cv2.circle(color, last_tcp_px, 8, (0, 255, 0), 2)
            cv2.putText(color, "click tip, SPACE=measure, Q=quit  red=click green=robot TCP",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("pointer test", color)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != ord(" "):
                continue
            if clicked["uv"] is None:
                print("Click the tip in the image first")
                continue

            u, v = clicked["uv"]
            measured_depth = depth_frame.get_distance(u, v)
            tcp = read_tcp_pose(args.robot_ip)
            actual = tcp[:3]
            tcp_cam = (T_cam_base @ np.append(actual, 1.0))[:3]
            tcp_px = project(tcp_cam, intr)
            last_tcp_px = (int(round(tcp_px[0])), int(round(tcp_px[1])))

            pred_fixed = (T_base_cam @ np.append(deproject(u, v, args.depth_mm / 1000.0, intr), 1.0))[:3]
            pred_meas = (T_base_cam @ np.append(deproject(u, v, measured_depth, intr), 1.0))[:3] \
                if measured_depth > 0 else None

            err_fixed = (pred_fixed - actual) * 1000
            print(f"\nSample {len(results) + 1}: pixel ({u}, {v})")
            print(f"  actual TCP base [m]            {np.round(actual, 4)}")
            print(f"  predicted (fixed {args.depth_mm:.0f} mm) [m]   {np.round(pred_fixed, 4)}  "
                  f"err [mm] {np.round(err_fixed, 1)}  |{np.linalg.norm(err_fixed):.1f}|")
            if pred_meas is not None:
                err_meas = (pred_meas - actual) * 1000
                print(f"  predicted (measured {measured_depth * 1000:.0f} mm) [m] {np.round(pred_meas, 4)}  "
                      f"err [mm] {np.round(err_meas, 1)}  |{np.linalg.norm(err_meas):.1f}|")
            print(f"  TCP depth in camera frame: {tcp_cam[2] * 1000:.0f} mm (fixed depth {args.depth_mm:.0f} mm)")
            print(f"  TCP reprojected to pixel ({tcp_px[0]:.0f}, {tcp_px[1]:.0f}) -> "
                  f"pixel error {np.hypot(tcp_px[0] - u, tcp_px[1] - v):.1f} px")

            results.append({
                "pixel": [u, v], "measured_depth_m": measured_depth, "fixed_depth_m": args.depth_mm / 1000.0,
                "tcp_pose": tcp.tolist(), "tcp_in_cam_m": tcp_cam.tolist(), "tcp_reprojected_px": list(tcp_px),
                "pred_fixed_base_m": pred_fixed.tolist(),
                "pred_measured_base_m": pred_meas.tolist() if pred_meas is not None else None,
            })
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if results:
        errs = np.array([np.array(r["pred_fixed_base_m"]) - np.array(r["tcp_pose"][:3]) for r in results]) * 1000
        print(f"\n{len(results)} samples, fixed-depth error [mm] mean {np.round(errs.mean(axis=0), 1)} "
              f"std {np.round(errs.std(axis=0), 1)}")
        print("Constant mean with small std -> TCP definition / calibration translation.")
        print("Error mostly along the camera viewing axis -> depth (tip not at the fixed depth).")
        os.makedirs(os.path.dirname(args.log), exist_ok=True)
        with open(args.log, "w", encoding="utf-8") as f:
            json.dump({"created": datetime.now().isoformat(), "samples": results}, f, indent=2)
        print(f"Saved {args.log}")


if __name__ == "__main__":
    main()
