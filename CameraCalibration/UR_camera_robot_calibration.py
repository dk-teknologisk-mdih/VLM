"""Eye-to-hand calibration for a UR robot with a static RealSense camera.

A ChArUco board (DICT_6X6, e.g. calib.io 8x11) is rigidly mounted on the robot
flange/gripper. For each sample the robot is jogged (pendant or freedrive) to a
new pose where the camera sees the board, then SPACE stores the image and the
actual TCP pose.

    python UR_camera_robot_calibration.py board        # writes a printable PNG
    python UR_camera_robot_calibration.py collect --robot-ip 192.168.1.10
    python UR_camera_robot_calibration.py calibrate

Result: T_base_cam (camera color optical frame -> robot base, meters).
"""

import argparse
import json
import os
import socket
import struct
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


# =========================================================
# Robot
# =========================================================

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("UR real-time interface closed the connection")
        buf += chunk
    return buf


def read_tcp_pose(robot_ip: str, port: int = 30003) -> np.ndarray:
    """Actual TCP pose [x, y, z, rx, ry, rz] (m, axis-angle rad) from the real-time interface."""
    with socket.create_connection((robot_ip, port), timeout=5) as sock:
        size = struct.unpack(">i", _recv_exact(sock, 4))[0]
        body = _recv_exact(sock, size - 4)
    # "Tool vector actual" starts at byte 444 of the packet (incl. the 4-byte length)
    return np.array(struct.unpack_from(">6d", body, 444 - 4))


def ur_pose_to_matrix(pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(np.asarray(pose[3:6], dtype=float))
    T[:3, 3] = pose[:3]
    return T


# =========================================================
# Camera
# =========================================================

def start_camera(resolution=(1280, 720)):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, 30)
    profile = pipeline.start(config)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intrinsics = {
        "width": intr.width, "height": intr.height,
        "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
        "model": str(intr.model), "coeffs": list(intr.coeffs),
    }
    for _ in range(30):  # let auto-exposure settle
        pipeline.wait_for_frames()
    return pipeline, intrinsics


def intrinsics_to_cv(intrinsics: dict):
    K = np.array([[intrinsics["fx"], 0, intrinsics["ppx"]],
                  [0, intrinsics["fy"], intrinsics["ppy"]],
                  [0, 0, 1]], dtype=float)
    return K, np.array(intrinsics["coeffs"], dtype=float)


# =========================================================
# Board detection
# =========================================================

def make_board(args):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    board = cv2.aruco.CharucoBoard((args.squares_x, args.squares_y), args.square_length,
                                   args.marker_length, dictionary)
    # calib.io and pre-4.6 OpenCV boards use the legacy layout (marker in the top-left square)
    board.setLegacyPattern(args.legacy)
    return board, cv2.aruco.CharucoDetector(board)


def find_board(gray: np.ndarray, board, detector, min_corners: int = 6):
    """Returns (object_points, image_points, charuco_corners, charuco_ids) or None."""
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) < min_corners:
        return None
    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    return obj_pts, img_pts, charuco_corners, charuco_ids


def save_board_image(args):
    board, _ = make_board(args)
    px_per_m = args.dpi / 0.0254
    margin = int(0.01 * px_per_m)
    size = (int(args.squares_x * args.square_length * px_per_m) + 2 * margin,
            int(args.squares_y * args.square_length * px_per_m) + 2 * margin)
    img = board.generateImage(size, marginSize=margin, borderBits=1)
    path = os.path.join(args.output_dir, "charuco_board.png")
    os.makedirs(args.output_dir, exist_ok=True)
    cv2.imwrite(path, img)
    print(f"Saved {path} ({size[0]}x{size[1]} px at {args.dpi} DPI). "
          "Print at 100% scale and measure a square to confirm --square-length.")


# =========================================================
# Collection
# =========================================================

def collect(args):
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    board, detector = make_board(args)
    pipeline, intrinsics = start_camera()
    with open(os.path.join(args.output_dir, "camera_intrinsics.json"), "w", encoding="utf-8") as f:
        json.dump(intrinsics, f, indent=2)

    poses_path = os.path.join(args.output_dir, "poses.json")
    samples = []
    if os.path.exists(poses_path):
        with open(poses_path, encoding="utf-8") as f:
            samples = json.load(f)

    print("Jog the robot so the markers are visible, then release it.")
    print("SPACE = capture sample, Q/ESC = finish")
    try:
        while True:
            frame = np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detection = find_board(gray, board, detector)
            preview = frame.copy()
            n_corners = 0
            if detection is not None:
                cv2.aruco.drawDetectedCornersCharuco(preview, detection[2], detection[3])
                n_corners = len(detection[3])
            cv2.putText(preview, f"samples: {len(samples)}  corners: {n_corners}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("UR eye-to-hand calibration", preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key != ord(" "):
                continue
            if detection is None:
                print("  Board not detected, sample skipped")
                continue

            pose = read_tcp_pose(args.robot_ip).tolist()
            name = f"image_{len(samples):03d}.png"
            cv2.imwrite(os.path.join(args.output_dir, "images", name), frame)
            samples.append({"image": name, "tcp_pose": pose})
            with open(poses_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2)
            print(f"  Sample {len(samples)}: tcp={np.round(pose, 4).tolist()}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print(f"Collected {len(samples)} samples in '{args.output_dir}'")


# =========================================================
# Calibration
# =========================================================

def _consistency_error(T_base_cam, T_base_tcp_list, T_cam_target_list):
    """For eye-to-hand the board pose in the TCP frame must be constant across samples."""
    T_tcp_target = [np.linalg.inv(Tb) @ T_base_cam @ Tc
                    for Tb, Tc in zip(T_base_tcp_list, T_cam_target_list)]
    t = np.array([T[:3, 3] for T in T_tcp_target])
    t_err = np.linalg.norm(t - t.mean(axis=0), axis=1)
    R_ref = T_tcp_target[0][:3, :3]
    r_err = [np.degrees(np.linalg.norm(cv2.Rodrigues(R_ref.T @ T[:3, :3])[0])) for T in T_tcp_target]
    return float(np.sqrt(np.mean(t_err ** 2))), float(np.sqrt(np.mean(np.square(r_err))))


def calibrate(args):
    with open(os.path.join(args.output_dir, "camera_intrinsics.json"), encoding="utf-8") as f:
        intrinsics = json.load(f)
    with open(os.path.join(args.output_dir, "poses.json"), encoding="utf-8") as f:
        samples = json.load(f)

    K, dist = intrinsics_to_cv(intrinsics)
    board, detector = make_board(args)

    R_base2tcp, t_base2tcp, R_target2cam, t_target2cam = [], [], [], []
    T_base_tcp_list, T_cam_target_list = [], []
    for s in samples:
        gray = cv2.imread(os.path.join(args.output_dir, "images", s["image"]), cv2.IMREAD_GRAYSCALE)
        detection = find_board(gray, board, detector)
        if detection is None:
            print(f"  {s['image']}: board not found, skipped")
            continue
        obj_pts, img_pts, _, _ = detection
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist, rvec, tvec)
        if not ok:
            continue

        T_cam_target = np.eye(4)
        T_cam_target[:3, :3], _ = cv2.Rodrigues(rvec)
        T_cam_target[:3, 3] = tvec.ravel()
        T_base_tcp = ur_pose_to_matrix(s["tcp_pose"])
        # Eye-to-hand: feed base->TCP (inverse robot pose) so OpenCV returns cam->base
        T_tcp_base = np.linalg.inv(T_base_tcp)

        R_base2tcp.append(T_tcp_base[:3, :3])
        t_base2tcp.append(T_tcp_base[:3, 3])
        R_target2cam.append(T_cam_target[:3, :3])
        t_target2cam.append(T_cam_target[:3, 3])
        T_base_tcp_list.append(T_base_tcp)
        T_cam_target_list.append(T_cam_target)

    n = len(T_base_tcp_list)
    if n < 3:
        raise SystemExit(f"Need at least 3 valid samples, got {n} (15-25 recommended)")

    best = None
    print(f"\nUsing {n} samples")
    for name, method in HAND_EYE_METHODS.items():
        R, t = cv2.calibrateHandEye(R_base2tcp, t_base2tcp, R_target2cam, t_target2cam, method=method)
        T_base_cam = np.eye(4)
        T_base_cam[:3, :3] = R
        T_base_cam[:3, 3] = t.ravel()
        t_rms, r_rms = _consistency_error(T_base_cam, T_base_tcp_list, T_cam_target_list)
        print(f"  {name:<10} translation RMS {t_rms * 1000:6.2f} mm   rotation RMS {r_rms:5.2f} deg")
        if best is None or t_rms < best[2]:
            best = (name, T_base_cam, t_rms, r_rms)

    name, T_base_cam, t_rms, r_rms = best
    result = {
        "description": "T_base_cam: RealSense color optical frame -> UR base frame (meters)",
        "T_base_cam": T_base_cam.tolist(),
        "method": name,
        "translation_rms_mm": t_rms * 1000,
        "rotation_rms_deg": r_rms,
        "num_samples": n,
        "board": {"type": "charuco", "dict": args.dict, "squares_x": args.squares_x,
                  "squares_y": args.squares_y, "square_length_m": args.square_length,
                  "marker_length_m": args.marker_length, "legacy": args.legacy},
        "camera_intrinsics": intrinsics,
        "calibrated_on": datetime.now().isoformat(),
    }
    out_path = os.path.join(args.output_dir, "T_base_cam.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nBest method: {name}")
    print(np.array2string(T_base_cam, precision=5, suppress_small=True))
    print(f"Saved to {out_path}")


# =========================================================
# Using the result
# =========================================================

def load_T_base_cam(path: str) -> np.ndarray:
    with open(path, encoding="utf-8") as f:
        return np.array(json.load(f)["T_base_cam"])


def camera_point_to_base(T_base_cam: np.ndarray, p_cam) -> np.ndarray:
    """p_cam in meters in the color optical frame, e.g. from rs.rs2_deproject_pixel_to_point."""
    return (T_base_cam @ np.append(np.asarray(p_cam, dtype=float), 1.0))[:3]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["board", "collect", "calibrate"])
    parser.add_argument("--robot-ip", help="UR controller IP (collect mode)")
    parser.add_argument("--output-dir", default="UR_eye_to_hand_calib")
    parser.add_argument("--dict", default="DICT_6X6_50",
                        choices=["DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000"])
    parser.add_argument("--squares-x", type=int, default=11, help="chessboard squares along board width")
    parser.add_argument("--squares-y", type=int, default=8, help="chessboard squares along board height")
    parser.add_argument("--square-length", type=float, default=0.02, help="printed square side in meters")
    parser.add_argument("--marker-length", type=float, default=0.015, help="printed marker side in meters")
    parser.add_argument("--no-legacy", dest="legacy", action="store_false",
                        help="use the OpenCV >=4.6 ChArUco layout instead of the calib.io/legacy one")
    parser.add_argument("--dpi", type=int, default=300, help="print resolution (board mode)")
    args = parser.parse_args()

    if args.mode == "board":
        save_board_image(args)
    elif args.mode == "collect":
        if not args.robot_ip:
            parser.error("--robot-ip is required for collect")
        collect(args)
    else:
        calibrate(args)


if __name__ == "__main__":
    main()
