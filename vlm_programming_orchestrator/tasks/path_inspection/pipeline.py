"""Pipeline for the path-inspection task: detect an object, plan an ordered
inspection path along it.

The target object (default "bike frame") is fully configurable via
`gui_config["target_object"]` — swap it to inspect any other object without
touching code.
"""

import json
import os

import numpy as np
from PIL import Image

from ...vision.registry import get_detector_backend
from ..common import add_depth_to_detections, convert_plan_to_3d, load_prompt_template

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_CALIB_FILE = os.path.join(os.path.dirname(__file__), "T_base_cam.json")
_DEFAULT_TARGET_OBJECT = "bike frame (not wheels, handlebars, or seat)"


def plan_inspection_path(
    api_key,
    base_url,
    target_object=_DEFAULT_TARGET_OBJECT,
    config=None,
    vision_backend="gemini_realsense",
    vlm_model="gemini-robotics-er-2-preview",
    vlm_prompt_file=None,
    image_path=None,
):
    """Detect `target_object` and plan an ordered 3D inspection path along it.

    Args:
        api_key (str): API key for the VLM proxy.
        base_url (str): Base URL for the proxy server.
        target_object (str): Description of the object to inspect (e.g. "bike frame").
        config (dict, optional): GUI/config dict; `target_object` and camera
            settings (`exposure`, `contrast`, `brightness_target`) are read from it.
        vision_backend (str): Name of the registered vision/detector backend to use.
        vlm_prompt_file (str or Path, optional): Overrides the bundled default
            VLM detection prompt template.
        image_path (str or Path, optional): Image sent to the VLM instead of
            the color image captured from RealSense.

    Returns:
        dict or None: Contains `plan_data` (ordered 3D waypoints), `detections`,
        and `target_object`; None if no waypoints were detected.
    """
    if config and config.get('target_object'):
        target_object = config['target_object']

    exposure = config.get('exposure', 800) if config else 2500
    contrast = config.get('contrast', 70) if config else 70
    brightness_target = config.get('brightness_target', 20) if config else 20

    detector = get_detector_backend(
        vision_backend, api_key,
        exposure=exposure,
        contrast=contrast,
        brightness_target=brightness_target,
        model_id=vlm_model,
        use_realsense=image_path is None,
    )
    if image_path:
        image = Image.open(image_path).convert("RGB")
        depth_frame = None
        depth_intrinsic = None
    else:
        image, depth_frame, depth_intrinsic = detector.capture_realsense_image(
            "0_raw_realsense_capture.png")
    image_size = image.size
    depth_image = (
        np.asanyarray(depth_frame.get_data()) if depth_frame is not None else None
    )

    prompt_path = vlm_prompt_file or os.path.join(
        _PROMPTS_DIR, "vlm_detect_path.txt")
    detect_prompt = load_prompt_template(
        prompt_path, target_object=target_object)

    waypoints_pixel = detector.detect_objects(
        image, detect_prompt, save_path="1_inspection_path_pixel.yaml")
    if not waypoints_pixel:
        print(f"Warning: could not detect any waypoints for {target_object}")
        return None

    add_depth_to_detections(
        waypoints_pixel, depth_image, image_size,
        hardcoded=True)
    detector.draw_points_on_image(
        image, waypoints_pixel, output_path="1_inspection_path.png")

    # Wrap as a single "step" so it reuses the generic step/trajectory/waypoint
    # schema shared with other tasks (see tasks/common.py::convert_plan_to_3d).
    plan_pixel = [{
        "step": 1,
        "action": "inspect",
        "trajectory": [
            {"point": wp["point"], "label": wp.get("label", "inspect_point")}
            for wp in waypoints_pixel
        ],
    }]
    with open(_CALIB_FILE, encoding="utf-8") as f:
        intrinsics = json.load(f)["camera_intrinsics"]
    plan_data = convert_plan_to_3d(
        detector, plan_pixel, image_size, depth_frame, depth_intrinsic,
        intrinsics=intrinsics)

    print(
        f"\n=== Summary ===\nPlanned inspection path with {len(waypoints_pixel)} "
        f"waypoints for {target_object}"
    )

    return {
        "plan_data": plan_data,
        "detections": waypoints_pixel,
        "target_object": target_object,
        "image_size": image_size,
    }
