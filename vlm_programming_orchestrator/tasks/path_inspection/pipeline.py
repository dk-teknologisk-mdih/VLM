"""Pipeline for the path-inspection task: detect an object, plan an ordered
inspection path along it.

The target object (default "bike frame") is fully configurable via
`gui_config["target_object"]` — swap it to inspect any other object without
touching code.
"""

from copy import deepcopy
import json
import logging
import os

import numpy as np
import yaml
from PIL import Image

from ...vision.registry import get_detector_backend
from ..common import add_depth_to_detections, convert_plan_to_3d, load_prompt_template

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_CALIB_FILE = os.path.join(os.path.dirname(__file__), "T_base_cam.json")
_DEFAULT_TARGET_OBJECT = "bike frame (not wheels, handlebars, or seat)"
logger = logging.getLogger("orchestrator.path_inspection")


class WaypointVerificationError(RuntimeError):
    """The inspection waypoints could not be verified by the VLM."""


class WaypointOrderingError(RuntimeError):
    """The VLM could not provide a valid continuous tube traversal."""


def _validate_attempt_limit(max_attempts):
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("vlm_verification_max_attempts must be a positive integer")


def _validate_waypoints(waypoints):
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError("waypoints must be a nonempty list")
    for index, waypoint in enumerate(waypoints, start=1):
        if not isinstance(waypoint, dict):
            raise ValueError(f"Waypoint {index} must be a mapping")
        point = waypoint.get("point")
        if (
            not isinstance(point, list)
            or len(point) != 2
            or not all(
                type(coordinate) in (int, float) and 0 <= coordinate <= 1000
                for coordinate in point
            )
        ):
            raise ValueError(
                f"Waypoint {index} must have two finite [y, x] coordinates in 0-1000")
        label = waypoint.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Waypoint {index} must have a nonempty string label")


def verify_inspection_waypoints(detector, image, waypoints, detect_prompt, max_attempts=3):
    """Review the latest valid 2D waypoints until accepted, or fail closed."""
    _validate_attempt_limit(max_attempts)
    try:
        _validate_waypoints(waypoints)
    except ValueError as error:
        raise WaypointVerificationError(f"Invalid detected waypoints: {error}") from error

    prompt_path = os.path.join(_PROMPTS_DIR, "vlm_verify_path.txt")
    feedback = "None; this is the first verification attempt."
    detector.draw_points_on_image(
        image, waypoints, output_path="1_inspection_path.png")
    for attempt in range(1, max_attempts + 1):
        prompt = load_prompt_template(
            prompt_path,
            detection_criteria=detect_prompt,
            waypoints_yaml=yaml.safe_dump(waypoints, sort_keys=False),
            previous_feedback=feedback,
        )
        logger.info("Verifying inspection waypoints: attempt %s/%s", attempt, max_attempts)
        response = detector.detect_objects(
            image, prompt,
            save_path=f"1_inspection_path_verification_{attempt}.yaml",
        )
        path_accepted = False
        try:
            if not isinstance(response, dict):
                raise ValueError("Verification response must be a YAML mapping")
            accepted = response.get("accepted")
            if type(accepted) is not bool:
                raise ValueError("accepted must be a YAML boolean (true or false)")
            reason = response.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("reason must be a nonempty string")
            reviewed_waypoints = response.get("waypoints")
            _validate_waypoints(reviewed_waypoints)
        except ValueError as error:
            feedback = f"Invalid verification response: {error}. Return the required schema."
            logger.warning("%s", feedback)
        else:
            logger.info("Waypoint review: accepted=%s; %s", accepted, reason)
            path_accepted = accepted and reviewed_waypoints == waypoints
            feedback = reason
            if accepted and not path_accepted:
                feedback += " You changed the waypoints; verify these corrections before accepting."
            waypoints = reviewed_waypoints

        detector.draw_points_on_image(
            image, waypoints,
            output_path=f"1_inspection_path_verification_{attempt}.png")
        detector.draw_points_on_image(
            image, waypoints, output_path="1_inspection_path.png")
        if path_accepted:
            logger.info("Inspection waypoints accepted on verification attempt %s", attempt)
            return waypoints

    raise WaypointVerificationError(
        f"Inspection waypoints were not accepted after {max_attempts} verification "
        f"attempt(s). Last feedback: {feedback}")


def order_inspection_waypoints(detector, image, waypoints, detect_prompt):
    """Ask the VLM for a tube-covering order; validate without calculating a route."""
    try:
        _validate_waypoints(waypoints)
    except ValueError as error:
        raise WaypointOrderingError(f"Invalid verified waypoints: {error}") from error

    indexed_waypoints = [
        {"id": index, "point": waypoint["point"], "label": waypoint["label"]}
        for index, waypoint in enumerate(waypoints, start=1)
    ]
    prompt = load_prompt_template(
        os.path.join(_PROMPTS_DIR, "vlm_order_path.txt"),
        detection_criteria=detect_prompt,
        waypoints_yaml=yaml.safe_dump(indexed_waypoints, sort_keys=False),
    )
    logger.info("Requesting a VLM tube-following inspection order")
    response = detector.detect_objects(
        image, prompt, save_path="1_inspection_path_ordering.yaml")
    try:
        if not isinstance(response, dict):
            raise ValueError("Ordering response must be a YAML mapping")
        routable = response.get("routable")
        if type(routable) is not bool:
            raise ValueError("routable must be a YAML boolean (true or false)")
        reason = response.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a nonempty string")
        if not routable:
            raise ValueError(f"VLM could not find a continuous tube traversal: {reason}")

        valid_ids = set(range(1, len(waypoints) + 1))
        order = response.get("waypoint_order")
        if (
            not isinstance(order, list) or not order
            or any(type(point_id) is not int or point_id not in valid_ids
                   for point_id in order)
        ):
            raise ValueError("waypoint_order must be a nonempty list of valid integer IDs")
        if set(order) != valid_ids:
            raise ValueError("waypoint_order must visit every verified waypoint")

        segments = response.get("tube_segments")
        if not isinstance(segments, list):
            raise ValueError("tube_segments must be a list of waypoint ID pairs")
        declared_segments = set()
        for segment in segments:
            if (
                not isinstance(segment, list) or len(segment) != 2
                or any(type(point_id) is not int or point_id not in valid_ids
                       for point_id in segment)
                or segment[0] == segment[1]
            ):
                raise ValueError("Each tube segment must contain two distinct valid integer IDs")
            edge = frozenset(segment)
            if edge in declared_segments:
                raise ValueError("tube_segments must not contain duplicate undirected segments")
            declared_segments.add(edge)

        traversed_segments = {
            frozenset((start_id, end_id))
            for start_id, end_id in zip(order, order[1:])
        }
        if not traversed_segments <= declared_segments:
            raise ValueError("waypoint_order contains a move outside the declared tube segments")
        if traversed_segments != declared_segments:
            raise ValueError("waypoint_order must traverse every declared tube segment")
    except ValueError as error:
        raise WaypointOrderingError(f"Invalid inspection ordering: {error}") from error

    logger.info("VLM inspection order: %s visits; %s", len(order), reason)
    return [deepcopy(waypoints[point_id - 1]) for point_id in order]


def plan_inspection_path(
    api_key,
    base_url,
    target_object=_DEFAULT_TARGET_OBJECT,
    config=None,
    vision_backend="gemini_realsense",
    vlm_model="gemini-robotics-er-2-preview",
    vlm_prompt_file=None,
    image_path=None,
    vlm_verification_max_attempts=3,
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
        vlm_verification_max_attempts (int): Maximum review calls after detection.

    Returns:
        dict or None: Contains `plan_data` (ordered 3D waypoints), `detections`,
        and `target_object`; None if no waypoints were detected.

    Raises:
        WaypointVerificationError: The detected waypoints were not accepted.
        WaypointOrderingError: The VLM did not provide a valid tube traversal.
    """
    _validate_attempt_limit(vlm_verification_max_attempts)
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
        image, detect_prompt, save_path="1_inspection_path_initial.yaml")
    if not waypoints_pixel:
        print(f"Warning: could not detect any waypoints for {target_object}")
        return None

    waypoints_pixel = verify_inspection_waypoints(
        detector, image, waypoints_pixel, detect_prompt,
        max_attempts=vlm_verification_max_attempts)
    waypoints_pixel = order_inspection_waypoints(
        detector, image, waypoints_pixel, detect_prompt)
    detector.draw_points_on_image(
        image, waypoints_pixel, output_path="1_inspection_path.png")
    os.makedirs("VLM_output", exist_ok=True)
    with open("VLM_output/1_inspection_path_pixel.yaml", "w", encoding="utf-8") as output:
        yaml.safe_dump(waypoints_pixel, output, sort_keys=False)

    add_depth_to_detections(
        waypoints_pixel, depth_image, image_size,
        hardcoded=True)

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
