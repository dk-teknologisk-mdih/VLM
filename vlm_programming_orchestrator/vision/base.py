"""Structural interface for camera capture + VLM-based object detection."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ObjectDetector(Protocol):
    """Captures a scene and detects/points at objects in it via a VLM."""

    def capture_realsense_image(self, save_path=None):
        """Returns (image, depth_frame, depth_intrinsic)."""
        ...

    def detect_objects(self, img, prompt, temperature: float = 0.5,
                       use_thinking: bool = False, save_path=None):
        """Returns parsed point/label detections for the given prompt."""
        ...

    def draw_points_on_image(self, image, points_data, output_path=None):
        """Annotates and optionally saves an image with detected points."""
        ...
