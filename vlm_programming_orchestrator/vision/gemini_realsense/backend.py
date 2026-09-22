"""RealSense camera + Gemini Robotics-ER object detector backend."""

from .detector import GeminiRoboticsDetector

VISION_BACKEND = "gemini_realsense"


def create_backend(api_key, **kwargs) -> GeminiRoboticsDetector:
    """Build the Gemini Robotics-ER + RealSense `ObjectDetector`."""
    return GeminiRoboticsDetector(api_key, **kwargs)
