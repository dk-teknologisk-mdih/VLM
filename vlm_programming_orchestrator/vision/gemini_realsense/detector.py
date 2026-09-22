"""Detector class for object detection and trajectory planning using Gemini Robotics-ER."""

import base64
import json
import os
import re
import time
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
import requests
# Disable SSL warnings for self-signed certificates
import urllib3
import yaml
from PIL import Image, ImageDraw, ImageFont

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class GeminiRoboticsDetector:
    """Detector class for object detection and trajectory planning using Gemini Robotics-ER."""

    def __init__(
        self,
        api_key,
        base_url="https://yoda.teknologisk.dk/public/api-gateway/gemini/",
        exposure=450,
        contrast=0,
        brightness_target=20
    ):
        """
        Initialize the detector with API key and proxy URL.

        Args:
            api_key (str): API key for authentication
            base_url (str): Base URL for the proxy server
            exposure (int): Camera exposure value
            contrast (int): Camera contrast value
            brightness_target (int): Target mean brightness for auto-exposure
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model_id = "gemini-robotics-er-1.6-preview"
        self.exposure = exposure
        self.exposure_time = exposure
        self.contrast = contrast
        self.brightness_target = brightness_target
        self.pipeline, self.config, self.align = self.init_realsense(
            aligned=True)

    def init_realsense(self, aligned=True, resolution=(1280, 720)):
        """Initialize RealSense camera pipeline."""
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color, resolution[0], resolution[1], rs.format.rgb8, 30)
        config.enable_stream(
            rs.stream.depth, resolution[0], resolution[1], rs.format.z16, 30)

        if aligned:
            align_to = rs.stream.color
            align = rs.align(align_to)
        else:
            align = None

        pipeline.start(config)
        sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
        sensor.set_option(rs.option.exposure, self.exposure)
        # sensor.set_option(rs.option.contrast, self.contrast)
        try:
            for _ in range(30):
                pipeline.wait_for_frames()
        except RuntimeError:
            pipeline.stop()
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            time.sleep(2)
            pipeline = rs.pipeline()
            pipeline.start(config)
            print("Device reset")

        return pipeline, config, align

    def aligned_frames_and_images(self):
        """Get aligned color and depth frames."""
        frames = self.pipeline.wait_for_frames(5000)
        aligned_frames = self.align.process(frames) # type: ignore
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Could not acquire depth or color frame.")

        color_intrinsic = color_frame.profile.as_video_stream_profile().intrinsics
        depth_intrinsic = depth_frame.profile.as_video_stream_profile().intrinsics
        return color_frame, depth_frame, color_intrinsic, depth_intrinsic

    def img_point_to_cam_coord_realsense(self, image_point, depth_frame, depth_intrinsic):
        """Transform image point to 3D camera coordinates."""
        depth = depth_frame.get_distance(
            int(image_point[0]), int(image_point[1]))
        # print('Depth ', depth)
        point = rs.rs2_deproject_pixel_to_point(
            depth_intrinsic, [image_point[0], image_point[1]], depth)
        return point, depth

    def auto_exposure(self, brightness_target=20, tolerance=8, white_threshold=200, max_iter=25, flush_frames=15):
        """Automatically adjust camera exposure by metering on non-white pixels.

        Masks out bright/white pixels so they don't skew the measurement.
        Iteratively adjusts exposure until the mean brightness of the remaining
        pixels is within *tolerance* of *brightness_target*.

        Args:
            brightness_target (int): Desired mean brightness of non-white pixels.
            tolerance (int): Acceptable error around the target.
            white_threshold (int): Pixels brighter than this are ignored.
            max_iter (int): Maximum adjustment iterations.
            flush_frames (int): Frames to discard after each exposure change.
        """
        sensor = self.pipeline.get_active_profile(
        ).get_device().query_sensors()[1]

        for iteration in range(max_iter):
            # Flush stale frames so the sensor settles on the new exposure
            for _ in range(flush_frames):
                try:
                    self.pipeline.wait_for_frames(timeout_ms=500)
                except RuntimeError:
                    pass

            # Measure – average over a few frames to reduce noise
            measurements = []
            for _ in range(3):
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=500)
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    color_image = np.asanyarray(color_frame.get_data())
                    gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
                    non_white = gray[gray < white_threshold]
                    if len(non_white) < 100:
                        measurements.append(float(np.mean(gray)))
                    else:
                        measurements.append(float(np.mean(non_white)))
                except RuntimeError:
                    pass

            if not measurements:
                continue

            metered = sum(measurements) / len(measurements)
            error = metered - brightness_target
            print(
                f"Auto-exposure [{iteration}]: masked_mean={metered:.1f}, "
                f"target={brightness_target}, "
                f"exposure={self.exposure}, "
                f"error={error:.1f}"
            )

            if abs(error) <= tolerance:
                print(
                    f"Auto-exposure converged at exposure={self.exposure} (masked_mean={metered:.1f})")
                return

            # Proportional adjustment with damping (60 % towards ideal, 40 % current)
            if metered > 0:
                ideal_exposure = self.exposure * (brightness_target / metered)
                new_exposure = int(self.exposure * 0.4 + ideal_exposure * 0.6)
            else:
                new_exposure = self.exposure + 50

            new_exposure = max(1, min(new_exposure, 10000))
            self.exposure = new_exposure
            sensor.set_option(rs.option.exposure, self.exposure)

        print(
            f"Auto-exposure reached max iterations ({max_iter}), exposure={self.exposure}")

    def capture_realsense_image(self, save_path=None):
        """Capture an image from RealSense camera."""

        self.auto_exposure(brightness_target=self.brightness_target)

        color_frame, depth_frame, _color_intrinsic, depth_intrinsic = self.aligned_frames_and_images()
        color_image = np.asanyarray(color_frame.get_data())
        pil_image = Image.fromarray(color_image)

        if save_path:
            save_path = self.ensure_output_dir(save_path)
            pil_image.save(save_path)
            print(f"Image saved to: {save_path}")

            depth_image = np.asanyarray(depth_frame.get_data())

            depth_txt_path = save_path.rsplit('.', 1)[0] + '_depth.txt'
            np.savetxt(depth_txt_path, depth_image, fmt='%d')
            print(f"Depth array saved to: {depth_txt_path}")

            depth_save_path = save_path.rsplit('.', 1)[0] + '_depth.png'
            plt.figure()
            plt.imshow(depth_image, cmap='viridis', vmin=0, vmax=600)
            plt.colorbar(label='Depth (mm)')
            plt.title('Depth Image Visualization')
            plt.savefig(depth_save_path)
            plt.close()
            print(f"Depth image saved to: {depth_save_path}")

        print("Image captured successfully")
        return pil_image, depth_frame, depth_intrinsic

    def resize_image(self, img_path, max_width=800):
        """Resize image for faster processing."""
        img = Image.open(img_path)
        img = img.resize(
            (max_width, int(max_width * img.size[1] / img.size[0])),
            Image.Resampling.LANCZOS
        )
        return img

    def _image_to_base64(self, img):
        """
        Convert PIL Image to base64 string.

        Args:
            img (PIL.Image): Image to convert

        Returns:
            tuple: (base64_string, mime_type)
        """
        buffer = BytesIO()

        # Convert to RGB if necessary (handles RGBA, P mode, etc.)
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert('RGB')

        # Save as JPEG for efficiency
        img.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)

        base64_string = base64.b64encode(buffer.read()).decode("utf-8")
        return base64_string, "image/jpeg"

    def _load_image_as_base64(self, img_or_path):
        """
        Load an image (from path or PIL.Image) and return base64 encoded data.

        Args:
            img_or_path (str or PIL.Image): Image path or PIL Image object

        Returns:
            tuple: (base64_string, mime_type, PIL.Image)
        """
        if isinstance(img_or_path, str):
            img = self.resize_image(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f"Unsupported image type: {type(img_or_path)}")

        base64_string, mime_type = self._image_to_base64(img)
        return base64_string, mime_type, img

    def parse_yaml(self, yaml_output):
        """Parse YAML output from the model, removing markdown fencing if present."""
        text = yaml_output.strip()

        # Handle markdown code fencing
        if text.startswith("```"):
            match = re.search(r'```(?:yaml|json)?\s*([\s\S]*?)\s*```', text)
            if match:
                text = match.group(1).strip()
        else:
            # Alternative parsing method
            lines = yaml_output.splitlines()
            for i, line in enumerate(lines):
                if line == "```yaml" or line == "```json":
                    text = "\n".join(lines[i + 1:])
                    text = text.split("```", maxsplit=1)[0]
                    break

        # Fix common YAML issues: unquoted strings with colons
        # Quote description values that contain colons
        fixed_lines = []
        for line in text.splitlines():
            # Match lines like "  description: some text with: colons"
            desc_match = re.match(r'^(\s*description:\s*)(.+)$', line)
            if desc_match:
                prefix, value = desc_match.groups()
                # If value contains colon and isn't already quoted
                if ':' in value and not (value.startswith('"') or value.startswith("'")):
                    value = f'"{value}"'
                fixed_lines.append(f"{prefix}{value}")
            else:
                fixed_lines.append(line)

        return "\n".join(fixed_lines)

    def ensure_output_dir(self, filepath=None):
        """Ensure VLM_output directory exists and prepend it to filepath if provided."""
        output_dir = "VLM_output"
        os.makedirs(output_dir, exist_ok=True)
        if filepath:
            return os.path.join(output_dir, os.path.basename(filepath))
        return output_dir

    def detect_objects(self, img, prompt, temperature=0.5, use_thinking=False, save_path=None):
        """
        Detect objects in an image using a custom prompt via the proxy.

        Args:
            img (PIL.Image or str): Image object or path to image
            prompt (str): Detection prompt
            temperature (float): Model temperature (0.0-1.0)
            use_thinking (bool): Enable thinking mode for complex reasoning
            save_path (str, optional): Path to save raw response

        Returns:
            list or dict: Parsed response data
        """
        # Load and encode image
        image_base64, mime_type, _pil_img = self._load_image_as_base64(img)

        # Configure thinking budget
        thinking_budget = -1 if use_thinking else 0

        # Build the request payload
        payload = {
            "contents": [{
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": image_base64,
                        },
                    },
                    {"text": prompt},
                ],
            }],
            "generationConfig": {
                "temperature": temperature,
                "thinkingConfig": {"thinkingBudget": thinking_budget},
            },
        }

        # Make the request to the proxy
        url = f"{self.base_url}/v1beta/models/{self.model_id}:generateContent"
        headers = {
            "x-goog-api-key": f"{self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                verify=False,  # Disable SSL verification for self-signed certs
                timeout=120,   # 2 minute timeout for large images
            )
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return []

        # Check for errors
        if response.status_code != 200:
            print(f"Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return []

        # Parse response
        try:
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Error extracting text from response: {e}")
            print(f"Full response: {response.text[:500]}")
            return []

        # Parse YAML from response
        yaml_output = self.parse_yaml(text)

        try:
            data = yaml.safe_load(yaml_output)
            if save_path:
                save_path = self.ensure_output_dir(save_path)
                with open(save_path, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, default_flow_style=False,
                              sort_keys=False)
                    print(f"Raw response saved to: {save_path}")
            return data
        except yaml.YAMLError as e:
            print(f"Error decoding YAML: {e}")
            print(f"Raw response: {text}")
            return []

    def _check_bbox_overlap(self, bbox1, bbox2):
        """Check if two bounding boxes overlap."""
        return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                    bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

    def _find_non_overlapping_position(self, initial_bbox, occupied_rects, max_attempts=20):
        """Find a non-overlapping position by shifting down."""
        bbox = initial_bbox
        shift_amount = 5

        for _attempt in range(max_attempts):
            overlaps = False
            for occupied in occupied_rects:
                if self._check_bbox_overlap(bbox, occupied):
                    overlaps = True
                    break

            if not overlaps:
                return bbox

            bbox = (bbox[0], bbox[1] + shift_amount,
                    bbox[2], bbox[3] + shift_amount)

        return bbox

    def draw_points_on_image(self, image, points_data, output_path=None):
        """Draw detected points on the image."""
        img = image.copy()
        if isinstance(img, str):
            img = self.resize_image(img)

        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size

        try:
            font = ImageFont.truetype("arial.ttf", size=20)
        except Exception:
            font = ImageFont.load_default()

        # Map label keywords to display colors
        label_color_map = {
            "red": "red",
            "green": "green",
            "blue": "blue",
            "dark blue": "darkblue",
            "light blue": "deepskyblue",
            "yellow": "goldenrod",
            "black": "black",
            "orange": "orange",
            "white": "gray",
            "brown": "brown",
            "pink": "deeppink",
            "purple": "purple",
            "cyan": "cyan",
            "target": "crimson",
        }

        def _color_from_label(label):
            """Derive a display color from the label text."""
            label_lower = label.lower()
            # Check longest keys first to match "dark blue" before "blue"
            for key in sorted(label_color_map, key=len, reverse=True):
                if key in label_lower:
                    return label_color_map[key]
            return "gray"

        occupied_rects = []

        for _, point_info in enumerate(points_data):
            if "point" in point_info and "label" in point_info:
                y_norm, x_norm, _ = point_info["point"]
                label = point_info["label"]

                abs_x = int(x_norm / 1000.0 * width)
                abs_y = int(y_norm / 1000.0 * height)

                color = _color_from_label(label)

                point_radius = 4
                draw.ellipse(
                    (abs_x - point_radius, abs_y - point_radius,
                     abs_x + point_radius, abs_y + point_radius),
                    fill=color, outline="white", width=2
                )

                label_pos_x = abs_x + point_radius + 5
                label_pos_y = abs_y - point_radius - 5 if abs_y > 20 else abs_y + point_radius + 5

                initial_bbox = draw.textbbox(
                    (label_pos_x, label_pos_y), label, font=font)
                final_bbox = self._find_non_overlapping_position(
                    initial_bbox, occupied_rects)

                final_label_pos_x = final_bbox[0]
                final_label_pos_y = final_bbox[1]

                draw.rectangle(final_bbox, fill="white",
                               outline=color, width=2)
                draw.text((final_label_pos_x, final_label_pos_y),
                          label, fill=color, font=font)

                occupied_rects.append(final_bbox)

        if output_path:
            output_path = self.ensure_output_dir(output_path)
            img.save(output_path)
            print(f"Annotated image saved to: {output_path}")

        return img
