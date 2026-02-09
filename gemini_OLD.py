"""
Gemini Robotics Object Detection Script

This script allows you to detect objects in images using Gemini Robotics-ER 1.5 model
and visualize the detected points on the image.
"""

import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types
import numpy as np
import os
import time

import pyrealsense2 as rs


class GeminiRoboticsDetector:
    """Main class for object detection using Gemini Robotics-ER model."""
    
    def __init__(self, api_key):
        """
        Initialize the detector with API key.
        
        Args:
            api_key (str): Google API key for Gemini
        """
        os.environ["ALL_PROXY"] = "http://squid2.localdom.net:3128"
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-robotics-er-1.5-preview"
        
        # Initialize RealSense camera 
        self.pipeline, self.config, self.align = self.init_realsense(aligned=True)
        self.depth_frame, self.color_frame, self.color_image, self.depth_image, self.color_intrinsic, self.depth_intrinsic = None, None, None, None, None, None
        
    def init_realsense(self, aligned=True, resolution=(1280, 720)):
        # Create a pipeline for realsense camera
        pipeline = rs.pipeline()
        # Create a config object
        config = rs.config()

        # Enable the color and depth streams frm the camera
        config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, 30)
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        if aligned:
            align_to = rs.stream.color
            align = rs.align(align_to)
        else:
            align = None

        
        # Start the pipeline
        pipeline.start(config)
        try:
            # Wait for camera to warm up and get stable frames
            for _ in range(30):
                pipeline.wait_for_frames()
                
        except RuntimeError:
            #Reset device and try again

            pipeline.stop()
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()

            time.sleep(2)
            pipeline = rs.pipeline()
            pipeline.start(config)
            self.camera_pipeline_running = True
            print("Device reset")

        return pipeline, config, align
    
    def aligned_frames_and_images(self):
        # Get frameset of color and depth
        self.frames = self.pipeline.wait_for_frames(5000)

        # Align the depth frame to color frame
        aligned_frames = self.align.process(self.frames)

        # Get aligned frames
        self.depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a self.camera_width x self.camera_height depth image
        self.color_frame = aligned_frames.get_color_frame()
        
        if not self.depth_frame or not self.color_frame:
            raise RuntimeError("Could not acquire depth or color frame.")
        
        # Convert images to numpy arrays
        self.color_image = np.asanyarray(self.color_frame.get_data())
        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        
        # Get realsense camera intrinsic
        self.color_intrinsic = self.color_frame.profile.as_video_stream_profile().intrinsics
        self.depth_intrinsic = self.depth_frame.profile.as_video_stream_profile().intrinsics
        
        # self.depth_intrinsic = self.intrinsic_from_calibration()
        # print("depth intrinsic ", self.depth_intrinsic)
        # print("color intrinsic ", self.color_intrinsic)
        
    def img_point_to_cam_coord_realsense(self, image_point, depth_frame):
        # Calculate the depth of the image point
        depth = depth_frame.get_distance(int(image_point[0]), int(image_point[1]))
        # print('Depth ', depth)
        # Transform the image point to 3D point in camera coordinates
        point = rs.rs2_deproject_pixel_to_point(self.depth_intrinsic, [image_point[0], image_point[1]], depth)
        return point, depth
    
    def capture_realsense_image(self, save_path=None):
        """
        Capture an image from RealSense camera.
        
        Args:
            save_path (str, optional): Path to save the captured image
            resolution (tuple): Resolution (width, height) for capture
        """
        # Get aligned frames and images
        self.aligned_frames_and_images()
        
        # Convert to PIL Image
        pil_image = Image.fromarray(self.color_image)
        
        # Save if path provided
        if save_path:
            pil_image.save(save_path)
            print(f"Image saved to: {save_path}")
        
        print("Image captured successfully")
        return pil_image, self.depth_image
        
    def resize_image(self, img_path, max_width=800):
        """
        Resize image for faster processing.
        
        Args:
            img_path (str): Path to the image file
            max_width (int): Maximum width for resizing
            
        Returns:
            PIL.Image: Resized image
        """
        img = Image.open(img_path)
        img = img.resize(
            (max_width, int(max_width * img.size[1] / img.size[0])), 
            Image.Resampling.LANCZOS
        )
        return img
    
    def parse_json(self, json_output):
        """
        Parse JSON output from the model, removing markdown fencing if present.
        
        Args:
            json_output (str): Raw JSON output from model
            
        Returns:
            str: Cleaned JSON string
        """
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i + 1:])
                json_output = json_output.split("```")[0]
                break
        return json_output
    
    def detect_objects(self, img, prompt, temperature=0.5, use_thinking=False):
        """
        Detect objects in an image using a custom prompt.
        
        Args:
            img (PIL.Image or str): Image object or path to image
            prompt (str): Detection prompt
            temperature (float): Model temperature (0.0-1.0)
            use_thinking (bool): Enable thinking mode for complex reasoning
            
        Returns:
            list: List of detected objects with coordinates and labels
        """
        # Load image if path is provided
        if isinstance(img, str):
            img = self.resize_image(img)
        
        # Configure thinking budget
        thinking_budget = -1 if use_thinking else 0
        
        # Generate content
        config = types.GenerateContentConfig(
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
        )
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[img, prompt],
            config=config,
        )
        
        # Parse response
        json_output = self.parse_json(response.text)
        
        try:
            data = json.loads(json_output)
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Raw response: {response.text}")
            return []
    
    def draw_points_on_image(self, img, points_data, output_path=None):
        """
        Draw detected points on the image.
        
        Args:
            img (PIL.Image or str): Image object or path to image
            points_data (list): List of point dictionaries with 'point' and 'label' keys
            output_path (str, optional): Path to save annotated image
            
        Returns:
            PIL.Image: Annotated image
        """
        # Load image if path is provided
        if isinstance(img, str):
            img = self.resize_image(img)
        
        # Convert to RGB for drawing
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", size=14)
        except:
            font = ImageFont.load_default()
        
        # Define colors
        colors = [
            "red", "green", "blue", "yellow", "orange", "pink", "purple",
            "brown", "cyan", "magenta", "lime", "navy", "teal", "gold"
        ]
        
        # Draw each point
        for i, point_info in enumerate(points_data):
            if "point" in point_info and "label" in point_info:
                y_norm, x_norm, _ = point_info["point"]
                label = point_info["label"]
                
                # Convert normalized coordinates to absolute pixel coordinates
                abs_x = int(x_norm / 1000.0 * width)
                abs_y = int(y_norm / 1000.0 * height)
                
                # Select color
                color = colors[i % len(colors)]
                
                # Draw a circle at the point
                point_radius = 4
                draw.ellipse(
                    (
                        abs_x - point_radius,
                        abs_y - point_radius,
                        abs_x + point_radius,
                        abs_y + point_radius,
                    ),
                    fill=color,
                    outline="white",
                    width=2
                )
                
                # Draw the label
                label_pos_x = abs_x + point_radius + 5
                label_pos_y = abs_y - point_radius - 5 if abs_y > 20 else abs_y + point_radius + 5
                
                # Draw text background for better readability
                bbox = draw.textbbox((label_pos_x, label_pos_y), label, font=font)
                draw.rectangle(bbox, fill="white", outline=color, width=2)
                draw.text((label_pos_x, label_pos_y), label, fill=color, font=font)
        
        # Save if output path is provided
        if output_path:
            img.save(output_path)
            print(f"Annotated image saved to: {output_path}")
        
        return img
    
    def draw_bounding_boxes(self, img, bounding_boxes_data, output_path=None):
        """
        Draw bounding boxes on the image.
        
        Args:
            img (PIL.Image or str): Image object or path to image
            bounding_boxes_data (list): List of bounding box dictionaries
            output_path (str, optional): Path to save annotated image
            
        Returns:
            PIL.Image: Annotated image
        """
        # Load image if path is provided
        if isinstance(img, str):
            img = self.resize_image(img)
        
        width, height = img.size
        draw = ImageDraw.Draw(img)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", size=14)
        except:
            font = ImageFont.load_default()
        
        # Define colors
        colors = [
            "red", "green", "blue", "yellow", "orange", "pink", "purple",
            "brown", "cyan", "magenta", "lime", "navy", "teal", "gold"
        ]
        
        # Draw each bounding box
        for i, bbox in enumerate(bounding_boxes_data):
            color = colors[i % len(colors)]
            
            # Convert normalized coordinates to absolute coordinates
            abs_y1 = int(bbox["box_2d"][0] / 1000 * height)
            abs_x1 = int(bbox["box_2d"][1] / 1000 * width)
            abs_y2 = int(bbox["box_2d"][2] / 1000 * height)
            abs_x2 = int(bbox["box_2d"][3] / 1000 * width)
            
            # Ensure coordinates are in correct order
            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1
            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1
            
            # Draw the bounding box
            draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)
            
            # Draw the label
            if "label" in bbox:
                draw.text((abs_x1 + 8, abs_y1 + 6), bbox["label"], fill=color, font=font)
        
        # Save if output path is provided
        if output_path:
            img.save(output_path)
            print(f"Annotated image saved to: {output_path}")
        
        return img


def detect_all_objects_with_points(image_path, api_key, max_objects=10, output_path=None):
    """
    Detect all objects in an image and return points.
    
    Args:
        image_path (str): Path to the image file
        api_key (str): Google API key for Gemini
        max_objects (int): Maximum number of objects to detect
        output_path (str, optional): Path to save annotated image
        
    Returns:
        tuple: (list of detected objects, annotated PIL.Image)
    """
    detector = GeminiRoboticsDetector(api_key)
    
    prompt = f"""Point to no more than {max_objects} items in the image. The label returned should be an
identifying name for the object detected.

The answer should follow the JSON format:
[{{"point": <point>, "label": <label1>}}, ...]

The points are in [y, x] format normalized to 0-1000."""
    
    points = detector.detect_objects(image_path, prompt)
    annotated_img = detector.draw_points_on_image(image_path, points, output_path)
    
    return points, annotated_img


def detect_specific_objects_with_points(image_path, api_key, object_names, output_path=None):
    """
    Detect specific objects in an image and return points.
    
    Args:
        image_path (str): Path to the image file
        api_key (str): Google API key for Gemini
        object_names (list or str): List of object names to detect, or comma-separated string
        output_path (str, optional): Path to save annotated image
        
    Returns:
        tuple: (list of detected objects, annotated PIL.Image)
    """
    detector = GeminiRoboticsDetector(api_key)
    
    # Handle both list and string input
    if isinstance(object_names, list):
        objects_str = ", ".join(object_names)
    else:
        objects_str = object_names
    
    prompt = f"""Get all points matching the following objects: {objects_str}. 
The label returned should be an identifying name for the object detected.

The answer should follow the JSON format:
[{{"point": <point>, "label": <label1>}}, ...]

The points are in [y, x] format normalized to 0-1000."""
    
    points = detector.detect_objects(image_path, prompt)
    annotated_img = detector.draw_points_on_image(image_path, points, output_path)
    
    return points, annotated_img


def detect_objects_with_bounding_boxes(image_path, api_key, max_objects=10, output_path=None):
    """
    Detect objects in an image and return bounding boxes.
    
    Args:
        image_path (str): Path to the image file
        api_key (str): Google API key for Gemini
        max_objects (int): Maximum number of objects to detect
        output_path (str, optional): Path to save annotated image
        
    Returns:
        tuple: (list of detected objects with bounding boxes, annotated PIL.Image)
    """
    detector = GeminiRoboticsDetector(api_key)
    
    prompt = f"""Return bounding boxes as a JSON array with labels. Never return masks or
code fencing. Limit to {max_objects} objects. Include objects you can identify in the image.

The format should be as follows:
[{{"box_2d": [ymin, xmin, ymax, xmax], "label": <label for the object>}}]
normalized to 0-1000. The values in box_2d must only be integers."""
    
    bboxes = detector.detect_objects(image_path, prompt)
    annotated_img = detector.draw_bounding_boxes(image_path, bboxes, output_path)
    
    return bboxes, annotated_img


def detect_specific_objects_with_bounding_boxes(image_path, api_key, object_names, output_path=None):
    """
    Detect specific objects in an image and return bounding boxes.
    
    Args:
        image_path (str): Path to the image file
        api_key (str): Google API key for Gemini
        object_names (list or str): List of object names to detect, or comma-separated string
        output_path (str, optional): Path to save annotated image
        
    Returns:
        tuple: (list of detected objects with bounding boxes, annotated PIL.Image)
    """
    detector = GeminiRoboticsDetector(api_key)
    
    # Handle both list and string input
    if isinstance(object_names, list):
        objects_str = ", ".join(object_names)
    else:
        objects_str = object_names
    
    prompt = f"""Return bounding boxes as a JSON array with labels for the following specific objects: {objects_str}.
Never return masks or code fencing. Only include the requested objects if they are present in the image.

The format should be as follows:
[{{"box_2d": [ymin, xmin, ymax, xmax], "label": <label for the object>}}]
normalized to 0-1000. The values in box_2d must only be integers."""
    
    bboxes = detector.detect_objects(image_path, prompt)
    annotated_img = detector.draw_bounding_boxes(image_path, bboxes, output_path)
    
    return bboxes, annotated_img


def detect_from_realsense(api_key, detection_type="points", object_names=None, 
                          max_objects=10, output_path=None, save_capture=None):
    """
    Capture image from RealSense camera and detect objects.
    
    Args:
        api_key (str): Google API key for Gemini
        detection_type (str): Type of detection - "points", "bboxes", "specific_points", "specific_bboxes"
        object_names (list or str, optional): Specific objects to detect (required for specific detection)
        max_objects (int): Maximum number of objects to detect
        output_path (str, optional): Path to save annotated image
        save_capture (str, optional): Path to save the raw captured image
        
    Returns:
        tuple: (list of detected objects, annotated PIL.Image, raw captured image)
    """
    # Capture image from RealSense
    captured_image = capture_realsense_image(save_path=save_capture)
    
    # Create temporary path for processing
    temp_path = "temp_realsense_capture.png"
    captured_image.save(temp_path)
    
    # Detect based on type
    if detection_type == "points":
        results, annotated_img = detect_all_objects_with_points(
            temp_path, api_key, max_objects=max_objects, output_path=output_path
        )
    elif detection_type == "bboxes":
        results, annotated_img = detect_objects_with_bounding_boxes(
            temp_path, api_key, max_objects=max_objects, output_path=output_path
        )
    elif detection_type == "specific_points":
        if object_names is None:
            raise ValueError("object_names must be provided for specific detection")
        results, annotated_img = detect_specific_objects_with_points(
            temp_path, api_key, object_names, output_path=output_path
        )
    elif detection_type == "specific_bboxes":
        if object_names is None:
            raise ValueError("object_names must be provided for specific detection")
        results, annotated_img = detect_specific_objects_with_bounding_boxes(
            temp_path, api_key, object_names, output_path=output_path
        )
    else:
        raise ValueError(f"Invalid detection_type: {detection_type}")
    
    return results, annotated_img, captured_image


def plan_stacking_trajectory(api_key, object_to_stack="red block", where_to_stack="first block", output_path:str | None = None):
    """
    Plan a trajectory for stacking objects on top of each other.
    
    Args:
        image_path (str or PIL.Image): Path to image or PIL Image object
        api_key (str): Google API key for Gemini
        object_to_stack (str): Name of the objects to stack (e.g., "red block")
        num_waypoints (int): Number of waypoints for each pick-and-place trajectory
        output_path (str, optional): Path to save annotated image
        
    Returns:
        dict: Dictionary containing:
            - 'blocks': List of detected block locations
            - 'stacking_plan': List of steps with trajectories
            - 'annotated_image': PIL Image with trajectory visualization
    """
    detector = GeminiRoboticsDetector(api_key)
    
    image, depth_map = detector.capture_realsense_image()
    image_path = "temp_realsense_capture.png"
    image.save(image_path)
    
    # First, detect all blocks
    detect_prompt = f"""Detect all {object_to_stack}s in the image and return their locations.

The answer should follow the JSON format:
[{{"point": <point>, "label": <label>}}]

The points are in [y, x] format normalized to 0-1000."""
    
    blocks = detector.detect_objects(image_path, detect_prompt)
    
    # if len(blocks) < 2:
    #     print(f"Warning: Only found {len(blocks)} block(s). Need at least 2 for stacking.")
    #     return {'blocks': blocks, 'stacking_plan': [], 'annotated_image': None}
    
    
    
    # get the height of each block from the realsense ir depth map
    for block in blocks:
        y_norm, x_norm = block["point"]
        width, height = image.size
        abs_x = int(x_norm / 1000.0 * width)
        abs_y = int(y_norm / 1000.0 * height)
        depth_value = depth_map[abs_y, abs_x]  # depth map is in mm
        print(f"Block {block['label']} at pixel ({abs_x}, {abs_y}) has depth {depth_value} mm")
        block['point'].extend([int(depth_value)])
    
    detector.draw_points_on_image(image_path, blocks, "part1" + output_path)
    
    # Now plan the stacking trajectory
    trajectory_prompt = f"""You are controlling a robot arm to stack {len(blocks)} {object_to_stack}s on top of each other.

The blocks are located at these positions:
{chr(10).join([f"Block {i+1}: {block['point']}" for i, block in enumerate(blocks)])}

Plan a stacking operation where you:
1. Pick up each block one by one
2. Stack them all at the location of {where_to_stack}
3. For each pick operation, provide waypoints from the gripper starting position to the block 
4. For each place operation, provide waypoints from the picked block to the stacking location
5. Any time a height change occurs (picking up or placing down), include a single waypoint above the target location to avoid collisions

Return a JSON array with the stacking plan:
[
  {{
    "step": 1,
    "action": "pick",
    "block": "Block 2",
    "trajectory": [{{"point": [x, y, z], "label": "waypoint_1"}}, ...],
    "description": "Move to Block 2 and pick it up"
  }},
  {{
    "step": 2,
    "action": "place",
    "block": "Block 2",
    "trajectory": [{{"point": [x, y, z], "label": "waypoint_1"}}, ...],
    "description": "Move Block 2 to stack location and place it on Block 1"
  }},
  ...
]

The points are in [x, y, z] format normalized to 0-1000.
Label waypoints sequentially for each trajectory segment."""
    
    stacking_plan = detector.detect_objects(image_path, trajectory_prompt)
    
    # Visualize the complete trajectory
    all_waypoints = []
    for step in stacking_plan:
        if 'trajectory' in step:
            all_waypoints.extend(step['trajectory'])
    
    # Add the detected blocks to waypoints
    all_waypoints.extend(blocks)
    
    annotated_img = detector.draw_points_on_image(image_path, all_waypoints, output_path)
    
    return {
        'blocks': blocks,
        'stacking_plan': stacking_plan,
        'annotated_image': annotated_img,
        "raw_plan": stacking_plan
    }


def plan_stacking_from_realsense(api_key, object_to_stack="red block", where_to_stack="first block",
                                 num_waypoints=15, output_path=None, save_capture=None):
    """
    Capture image from RealSense camera and plan stacking trajectory.
    
    Args:
        api_key (str): Google API key for Gemini
        object_to_stack (str): Name of the objects to stack
        num_waypoints (int): Number of waypoints for trajectories
        output_path (str, optional): Path to save annotated image
        save_capture (str, optional): Path to save the raw captured image
        
    Returns:
        dict: Dictionary with blocks, stacking_plan, annotated_image, and raw_image
    """
    # Capture image from RealSense
    captured_image = capture_realsense_image(save_path=save_capture)
    
    # Create temporary path for processing
    temp_path = "temp_realsense_capture.png"
    captured_image.save(temp_path)
    
    # Plan stacking trajectory
    result = plan_stacking_trajectory(
        temp_path, api_key, object_to_stack, num_waypoints, output_path
    )
    
    result['raw_image'] = captured_image
    return result


def main():
    """Example usage of the generic detection functions."""
    
    API_KEY = "AIzaSyDlGRUwor17QRWdbTFds_Gg_5opCjr8c4g"  # Replace with your actual API key
    image_path = "red_blocks.JPG"  # Replace with your image path
    
    # # Example with RealSense camera
    # print("=== RealSense Camera Example ===")
    # try:
    #     # Capture and detect specific objects with bounding boxes
    #     results, annotated_img, raw_img = detect_from_realsense(
    #         API_KEY,
    #         detection_type="specific_bboxes",
    #         object_names=["flag", "birthday boy"],
    #         output_path="realsense_detection.png",
    #         save_capture="realsense_raw.png"
    #     )
    #     print(f"Detected {len(results)} objects from RealSense camera")
    #     for item in results:
    #         print(f"  - {item.get('label', 'unknown')}: {item.get('box_2d', item.get('point', []))}")
    # except Exception as e:
    #     print(f"RealSense example failed: {e}")
    #     print("Falling back to file-based examples...")
    
    # print()
    
    # Example 5: Plan stacking trajectory
    print("=== Example 5: Planning Stacking Trajectory ===")
    result = plan_stacking_trajectory(
        API_KEY,
        object_to_stack="red block",
        where_to_stack="the middle of the white block",
        output_path="stacking_trajectory.png"
    )
    print(f"Found {len(result['blocks'])} blocks to stack")
    print(f"Generated {len(result['stacking_plan'])} steps in stacking plan:")
    for step in result['stacking_plan']:
        print(f"  Step {step.get('step', '?')}: {step.get('action', 'unknown')} - {step.get('description', '')}")
        if 'trajectory' in step:
            print(f"    Trajectory has {len(step['trajectory'])} waypoints")
    
    with open("stacking_plan.json", "w") as f:
        json.dump(result["raw_plan"], f, indent=2)


if __name__ == "__main__":
    main()
