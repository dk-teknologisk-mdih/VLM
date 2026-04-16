"""
Gemini Robotics Object Detection Script

This script allows you to detect objects in images using Gemini Robotics-ER 1.5 model
and visualize the detected points on the image.
"""

import json
import PIL
import yaml
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
        self.pipeline, self.config, self.align = self._init_realsense(aligned=True)
        
    def _init_realsense(self, aligned=True, resolution=(1280, 720)):
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
    
    def _get_images(self):
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames(5000)

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a self.camera_width x self.camera_height depth image
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            raise RuntimeError("Could not acquire depth or color frame.")
        
        
        # Get realsense camera intrinsic
        color_intrinsic = color_frame.profile.as_video_stream_profile().intrinsics
        depth_intrinsic = depth_frame.profile.as_video_stream_profile().intrinsics
        
        return depth_frame, color_frame, color_intrinsic, depth_intrinsic

    def img_point_to_cam_coord_realsense(self, image_point, depth_frame, depth_intrinsic):
        # Calculate the depth of the image point
        depth = depth_frame.get_distance(int(image_point[0]), int(image_point[1]))
        # print('Depth ', depth)
        # Transform the image point to 3D point in camera coordinates
        point = rs.rs2_deproject_pixel_to_point(depth_intrinsic, [image_point[0], image_point[1]], depth)
        return point, depth
    
    def capture_realsense_image(self, save_path=None):
        """
        Capture an image from RealSense camera.
        
        Args:
            save_path (str, optional): Path to save the captured image
            resolution (tuple): Resolution (width, height) for capture
        """
        # Get aligned frames and images
        depth_frame, color_frame, color_intrinsic, depth_intrinsic = self._get_images()
        
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert to PIL Image
        pil_image = Image.fromarray(color_image)
        
        # Save if path provided
        if save_path:
            save_path = self.ensure_output_dir(save_path)
            pil_image.save(save_path)
            print(f"Image saved to: {save_path}")
        
        return pil_image, depth_frame, depth_intrinsic, depth_image
        
    def resize_image(self, img_path, max_width=800):
        """
        Resize image for faster processing.
        
        Args:
            img_path (str): Path to the image file
            max_width (int): Maximum width for resizing
            
        Returns:
            PIL.Image: Resized image
        """
        if isinstance(img_path, Image.Image):
            img = img_path
        else:
            img = Image.open(img_path)
        img = img.resize(
            (max_width, int(max_width * img.size[1] / img.size[0])), 
            Image.Resampling.LANCZOS
        )
        return img
    
    def parse_yaml(self, yaml_output):
        """
        Parse YAML output from the model, removing markdown fencing if present.
        
        Args:
            yaml_output (str): Raw YAML output from model
            
        Returns:
            str: Cleaned YAML string
        """
        lines = yaml_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```yaml" or line == "```json":
                yaml_output = "\n".join(lines[i + 1:])
                yaml_output = yaml_output.split("```")[0]
                break
        return yaml_output
    
    def ensure_output_dir(self, filepath=None):
        """Ensure VLM_output directory exists and prepend it to filepath if provided."""
        output_dir = "VLM_output"
        os.makedirs(output_dir, exist_ok=True)
        if filepath:
            return os.path.join(output_dir, os.path.basename(filepath))
        return output_dir
    
    def detect_objects(self, img, prompt, temperature=0.5, use_thinking=False, save_path=None):
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
        # if isinstance(img, str):
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
        yaml_output = self.parse_yaml(response.text)
        
        try:
            data = yaml.safe_load(yaml_output)
            if save_path:
                save_path = self.ensure_output_dir(save_path)
                with open(save_path, "w") as f:
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
                    print(f"Raw response saved to: {save_path}")
            return data
        except yaml.YAMLError as e:
            print(f"Error decoding YAML: {e}")
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
        # if isinstance(img, str):
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
                x_pixel = int(point_info["x"] / 1000 * img.width)
                y_pixel = int(point_info["y"] / 1000 * img.height)
                label = point_info["label"]
                
                # Select color
                color = colors[i % len(colors)]
                
                # Draw a circle at the point
                point_radius = 4
                draw.ellipse(
                    (
                        x_pixel - point_radius,
                        y_pixel - point_radius,
                        x_pixel + point_radius,
                        y_pixel + point_radius,
                    ),
                    fill=color,
                    outline="white",
                    width=2
                )
                
                # Draw the label
                label_pos_x = x_pixel + point_radius + 5
                label_pos_y = y_pixel - point_radius - 5 if y_pixel > 20 else y_pixel + point_radius + 5
                
                # Draw text background for better readability
                bbox = draw.textbbox((label_pos_x, label_pos_y), label, font=font)
                draw.rectangle(bbox, fill="white", outline=color, width=2)
                draw.text((label_pos_x, label_pos_y), label, fill=color, font=font)
        
        # Save if output path is provided
        if output_path:
            output_path = self.ensure_output_dir(output_path)
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
            output_path = self.ensure_output_dir(output_path)
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

The answer should follow the YAML format:
- point: <point>
  label: <label1>
- point: <point>
  label: <label2>

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

The answer should follow the YAML format:
- point: <point>
  label: <label1>
- point: <point>
  label: <label2>

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
    
    prompt = f"""Return bounding boxes as a YAML array with labels. Never return masks or
code fencing. Limit to {max_objects} objects. Include objects you can identify in the image.

The format should be as follows:
- box_2d: [ymin, xmin, ymax, xmax]
  label: <label for the object>
- box_2d: [ymin, xmin, ymax, xmax]
  label: <label for the object>

Normalized to 0-1000. The values in box_2d must only be integers."""
    
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
    
    prompt = f"""Return bounding boxes as a YAML array with labels for the following specific objects: {objects_str}.
Never return masks or code fencing. Only include the requested objects if they are present in the image.

The format should be as follows:
- box_2d: [ymin, xmin, ymax, xmax]
  label: <label for the object>
- box_2d: [ymin, xmin, ymax, xmax]
  label: <label for the object>

Normalized to 0-1000. The values in box_2d must only be integers."""
    
    bboxes = detector.detect_objects(image_path, prompt)
    annotated_img = detector.draw_bounding_boxes(image_path, bboxes, output_path)
    
    return bboxes, annotated_img

def plan_stacking_trajectory(api_key, object_to_stack="red block", where_to_stack="first block"):
    """
    Plan a trajectory for stacking objects on top of each other.
    
    Args:
        image (PIL.Image): PIL Image object
        api_key (str): Google API key for Gemini
        object_to_stack (str): Name of the objects to stack (e.g., "red block")
        where_to_stack (str): Description of where to stack (e.g., "the middle of the white box")
        
    Returns:
        dict: Dictionary containing:
            - 'blocks': List of detected block locations
            - 'stacking_plan': List of steps with trajectories
            - 'annotated_image': PIL Image with trajectory visualization
    """
    detector = GeminiRoboticsDetector(api_key)
    
    image, depth_frame, depth_intrinsic, depth_image = detector.capture_realsense_image("0_raw_realsense_capture.png")
    
    # First, detect all blocks
    detect_prompt = f"""Detect all {object_to_stack}s in the image and return their locations.

The answer should follow the YAML format:
- point: <point>
  label: <label>
- point: <point>
  label: <label>

The points are in [y, x] format normalized to 0-1000."""
    
    blocks = detector.detect_objects(image, detect_prompt, save_path="1_detected_blocks.yaml")
    
    # if len(blocks) < 2:
    #     print(f"Warning: Only found {len(blocks)} block(s). Need at least 2 for stacking.")
    #     return {'blocks': blocks, 'stacking_plan': [], 'annotated_image': None}
    
    # get the depth of each block from the realsense ir depth map
    for block in blocks:
        block["x"] = block["point"][1]
        block["y"] = block["point"][0]
        #normalize back to pixel coordinates
        block["x"] = int(block["x"] / 1000 * image.width)
        block["y"] = int(block["y"] / 1000 * image.height)
        point_cam_coord, depth_value = detector.img_point_to_cam_coord_realsense((block["x"], block["y"]), depth_frame, depth_intrinsic)
        print(f"1. Block {block['label']} at pixel ({block['x']}, {block['y']}) has depth {depth_value:.2f} meters and camera coordinates {point_cam_coord}")

        y_norm, x_norm = block["point"]
        width, height = image.size
        abs_x = int(x_norm / 1000.0 * width)
        abs_y = int(y_norm / 1000.0 * height)
        depth_value = depth_image[abs_y, abs_x]  # depth map is in mm
        print(f"2. Block {block['label']} at pixel ({abs_x}, {abs_y}) has depth {depth_value} mm")

        block['point'].extend([int(depth_value)])
    
    img_with_detected_objects = detector.draw_points_on_image(image, blocks, "1_detected_objects.png")
    
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

Return a YAML array with the stacking plan:
- step: 1
  action: pick
  block: Block 2
  trajectory:
    - point: [x, y, z]
      label: waypoint_1
    - point: [x, y, z]
      label: waypoint_2
  description: Move to Block 2 and pick it up
- step: 2
  action: place
  block: Block 2
  trajectory:
    - point: [x, y, z]
      label: waypoint_1
    - point: [x, y, z]
      label: waypoint_2
  description: Move Block 2 to stack location and place it on Block 1

The points are in [x, y, z] format in pixel coordinates with depth in mm.
Label waypoints sequentially for each trajectory segment."""
    
    stacking_plan = detector.detect_objects(img_with_detected_objects, trajectory_prompt, save_path="2_stacking_plan.yaml")
    
    # Visualize the complete trajectory
    all_waypoints = []
    for step in stacking_plan:
        if 'trajectory' in step:
            for waypoint in step['trajectory']:
                waypoint["x"] = waypoint["point"][0]
                waypoint["y"] = waypoint["point"][1]
            all_waypoints.extend(step['trajectory'])
    
    # Add the detected blocks to waypoints
    all_waypoints.extend(blocks)
    
    img_with_all_waypoints = detector.draw_points_on_image(img_with_detected_objects, all_waypoints, "2_trajectory_with_waypoints.png")
    
    print(f"Found {len(blocks)} blocks to stack")
    print(f"Generated {len(stacking_plan)} steps in stacking plan:")
    for step in stacking_plan:
        print(f"  Step {step.get('step', '?')}: {step.get('action', 'unknown')} - {step.get('description', '')}")
        if 'trajectory' in step:
            print(f"    Trajectory has {len(step['trajectory'])} waypoints")


def main():
    """Example usage of the generic detection functions."""
    
    API_KEY = "AIzaSyDlGRUwor17QRWdbTFds_Gg_5opCjr8c4g"  # Replace with your actual API key
    
    # Example 5: Plan stacking trajectory
    print("=== Planning Stacking Trajectory ===")
    result = plan_stacking_trajectory(
        API_KEY,
        object_to_stack="red block",
        where_to_stack="the middle of the white box",
    )

if __name__ == "__main__":
    main()
