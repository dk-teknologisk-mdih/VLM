"""
Gemini Robotics Object Detection Script

This script allows you to detect objects in images using Gemini Robotics-ER 1.5 model
and visualize the detected points on the image.
"""

import json
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
        self.pipeline, self.config, self.align = self.init_realsense(aligned=True)
        
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
        
        # self.depth_intrinsic = self.intrinsic_from_calibration()
        # print("depth intrinsic ", self.depth_intrinsic)
        # print("color intrinsic ", self.color_intrinsic)
        return color_frame, depth_frame, color_intrinsic, depth_intrinsic
        
    def img_point_to_cam_coord_realsense(self, image_point, depth_frame, depth_intrinsic):
        # Calculate the depth of the image point
        depth = depth_frame.get_distance(int(image_point[0]), int(image_point[1]))
        print('Depth ', depth)
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
        color_frame, depth_frame, color_intrinsic, depth_intrinsic = self.aligned_frames_and_images()
        
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        # Convert to PIL Image
        pil_image = Image.fromarray(color_image)
        
        # Save if path provided
        if save_path:
            save_path = self.ensure_output_dir(save_path)
            pil_image.save(save_path)
            print(f"Image saved to: {save_path}")

        # save depth frame as numpy array
        print(type(depth_frame))
        print(depth_frame)
        
        print("Image captured successfully")
        return pil_image, depth_frame, depth_intrinsic
        
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
    
    def _check_bbox_overlap(self, bbox1, bbox2):
        """Check if two bounding boxes overlap"""
        return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or 
                   bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])
    
    def _find_non_overlapping_position(self, initial_bbox, occupied_rects, max_attempts=20):
        """Find a non-overlapping position by shifting down"""
        bbox = initial_bbox
        shift_amount = 5  # pixels to shift each attempt
        
        for attempt in range(max_attempts):
            # Check if current position overlaps with any occupied rect
            overlaps = False
            for occupied in occupied_rects:
                if self._check_bbox_overlap(bbox, occupied):
                    overlaps = True
                    break
            
            if not overlaps:
                return bbox
            
            # Shift down for next attempt
            bbox = (bbox[0], bbox[1] + shift_amount, bbox[2], bbox[3] + shift_amount)
        
        # If still overlapping after max attempts, return last position
        return bbox
    
    def draw_points_on_image(self, image, points_data, output_path=None):
        """
        Draw detected points on the image.
        
        Args:
            image (PIL.Image or str): Image object or path to image
            points_data (list): List of point dictionaries with 'point' and 'label' keys
            output_path (str, optional): Path to save annotated image
            
        Returns:
            PIL.Image: Annotated image
        """
        img = image.copy()
        # Load image if path is provided
        if isinstance(img, str):
            img = self.resize_image(img)
        
        # Convert to RGB for drawing
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", size=20)
        except:
            font = ImageFont.load_default()
        
        # Define colors
        colors = [
            "red", "green", "blue", "purple", "pink", "brown", "navy",
            "magenta", "cyan", "yellow", "lime", "orange", "teal", "gold"
        ]
        
        # Track occupied label bounding boxes to avoid overlaps
        occupied_rects = []
        
        # Draw each point
        for i, point_info in enumerate(points_data):
            if "point" in point_info and "label" in point_info:
                y_norm, x_norm, _ = point_info["point"]
                label = point_info["label"]
                
                # Convert normalized coordinates to absolute pixel coordinates
                abs_x = int(x_norm / 1000.0 * width)
                abs_y = int(y_norm / 1000.0 * height)
                
                # Select color based on step number if available, otherwise use index
                if "step" in point_info:
                    color = colors[point_info["step"] % len(colors)]
                else:
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
                
                # Calculate initial label position
                label_pos_x = abs_x + point_radius + 5
                label_pos_y = abs_y - point_radius - 5 if abs_y > 20 else abs_y + point_radius + 5
                
                # Get initial text bounding box
                initial_bbox = draw.textbbox((label_pos_x, label_pos_y), label, font=font)
                
                # Find non-overlapping position
                final_bbox = self._find_non_overlapping_position(initial_bbox, occupied_rects)
                
                # Calculate final label position from adjusted bbox
                final_label_pos_x = final_bbox[0]
                final_label_pos_y = final_bbox[1]
                
                # Draw text background for better readability
                draw.rectangle(final_bbox, fill="white", outline=color, width=2)
                draw.text((final_label_pos_x, final_label_pos_y), label, fill=color, font=font)
                
                # Add this label's bbox to occupied list
                occupied_rects.append(final_bbox)
        
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
    
    prompt = f"""Return bounding boxes as a YAML array with labels for the following specific objects: {objects_str}.
Never return masks or code fencing. Only include the requested objects if they are present in the image.

The format should be as follows:
- box_2d: [ymin, xmin, ymax, xmax]
  label: <label for the object>
- box_2d: [ymin, xmin, ymax, xmax]
  label: <label for the object>

normalized to 0-1000. The values in box_2d must only be integers."""
    
    bboxes = detector.detect_objects(image_path, prompt)
    annotated_img = detector.draw_bounding_boxes(image_path, bboxes, output_path)
    
    return bboxes, annotated_img

def plan_stacking_trajectory(api_key, object_to_stack="red block", where_to_stack="first block"):
    """
    Plan a trajectory for stacking objects on top of each other.
    
    Args:
        api_key (str): Google API key for Gemini
        object_to_stack (str): Name of the objects to stack (e.g., "red block")
        where_to_stack (str): Target location description (e.g., "white paper", "green block")
    """
    detector = GeminiRoboticsDetector(api_key)
    
    image, depth_frame, depth_intrinsic = detector.capture_realsense_image("0_raw_realsense_capture.png")
    width, height = image.size
    
    # Detect all blocks to stack
    detect_blocks_prompt = f"""Detect all {object_to_stack}s in the image and return their locations.

The answer should follow the YAML format:
- point: <point>
  label: <label>
- point: <point>
  label: <label>

The points are in [y, x] format normalized to 0-1000."""
    
    blocks = detector.detect_objects(image, detect_blocks_prompt)
    
    # Detect the target stacking location
    detect_target_prompt = f"""Detect the {where_to_stack} in the image and return its location.

The answer should follow the YAML format:
- point: <point>
  label: <label>

The point is in [y, x] format normalized to 0-1000."""
    
    target_locations = detector.detect_objects(image, detect_target_prompt)
    
    if not target_locations:
        print(f"Warning: Could not find {where_to_stack}")
        target_location = None
    else:
        target_location = target_locations[0]
    
    # Get depth image for calculating depth values
    depth_image = np.asanyarray(depth_frame.get_data())
    print("Depth image shape: ", depth_image.shape)
    print(depth_image)  # Print the depth image array to verify its contents
    # Convert normalized coordinates to pixel coordinates and get depth for blocks
    for block in blocks:
        y_norm, x_norm = block["point"]
        abs_x = int(x_norm / 1000.0 * width)
        abs_y = int(y_norm / 1000.0 * height)
        depth_value = depth_image[abs_y, abs_x]  # depth map is in mm
        # Add depth as third element for visualization
        block['point'].append(int(depth_value))
        print(f"\n{block['label']} at pixel ({abs_x}, {abs_y}) with depth {depth_value} mm")
    
    # Convert target location to pixel coordinates and get depth if found
    if target_location:
        y_norm, x_norm = target_location["point"]
        target_pixel_x = int(x_norm / 1000.0 * width)
        target_pixel_y = int(y_norm / 1000.0 * height)
        target_depth = depth_image[target_pixel_y, target_pixel_x]  # depth map is in mm
        # Add depth as third element for visualization
        target_location['point'].append(int(target_depth))
        print(f"\n{target_location['label']} at pixel ({target_pixel_x}, {target_pixel_y}) with depth {target_depth} mm")
    
    # Combine all detected objects
    all_detected = blocks.copy()
    if target_location:
        all_detected.append(target_location)
    
    # Save combined detected objects to single YAML file
    save_path = detector.ensure_output_dir("1_detected_objects.yaml")
    with open(save_path, "w") as f:
        yaml.dump(all_detected, f, default_flow_style=False, sort_keys=False)
        print(f"\nDetected objects saved to: {save_path}")
    
    # Draw all detected objects in single PNG
    detector.draw_points_on_image(image, all_detected, output_path="1_detected_objects.png")
    
    # Now plan the stacking trajectory using normalized coordinates with depth
    blocks_info = chr(10).join([f"Block {i+1} ({block['label']}): y={block['point'][0]}, x={block['point'][1]} (normalized 0-1000), depth={block['point'][2]} mm" for i, block in enumerate(blocks)])
    target_info = f"Target location ({target_location['label']}): y={target_location['point'][0]}, x={target_location['point'][1]} (normalized 0-1000), depth={target_location['point'][2]} mm" if target_location else f"Target: {where_to_stack} (location not precisely detected)"
    
    trajectory_prompt = f"""You are controlling a robot arm to stack {len(blocks)} {object_to_stack}s on top of each other.

The blocks are located at these positions:
{blocks_info}

{target_info}

Plan a stacking operation where you:
1. Pick up each block one by one
2. Stack them all at the target location: {where_to_stack}
3. For each pick operation, provide waypoints from the gripper starting position to the block 
4. For each place operation, provide waypoints from the picked block to the stacking location

Any time a height change occurs (picking up or placing down), include a single waypoint above the target location before and after the target location to avoid collisions

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
  description: Move Block 2 to stack location and place it on target

The points should be in [y, x, z] format normalized to 0-1000 for y and x, and z as depth in millimeters.
Label waypoints sequentially for each trajectory segment."""
    
    stacking_plan_pixel = detector.detect_objects(image, trajectory_prompt, save_path="2_stacking_plan_pixel.yaml")
    
    # Visualize the complete trajectory - waypoints are in normalized coords, convert to absolute for drawing
    all_waypoints = []
    for step in stacking_plan_pixel:
        if 'trajectory' in step:
            step_number = step.get('step', 0)
            for waypoint in step['trajectory']:
                # Add step number to waypoint for color grouping
                waypoint_with_step = waypoint.copy()
                waypoint_with_step['step'] = step_number
                all_waypoints.append(waypoint_with_step)
    
    # Add the detected blocks and target to waypoints for visualization
    all_waypoints.extend(all_detected)
    
    annotated_img = detector.draw_points_on_image(image, all_waypoints, output_path="2_trajectory_with_waypoints.png")
    
    # Convert all waypoints from normalized to 3D camera coordinates
    stacking_plan_3d = []
    for step in stacking_plan_pixel:
        step_3d = step.copy()
        
        if 'trajectory' in step:
            trajectory_3d = []
            for waypoint in step['trajectory']:
                waypoint_3d = waypoint.copy()
                if 'point' in waypoint and len(waypoint['point']) >= 2:
                    # Convert normalized coordinates to absolute pixel coordinates
                    y_norm, x_norm = waypoint['point'][0], waypoint['point'][1]
                    abs_x = int(x_norm / 1000.0 * width)
                    abs_y = int(y_norm / 1000.0 * height)
                    
                    print(f"point before {waypoint['label']}: normalized ({y_norm}, {x_norm}), pixel ({abs_x}, {abs_y}), z {waypoint['point'][2]} mm")

                    # Convert pixel coordinates to 3D camera coordinates
                    point_3d, depth = detector.img_point_to_cam_coord_realsense(
                        [abs_x, abs_y], 
                        depth_frame, 
                        depth_intrinsic
                    )
                    waypoint_3d['point'] = [float(point_3d[0]), float(point_3d[1]), float(point_3d[2])]
                    print(f"point after {waypoint['label']}: 3D camera coordinates {waypoint_3d['point']}")
                trajectory_3d.append(waypoint_3d)
            step_3d['trajectory'] = trajectory_3d
        
        stacking_plan_3d.append(step_3d)
    
    # Save the 3D stacking plan
    save_path = detector.ensure_output_dir("3_stacking_trajectory_plan.yaml")
    with open(save_path, "w") as f:
        yaml.dump(stacking_plan_3d, f, default_flow_style=False, sort_keys=False)
        print(f"Stacking plan with 3D coordinates saved to: {save_path}")
    
    print(f"\n=== Summary ===")
    print(f"Found {len(blocks)} blocks to stack")
    if target_location:
        print(f"Target location: {target_location['label']}")
    print(f"Generated {len(stacking_plan_3d)} steps in stacking plan:")
    for step in stacking_plan_3d:
        print(f"  Step {step.get('step', '?')}: {step.get('action', 'unknown')} - {step.get('description', '')}")
        if 'trajectory' in step:
            print(f"    Trajectory has {len(step['trajectory'])} waypoints (in 3D camera coordinates)")

def TEST(api_key):
     
    detector = GeminiRoboticsDetector(api_key)
    
    image, depth_frame, depth_intrinsic = detector.capture_realsense_image()
    width, height = image.size

    # Load stacking plan pixel data from a YAML file
    yaml_file_path = detector.ensure_output_dir("2_stacking_plan_pixel.yaml")
    with open(yaml_file_path, "r") as f:
        stacking_plan_pixel = yaml.safe_load(f)
     
    # Convert all waypoints from normalized to 3D camera coordinates
    stacking_plan_3d = []
    for step in stacking_plan_pixel:
        step_3d = step.copy()
        
        if 'trajectory' in step:
            trajectory_3d = []
            for waypoint in step['trajectory']:
                waypoint_3d = waypoint.copy()
                if 'point' in waypoint and len(waypoint['point']) >= 2:
                    # Convert normalized coordinates to absolute pixel coordinates
                    y_norm, x_norm = waypoint['point'][0], waypoint['point'][1]
                    abs_x = int(x_norm / 1000.0 * width)
                    abs_y = int(y_norm / 1000.0 * height)
                    
                    print(f"point before {waypoint['label']}: normalized ({y_norm}, {x_norm}), pixel ({abs_x}, {abs_y}), z {waypoint['point'][2]} mm")

                    # Convert pixel coordinates to 3D camera coordinates
                    point_3d, depth = detector.img_point_to_cam_coord_realsense(
                        [abs_x, abs_y], 
                        depth_frame, 
                        depth_intrinsic
                    )
                    waypoint_3d['point'] = [float(point_3d[0]), float(point_3d[1]), float(waypoint['point'][2]/1000.0)]
                    print(f"point after {waypoint['label']}: 3D camera coordinates {waypoint_3d['point']}")
                trajectory_3d.append(waypoint_3d)
            step_3d['trajectory'] = trajectory_3d
        
        stacking_plan_3d.append(step_3d)
    
    # Save the 3D stacking plan
    save_path = detector.ensure_output_dir("TEST.yaml")
    with open(save_path, "w") as f:
        yaml.dump(stacking_plan_3d, f, default_flow_style=False, sort_keys=False)
        print(f"Stacking plan with 3D coordinates saved to: {save_path}")
    


def main():
    """Example usage of the generic detection functions."""
    
    API_KEY = "AIzaSyDlGRUwor17QRWdbTFds_Gg_5opCjr8c4g"  # Replace with your actual API key
    # TEST(API_KEY)
    # Example 5: Plan stacking trajectory
    print("=== Planning Stacking Trajectory ===")
    result = plan_stacking_trajectory(
        API_KEY,
        object_to_stack="red block",
        where_to_stack="the middle of the white paper",
    )



if __name__ == "__main__":
    main()
