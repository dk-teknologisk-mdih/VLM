import socket
import time
import os
import json
import numpy as np
import cv2
from datetime import datetime
from scipy.spatial.transform import Rotation
import pyrealsense2 as rs


class RobotDataCollector:
    """
    Collect robot pose and camera image data
    Moves robot to various poses based on a reference pose where camera points at board
    """
    
    # Reference pose where camera points DOWN at the board
    # This is the "base" orientation - all other poses are variations of this
    REFERENCE_POSE = {
        'x': 229.25,
        'y': 222.98,
        'z': 570.85,
        'q1': 0.25415,   # w
        'q2': -0.66263,  # x
        'q3': -0.64590,  # y
        'q4': 0.28134    # z
    }
    
    def __init__(self, robot_ip: str, robot_port: int = 50000, output_dir: str = "gripper_camera_calib", 
                 use_camera: bool = False, radius: float = 400.0, max_angle: float = 30.0):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.output_dir = output_dir
        self.socket = None
        self.use_camera = use_camera
        self.radius = radius
        self.max_angle = max_angle
        
        # Check camera availability
        self.camera_available = False
        if self.use_camera:
            try:
                ctx = rs.context()
                if len(ctx.devices) > 0:
                    self.camera_available = True
                    print("RealSense camera detected and ready.")
                else:
                    print("Warning: No RealSense camera detected. Images will not be captured.")
            except Exception as e:
                print(f"Warning: Could not initialize RealSense: {e}")
        
        # Collected data
        self.gripper_poses = []
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "poses"), exist_ok=True)
        
        # Extract reference orientation as rotation object
        self.reference_rotation = self._get_reference_rotation()
        
        print(f"\nReference pose (camera pointing at board):")
        print(f"  Position: ({self.REFERENCE_POSE['x']:.2f}, {self.REFERENCE_POSE['y']:.2f}, {self.REFERENCE_POSE['z']:.2f})")
        print(f"  Quaternion: [{self.REFERENCE_POSE['q1']:.4f}, {self.REFERENCE_POSE['q2']:.4f}, {self.REFERENCE_POSE['q3']:.4f}, {self.REFERENCE_POSE['q4']:.4f}]")
    
    def _get_reference_rotation(self) -> Rotation:
        """Get the reference orientation as a scipy Rotation object"""
        # ABB format: q1=w, q2=x, q3=y, q4=z
        # scipy format: [x, y, z, w]
        q = self.REFERENCE_POSE
        return Rotation.from_quat([q['q2'], q['q3'], q['q4'], q['q1']])
    
    # =========================================================
    # Robot Communication
    # =========================================================
    
    def connect_robot(self):
        """Connect to robot"""
        print(f"Connecting to robot at {self.robot_ip}:{self.robot_port}...")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(60)
        self.socket.connect((self.robot_ip, self.robot_port))
        print("Connected!")
    
    def send_command(self, command: str) -> str:
        """Send command and get response"""
        self.socket.send((command + "\n").encode('utf-8'))
        response = self.socket.recv(1024).decode('utf-8').strip()
        return response
    
    def get_gripper_pose(self) -> np.ndarray:
        """Get current gripper TCP pose as 4x4 matrix"""
        response = self.send_command("POSE")
        
        if response.startswith("GRIP:"):
            values = [float(v) for v in response[5:].split(",")]
            x, y, z, q1, q2, q3, q4 = values
            return self.pose_to_matrix(x, y, z, q1, q2, q3, q4)
        return None
    
    def move_robot(self, x, y, z, q1, q2, q3, q4) -> bool:
        """Move robot to pose with normalized quaternion"""
        # Normalize quaternion before sending
        norm = np.sqrt(q1**2 + q2**2 + q3**2 + q4**2)
        if norm < 0.0001:
            print("ERROR: Invalid quaternion (near zero magnitude)")
            return False
        
        q1, q2, q3, q4 = q1/norm, q2/norm, q3/norm, q4/norm
        
        # Log what we're sending
        print(f"    Sending: X={x:.1f}, Y={y:.1f}, Z={z:.1f}")
        print(f"    Quat: [{q1:.4f}, {q2:.4f}, {q3:.4f}, {q4:.4f}]")
        
        # Send with high precision
        command = f"MOVE:{x:.4f},{y:.4f},{z:.4f},{q1:.8f},{q2:.8f},{q3:.8f},{q4:.8f}"
        response = self.send_command(command)
        return response == "MOVE_DONE"
    
    def go_home(self):
        """Move to home"""
        self.send_command("HOME")
    
    # =========================================================
    # Pose Generation - Based on Reference Pose
    # =========================================================
    
    def generate_calibration_poses(self, board_center: dict, num_poses: int = 15, 
                                    radius: float = 300.0, max_angle: float = 30.0) -> list:
        """
        Generate calibration poses based on the reference pose.
        
        The reference pose defines the orientation where camera points at the board.
        We generate variations by:
        1. Moving the position on a hemisphere around the board center
        2. Adjusting the orientation to keep camera pointing at board center
        
        Args:
            board_center: Dictionary with 'x', 'y', 'z' coordinates of board center
            num_poses: Number of poses to generate
            radius: Distance from board center to gripper TCP (mm)
            max_angle: Maximum angle from vertical in degrees
            
        Returns:
            List of pose dictionaries
        """
        poses = []
        cx, cy, cz = board_center['x'], board_center['y'], board_center['z']
        
        # First pose is the reference pose (straight down view)
        # We'll compute what the "straight down" position should be based on reference
        
        # From reference pose, calculate the viewing direction
        ref_pos = np.array([self.REFERENCE_POSE['x'], self.REFERENCE_POSE['y'], self.REFERENCE_POSE['z']])
        board_pos = np.array([cx, cy, cz])
        
        # This is the direction from camera to board in reference pose
        ref_view_dir = board_pos - ref_pos
        ref_distance = np.linalg.norm(ref_view_dir)
        ref_view_dir_normalized = ref_view_dir / ref_distance
        
        print(f"\nReference view analysis:")
        print(f"  Board center: ({cx:.1f}, {cy:.1f}, {cz:.1f})")
        print(f"  Reference TCP: ({ref_pos[0]:.1f}, {ref_pos[1]:.1f}, {ref_pos[2]:.1f})")
        print(f"  Distance to board: {ref_distance:.1f} mm")
        print(f"  View direction: ({ref_view_dir_normalized[0]:.3f}, {ref_view_dir_normalized[1]:.3f}, {ref_view_dir_normalized[2]:.3f})")
        
        # Use specified radius or reference distance
        if radius is None:
            radius = ref_distance
        
        print(f"  Using radius: {radius:.1f} mm")
        print(f"  Max angle: {max_angle:.1f}°")
        
        # Generate poses on hemisphere
        # Convert max_angle to radians
        max_theta = np.radians(max_angle)
        
        # Define pose pattern: rings at different angles from vertical
        # Ring 0: center (theta=0)
        # Ring 1: small angle
        # Ring 2: medium angle
        ring_configs = [
            (0.0, 1),                    # Center: 1 pose at theta=0
            (max_theta * 0.4, 4),        # Inner ring: 4 poses
            (max_theta * 0.7, 6),        # Middle ring: 6 poses
            (max_theta * 1.0, 6),        # Outer ring: 6 poses
        ]
        
        pose_count = 0
        
        for ring_idx, (theta, num_in_ring) in enumerate(ring_configs):
            if pose_count >= num_poses:
                break
            
            for i in range(num_in_ring):
                if pose_count >= num_poses:
                    break
                
                # Azimuthal angle (rotation around vertical axis)
                if num_in_ring > 1:
                    phi = (2 * np.pi * i / num_in_ring) + (ring_idx * np.pi / 8)
                else:
                    phi = 0
                
                # Calculate position offset from board center (spherical coordinates)
                # Camera position is ABOVE and AROUND the board
                dx = radius * np.sin(theta) * np.cos(phi)
                dy = radius * np.sin(theta) * np.sin(phi)
                dz = radius * np.cos(theta)  # Always positive (above board)
                
                # Gripper TCP position
                x = cx + dx
                y = cy + dy
                z = cz + dz
                
                # Calculate orientation to keep camera pointing at board
                # We need to rotate the reference orientation to account for the new position
                q1, q2, q3, q4 = self._calculate_orientation_for_position(
                    np.array([x, y, z]), 
                    board_pos,
                    theta, 
                    phi
                )
                
                poses.append({
                    'x': x, 'y': y, 'z': z,
                    'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4
                })
                
                print(f"  Pose {pose_count}: pos=({x:.0f},{y:.0f},{z:.0f}) theta={np.degrees(theta):.1f}° phi={np.degrees(phi):.1f}°")
                
                pose_count += 1
        
        return poses
    
    def _calculate_orientation_for_position(self, tcp_pos: np.ndarray, board_pos: np.ndarray,
                                             theta: float, phi: float) -> tuple:
        """
        Calculate gripper orientation to keep camera pointing at board.
        
        Uses the reference orientation and applies relative rotations based on 
        how much we've moved from the reference position.
        
        Args:
            tcp_pos: Target TCP position
            board_pos: Board center position
            theta: Angle from vertical (radians)
            phi: Azimuthal angle (radians)
            
        Returns:
            Quaternion (q1, q2, q3, q4) in ABB format
        """
        # Method: Apply incremental rotation to reference orientation
        # The camera needs to "tilt" to keep looking at the board center
        
        # At reference pose (theta=0), we use reference orientation directly
        # As we move around the hemisphere, we rotate the gripper accordingly
        
        # Rotation needed: rotate around the axis perpendicular to the movement
        # This keeps the camera pointed at the board center
        
        if theta < 0.001:
            # At center - use reference orientation
            R_total = self.reference_rotation
        else:
            # We need to rotate the gripper to compensate for position change
            # The rotation axis is perpendicular to the radial direction in the XY plane
            
            # Rotation axis (tangent to the hemisphere, perpendicular to radial direction)
            # For a point at (theta, phi), the tilt axis is in the tangent plane
            rot_axis = np.array([-np.sin(phi), np.cos(phi), 0])
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
            
            # Create rotation to tilt the gripper
            R_tilt = Rotation.from_rotvec(theta * rot_axis)
            
            # Apply tilt to reference orientation
            # The order matters: first reference, then tilt
            R_total = R_tilt * self.reference_rotation
        
        # Convert to quaternion
        q = R_total.as_quat()  # [x, y, z, w]
        
        # Normalize
        norm = np.linalg.norm(q)
        q = q / norm
        
        # Return in ABB format: q1=w, q2=x, q3=y, q4=z
        return float(q[3]), float(q[0]), float(q[1]), float(q[2])
    
    # =========================================================
    # Math Utilities
    # =========================================================
    
    def pose_to_matrix(self, x, y, z, q1, q2, q3, q4) -> np.ndarray:
        """Convert ABB pose to 4x4 matrix"""
        rot = Rotation.from_quat([q2, q3, q4, q1])
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = [x, y, z]
        return T
    
    def matrix_to_pose(self, T: np.ndarray) -> dict:
        """Convert 4x4 matrix to pose dict"""
        rot = Rotation.from_matrix(T[:3, :3])
        q = rot.as_quat()
        return {
            'x': T[0, 3], 'y': T[1, 3], 'z': T[2, 3],
            'q1': q[3], 'q2': q[0], 'q3': q[1], 'q4': q[2]
        }
    
    def matrix_to_euler_pose(self, T: np.ndarray) -> dict:
        """Convert 4x4 matrix to euler pose"""
        rot = Rotation.from_matrix(T[:3, :3])
        euler = rot.as_euler('xyz', degrees=True)
        return {
            'x': T[0, 3], 'y': T[1, 3], 'z': T[2, 3],
            'rx': euler[0], 'ry': euler[1], 'rz': euler[2]
        }
    
    # =========================================================
    # Camera Functions
    # =========================================================
    
    def get_camera_intrinsics(self, resolution=(1280, 720)):
        """Get camera intrinsic parameters from RealSense camera"""
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, 30)
        
        try:
            pipeline.start(config)
            sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
            sensor.set_option(rs.option.exposure, 200)
            
            # Get the active profile
            profile = pipeline.get_active_profile()
            color_stream = profile.get_stream(rs.stream.color)
            intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            # Extract intrinsic parameters
            camera_params = {
                'width': intrinsics.width,
                'height': intrinsics.height,
                'fx': intrinsics.fx,  # Focal length in x
                'fy': intrinsics.fy,  # Focal length in y
                'ppx': intrinsics.ppx,  # Principal point x
                'ppy': intrinsics.ppy,  # Principal point y
                'model': str(intrinsics.model),  # Distortion model
                'coeffs': intrinsics.coeffs  # Distortion coefficients [k1, k2, p1, p2, k3]
            }
            
            return camera_params
        finally:
            pipeline.stop()
    
    def save_camera_intrinsics(self, filepath: str, resolution=(1280, 720)):
        """Load camera intrinsic parameters and save them to a txt file"""
        if not self.camera_available:
            print("Warning: No camera available to get intrinsics")
            return False
        
        try:
            print("Loading camera intrinsic parameters...")
            intrinsics = self.get_camera_intrinsics(resolution)
            
            # Save to text file
            with open(filepath, 'w') as f:
                f.write("# RealSense Camera Intrinsic Parameters\n")
                f.write(f"# Captured on: {datetime.now().isoformat()}\n")
                f.write("# " + "="*50 + "\n\n")
                
                f.write(f"Image Resolution:\n")
                f.write(f"  Width:  {intrinsics['width']} pixels\n")
                f.write(f"  Height: {intrinsics['height']} pixels\n\n")
                
                f.write(f"Camera Matrix (K):\n")
                f.write(f"  fx = {intrinsics['fx']:.6f}  (focal length x)\n")
                f.write(f"  fy = {intrinsics['fy']:.6f}  (focal length y)\n")
                f.write(f"  cx = {intrinsics['ppx']:.6f}  (principal point x)\n")
                f.write(f"  cy = {intrinsics['ppy']:.6f}  (principal point y)\n\n")
                
                f.write(f"Camera Matrix Format:\n")
                f.write(f"  [[{intrinsics['fx']:.6f}, 0.0, {intrinsics['ppx']:.6f}],\n")
                f.write(f"   [0.0, {intrinsics['fy']:.6f}, {intrinsics['ppy']:.6f}],\n")
                f.write(f"   [0.0, 0.0, 1.0]]\n\n")
                
                f.write(f"Distortion Model: {intrinsics['model']}\n")
                f.write(f"Distortion Coefficients [k1, k2, p1, p2, k3]:\n")
                for i, coeff in enumerate(intrinsics['coeffs']):
                    f.write(f"  {['k1', 'k2', 'p1', 'p2', 'k3'][i]} = {coeff:.8f}\n")
                
                f.write(f"\nDistortion Coefficients Array:\n")
                f.write(f"  {list(intrinsics['coeffs'])}\n")
            
            # Also save as JSON for easy loading
            json_filepath = filepath.replace('.txt', '.json')
            with open(json_filepath, 'w') as f:
                # Convert numpy array to list for JSON serialization
                intrinsics_json = intrinsics.copy()
                intrinsics_json['coeffs'] = list(intrinsics['coeffs'])
                json.dump(intrinsics_json, f, indent=2)
            
            print(f"✓ Camera intrinsics saved to:")
            print(f"  - {filepath}")
            print(f"  - {json_filepath}")
            print(f"\nCamera parameters:")
            print(f"  Resolution: {intrinsics['width']} x {intrinsics['height']}")
            print(f"  Focal length: fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}")
            print(f"  Principal point: cx={intrinsics['ppx']:.2f}, cy={intrinsics['ppy']:.2f}")
            
            return True
            
        except Exception as e:
            print(f"Error getting camera intrinsics: {e}")
            return False
    
    def capture_realsense_image(self, resolution=(1280, 720)):
        """Capture an image from RealSense camera"""
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, 30)
        
        try:
            pipeline.start(config)
            for _ in range(30):
                pipeline.wait_for_frames()
            
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                raise RuntimeError("Failed to capture color frame")
            
            return np.asanyarray(color_frame.get_data())
        finally:
            pipeline.stop()
    
    # =========================================================
    # Data Capture Functions
    # =========================================================
    
    def save_pose_to_txt(self, gripper_pose: np.ndarray, filepath: str):
        """Save gripper TCP pose to text file"""
        with open(filepath, 'w') as f:
            f.write("# Robot TCP Pose Data\n")
            f.write("# Format: x,y,z,rx,ry,rz\n")
            f.write("# Units: mm, degrees (euler xyz)\n\n")
            
            gripper_euler = self.matrix_to_euler_pose(gripper_pose)
            f.write(f"{gripper_euler['x']:.6f},{gripper_euler['y']:.6f},{gripper_euler['z']:.6f},")
            f.write(f"{gripper_euler['rx']:.6f},{gripper_euler['ry']:.6f},{gripper_euler['rz']:.6f}\n")
    
    def capture_single_pose(self, pose_index: int, target_pose: dict) -> bool:
        """Capture data for a single pose"""
        print(f"\n--- Pose {pose_index+1} ---")
        
        success = self.move_robot(
            target_pose['x'], target_pose['y'], target_pose['z'],
            target_pose['q1'], target_pose['q2'], target_pose['q3'], target_pose['q4']
        )
        
        if not success:
            print("  Move failed, skipping")
            return False
        
        time.sleep(1)
        
        T_gripper = self.get_gripper_pose()
        if T_gripper is None:
            print("  Could not get robot pose")
            return False
        
        # Capture camera image if available
        camera_image = None
        if self.camera_available:
            try:
                camera_image = self.capture_realsense_image()
            except Exception as e:
                print(f"  Warning: Failed to capture image: {e}")
        
        # Save files
        pose_filename = f"pose_{pose_index:03d}.txt"
        image_filename = f"image_{pose_index:03d}.png"
        
        pose_filepath = os.path.join(self.output_dir, "poses", pose_filename)
        image_filepath = os.path.join(self.output_dir, "images", image_filename)
        
        self.save_pose_to_txt(T_gripper, pose_filepath)
        
        if camera_image is not None:
            cv2.imwrite(image_filepath, camera_image)
            print(f"  ✓ Saved image: {image_filename}")
        
        self.gripper_poses.append(T_gripper)
        print(f"  ✓ Captured pose {pose_index}")
        
        return True
    
    # =========================================================
    # Main Collection Routine
    # =========================================================
    
    def collect_data(self, board_center: dict, num_poses: int = 15):
        """Run robot data collection"""
        print("=" * 60)
        print("ROBOT DATA COLLECTION")
        print("Using reference pose for camera orientation")
        print("=" * 60)
        
        # Save camera intrinsics if camera is available
        if self.camera_available:
            print("\n[0] Saving camera intrinsic parameters...")
            intrinsics_path = os.path.join(self.output_dir, "camera_intrinsics.txt")
            self.save_camera_intrinsics(intrinsics_path)
        
        print("\n[1] Connecting to robot...")
        self.connect_robot()
        
        print("\n[2] Generating calibration poses from reference...")
        target_poses = self.generate_calibration_poses(
            board_center, 
            num_poses,
            radius=self.radius,
            max_angle=self.max_angle
        )
        
        print("\n[3] Moving to home...")
        self.go_home()
        time.sleep(2)
        
        print("\n[4] Collecting data...")
        
        successful_captures = 0
        for i, target in enumerate(target_poses):
            if self.capture_single_pose(i, target):
                successful_captures += 1
        
        print(f"\n  Total successful captures: {successful_captures}/{len(target_poses)}")
        
        # print("\n[5] Returning home...")
        # self.go_home()
        
        print("\n[6] Saving collected data...")
        if len(self.gripper_poses) > 0:
            self._save_raw_data()
        else:
            print("ERROR: No poses collected!")
        
        print("\n[7] Cleaning up...")
        self.send_command("QUIT")
        self.socket.close()
        
        print("\n" + "=" * 60)
        print("DATA COLLECTION COMPLETE")
        print("=" * 60)
        print(f"Collected {len(self.gripper_poses)} poses")
        print(f"Data saved to '{self.output_dir}/'")
    
    def _save_raw_data(self):
        """Save raw collected poses"""
        poses_txt_path = os.path.join(self.output_dir, "gripper_poses.txt")
        with open(poses_txt_path, 'w') as f:
            f.write("# All Robot TCP Poses\n")
            f.write("# Format: x,y,z,rx,ry,rz\n")
            f.write("# Units: mm, degrees (euler xyz)\n\n")
            
            for T in self.gripper_poses:
                euler_pose = self.matrix_to_euler_pose(T)
                f.write(f"{euler_pose['x']:.6f},{euler_pose['y']:.6f},{euler_pose['z']:.6f},")
                f.write(f"{euler_pose['rx']:.6f},{euler_pose['ry']:.6f},{euler_pose['rz']:.6f}\n")
        
        data = {
            'reference_pose': self.REFERENCE_POSE,
            'num_poses': len(self.gripper_poses),
            'collection_time': datetime.now().isoformat(),
            'gripper_poses': [self.matrix_to_pose(T) for T in self.gripper_poses]
        }
        
        with open(os.path.join(self.output_dir, "collected_data.json"), 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  Saved {len(self.gripper_poses)} poses")


def main():
    # Configuration
    ROBOT_IP = "192.168.125.1"  # RobotStudio physical robot
    # ROBOT_IP = "127.0.0.1"  # RobotStudio simulation
    USE_CAMERA = True
    
    # Board center - estimate from reference pose
    # The reference pose looks at the board from above
    # Estimate board center is roughly below the reference position
    BOARD_CENTER = {
        'x': 229.25,   # Same X as reference (camera looking straight down)
        'y': 222.98,   # Same Y as reference
        'z': 0         # Board is on the table (Z=0 or adjust as needed)
    }
  
    NUM_POSES = 10
    HEMISPHERE_RADIUS = 600  # Approximate distance from reference pose to board
    MAX_ANGLE_FROM_VERTICAL = 30  # Maximum tilt angle in degrees
    
    collector = RobotDataCollector(
        robot_ip=ROBOT_IP,
        output_dir="ABB_gripper_camera_calib",
        use_camera=USE_CAMERA,
        radius=HEMISPHERE_RADIUS,
        max_angle=MAX_ANGLE_FROM_VERTICAL
    )
    
    collector.collect_data(
        board_center=BOARD_CENTER,
        num_poses=NUM_POSES
    )


if __name__ == "__main__":
    main()
