"""
ABB Robot Current Pose Reader

Simple script to connect to an ABB robot and get the current TCP pose.
"""

import socket
import numpy as np
from scipy.spatial.transform import Rotation


def connect_robot(robot_ip: str, robot_port: int = 50000, timeout: int = 10) -> socket.socket:
    """Connect to the ABB robot via socket."""
    print(f"Connecting to robot at {robot_ip}:{robot_port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((robot_ip, robot_port))
    print("Connected!")
    return sock


def send_command(sock: socket.socket, command: str) -> str:
    """Send command to robot and get response."""
    sock.send((command + "\n").encode('utf-8'))
    response = sock.recv(1024).decode('utf-8').strip()
    return response


def get_current_pose(sock: socket.socket) -> dict:
    """
    Get current TCP pose from the robot.
    
    Returns:
        Dictionary with position (x, y, z) in mm and quaternion (q1, q2, q3, q4)
        ABB quaternion format: q1=w, q2=x, q3=y, q4=z
    """
    response = send_command(sock, "POSE")
    
    if response.startswith("GRIP:"):
        values = [float(v) for v in response[5:].split(",")]
        x, y, z, q1, q2, q3, q4 = values
        return {
            'x': x,
            'y': y,
            'z': z,
            'q1': q1,  # w
            'q2': q2,  # x
            'q3': q3,  # y
            'q4': q4   # z
        }
    else:
        print(f"Unexpected response: {response}")
        return None


def pose_to_euler(pose: dict) -> dict:
    """
    Convert quaternion pose to Euler angles (degrees).
    
    Args:
        pose: Dictionary with q1, q2, q3, q4 (ABB format: q1=w, q2=x, q3=y, q4=z)
        
    Returns:
        Dictionary with roll, pitch, yaw in degrees
    """
    # Convert ABB quaternion (w, x, y, z) to scipy format (x, y, z, w)
    rotation = Rotation.from_quat([pose['q2'], pose['q3'], pose['q4'], pose['q1']])
    euler = rotation.as_euler('xyz', degrees=True)
    return {
        'roll': euler[0],
        'pitch': euler[1],
        'yaw': euler[2]
    }


def print_pose(pose: dict):
    """Print pose in a readable format."""
    print("\n" + "=" * 50)
    print("Current Robot TCP Pose")
    print("=" * 50)
    print(f"\nPosition (mm):")
    print(f"  X: {pose['x']:.3f}")
    print(f"  Y: {pose['y']:.3f}")
    print(f"  Z: {pose['z']:.3f}")
    print(f"\nOrientation (quaternion):")
    print(f"  q1 (w): {pose['q1']:.6f}")
    print(f"  q2 (x): {pose['q2']:.6f}")
    print(f"  q3 (y): {pose['q3']:.6f}")
    print(f"  q4 (z): {pose['q4']:.6f}")
    
    euler = pose_to_euler(pose)
    print(f"\nOrientation (Euler angles, degrees):")
    print(f"  Roll:  {euler['roll']:.3f}")
    print(f"  Pitch: {euler['pitch']:.3f}")
    print(f"  Yaw:   {euler['yaw']:.3f}")
    print("=" * 50 + "\n")


def main():
    # Robot connection settings
    ROBOT_IP = "192.168.125.1"  # Change to your robot's IP
    ROBOT_PORT = 50000
    
    sock = None
    try:
        # Connect to robot
        sock = connect_robot(ROBOT_IP, ROBOT_PORT)
        
        # Get and print current pose
        pose = get_current_pose(sock)
        if pose:
            print_pose(pose)
        else:
            print("Failed to get pose from robot.")
            
    except socket.timeout:
        print("Connection timed out. Check robot IP and ensure server is running.")
    except ConnectionRefusedError:
        print("Connection refused. Ensure robot server is running.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if sock:
            sock.close()
            print("Disconnected from robot.")


if __name__ == "__main__":
    main()
