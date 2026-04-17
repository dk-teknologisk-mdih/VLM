"""LLM-based robot code generation using Claude."""

import yaml
import json
import os
import requests

# Disable SSL warnings for self-signed certificates
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def call_claude_for_robot_code(api_key, base_url, stacking_plan_3d, robot_type="ABB", code_language="Rapid"):
    """
    Send stacking plan to Claude model to generate robot control code.
    
    Args:
        api_key (str): API key for the proxy
        base_url (str): Base URL for the proxy server
        stacking_plan_3d (list): Stacking plan with 3D coordinates
        robot_type (str): Type of robot (e.g., "ABB", "UR", "Fanuc")
        code_language (str): Programming language for the robot code (e.g., "Rapid", "Python")
    Returns:
        str: Generated robot control code
    """
    model_id = "claude-opus-4-6"  # Claude model
    
    # Format the stacking plan as YAML for the prompt
    stacking_plan_yaml = yaml.dump(stacking_plan_3d, default_flow_style=False, sort_keys=False)
    
    prompt = f"""You are an expert robotics engineer. Generate {code_language} code to control a {robot_type} robot arm to execute the following stacking trajectory plan.

The trajectory plan contains pick and place operations with 3D waypoints in camera coordinates (meters).
Each step has:
- action: "pick" or "place"
- block: which block is being manipulated
- trajectory: list of waypoints with [x, y, z] coordinates in camera frame
- description: human-readable description

## Stacking Plan (3D Camera Coordinates):
```yaml
{stacking_plan_yaml}
```
"""

    # Build the request payload for Claude
    payload = {
        "model": model_id,
        "max_tokens": 4096,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    # Make the request to the proxy
    url = f"{base_url}/v1/messages"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    print("\n=== Calling Claude model for robot code generation ===")
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            verify=False,
            timeout=180,
        )
    except requests.exceptions.RequestException as e:
        print(f"Request to Claude failed: {e}")
        return None
    
    if response.status_code != 200:
        print(f"Error: Status code {response.status_code}")
        print(f"Response: {response.text}")
        return None
    
    try:
        result = response.json()
        generated_code = result["content"][0]["text"]
        
        # Save the generated code
        output_path = os.path.join("VLM_output", "4_robot_control_code.txt")
        with open(output_path, "w") as f:
            f.write(generated_code)
        print(f"Generated robot code saved to: {output_path}")
        
        return generated_code
        
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Error extracting response from Claude: {e}")
        print(f"Full response: {response.text[:500]}")
        return None
