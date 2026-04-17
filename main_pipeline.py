"""
Main Pipeline Coordinator

Orchestrates the full block stacking workflow:
  1. GUI Input       – User selects task, blocks, stacking order, and target via GUI
  2. VLM Planning    – Gemini Robotics-ER detects objects, plans stacking trajectory
  3. Robot Code Gen  – Claude generates robot control code from the 3D trajectory
"""

import yaml
import os
import sys
import json
import requests
from dotenv import load_dotenv

from GUI_VLM_input import get_user_input
from vlm_stack_blocks import plan_stacking_trajectory

# Disable SSL warnings for self-signed certificates
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
API_KEY = os.environ["API_KEY"]
BASE_URL = os.environ["BASE_URL"]

GENERATE_ROBOT_CODE = True        # Set True to also call Claude for ABB Rapid code
ROBOT_TYPE = "ABB"
CODE_LANGUAGE = "Rapid"
# ─────────────────────────────────────────────────────────────────────────────


def call_assistant_api(stacking_plan_3d):
    """
    Call the assistant API to generate robot control code from a 3D stacking plan.
    """
    stacking_plan_yaml = yaml.dump(stacking_plan_3d, default_flow_style=False, sort_keys=False)

    prompt = (
        f"Solve the task in ABB_task_description_VLM.txt.\n"
        f"Only write the code. \n"
        f"This is the task plan given in 3D camera coordinates: \n"
        f"\n{stacking_plan_yaml}\n"
    )

#     payload = {
#   "prompt": "Solve the task in abb_task_description_vlm.txt\nThis is the task plan: \nyaml\n- step: 1\n  action: pick\n  block: Block 1\n  trajectory:\n  - point:\n    - 0.035975825041532516\n    - -0.06488277018070221\n    - 0.37\n    label: approach_point\n  - point:\n    - 0.035975825041532516\n    - -0.06488277018070221\n    - 0.57\n    label: pick_point\n  - point:\n    - 0.035975825041532516\n    - -0.06488277018070221\n    - 0.37\n    label: retract_point\n- step: 2\n  action: place\n  block: Block 1\n  trajectory:\n  - point:\n    - -0.0\n    - 0.0\n    - 0.37\n    label: approach_point\n  - point:\n    - -0.0\n    - 0.0\n    - 0.57\n    label: place_point\n  - point:\n    - -0.0\n    - 0.0\n    - 0.37\n    label: retract_point\n- step: 3\n  action: pick\n  block: Block 2\n  trajectory:\n  - point:\n    - 0.12560449540615082\n    - -0.06459253281354904\n    - 0.37\n    label: approach_point\n  - point:\n    - 0.12560449540615082\n    - -0.06459253281354904\n    - 0.57\n    label: pick_point\n  - point:\n    - 0.12560449540615082\n    - -0.06459253281354904\n    - 0.37\n    label: retract_point\n- step: 4\n  action: place\n  block: Block 2\n  trajectory:\n  - point:\n    - -0.0\n    - 0.0\n    - 0.37\n    label: approach_point\n  - point:\n    - -0.0\n    - 0.0\n    - 0.53\n    label: place_point\n  - point:\n    - -0.0\n    - 0.0\n    - 0.37\n    label: retract_point\n- step: 5\n  action: pick\n  block: Block 3\n  trajectory:\n  - point:\n    - -0.03590773791074753\n    - -0.06310895830392838\n    - 0.37\n    label: approach_point\n  - point:\n    - -0.03590773791074753\n    - -0.06310895830392838\n    - 0.57\n    label: pick_point\n  - point:\n    - -0.03590773791074753\n    - -0.06310895830392838\n    - 0.37\n    label: retract_point\n- step: 6\n  action: place\n  block: Block 3\n  trajectory:\n  - point:\n    - -0.0\n    - 0.0\n    - 0.37\n    label: approach_point\n  - point:\n    - -0.0\n    - 0.0\n    - 0.49\n    label: place_point\n  - point:\n    - -0.0\n    - 0.0\n    - 0.37\n    label: retract_point\n- step: 7\n  action: wait\n  duration: 5\n- step: 8\n  action: pick\n  block: Block 3\n  trajectory:\n  - point:\n    - -0.0\n    - 0.0\n    - 0.37\n    label: approach_point\n  - point:\n    - -0.0\n    - 0.0\n    - 0.49\n    label: pick_point\n  - point:\n    - -0.0\n    - 0.0\n    - 0.37\n    label: retract_point\n- step: 9\n  action: place\n  block: Block 3\n  trajectory:\n  - point:\n    - -0.03590773791074753\n    - -0.06310895830392838\n    - 0.37\n    label: approach_point\n  - point:\n    - -0.03590773791074753\n    - -0.06310895830392838\n    - 0.57\n    label: place_point\n  - point:\n    - -0.03590773791074753\n    - -0.06310895830392838\n    - 0.37\n    label: retract_point\n- step: 10\n  action: pick\n  block: Block 2\n  trajectory:\n  - point:\n    - -0.0\n    - 0.0\n    - 0.37\n    label: approach_point\n  - point:\n    - -0.0\n    - 0.0\n    - 0.53\n    label: pick_point\n  - point:\n    - -0.0\n    - 0.0\n    - 0.37\n    label: retract_point\n- step: 11\n  action: place\n  block: Block 2\n  trajectory:\n  - point:\n    - 0.12560449540615082\n    - -0.06459253281354904\n    - 0.37\n    label: approach_point\n  - point:\n    - 0.12560449540615082\n    - -0.06459253281354904\n    - 0.57\n    label: place_point\n  - point:\n    - 0.12560449540615082\n    - -0.06459253281354904\n    - 0.37\n    label: retract_point\n- step: 12\n  action: pick\n  block: Block 1\n  trajectory:\n  - point:\n    - -0.0\n    - 0.0\n    - 0.37\n    label: approach_point\n  - point:\n    - -0.0\n    - 0.0\n    - 0.57\n    label: pick_point\n  - point:\n    - -0.0\n    - 0.0\n    - 0.37\n    label: retract_point\n- step: 13\n  action: place\n  block: Block 1\n  trajectory:\n  - point:\n    - 0.035975825041532516\n    - -0.06488277018070221\n    - 0.37\n    label: approach_point\n  - point:\n    - 0.035975825041532516\n    - -0.06488277018070221\n    - 0.57\n    label: place_point\n  - point:\n    - 0.035975825041532516\n    - -0.06488277018070221\n    - 0.37\n    label: retract_point\n\n"
# }

    # Save the full prompt to file
    prompt_output_path = os.path.join("LLM_input", "prompt.txt")
    with open(prompt_output_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"Full prompt saved to: {prompt_output_path}")

    url = f"https://yoda.localdom.net:8443/organization/C040/ai/create_assistent_api/"
    payload = {"prompt": prompt}
    print("\n=== Calling assistant API for robot code generation ===")

    #print(prompt[:500])  # Print the prompt being sent (truncated for readability)

    try:
        response = requests.post(url, json=payload, verify=False, timeout=180)
    except requests.exceptions.RequestException as e:
        print(f"Request to assistant API failed: {e}")
        return None

    if response.status_code != 200:
        print(f"Error: Status code {response.status_code}")
        print(f"Response: {response.text}")
        return None

    try:
        result = response.text
    
        output_path = os.path.join("LLM_output", "robot_control_code.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Generated robot code saved to: {output_path}")

        return result
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing assistant API response: {e}")
        print(f"Raw response: {response.text[:500]}")
        return None
    
def call_assistant_chat_api(stacking_plan):
    """
    Example function to call a chat-based assistant API endpoint.
    Not currently used in the main pipeline, but can be adapted for interactive conversations.
    """

    stacking_plan_yaml = yaml.dump(stacking_plan, default_flow_style=False, sort_keys=False)


    url = f"https://yoda.localdom.net:8443/organization/C040/ai/create_assistent_api/chat"
    messages = [
        {"role": "user", "content": f"Solve the task in 'ABB_task_description_VLM.txt'. Only write the code. Remember to adhere to best practices (ABB_Best_Practices.txt)  \nThis is the task plan given in 3D camera coordinates: \n{stacking_plan_yaml}"}
    ]

    payload = {"messages": messages}
    # save the message content (prompt) to a txt file for reference
    prompt_output_path = os.path.join("LLM_input", "chat_api_prompt.txt")
    with open(prompt_output_path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(f"{msg['content']}\n")
    print(f"Chat API prompt saved to: {prompt_output_path}")

    try:
        response = requests.post(url, json=payload, verify=False, timeout=180)
        response.raise_for_status()
        result = response.json()  # Assuming the response is in JSON format
        # print("Assistant response:", result)
        # save result to txt file 
        output_path = os.path.join("LLM_output", "chat_api_response.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(result["content"], indent=2))
        print(f"Assistant response saved to: {output_path}")

        if "content" in result:
            code_output_path = os.path.join("LLM_output", "robot_control_code.md")
            with open(code_output_path, "w", encoding="utf-8") as f:
                f.write(result["content"])
            print(f"Robot control code saved to: {code_output_path}")

        return result
    except requests.exceptions.RequestException as e:
        print(f"Request to assistant API failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding assistant API response: {e}")
        print(f"Raw response: {response.text[:500]}")
        return None


def load_config_from_file(path="VLM_input/GUI_output.yaml"):
    """Load a previously saved GUI configuration from YAML."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    # Convert target_position list back to tuple if present
    if "target_position" in config and isinstance(config["target_position"], list):
        config["target_position"] = tuple(config["target_position"])
    return config


def run_pipeline(config):
    """
    Execute the VLM planning (and optionally robot code generation) pipeline.

    Args:
        config: Task configuration dict produced by the GUI.

    Returns:
        dict with trajectory_result and optionally robot_code.
    """
    task = config.get("task", "stack blocks")
    print(f"\n{'='*60}")
    print(f"  Task: {task}")
    print(f"{'='*60}")

    if task == "stack blocks":
        # ── Step 2: VLM trajectory planning ──────────────────────────
        print("\n>>> Step 2/3: Running VLM trajectory planning …")
        trajectory_result = plan_stacking_trajectory(
            API_KEY, BASE_URL, config=config
        )

        # ── Step 3 (optional): Robot code generation ─────────────────
        robot_code = None
        if GENERATE_ROBOT_CODE:
            print("\n>>> Step 3/3: Generating robot control code …")
            robot_code = call_assistant_chat_api(
                stacking_plan=trajectory_result["stacking_plan_3d"]
            )
        else:
            print("\n>>> Step 3/3: Robot code generation skipped (GENERATE_ROBOT_CODE=False)")

        return {"trajectory_result": trajectory_result, "robot_code": robot_code}

    elif task == "sort blocks":
        print("Sort-blocks pipeline is not yet implemented.")
        return None

    else:
        print(f"Unknown task: {task}")
        return None


def main():
    """
    Full pipeline entry-point.

    Usage:
        python main_pipeline.py              # launch GUI then run pipeline
        python main_pipeline.py --from-file  # reuse last saved GUI config
    """
    use_saved = "--from-file" in sys.argv

    if use_saved:
        config_path = "VLM_input/GUI_output.yaml"
        if not os.path.exists(config_path):
            print(f"Error: saved config not found at {config_path}")
            print("Run without --from-file to open the GUI first.")
            sys.exit(1)
        print(">>> Step 1/3: Loading saved GUI configuration …")
        config = load_config_from_file(config_path)
    else:
        # ── Step 1: GUI input ────────────────────────────────────────
        print(">>> Step 1/3: Opening GUI for task configuration …")
        config = get_user_input()
        if config is None:
            print("User cancelled – exiting.")
            sys.exit(0)

    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    result = run_pipeline(config)

    if result:
        print(f"\n{'='*60}")
        print("  Pipeline complete ✓")
        print(f"{'='*60}")


def test():
    #load stacking plan from file for testing
    with open("./VLM_output/3_stacking_trajectory_plan.yaml", "r") as f:
        stacking_plan_3d = yaml.safe_load(f)

    # call_assistant_api(stacking_plan_3d)
    output = call_assistant_chat_api(stacking_plan_3d)

    # save result to file
    if output:
        output_path = os.path.join("LLM_output", "chat_api_response.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Chat API response saved to: {output_path}")

        if "content" in output:
            code_output_path = os.path.join("LLM_output", "robot_control_code.md")
            with open(code_output_path, "w", encoding="utf-8") as f:
                f.write(output["content"])
            print(f"Robot control code saved to: {code_output_path}")


if __name__ == "__main__":
    #test()
    main()
