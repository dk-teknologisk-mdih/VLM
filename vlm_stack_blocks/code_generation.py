"""LLM-based robot code generation using Claude.

Note: `ABB_task_description_VLM.txt` and `ABB_Best_Practices.txt` live in the
LLM's RAG system — we reference them by name so the model retrieves them itself.
We do NOT inject their contents into the prompt.
"""

import re
import yaml
import json
import os
import requests

# Disable SSL warnings for self-signed certificates
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def build_stacking_prompt(
    stacking_plan_3d,
    robot_type="ABB",
    code_language="Rapid",
    task_description_file="ABB_task_description_VLM.txt",
    best_practices_file="ABB_Best_Practices.txt",
):
    """Build the initial prompt for stacking-trajectory robot code generation.

    References the task-description and best-practices files by name (the LLM
    retrieves them via RAG); their contents are NOT embedded here.
    """
    stacking_plan_yaml = yaml.dump(
        stacking_plan_3d, default_flow_style=False, sort_keys=False
    )
    prompt = f"""You are an expert robotics engineer. Generate {code_language} code to control a {robot_type} robot arm.

Solve the task described in `{task_description_file}` (available via RAG).
Adhere to the conventions in `{best_practices_file}` (available via RAG).
Only output the code.

The trajectory plan below gives pick and place operations with 3D waypoints in
camera coordinates (meters). Each step has:
- action: "pick" or "place"
- block: which block is being manipulated
- trajectory: list of waypoints with [x, y, z] in the camera frame
- description: human-readable description

## Stacking Plan (3D Camera Coordinates):
```yaml
{stacking_plan_yaml}
```
"""
    return prompt


def extract_code_block(text):
    """Strip ```rapid / ``` fences from an LLM response. Returns the raw code."""
    if not text:
        return text
    m = re.search(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _post_claude(api_key, base_url, messages, model="claude-opus-4-6", max_tokens=4096):
    """POST to the Claude proxy and return the response text (or None on failure)."""
    payload = {"model": model, "max_tokens": max_tokens, "messages": messages}
    url = f"{base_url}/v1/messages"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    try:
        response = requests.post(
            url, headers=headers, json=payload, verify=False, timeout=180
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
        return result["content"][0]["text"]
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Error extracting response from Claude: {e}")
        print(f"Full response: {response.text[:500]}")
        return None


def call_claude_for_robot_code(
    api_key,
    base_url,
    stacking_plan_3d,
    robot_type="ABB",
    code_language="Rapid",
    task_description_file="ABB_task_description_VLM.txt",
    best_practices_file="ABB_Best_Practices.txt",
    model="claude-opus-4-6",
):
    """Generate robot control code from a 3D stacking plan.

    Returns (generated_text, prompt). `prompt` can be passed back to
    `call_claude_to_fix_code` so the fix request carries the original context.
    """
    prompt = build_stacking_prompt(
        stacking_plan_3d,
        robot_type=robot_type,
        code_language=code_language,
        task_description_file=task_description_file,
        best_practices_file=best_practices_file,
    )

    print("\n=== Calling Claude for robot code generation ===")
    generated = _post_claude(
        api_key, base_url, [{"role": "user", "content": prompt}], model=model
    )
    if generated is None:
        return None, prompt

    os.makedirs("VLM_output", exist_ok=True)
    output_path = os.path.join("VLM_output", "4_robot_control_code.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(generated)
    print(f"Generated robot code saved to: {output_path}")
    return generated, prompt


def call_claude_to_fix_code(
    api_key,
    base_url,
    current_code,
    error_message,
    original_prompt,
    robot_type="ABB",
    code_language="Rapid",
    model="claude-opus-4-6",
):
    """Ask Claude to fix the given code based on an error/feedback message.

    The original prompt is replayed as prior context so the model still knows
    about the RAG files and the stacking plan.
    """
    fix_message = (
        f"The {code_language} code you produced for the {robot_type} robot has a problem.\n\n"
        #f"```{code_language.lower()}\n{current_code}\n```\n\n"
        f"Problem / error message:\n{error_message}\n\n"
        f"Return the corrected full module only, inside a single code block."
    )

    messages = [
        {"role": "user", "content": original_prompt},
        {"role": "assistant", "content": current_code},
        {"role": "user", "content": fix_message},
    ]

    print("\n=== Calling Claude to fix robot code ===")
    fixed = _post_claude(api_key, base_url, messages, model=model)
    if fixed is None:
        return None

    os.makedirs("VLM_output", exist_ok=True)
    output_path = os.path.join("VLM_output", "4_robot_control_code.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(fixed)
    print(f"Fixed robot code saved to: {output_path}")
    return fixed
