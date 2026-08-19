"""Generic (robot/LLM-agnostic) prompt templating for robot code generation.

The LLM call itself now lives in the swappable `llm/` backends (see
`llm/claude_proxy/backend.py`); this module only builds prompt text and
parses code fences, parameterized by `robot_type`/`code_language`.
"""

import re

import yaml


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
    
    # Remove em-dashes and en-dashes that may be used as fences instead of backticks
    text = text.replace("—", "-").replace("–", "-")

    m = re.search(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()



