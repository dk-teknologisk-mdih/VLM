"""Task plugin: detect an object (e.g. a bike frame) and plan an inspection path along it."""

import os

import yaml

from ..common import load_prompt_template
from .pipeline import plan_inspection_path

TASK_TYPE = "path_inspection"

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


class PathInspectionTaskBackend:
    """`TaskBackend` for the object-inspection-path task."""

    def __init__(self, config):
        self.config = config

    def plan(self, api_key, base_url, gui_config, vision_backend):
        return plan_inspection_path(
            api_key, base_url, config=gui_config, vision_backend=vision_backend,
            vlm_prompt_file=self.config.vlm_prompt_file,
        )

    def build_prompt(self, plan_data, robot_type, code_language, task_description_file, best_practices_file):
        template_path = self.config.llm_prompt_file or os.path.join(
            _PROMPTS_DIR, "llm_codegen_prompt.txt")
        plan_yaml = yaml.dump(
            plan_data, default_flow_style=False, sort_keys=False)
        return load_prompt_template(
            template_path,
            plan_yaml=plan_yaml,
            robot_type=robot_type,
            code_language=code_language,
            task_description_file=task_description_file,
            best_practices_file=best_practices_file,
        )


def create_backend(config) -> PathInspectionTaskBackend:
    """Build the `TaskBackend` for the path-inspection task."""
    return PathInspectionTaskBackend(config)
