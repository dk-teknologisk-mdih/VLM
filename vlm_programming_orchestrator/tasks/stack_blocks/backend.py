"""Task plugin: detect blocks + a target location, plan a stack/destack trajectory."""

import os

import yaml

from ..common import load_prompt_template
from .pipeline import plan_stacking_trajectory

TASK_TYPE = "stack_blocks"

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


class StackBlocksTaskBackend:
    """`TaskBackend` for the block-stacking task."""

    def __init__(self, config):
        self.config = config

    def plan(self, api_key, base_url, gui_config, vision_backend):
        return plan_stacking_trajectory(
            api_key, base_url, config=gui_config, vision_backend=vision_backend,
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


def create_backend(config) -> StackBlocksTaskBackend:
    """Build the `TaskBackend` for the block-stacking task."""
    return StackBlocksTaskBackend(config)
