"""Structural interface for task plugins.

A "task" owns the VLM side of the pipeline (what to detect and how to turn
detections into a plan) and the wording of the LLM code-gen prompt built from
that plan. Everything downstream (LLM call, robot validation/simulation/
execution) stays generic and is unaware of which task produced the plan.

A task plugin implements this Protocol as a plain class (no inheritance
required) and exposes it via a `TASK_TYPE` marker + `create_backend(config)`
in its `backend.py`, discovered by `tasks/registry.py` — same convention as
`robots/`, `llm/`, and `vision/`.
"""

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class TaskBackend(Protocol):
    """Plans a task from VLM detections and builds the LLM code-gen prompt for it."""

    def plan(
        self,
        api_key: str,
        base_url: str,
        gui_config: Optional[dict],
        vision_backend: str,
    ) -> Optional[dict]:
        """Runs VLM detection/planning; returns a dict with at least "plan_data",
        or None if planning failed."""
        ...

    def build_prompt(
        self,
        plan_data,
        robot_type: str,
        code_language: str,
        task_description_file: str,
        best_practices_file: str,
    ) -> str:
        """Builds the LLM code-generation prompt from the planned task data."""
        ...
