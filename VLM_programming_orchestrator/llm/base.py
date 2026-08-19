"""Structural interface for LLM-based robot code generation/fixing."""

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class CodeGenBackend(Protocol):
    """Generates and fixes robot code from a task/trajectory plan."""

    def generate_code(
        self,
        stacking_plan_3d,
        robot_type: str,
        code_language: str,
        task_description_file: str,
        best_practices_file: str,
        events=None,
    ) -> tuple[Optional[str], str]:
        """Returns (generated_text_or_None, prompt_used)."""
        ...

    def fix_code(
        self,
        current_code: str,
        error_message: str,
        original_prompt: str,
        robot_type: str,
        code_language: str,
        events=None,
    ) -> Optional[str]:
        """Returns the fixed generated_text, or None on failure."""
        ...
