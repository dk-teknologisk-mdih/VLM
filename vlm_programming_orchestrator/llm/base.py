"""Structural interface for LLM-based robot code generation/fixing."""

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class CodeGenBackend(Protocol):
    """Generates and fixes robot code from an already-built prompt.

    Prompt construction is the task plugin's responsibility (see
    `tasks/base.py::TaskBackend.build_prompt`) — this backend only knows how
    to talk to an LLM, not what task produced the prompt.
    """

    def generate_code(self, prompt: str, events=None) -> Optional[str]:
        """Returns the generated text, or None on failure."""
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
