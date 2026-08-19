"""Rewrites LLM-generated code to match the robot's entry-point convention."""


class RapidModulePreparer:
    """RAPID modules run via `run_task()`, but the LLM is prompted to write `main()`."""

    def prepare(self, code: str) -> str:
        return code.replace("main()", "run_task()")
