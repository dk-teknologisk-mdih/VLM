"""
Human review utilities for the VLM programming orchestrator.
"""
# pylint: disable=W1203, W0718

import logging
import tkinter as tk
from tkinter import messagebox, simpledialog

logger = logging.getLogger("orchestrator")


def ask_human_review() -> bool:
    """Pop-up asking if the simulation looked correct. Returns True if approved."""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        result = messagebox.askyesno(
            "Human Review",
            "The simulation has completed.\n\n"
            "Did the robot behavior look correct?\n\n"
            "Click 'Yes' to send the code to the physical robot.\n"
            "Click 'No' to regenerate the code.",
            parent=root,
        )
        root.destroy()
        return result
    except Exception as e:
        logger.error(f"Human review dialog failed: {e}")
        response = input(
            "Did the simulation look correct? (y/n): ").strip().lower()
        return response in ("y", "yes")


def ask_human_review_with_feedback() -> tuple[bool, str]:
    """
    Pop-up asking if the simulation looked correct.
    On rejection, prompts for free-text feedback to pass to the LLM.
    Returns (approved, feedback_text).
    """
    approved = ask_human_review()
    if approved:
        return True, ""

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        feedback = simpledialog.askstring(
            "What went wrong?",
            "Describe what was wrong with the simulation so the LLM can fix it:",
            parent=root,
        )
        root.destroy()
        return False, (feedback or "").strip()
    except Exception as e:
        logger.error(f"Feedback dialog failed: {e}")
        feedback = input("Describe what was wrong: ").strip()
        return False, feedback


def ask_skip_validation() -> bool:
    """Ask the user whether to skip RobotStudio validation. Returns True to skip."""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        result = messagebox.askyesno(
            "RobotStudio Not Available",
            "Could not connect to RobotStudio for validation.\n\n"
            "Do you want to skip validation and proceed anyway?",
            parent=root,
        )
        root.destroy()
        return result
    except Exception as e:
        logger.error(f"Skip-validation dialog failed: {e}")
        response = input(
            "RobotStudio not available. Skip validation? (y/n): "
        ).strip().lower()
        return response in ("y", "yes")


if __name__ == "__main__":
    ok, fb = ask_human_review_with_feedback()
    print(f"Approved: {ok}, Feedback: {fb!r}")
