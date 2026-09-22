"""
Configuration and constants for the VLM programming orchestrator.
"""

# pylint: disable=W1203, W0718

import logging
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """All tuneable parameters for the VLM pipeline in one place."""

    # ----- LLM / proxy -----
    # API_KEY and BASE_URL are loaded from the environment at startup.
    api_key: str = field(default_factory=lambda: os.environ.get("API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.environ.get("BASE_URL", ""))
    llm_model: str = "claude-opus-4-6"
    llm_max_tokens: int = 4096
    max_retry_attempts: int = 3

    # ----- Robot code generation -----
    generate_robot_code: bool = True
    robot_type: str = "ABB"
    code_language: str = "Rapid"

    # ----- Swappable backends (see robots/, llm/, vision/, tasks/ registries) -----
    llm_backend: str = "claude_proxy"
    vision_backend: str = "gemini_realsense"

    # ----- Task selection (see tasks/ registry) -----
    # e.g. "stack_blocks" or "path_inspection". Determines what the VLM
    # detects/plans and how the LLM prompt is built; robot/LLM/vision
    # backends stay unaware of which task is active.
    task_type: str = "stack_blocks"

    # Offline-editable prompt template overrides for the active task. When
    # set, the task backend loads these instead of its bundled default
    # `prompts/*.txt` template. None means "use the task's own default".
    vlm_prompt_file: Optional[Path] = None
    llm_prompt_file: Optional[Path] = None

    # ----- RAG-accessible filenames (referenced by name only, not read) -----
    task_description_file: str = "ABB_task_description_VLM.txt"
    best_practices_file: str = "ABB_Best_Practices.txt"

    # ----- File paths -----
    local_output_dir: Path = Path("./generated_modules")
    ftp_shared_dir: Path = Path("C:\\Users\\sem\\Documents\\AIMatters\\FTP_RobotPrograms")
    module_filename: str = "generated_task.mod"

    vlm_output_dir: Path = Path("VLM_output")
    vlm_input_dir: Path = Path("VLM_input")
    llm_output_dir: Path = Path("LLM_output")
    llm_input_dir: Path = Path("LLM_input")
    gui_config_path: Path = Path("VLM_input/GUI_output.yaml")

    # ----- Robot socket connection -----
    robot_ip: str = "192.168.125.1"
    robot_port: int = 1025
    socket_timeout: float = 120.0

    # ----- Robot validation and simulation -----
    # Despite the legacy name, this gates the selected backend's verifier and
    # simulator: RobotStudio for ABB, URScript validation and URSim for UR.
    use_robotstudio_validation: bool = True
    robotstudio_window_title: str = "AIMatters_stacking - RobotStudio"
    controller_id: str = "15000-500064"
    module_name: str = "MainModule"

    # ----- Touchscreen lockout during simulation -----
    # When True, the HID touchscreen device(s) matching `touch_device_filter`
    # are disabled while a RobotStudio simulation is running and re-enabled
    # when it stops. Requires the process to run elevated (Administrator),
    # otherwise the PowerShell call silently fails and touch stays enabled.
    disable_touch_during_simulation: bool = True
    # Case-insensitive substring matched against PnP device FriendlyName.
    touch_device_filter: str = "touch screen"

    # ----- UR / URSim -----
    ursim_ip: str = "127.0.0.1"
    ur_dashboard_port: int = 29999
    ur_dashboard_timeout: float = 10.0
    # Host dir bind-mounted into the URSim container's programs folder.
    ursim_programs_dir: Path = Path("./URSim_programs")
    # Static Polyscope program (user-authored once) with an Import-Script-File
    # node that loads `module_filename` from the shared programs folder.
    ur_container_program: str = "AIMatters_container.urp"
    # Real UR controller IP; blank disables real-robot execution.
    ur_robot_ip: str = ""
    ur_validate_image: str = "universalrobots/ursim_e-series"
    ur_validate_robot_model: str = "UR5"
    ur_validate_timeout: int = 10

    # ----- Human review -----
    require_human_review: bool = True

    # ----- Display / monitor selection -----
    # Name of the target monitor for the wizard GUI and review dialogs.
    # Matched as a case-insensitive substring against screeninfo monitor
    # names. On Windows these look like "\\.\DISPLAY1", "\\.\DISPLAY2", ...
    # which correspond to the display numbers in Windows Settings, so values
    # like "DISPLAY3" or just "3" both work. None falls back to the
    # VLM_GUI_DISPLAY env var, then the primary monitor.
    display_name: Optional[str] = "XWAYLAND0"

    # ----- Continuous loop -----
    cycle_delay_seconds: float = 2.0

    # ----- Free-form task description (overrides GUI if set) -----
    task_description: Optional[str] = None

    def __post_init__(self):
        self.robot_type = self.robot_type.strip().upper()
        if self.robot_type == "UR":
            if self.code_language == "Rapid":
                self.code_language = "URScript"
            if self.task_description_file == "ABB_task_description_VLM.txt":
                self.task_description_file = "UR_task_description.pdf"
            if self.best_practices_file == "ABB_Best_Practices.txt":
                self.best_practices_file = "UR_Best_Practices.docx"
            if self.module_filename == "generated_task.mod":
                self.module_filename = "generated_task.script"

        # Propagate display selection to env var so submodules (GUI, human
        # review dialogs) pick it up via display_utils.get_target_monitor().
        if self.display_name is not None:
            os.environ["VLM_GUI_DISPLAY"] = str(self.display_name)


class State(Enum):
    """States for the orchestrator's state machine."""
    RUN_GUI = auto()
    PLAN_TRAJECTORY = auto()
    GENERATE_CODE = auto()
    SAVE_MODULE = auto()
    VALIDATE_IN_ROBOTSTUDIO = auto()
    FIX_SYNTAX_ERRORS = auto()
    SIMULATE = auto()
    HUMAN_REVIEW = auto()
    TRANSFER_TO_ROBOT = auto()
    SEND_LOAD = auto()
    SEND_START = auto()
    CLEANUP = auto()
    DONE = auto()
    ERROR = auto()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("orchestrator.log"),
    ],
)
logger = logging.getLogger("orchestrator")
