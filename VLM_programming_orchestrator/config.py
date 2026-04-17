# pylint: disable=W1203, W0718

import os
import logging
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """All tuneable parameters for the VLM pipeline in one place."""

    # ----- LLM / proxy -----
    # API_KEY and BASE_URL are loaded from the environment at startup.
    api_key: str = field(default_factory=lambda: os.environ.get("API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("BASE_URL", ""))
    llm_model: str = "claude-opus-4-6"
    llm_max_tokens: int = 4096
    max_retry_attempts: int = 3

    # ----- Robot code generation -----
    generate_robot_code: bool = True
    robot_type: str = "ABB"
    code_language: str = "Rapid"

    # ----- RAG-accessible filenames (referenced by name only, not read) -----
    task_description_file: str = "ABB_task_description_VLM.txt"
    best_practices_file: str = "ABB_Best_Practices.txt"

    # ----- File paths -----
    local_output_dir: Path = Path("./generated_modules")
    ftp_shared_dir: Path = Path("C:/ftp_share")
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

    # ----- RobotStudio UI automation -----
    use_robotstudio_validation: bool = True
    robotstudio_window_title: str = "RobotStudio"
    controller_id: str = "15000-500064"
    module_name: str = "generated_task"

    # ----- Human review -----
    require_human_review: bool = True

    # ----- Continuous loop -----
    cycle_delay_seconds: float = 2.0

    # ----- Free-form task description (overrides GUI if set) -----
    task_description: Optional[str] = None


class State(Enum):
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
