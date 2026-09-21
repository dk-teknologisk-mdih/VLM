"""Real UR controller connection via the Dashboard Server.

Implements the RobotExecutor interface (see robots/base.py) for the UR robot
backend. Assumes the orchestrator's generic TRANSFER_TO_ROBOT step has already
copied the generated .script into `config.ftp_shared_dir`, a folder the
controller's static container .urp program (config.ur_container_program)
imports from.
"""
# pylint: disable=W1203, W0718

import logging
import time

from ...config import Config
from .dashboard_client import DashboardClient

logger = logging.getLogger("orchestrator")


class URRobotExecutor:
    """Loads/runs the static container program on a real UR controller."""

    def __init__(self, config: Config):
        self.config = config
        self.dashboard = DashboardClient(
            config.ur_robot_ip, config.ur_dashboard_port, config.ur_dashboard_timeout)

    def is_connected(self) -> bool:
        return self.dashboard.is_connected()

    def connect(self) -> bool:
        if not self.config.ur_robot_ip:
            logger.error("config.ur_robot_ip is not set; no real UR robot configured.")
            return False
        return self.dashboard.connect()

    def disconnect(self) -> None:
        self.dashboard.disconnect()

    def load_module(self, filename: str) -> tuple[bool, str]:
        # `filename` is the .script placed in the shared folder; the dashboard
        # loads the fixed container program that imports it, not the script itself.
        if filename != self.config.module_filename:
            logger.warning(
                f"load_module filename {filename!r} != config.module_filename "
                f"{self.config.module_filename!r}; loading container program anyway."
            )
        response = self.dashboard.load(self.config.ur_container_program)
        if response.lower().startswith("loading program"):
            logger.info(f"Container program '{self.config.ur_container_program}' loaded on robot.")
            return True, response
        logger.error(f"Failed to load container program: {response}")
        return False, response

    def start_execution(self) -> tuple[bool, str]:
        response = self.dashboard.play()
        if not response.lower().startswith("starting program"):
            logger.error(f"Failed to start execution: {response}")
            return False, response

        logger.info("Execution started on robot. Waiting for completion...")
        success, message = self._wait_until_stopped(timeout=600)
        if success:
            logger.info("Robot execution completed successfully.")
        else:
            logger.error(f"Robot execution error: {message}")
        return success, message

    def _wait_until_stopped(self, timeout: float) -> tuple[bool, str]:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self.dashboard.is_running():
                break
            time.sleep(2)
        else:
            return False, "Execution did not complete within the timeout period."

        status = self.dashboard.safety_status()
        if status == "NORMAL":
            return True, "DONE"
        self.dashboard.close_popup()
        self.dashboard.unlock_protective_stop()
        return False, f"EXEC_ERR:safety status {status}"
