"""URSim automation via the Dashboard Server.

Implements the Simulator and ErrorDescriber interfaces (see robots/base.py)
for the UR robot backend, driving a local URSim Docker container.
"""
# pylint: disable=W1203, W0718

import logging
import shutil
import time

from ...config import Config
from .dashboard_client import DashboardClient

logger = logging.getLogger("orchestrator")


class URSimSimulator:
    """Copies the generated .script into URSim's shared programs folder and runs it."""

    def __init__(self, config: Config):
        self.config = config
        self.dashboard = DashboardClient(
            config.ursim_ip, config.ur_dashboard_port, config.ur_dashboard_timeout)

    def start_simulation(self) -> bool:
        # URSim's dashboard `load` reads from its own mounted programs folder, so the
        # generated .script must land there before the static container .urp is played.
        src = self.config.local_output_dir / self.config.module_filename
        dst = self.config.ursim_programs_dir / self.config.module_filename
        try:
            self.config.ursim_programs_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except (IOError, OSError) as e:
            logger.error(f"Could not copy {src} to URSim programs dir: {e}")
            return False

        if not self.dashboard.is_connected() and not self.dashboard.connect():
            return False

        response = self.dashboard.load(self.config.ur_container_program)
        if not response.lower().startswith("loading program"):
            logger.error(f"URSim failed to load {self.config.ur_container_program}: {response}")
            return False

        response = self.dashboard.play()
        if not response.lower().startswith("starting program"):
            logger.error(f"URSim failed to start playback: {response}")
            return False

        logger.info("URSim simulation started.")
        return True

    def stop_simulation(self) -> bool:
        if not self.dashboard.is_connected():
            return True
        try:
            self.dashboard.stop()
            logger.info("URSim simulation stopped.")
            return True
        except Exception as e:
            logger.error(f"Could not stop URSim simulation: {e}")
            return False

    def wait_for_simulation_complete(self, timeout: float = 60 * 5) -> tuple[bool, str]:
        logger.info(f"Waiting for URSim simulation to complete (timeout={timeout}s)...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if not self.dashboard.is_running():
                    break
            except Exception as e:
                return False, f"Lost connection to URSim dashboard: {e}"
            time.sleep(2)
        else:
            elapsed = time.time() - start_time
            logger.warning(f"URSim simulation did not complete within {elapsed:.1f}s (timeout).")
            return False, "Simulation did not complete within the timeout period."

        elapsed = time.time() - start_time
        status = self.dashboard.safety_status()
        if status == "NORMAL":
            logger.info(f"URSim simulation completed successfully in {elapsed:.1f}s.")
            return True, "Simulation completed successfully."

        logger.error(f"URSim simulation ended with safety status {status} after {elapsed:.1f}s.")
        self.dashboard.close_popup()
        self.dashboard.unlock_protective_stop()
        return False, f"Simulation stopped with safety status: {status}"

    def describe_error(self, raw_error: str) -> str:
        """URSim's dashboard error text is already human-readable; pass through."""
        return raw_error
