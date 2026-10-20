"""URSim automation via the Dashboard Server.

Implements the Simulator and ErrorDescriber interfaces (see robots/base.py)
for the UR robot backend, driving a local URSim Docker container.
"""
# pylint: disable=W1203, W0718

import logging
import shutil
import time

from ...config import Config
from .client_interface import RobotMessageListener
from .dashboard_client import DashboardClient

logger = logging.getLogger("orchestrator")


class URSimSimulator:
    """Copies the generated .script into URSim's shared programs folder and runs it."""

    # Polyscope acknowledges `play` before the program actually reaches PLAYING.
    PROGRAM_START_TIMEOUT = 15.0
    POLL_INTERVAL = 1.0
    # An abort popup can arrive just after programState flips to STOPPED.
    MESSAGE_GRACE_PERIOD = 1.0

    def __init__(self, config: Config):
        self.config = config
        self.dashboard = DashboardClient(
            config.ursim_ip, config.ur_dashboard_port, config.ur_dashboard_timeout)
        self.messages = RobotMessageListener(config.ursim_ip, config.ur_secondary_port)

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

        self.messages.start()

        response = self.dashboard.play()
        if not response.lower().startswith("starting program"):
            logger.error(f"URSim failed to start playback: {response}")
            return False

        if not self._wait_for_program_start():
            return False

        logger.info("URSim simulation started.")
        return True

    def _wait_for_program_start(self) -> bool:
        """Blocks until the program reports PLAYING, so completion polling can't see a stale STOPPED."""
        deadline = time.time() + self.PROGRAM_START_TIMEOUT
        while time.time() < deadline:
            try:
                state = self.dashboard.program_state()
                safety = self.dashboard.safety_status()
            except Exception as e:
                logger.error(f"Lost connection to URSim dashboard while starting: {e}")
                return False

            if safety != "NORMAL":
                logger.error(f"URSim refused to run, safety status is {safety}.")
                return False
            if state == "PLAYING":
                return True
            if state == "PAUSED":
                logger.error("URSim paused immediately after play (runtime error popup).")
                return False
            time.sleep(0.2)

        logger.error(
            f"URSim program never reached PLAYING within {self.PROGRAM_START_TIMEOUT}s "
            f"of `play`; the container program may not have loaded the generated script."
        )
        return False

    def stop_simulation(self) -> bool:
        if not self.dashboard.is_connected():
            self.messages.stop()
            return True
        try:
            self.dashboard.stop()
            logger.info("URSim simulation stopped.")
            return True
        except Exception as e:
            logger.error(f"Could not stop URSim simulation: {e}")
            return False
        finally:
            self.messages.stop()

    def wait_for_simulation_complete(self, timeout: float = 60 * 5) -> tuple[bool, str]:
        logger.info(f"Waiting for URSim simulation to complete (timeout={timeout}s)...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                state = self.dashboard.program_state()
                safety = self.dashboard.safety_status()
            except Exception as e:
                return False, f"Lost connection to URSim dashboard: {e}"

            elapsed = time.time() - start_time
            if safety != "NORMAL":
                logger.error(f"URSim safety status became {safety} after {elapsed:.1f}s.")
                return False, self._recover(f"Simulation stopped with safety status: {safety}")
            reported = self.messages.errors()
            if reported:
                logger.error(f"URSim reported a program error after {elapsed:.1f}s.")
                return False, self._recover(self._error_message(reported))
            if state == "PAUSED":
                logger.error(f"URSim program paused after {elapsed:.1f}s (runtime error popup).")
                return False, self._recover(
                    "Simulation paused on a runtime error popup; the program did not finish.")
            if state == "STOPPED":
                break
            time.sleep(self.POLL_INTERVAL)
        else:
            elapsed = time.time() - start_time
            logger.warning(f"URSim simulation did not complete within {elapsed:.1f}s (timeout).")
            self.stop_simulation()
            return False, "Simulation did not complete within the timeout period."

        elapsed = time.time() - start_time
        time.sleep(self.MESSAGE_GRACE_PERIOD)
        try:
            safety = self.dashboard.safety_status()
            mode = self.dashboard.robot_mode()
        except Exception as e:
            return False, f"Lost connection to URSim dashboard: {e}"

        if safety != "NORMAL":
            logger.error(f"URSim simulation ended with safety status {safety} after {elapsed:.1f}s.")
            return False, self._recover(f"Simulation stopped with safety status: {safety}")
        if mode != "RUNNING":
            logger.error(f"URSim ended with robot mode {mode} after {elapsed:.1f}s.")
            return False, self._recover(f"Simulation ended with robot mode: {mode}")

        reported = self.messages.errors()
        if reported:
            logger.error(f"URSim program aborted with an error after {elapsed:.1f}s.")
            return False, self._recover(self._error_message(reported))

        logger.info(f"URSim simulation completed successfully in {elapsed:.1f}s.")
        self.messages.stop()
        return True, "Simulation completed successfully."

    @staticmethod
    def _error_message(reported: list[str]) -> str:
        return "The program was aborted by the controller:\n" + "\n".join(reported)

    def _recover(self, message: str) -> str:
        """Clears popups/protective stops so the next retry attempt starts from a clean state."""
        try:
            self.dashboard.stop()
            self.dashboard.close_popup()
            self.dashboard.unlock_protective_stop()
        except Exception as e:
            logger.warning(f"Could not reset URSim after failure: {e}")
        self.messages.stop()
        return message

    def describe_error(self, raw_error: str) -> str:
        """URSim's dashboard error text is already human-readable; pass through."""
        return raw_error
