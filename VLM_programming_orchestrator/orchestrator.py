# pylint: disable=W1203, W0718

"""VLM pipeline orchestrator.

Implements the full 12-step workflow as a state machine:
  1. RUN_GUI              -> get user input
  2. PLAN_TRAJECTORY      -> VLM trajectory planning
  3. GENERATE_CODE        -> Claude generates RAPID code (or fixes it)
  4. SAVE_MODULE          -> write .mod file
  5. VALIDATE_IN_ROBOTSTUDIO / FIX_SYNTAX_ERRORS
  6. SIMULATE             -> start sim + wait for completion
  7. HUMAN_REVIEW
  8. TRANSFER_TO_ROBOT    -> copy .mod to FTP share
  9. SEND_LOAD            -> LOAD + wait LOAD_OK
 10. SEND_START           -> START + wait DONE
 11. CLEANUP              -> done
 12. run_continuous loops back to step 1
"""

import time
import shutil
import logging
from pathlib import Path

from .config import Config, State
from .robot_connection import RobotConnection
from .robotstudio import RobotStudioAutomation
from .human_review import ask_human_review_with_feedback

# Late-imported (inside methods) to avoid pulling the GUI / VLM deps unless used:
#   from GUI_VLM_input import get_user_input
#   from vlm_stack_blocks import (plan_stacking_trajectory,
#                                 call_claude_for_robot_code,
#                                 call_claude_to_fix_code,
#                                 extract_code_block)

logger = logging.getLogger("orchestrator")


class VLMOrchestrator:
    """State machine driving GUI -> VLM -> LLM -> RobotStudio -> Robot."""

    def __init__(self, config: Config, gui_config: dict | None = None):
        self.config = config
        self.state = State.RUN_GUI if gui_config is None else State.PLAN_TRAJECTORY
        self.gui_config: dict | None = gui_config

        # Runtime artefacts
        self.stacking_plan_3d = None
        self.current_prompt: str = ""
        self.current_code: str = ""        # extracted/cleaned code
        self.current_raw_code: str = ""    # raw LLM reply (with fences)
        self.error_message: str = ""
        self.attempt: int = 0

        self.robot = RobotConnection(config)
        self.robotstudio = RobotStudioAutomation(config)

        config.local_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            config.ftp_shared_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            logger.warning(
                f"Could not ensure FTP shared dir {config.ftp_shared_dir}: {e}")

    # ------------------------------------------------------------------
    def run(self):
        logger.info("=" * 60)
        logger.info("VLM Orchestrator starting")
        logger.info("=" * 60)

        handlers = {
            State.RUN_GUI: self._state_run_gui,
            State.PLAN_TRAJECTORY: self._state_plan_trajectory,
            State.GENERATE_CODE: self._state_generate_code,
            State.SAVE_MODULE: self._state_save_module,
            State.VALIDATE_IN_ROBOTSTUDIO: self._state_validate,
            State.FIX_SYNTAX_ERRORS: self._state_fix_syntax,
            State.SIMULATE: self._state_simulate,
            State.HUMAN_REVIEW: self._state_human_review,
            State.TRANSFER_TO_ROBOT: self._state_transfer,
            State.SEND_LOAD: self._state_send_load,
            State.SEND_START: self._state_send_start,
            State.CLEANUP: self._state_cleanup,
        }

        while self.state not in (State.DONE, State.ERROR):
            logger.info(f"State: {self.state.name}")
            handler = handlers.get(self.state)
            if handler is None:
                logger.error(f"No handler for state {self.state}")
                self.state = State.ERROR
                break
            handler()

        if self.state == State.DONE:
            logger.info("Pipeline cycle completed successfully.")
        else:
            logger.error(
                f"Pipeline ended in error. Last error: {self.error_message}")

    # ---- State handlers ----------------------------------------------

    def _state_run_gui(self):
        from VLM.VLM_programming_orchestrator.GUI import get_user_input
        logger.info("Launching GUI for user input...")
        self.gui_config = get_user_input()
        if self.gui_config is None:
            logger.info("User cancelled the GUI.")
            self.state = State.DONE
            return
        logger.info(f"GUI configuration: {self.gui_config}")
        self.state = State.PLAN_TRAJECTORY

    def _state_plan_trajectory(self):
        from VLM.VLM_programming_orchestrator.vlm_stack_blocks import plan_stacking_trajectory
        if not self.config.api_key or not self.config.base_url:
            self.error_message = "API_KEY / BASE_URL not set in environment."
            logger.error(self.error_message)
            self.state = State.ERROR
            return
        logger.info("Running VLM trajectory planning...")
        result = plan_stacking_trajectory(
            self.config.api_key, self.config.base_url, config=self.gui_config
        )
        if not result:
            self.error_message = "VLM trajectory planning returned no result."
            self.state = State.ERROR
            return
        self.stacking_plan_3d = result["stacking_plan_3d"]
        self.state = State.GENERATE_CODE

    def _state_generate_code(self):
        from VLM.VLM_programming_orchestrator.vlm_stack_blocks import (
            call_claude_for_robot_code,
            call_claude_to_fix_code,
            extract_code_block,
        )
        self.attempt += 1
        if self.attempt > self.config.max_retry_attempts:
            logger.error("Max retry attempts reached.")
            self.state = State.ERROR
            return

        logger.info(
            f"Generating code (attempt {self.attempt}/{self.config.max_retry_attempts})..."
        )

        if self.attempt == 1:
            generated, prompt = call_claude_for_robot_code(
                self.config.api_key,
                self.config.base_url,
                self.stacking_plan_3d,
                robot_type=self.config.robot_type,
                code_language=self.config.code_language,
                task_description_file=self.config.task_description_file,
                best_practices_file=self.config.best_practices_file,
                model=self.config.llm_model,
            )
            self.current_prompt = prompt
        else:
            generated = call_claude_to_fix_code(
                self.config.api_key,
                self.config.base_url,
                self.current_raw_code or self.current_code,
                self.error_message,
                self.current_prompt,
                robot_type=self.config.robot_type,
                code_language=self.config.code_language,
                model=self.config.llm_model,
            )

        if not generated:
            self.error_message = "LLM returned empty code."
            logger.error(self.error_message)
            self.state = State.ERROR
            return

        self.current_raw_code = generated
        self.current_code = extract_code_block(generated)
        logger.info("Code generated successfully.")
        self.state = State.SAVE_MODULE

    def _state_save_module(self):
        local_path = self.config.local_output_dir / self.config.module_filename
        try:
            local_path.write_text(self.current_code, encoding="utf-8")
            logger.info(f"Module saved to {local_path}")
        except IOError as e:
            self.error_message = f"Failed to save module: {e}"
            logger.error(self.error_message)
            self.state = State.ERROR
            return

        if self.config.use_robotstudio_validation:
            self.state = State.VALIDATE_IN_ROBOTSTUDIO
        else:
            self.state = State.TRANSFER_TO_ROBOT

    def _state_validate(self):
        if not self.robotstudio.app:
            if not self.robotstudio.connect_to_robotstudio():
                logger.warning(
                    "RobotStudio not available. Skipping validation.")
                self.state = State.TRANSFER_TO_ROBOT
                return

        syntax_ok, error_msg = self.robotstudio.paste_code_and_apply(
            self.current_code)
        if syntax_ok:
            logger.info("Syntax validation passed.")
            self.state = State.SIMULATE
        else:
            logger.warning(f"Syntax errors: {error_msg}")
            self.error_message = error_msg
            self.state = State.FIX_SYNTAX_ERRORS

    def _state_fix_syntax(self):
        self.state = State.GENERATE_CODE

    def _state_simulate(self):
        if not self.robotstudio.start_simulation():
            self.error_message = "Failed to start simulation."
            logger.error(self.error_message)
            self.state = State.ERROR
            return

        success, errors = self.robotstudio.wait_for_simulation_complete()
        if success:
            logger.info("Simulation completed successfully.")
            if self.config.require_human_review:
                self.state = State.HUMAN_REVIEW
            else:
                self.state = State.TRANSFER_TO_ROBOT
        else:
            logger.warning(f"Simulation failed: {errors}")
            self.error_message = f"Simulation error: {errors}"
            self.state = State.FIX_SYNTAX_ERRORS

    def _state_human_review(self):
        approved, feedback = ask_human_review_with_feedback()
        if approved:
            logger.info("Human approved the simulation.")
            self.state = State.TRANSFER_TO_ROBOT
        else:
            logger.info(
                f"Human rejected the simulation. Feedback: {feedback!r}")
            self.error_message = (
                "Human review: the simulation behaviour was wrong.\n"
                f"User feedback: {feedback or '(none provided)'}"
            )
            self.state = State.GENERATE_CODE

    def _state_transfer(self):
        src = self.config.local_output_dir / self.config.module_filename
        dst = self.config.ftp_shared_dir / self.config.module_filename
        try:
            shutil.copy2(str(src), str(dst))
            logger.info(f"Module copied to FTP share: {dst}")
            self.state = State.SEND_LOAD
        except (IOError, OSError) as e:
            self.error_message = f"Failed to copy module to FTP share: {e}"
            logger.error(self.error_message)
            self.state = State.ERROR

    def _state_send_load(self):
        if not self.robot.sock:
            if not self.robot.connect():
                self.error_message = "Cannot connect to robot."
                logger.error(self.error_message)
                self.state = State.ERROR
                return

        success, message = self.robot.load_module(self.config.module_filename)
        if success:
            self.state = State.SEND_START
            return

        logger.error(f"Load failed: {message}")
        self.error_message = f"Robot load error: {message}"
        # Robot-side syntax/link errors -> try LLM fix
        if message in ("SYNTAX_ERROR", "LINK_ERROR"):
            self.state = State.GENERATE_CODE
        else:
            self.state = State.ERROR

    def _state_send_start(self):
        success, message = self.robot.start_execution()
        if success:
            self.state = State.CLEANUP
        else:
            self.error_message = f"Robot execution error: {message}"
            logger.error(self.error_message)
            self.state = State.ERROR

    def _state_cleanup(self):
        logger.info("Cycle complete. Cleaning up.")
        self.state = State.DONE

    # ------------------------------------------------------------------
    def shutdown(self):
        logger.info("Shutting down orchestrator...")
        self.robot.disconnect()


def run_continuous(config: Config, gui_config: dict | None = None):
    """Run the orchestrator in a loop. Each cycle re-opens the GUI.

    If `gui_config` is supplied, it is used only for the first cycle (useful
    for `--from-file` debugging); subsequent cycles re-run the GUI.
    """
    cycle = 0
    initial_gui = gui_config
    while True:
        cycle += 1
        logger.info("\n" + "=" * 60)
        logger.info(f"CYCLE {cycle}")
        logger.info("=" * 60)

        orch = VLMOrchestrator(
            config, gui_config=initial_gui if cycle == 1 else None)
        try:
            orch.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
            orch.shutdown()
            break
        except Exception as e:
            logger.exception(f"Unexpected error in cycle {cycle}: {e}")
        finally:
            orch.shutdown()

        logger.info(
            f"Waiting {config.cycle_delay_seconds}s before next cycle...")
        try:
            time.sleep(config.cycle_delay_seconds)
        except KeyboardInterrupt:
            logger.info("Interrupted while waiting.")
            break
